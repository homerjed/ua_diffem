from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from pathlib import Path
import argparse
import json
import sys

_REPO_ROOT = Path(__file__).resolve().parents[2]
_LINKED_FLOW_ROOT = _REPO_ROOT.parent / "linked_flow"
if __package__ in (None, ""):
    sys.path.insert(0, str(_REPO_ROOT))
if _LINKED_FLOW_ROOT.exists():
    sys.path.insert(0, str(_LINKED_FLOW_ROOT))


import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np

from ua_diffem.diffem import (
    DiffEMConfig,
    DiffEMState,
    e_step_reconstruct,
    m_step_train,
    save_training_state,
)
from ua_diffem.shear import reverse_standardize, standardize_targets
from ua_diffem.shear.glass_maps import (
    SphericalShearObservationChannel,
    default_lmax,
    generate_glass_lognormal_kappa_dataset,
    reorder_healpix_array,
    spherical_shear_numpy,
)
from ua_diffem.uncertainty_aware_flow import UAFlowConfig, build_ua_flow
from ua_diffem.utils import (
    count_parameters,
    make_data_parallel_sharding,
    make_optimizer,
    shard_replicated_tree,
)


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[ua_diffem.spherical {timestamp}] {message}", flush=True)


def parse_dim_mults(value: str) -> tuple[int, ...]:
    parts = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not parts:
        raise ValueError("`dim_mults` must contain at least one integer.")
    return parts


def _ring_batch(maps: np.ndarray, *, nside: int) -> np.ndarray:
    return reorder_healpix_array(
        np.asarray(maps),
        nside=nside,
        input_nest=True,
        output_nest=False,
    )


def save_spherical_preview(
    run_dir: Path,
    *,
    em_idx: int,
    nside: int,
    clean: np.ndarray,
    observed_condition: np.ndarray,
    kaiser_squires: np.ndarray,
    posterior: np.ndarray,
    uncertainty: np.ndarray,
    target_stats: dict[str, float],
    gamma_scale: float,
    max_columns: int = 4,
) -> None:
    import healpy as hp

    ncols = min(int(max_columns), int(clean.shape[0]))
    clean_phys = clean[:ncols]
    kaiser_squires_phys = reverse_standardize(kaiser_squires[:ncols], target_stats)
    posterior_phys = reverse_standardize(posterior[:ncols], target_stats)

    clean_ring = _ring_batch(clean_phys[:, 0], nside=nside)
    ks_ring = _ring_batch(kaiser_squires_phys[:, 0], nside=nside)
    posterior_ring = _ring_batch(posterior_phys[:, 0], nside=nside)

    kappa_range = np.concatenate([clean_ring, ks_ring, posterior_ring], axis=0)
    kappa_vmin = float(np.nanquantile(kappa_range, 0.01))
    kappa_vmax = float(np.nanquantile(kappa_range, 0.99))

    gamma = observed_condition[:ncols, :2] * gamma_scale
    mask = observed_condition[:ncols, 2]
    gamma_mag = np.sqrt(np.sum(gamma**2, axis=1))
    gamma_mag_ring = _ring_batch(gamma_mag, nside=nside)
    mask_ring = _ring_batch(mask, nside=nside) > 0.5
    gamma_mag_ring = gamma_mag_ring.copy()
    gamma_mag_ring[~mask_ring] = np.nan
    gamma_vmax = float(np.nanquantile(gamma_mag_ring, 0.98))
    if not np.isfinite(gamma_vmax) or gamma_vmax <= 0.0:
        gamma_vmax = 1.0

    uncertainty = np.sqrt(np.maximum(uncertainty[:ncols], 0.0)).mean(axis=1)
    uncertainty_ring = _ring_batch(uncertainty, nside=nside)
    uncertainty_vmax = float(np.nanquantile(uncertainty_ring, 0.99))
    if not np.isfinite(uncertainty_vmax) or uncertainty_vmax <= 0.0:
        uncertainty_vmax = 1.0

    rows = [
        (clean_ring, "truth kappa", "magma", kappa_vmin, kappa_vmax),
        (gamma_mag_ring, "|gamma obs|", "viridis", 0.0, gamma_vmax),
        (ks_ring, "Kaiser-Squires", "magma", kappa_vmin, kappa_vmax),
        (posterior_ring, "reconstruction", "magma", kappa_vmin, kappa_vmax),
        (uncertainty_ring, "uncertainty", "magma", 0.0, uncertainty_vmax),
    ]

    nrows = len(rows)
    fig = plt.figure(figsize=(3.2 * ncols, 8.0), dpi=220)
    for row_idx, (row_maps, label, cmap, vmin, vmax) in enumerate(rows):
        for col_idx in range(ncols):
            hp.mollview(
                row_maps[col_idx],
                fig=fig.number,
                sub=(nrows, ncols, row_idx * ncols + col_idx + 1),
                cmap=cmap,
                min=vmin,
                max=vmax,
                cbar=False,
                title="",
                badcolor="0.55",
            )
        fig.text(
            0.04,
            0.91 - row_idx * (0.82 / max(1, nrows - 1)),
            label,
            ha="right",
            va="center",
            fontsize=9,
        )

    fig.subplots_adjust(left=0.08, right=0.99, top=0.98, bottom=0.02, hspace=0.02, wspace=0.02)
    fig.savefig(run_dir / f"reconstructions_em_{em_idx:02d}.png", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def save_em_loss_plot(
    run_dir: Path,
    losses_by_em: list[list[float]],
    validation_steps_by_em: list[list[int]],
    validation_losses_by_em: list[list[float]],
) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 3.2), dpi=220)
    colors = ("tab:red", "tab:blue")
    start = 0

    for em_idx, losses in enumerate(losses_by_em):
        if not losses:
            continue
        losses_arr = np.asarray(losses, dtype=np.float32)
        steps = np.arange(start, start + len(losses_arr))
        color = colors[em_idx % len(colors)]
        ax.plot(steps, losses_arr, color=color, linewidth=1.2)

        if em_idx < len(validation_losses_by_em) and validation_losses_by_em[em_idx]:
            val_steps = start + np.asarray(validation_steps_by_em[em_idx], dtype=np.int32)
            val_losses = np.asarray(validation_losses_by_em[em_idx], dtype=np.float32)
            ax.scatter(
                val_steps,
                val_losses,
                color=color,
                edgecolor="black",
                linewidth=0.35,
                s=18,
                marker="o",
                zorder=3,
            )

        if em_idx > 0:
            ax.axvline(start, color="0.2", linewidth=0.8, linestyle="--", alpha=0.55)
        start += len(losses_arr)

    ax.set_xlabel("M-step gradient step")
    ax.set_ylabel("UA-flow loss")
    ax.set_yscale("log")
    ax.set_title(f"M-step train and validation losses over {len(losses_by_em)} EM iterations")
    ax.grid(alpha=0.2, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(run_dir / "losses_by_em.png", bbox_inches="tight")
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DiffEM + uncertainty-aware flow matching for spherical weak-lensing maps."
    )
    parser.add_argument("--run_dir", type=Path, default=Path("runs/ua_diffem/spherical_basic"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train_size", type=int, default=1024)
    parser.add_argument("--preview_size", type=int, default=4)
    parser.add_argument("--dataset_progress_every", type=int, default=25)

    parser.add_argument("--nside", type=int, default=16)
    parser.add_argument("--lmax", type=int, default=0)
    parser.add_argument("--glass_amplitude", type=float, default=0.9)
    parser.add_argument("--spectral_index", type=float, default=1.25)
    parser.add_argument("--glass_damping", type=float, default=0.006)
    parser.add_argument("--gaussian_sigma", type=float, default=0.8)
    parser.add_argument("--noise_std", type=float, default=0.55)
    parser.add_argument("--mask_fraction", type=float, default=1.0)
    parser.add_argument("--hole_radius_deg", type=float, default=5.5)
    parser.add_argument("--num_holes", type=int, default=22)

    parser.add_argument("--em_steps", type=int, default=500)
    parser.add_argument("--m_steps_per_em", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--posterior_batch_size", type=int, default=4)
    parser.add_argument("--posterior_sample_steps", type=int, default=15)
    parser.add_argument("--posterior_solver", choices=("euler", "heun"), default="euler")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--no_bootstrap_first_e_step", action="store_true")
    parser.add_argument("--m_step_early_stopping", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--m_step_min_steps", type=int, default=50)
    parser.add_argument("--m_step_patience", type=int, default=50)
    parser.add_argument("--m_step_min_delta", type=float, default=1e-4)
    parser.add_argument("--m_step_validation_fraction", type=float, default=0.1)
    parser.add_argument("--m_step_validation_freq", type=int, default=10)
    parser.add_argument("--m_step_validation_batches", type=int, default=2)
    parser.add_argument("--use_ema", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ema_rate", type=float, default=0.999)

    parser.add_argument("--model_name", choices=("spherical_unet",), default="spherical_unet")
    parser.add_argument("--model_dim", type=int, default=32)
    parser.add_argument("--dim_mults", type=str, default="1,2")
    parser.add_argument("--time_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--chebyshev_order", type=int, default=3)
    parser.add_argument("--data_parallel_sharding", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--covariance_mode", choices=("zero", "jvp"), default="zero")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    states_dir = args.run_dir / "states"
    reconstructions_dir = args.run_dir / "reconstructions"
    out_dir = args.run_dir / "out"
    states_dir.mkdir(parents=True, exist_ok=True)
    reconstructions_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.train_size < 2:
        raise ValueError("`train_size` must be at least 2 for M-step validation.")
    if args.preview_size < 1:
        raise ValueError("`preview_size` must be at least 1.")
    if args.nside < 1:
        raise ValueError("`nside` must be positive.")
    if args.em_steps < 1 or args.m_steps_per_em < 1:
        raise ValueError("`em_steps` and `m_steps_per_em` must be at least 1.")
    if args.batch_size < 1 or args.posterior_batch_size < 1:
        raise ValueError("batch sizes must be at least 1.")
    if args.posterior_sample_steps < 1:
        raise ValueError("`posterior_sample_steps` must be at least 1.")
    if not 0.0 < args.m_step_validation_fraction < 1.0:
        raise ValueError("`m_step_validation_fraction` must be in (0, 1).")
    if args.m_step_validation_freq < 1 or args.m_step_validation_batches < 1:
        raise ValueError("validation frequency and batch count must be at least 1.")
    if args.m_step_min_steps < 1 or args.m_step_patience < 1:
        raise ValueError("early-stopping step and patience values must be at least 1.")
    if args.m_step_min_delta < 0.0:
        raise ValueError("`m_step_min_delta` must be non-negative.")

    local_devices = jax.local_devices()
    log(f"JAX backend={jax.default_backend()} devices={[str(d) for d in local_devices]}")
    sharding = make_data_parallel_sharding() if args.data_parallel_sharding else None
    if sharding is not None:
        if args.batch_size % len(local_devices) != 0:
            raise ValueError(
                f"`batch_size`={args.batch_size} must be divisible by "
                f"local_device_count={len(local_devices)} when data-parallel sharding is enabled."
            )
        log(f"Using data-parallel sharding across {len(local_devices)} local devices.")
    elif args.data_parallel_sharding:
        log("Data-parallel sharding requested, but only one local device is visible.")

    lmax = default_lmax(args.nside) if args.lmax <= 0 else int(args.lmax)
    dataset_progress_every = max(0, int(args.dataset_progress_every))
    log(
        "Generating GLASS log-normal convergence maps on the HEALPix sphere "
        f"(train_size={args.train_size}, nside={args.nside}, lmax={lmax})."
    )
    kappa_train_np = generate_glass_lognormal_kappa_dataset(
        n_samples=args.train_size,
        nside=args.nside,
        lmax=lmax,
        seed=args.seed,
        amplitude=args.glass_amplitude,
        spectral_index=args.spectral_index,
        damping=args.glass_damping,
        gaussian_sigma=args.gaussian_sigma,
        nest=True,
        progress_every=dataset_progress_every,
        progress=log,
    )
    x_train_np, target_stats = standardize_targets(kappa_train_np)
    log("Computing spherical shear maps for gamma scaling and initial observations.")
    gamma_true = spherical_shear_numpy(
        kappa_train_np[:, 0],
        nside=args.nside,
        lmax=lmax,
        input_nest=True,
        output_nest=True,
        progress_every=dataset_progress_every,
        progress=log,
    )
    gamma_scale = float(np.std(gamma_true) + args.noise_std + 1e-6)
    x_train = jnp.asarray(x_train_np)
    data_shape = tuple(x_train.shape[1:])
    npix = int(data_shape[-1])

    channel = SphericalShearObservationChannel(
        nside=args.nside,
        lmax=lmax,
        noise_std=args.noise_std,
        mask_fraction=args.mask_fraction,
        hole_radius_deg=args.hole_radius_deg,
        num_holes=args.num_holes,
        target_mean=target_stats["mean"],
        target_std=target_stats["std"],
        gamma_scale=gamma_scale,
        nest=True,
        progress_every=dataset_progress_every,
        progress=log,
    )

    key = jr.key(args.seed)
    key, key_obs, key_model = jr.split(key, 3)
    log("Sampling initial spherical shear noise and masks.")
    observed_train = channel.sample_from_shear(key_obs, gamma_true)

    ua_config = UAFlowConfig(
        image_size=npix,
        nside=args.nside,
        channels=1,
        cond_channels=channel.condition_channels,
        model_name=args.model_name,
        model_dim=args.model_dim,
        dim_mults=parse_dim_mults(args.dim_mults),
        time_dim=args.time_dim,
        dropout=args.dropout,
        spherical_chebyshev_order=args.chebyshev_order,
        covariance_mode=args.covariance_mode,
    )
    diffem_config = DiffEMConfig(
        em_steps=args.em_steps,
        m_steps_per_em=args.m_steps_per_em,
        batch_size=args.batch_size,
        posterior_batch_size=args.posterior_batch_size,
        posterior_sample_steps=args.posterior_sample_steps,
        posterior_solver=args.posterior_solver,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        bootstrap_first_e_step=not args.no_bootstrap_first_e_step,
        m_step_early_stopping=args.m_step_early_stopping,
        m_step_min_steps=args.m_step_min_steps,
        m_step_patience=args.m_step_patience,
        m_step_min_delta=args.m_step_min_delta,
        m_step_validation_fraction=args.m_step_validation_fraction,
        m_step_validation_freq=args.m_step_validation_freq,
        m_step_validation_batches=args.m_step_validation_batches,
        use_ema=args.use_ema,
        ema_rate=args.ema_rate,
    )

    config_payload = {
        "dataset": {
            "nside": args.nside,
            "npix": npix,
            "lmax": lmax,
            "glass_amplitude": args.glass_amplitude,
            "spectral_index": args.spectral_index,
            "glass_damping": args.glass_damping,
            "gaussian_sigma": args.gaussian_sigma,
            "noise_std": args.noise_std,
            "mask_fraction": args.mask_fraction,
            "hole_radius_deg": args.hole_radius_deg,
            "num_holes": args.num_holes,
            "dataset_progress_every": dataset_progress_every,
            "target_stats": target_stats,
            "gamma_scale": gamma_scale,
            "healpix_order": "NESTED",
        },
        "ua_flow": ua_config.__dict__,
        "diffem": diffem_config.__dict__,
        "runtime": {
            "data_parallel_sharding": sharding is not None,
            "local_device_count": len(local_devices),
        },
        "n_parameters": None,
    }

    flow = build_ua_flow(ua_config, key=key_model)
    opt = make_optimizer(diffem_config)
    opt_state = opt.init(eqx.filter(flow, eqx.is_array))
    flow_ema = deepcopy(flow) if diffem_config.use_ema else None
    flow = shard_replicated_tree(flow, sharding)
    flow_ema = shard_replicated_tree(flow_ema, sharding)
    opt_state = shard_replicated_tree(opt_state, sharding)
    state = DiffEMState(flow=flow, opt_state=opt_state, flow_ema=flow_ema)

    n_parameters = count_parameters(flow.model)
    config_payload["n_parameters"] = int(n_parameters)
    (args.run_dir / "config.json").write_text(json.dumps(config_payload, indent=2, sort_keys=True))
    log(f"Posterior model parameters: {n_parameters:.3e}")

    all_losses: list[float] = []
    losses_by_em: list[list[float]] = []
    validation_steps_by_em: list[list[int]] = []
    validation_losses_by_em: list[list[float]] = []
    preview_size = min(args.preview_size, args.train_size)
    preview_cond = observed_train.condition[:preview_size]
    preview_clean = kappa_train_np[:preview_size]

    for em_idx in range(diffem_config.em_steps):
        log(f"EM {em_idx + 1}/{diffem_config.em_steps}: E-step.")
        bootstrap = channel if diffem_config.bootstrap_first_e_step and em_idx == 0 else None
        key, key_e = jr.split(key)
        recon = e_step_reconstruct(
            state.sampling_flow,
            observed_train.condition,
            key=key_e,
            data_shape=data_shape,
            batch_size=diffem_config.posterior_batch_size,
            sample_steps=diffem_config.posterior_sample_steps,
            solver=diffem_config.posterior_solver,
            bootstrap_channel=bootstrap,
        )

        log(
            f"EM {em_idx + 1}/{diffem_config.em_steps}: M-step. "
            f"First corrupting {recon.samples.shape[0]} reconstructions with spherical shear, "
            f"then training for {diffem_config.m_steps_per_em} steps."
        )
        key, key_m = jr.split(key)
        m_result = m_step_train(
            state,
            recon.samples,
            channel,
            diffem_config,
            key=key_m,
            opt=opt,
            sharding=sharding,
        )
        state = m_result.state
        losses = m_result.train_losses
        all_losses.extend(losses)
        losses_by_em.append(losses)
        validation_steps_by_em.append(m_result.validation_steps)
        validation_losses_by_em.append(m_result.validation_losses)
        val_message = (
            f", last validation={m_result.validation_losses[-1]:.6f}"
            if m_result.validation_losses
            else ""
        )
        stop_message = " early-stopped" if m_result.stopped_early else ""
        log(
            f"EM {em_idx + 1}: ran {len(losses)}/{diffem_config.m_steps_per_em} "
            f"M-step updates{stop_message}, last train={losses[-1]:.6f}{val_message}"
        )

        key, key_preview = jr.split(key)
        preview = state.sampling_flow.sample(
            key_preview,
            batch_size=preview_size,
            data_shape=data_shape,
            cond=preview_cond,
            steps=diffem_config.posterior_sample_steps,
            solver=diffem_config.posterior_solver,
        )
        preview_ks = channel.bootstrap_reconstruction(preview_cond)
        save_spherical_preview(
            reconstructions_dir,
            em_idx=em_idx + 1,
            nside=args.nside,
            clean=preview_clean,
            observed_condition=np.asarray(jax.device_get(preview_cond)),
            kaiser_squires=np.asarray(jax.device_get(preview_ks)),
            posterior=np.asarray(jax.device_get(preview.samples)),
            uncertainty=np.asarray(jax.device_get(preview.variance)),
            target_stats=target_stats,
            gamma_scale=gamma_scale,
            max_columns=preview_size,
        )

        save_training_state(states_dir / f"state_em_{em_idx + 1:02d}.eqx", state)
        np.save(out_dir / "losses.npy", np.asarray(all_losses, dtype=np.float32))
        np.savez(
            out_dir / "losses_by_em.npz",
            **{
                f"em_{idx + 1:02d}": np.asarray(em_losses, dtype=np.float32)
                for idx, em_losses in enumerate(losses_by_em)
            },
        )
        np.savez(
            out_dir / "validation_losses_by_em.npz",
            **{
                f"em_{idx + 1:02d}_steps": np.asarray(validation_steps_by_em[idx], dtype=np.int32)
                for idx in range(len(validation_steps_by_em))
            }
            | {
                f"em_{idx + 1:02d}_losses": np.asarray(validation_losses_by_em[idx], dtype=np.float32)
                for idx in range(len(validation_losses_by_em))
            },
        )
        save_em_loss_plot(
            out_dir,
            losses_by_em,
            validation_steps_by_em,
            validation_losses_by_em,
        )

    save_training_state(states_dir / "state.eqx", state)
    log(f"Finished. Outputs are in {args.run_dir}.")


if __name__ == "__main__":
    main()
