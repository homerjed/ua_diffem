"""Shared loading and reconstruction helpers for saved shear evaluations.

This module keeps the per-test scripts small by centralizing the common pieces:

- loading a saved run and checkpoint by `run_name`
- rebuilding the trained UA-DiffEM posterior and its observation channel
- regenerating a held-out synthetic weak-lensing dataset
- corrupting that dataset into noisy / masked shear observations
- sampling posterior reconstructions with the saved model

The most important user-facing convention lives here too: each evaluation writes
into a same-named folder under `runs/ua_diffem/<run_name>/`.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import argparse
import json

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from ua_diffem.diffem import DiffEMConfig, DiffEMState, e_step_reconstruct, load_training_state
from ua_diffem.shear import (
    ShearObservationChannel,
    apply_standardization,
    generate_lognormal_kappa_dataset,
    kaiser_squires_shear_numpy,
    reverse_standardize,
)
from ua_diffem.uncertainty_aware_flow import UAFlowConfig, build_ua_flow
from ua_diffem.utils import REPO_ROOT, make_optimizer


RUNS_ROOT = REPO_ROOT / "runs" / "ua_diffem"


@dataclass(frozen=True)
class LoadedShearRun:
    run_name: str
    run_dir: Path
    output_dir: Path
    checkpoint_path: Path
    config: dict[str, object]
    ua_config: UAFlowConfig
    diffem_config: DiffEMConfig
    channel: ShearObservationChannel
    state: DiffEMState
    target_stats: dict[str, float]
    gamma_scale: float
    data_shape: tuple[int, int, int]

    @property
    def flow(self):
        return self.state.sampling_flow


@dataclass(frozen=True)
class ShearTestBatch:
    kappa_true_phys: np.ndarray
    x_true_std: jax.Array
    condition: jax.Array
    mask: jax.Array
    gamma_true_phys: np.ndarray
    gamma_obs_phys: np.ndarray
    kappa_ks_std: jax.Array
    kappa_ks_phys: np.ndarray


@dataclass(frozen=True)
class ShearReconstruction:
    samples_std: np.ndarray
    samples_phys: np.ndarray
    variance_std: np.ndarray
    variance_phys: np.ndarray
    uncertainty_score: np.ndarray


def add_shared_eval_args(
    parser: argparse.ArgumentParser,
    *,
    default_test_size: int = 128,
    default_test_seed: int = 123,
    default_observation_seed: int = 456,
) -> argparse.ArgumentParser:
    """Attach the shared CLI arguments used by all shear evaluation scripts.

    Parameters
    ----------
    parser:
        Parser to extend in-place.
    default_test_size:
        Default number of held-out synthetic examples to generate.
    default_test_seed:
        Default seed for generating the clean convergence fields.
    default_observation_seed:
        Default seed for drawing the noisy / masked shear observations from the
        known forward model.

    Notes
    -----
    The arguments added here have the following meaning:

    `run_name`
        Name of the saved run under `runs/ua_diffem/`.
    `checkpoint_name`
        Optional checkpoint filename under `states/`. If omitted, the loader
        prefers `state.eqx` and otherwise falls back to the latest
        `state_em_*.eqx`.
    `test_size`
        Number of held-out synthetic examples to evaluate.
    `test_seed`
        Seed for generating the clean synthetic kappa fields. Changing this
        changes which latent structures are tested.
    `observation_seed`
        Seed for the corruption process that creates noisy / masked observed
        shear from the clean fields.
    `posterior_batch_size`
        Sampling batch size used when reconstructing the held-out set.
    `posterior_sample_steps`
        Number of solver steps used by the UA-flow posterior sampler.
    `posterior_solver`
        Sampling solver, currently `euler` or `heun`.
    `cfg_max_scale`
        Optional classifier-free guidance cap for posterior sampling.
    `ucg_scale`
        Optional uncertainty-conditioned guidance strength during sampling.
    `use_raw_flow`
        Load the raw trained flow instead of the EMA-smoothed flow when an EMA
        checkpoint is available.
    """
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--checkpoint_name", type=str, default=None)
    parser.add_argument("--test_size", type=int, default=default_test_size)
    parser.add_argument("--test_seed", type=int, default=default_test_seed)
    parser.add_argument("--observation_seed", type=int, default=default_observation_seed)
    parser.add_argument("--posterior_batch_size", type=int, default=None)
    parser.add_argument("--posterior_sample_steps", type=int, default=None)
    parser.add_argument("--posterior_solver", choices=("euler", "heun"), default=None)
    parser.add_argument("--cfg_max_scale", type=float, default=None)
    parser.add_argument("--ucg_scale", type=float, default=None)
    parser.add_argument("--use_raw_flow", action=argparse.BooleanOptionalAction, default=False)
    return parser


def _resolve_checkpoint_path(run_dir: Path, checkpoint_name: str | None) -> Path:
    states_dir = run_dir / "states"
    if checkpoint_name is not None:
        path = states_dir / checkpoint_name
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint {path} does not exist.")
        return path

    default_path = states_dir / "state.eqx"
    if default_path.exists():
        return default_path

    candidates = sorted(states_dir.glob("state_em_*.eqx"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoints found under {states_dir}.")
    return candidates[-1]


def _load_config(run_dir: Path) -> dict[str, object]:
    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing shear run config at {config_path}.")
    return json.loads(config_path.read_text())


def _build_configs(config: dict[str, object]) -> tuple[UAFlowConfig, DiffEMConfig]:
    ua_payload = dict(config["ua_flow"])
    if "dim_mults" in ua_payload:
        ua_payload["dim_mults"] = tuple(ua_payload["dim_mults"])
    ua_config = UAFlowConfig(**ua_payload)

    diffem_payload = dict(config["diffem"])
    diffem_config = DiffEMConfig(**diffem_payload)
    return ua_config, diffem_config


def _build_channel(config: dict[str, object]) -> tuple[ShearObservationChannel, dict[str, float], float]:
    dataset_cfg = dict(config["dataset"])
    target_stats = dict(dataset_cfg["target_stats"])
    gamma_scale = float(dataset_cfg["gamma_scale"])
    channel = ShearObservationChannel(
        image_size=int(dataset_cfg["image_size"]),
        noise_std=float(dataset_cfg["noise_std"]),
        mask_fraction=float(dataset_cfg["mask_fraction"]),
        mask_size=int(dataset_cfg["mask_size"]),
        num_masks=int(dataset_cfg["num_masks"]),
        target_mean=float(target_stats["mean"]),
        target_std=float(target_stats["std"]),
        gamma_scale=gamma_scale,
    )
    return channel, target_stats, gamma_scale


def load_shear_run(
    *,
    run_name: str,
    output_name: str,
    checkpoint_name: str | None = None,
    use_raw_flow: bool = False,
) -> LoadedShearRun:
    """Load a saved shear run, rebuild its model, and prepare an output folder.

    Parameters
    ----------
    run_name:
        Name of the saved run under `runs/ua_diffem/`.
    output_name:
        Name of the evaluation-specific output directory to create inside the
        run directory.
    checkpoint_name:
        Optional specific checkpoint filename inside `states/`.
    use_raw_flow:
        If `True`, ignore the saved EMA model and evaluate the raw flow
        parameters instead.
    """
    run_dir = RUNS_ROOT / run_name
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory {run_dir} does not exist.")

    output_dir = run_dir / output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    config = _load_config(run_dir)
    ua_config, diffem_config = _build_configs(config)
    channel, target_stats, gamma_scale = _build_channel(config)
    checkpoint_path = _resolve_checkpoint_path(run_dir, checkpoint_name)

    flow = build_ua_flow(ua_config, key=jr.key(0))
    opt = make_optimizer(diffem_config)
    opt_state = opt.init(eqx.filter(flow, eqx.is_array))
    flow_ema = deepcopy(flow) if diffem_config.use_ema else None

    state = load_training_state(
        checkpoint_path,
        flow,
        opt_state,
        flow_ema_like=flow_ema,
    )
    if use_raw_flow:
        state = DiffEMState(
            flow=state.flow,
            opt_state=state.opt_state,
            flow_ema=None,
            em_step=state.em_step,
            m_step=state.m_step,
        )

    return LoadedShearRun(
        run_name=run_name,
        run_dir=run_dir,
        output_dir=output_dir,
        checkpoint_path=checkpoint_path,
        config=config,
        ua_config=ua_config,
        diffem_config=diffem_config,
        channel=channel,
        state=state,
        target_stats=target_stats,
        gamma_scale=gamma_scale,
        data_shape=(1, channel.image_size, channel.image_size),
    )


def generate_test_batch(
    loaded: LoadedShearRun,
    *,
    test_size: int,
    test_seed: int,
    observation_seed: int,
) -> ShearTestBatch:
    """Generate a fresh held-out synthetic shear dataset for evaluation.

    The clean kappa fields are sampled from the same log-normal synthetic model
    family used by training, but with a user-controlled seed so the evaluation
    set can be changed or kept fixed. The known corruption channel is then used
    to draw noisy / masked observed shear and a Kaiser-Squires bootstrap
    baseline.
    """
    dataset_cfg = dict(loaded.config["dataset"])
    kappa_true_phys = generate_lognormal_kappa_dataset(
        n_samples=test_size,
        image_size=int(dataset_cfg["image_size"]),
        spectral_index=float(dataset_cfg["spectral_index"]),
        gaussian_sigma=float(dataset_cfg["gaussian_sigma"]),
        seed=test_seed,
    )
    x_true_std = jnp.asarray(apply_standardization(kappa_true_phys, loaded.target_stats))
    observed = loaded.channel.sample(jr.key(observation_seed), x_true_std)

    gamma_true_phys = kaiser_squires_shear_numpy(kappa_true_phys[:, 0])
    gamma_obs_phys = np.asarray(jax.device_get(observed.condition)) * loaded.gamma_scale
    kappa_ks_std = loaded.channel.bootstrap_reconstruction(observed.condition)
    kappa_ks_phys = reverse_standardize(np.asarray(jax.device_get(kappa_ks_std)), loaded.target_stats)

    return ShearTestBatch(
        kappa_true_phys=kappa_true_phys.astype(np.float32),
        x_true_std=x_true_std,
        condition=observed.condition,
        mask=observed.mask,
        gamma_true_phys=gamma_true_phys.astype(np.float32),
        gamma_obs_phys=gamma_obs_phys.astype(np.float32),
        kappa_ks_std=kappa_ks_std,
        kappa_ks_phys=kappa_ks_phys.astype(np.float32),
    )


def reconstruct_test_batch(
    loaded: LoadedShearRun,
    batch: ShearTestBatch,
    *,
    posterior_batch_size: int | None = None,
    posterior_sample_steps: int | None = None,
    posterior_solver: str | None = None,
    cfg_max_scale: float | None = None,
    ucg_scale: float | None = None,
) -> ShearReconstruction:
    """Run posterior reconstruction on a held-out evaluation batch.

    Returns both standardized-space outputs, which match the model's internal
    training coordinates, and physical-space outputs, which are easier to
    interpret in plots and summary statistics.
    """
    posterior_batch_size = (
        loaded.diffem_config.posterior_batch_size
        if posterior_batch_size is None
        else posterior_batch_size
    )
    posterior_sample_steps = (
        loaded.diffem_config.posterior_sample_steps
        if posterior_sample_steps is None
        else posterior_sample_steps
    )
    posterior_solver = (
        loaded.diffem_config.posterior_solver
        if posterior_solver is None
        else posterior_solver
    )
    cfg_max_scale = loaded.diffem_config.cfg_max_scale if cfg_max_scale is None else cfg_max_scale
    ucg_scale = loaded.diffem_config.ucg_scale if ucg_scale is None else ucg_scale

    recon = e_step_reconstruct(
        loaded.flow,
        batch.condition,
        key=jr.key(loaded.state.em_step + 17),
        data_shape=loaded.data_shape,
        batch_size=posterior_batch_size,
        sample_steps=posterior_sample_steps,
        solver=posterior_solver,
        cfg_max_scale=cfg_max_scale,
        ucg_scale=ucg_scale,
    )
    samples_std = np.asarray(jax.device_get(recon.samples)).astype(np.float32)
    variance_std = np.asarray(jax.device_get(recon.variance)).astype(np.float32)
    samples_phys = reverse_standardize(samples_std, loaded.target_stats).astype(np.float32)
    variance_phys = (variance_std * (loaded.target_stats["std"] ** 2)).astype(np.float32)
    uncertainty_score = np.asarray(jax.device_get(recon.uncertainty_score)).astype(np.float32)

    return ShearReconstruction(
        samples_std=samples_std,
        samples_phys=samples_phys,
        variance_std=variance_std,
        variance_phys=variance_phys,
        uncertainty_score=uncertainty_score,
    )


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def write_eval_metadata(
    loaded: LoadedShearRun,
    path: Path,
    *,
    args: argparse.Namespace,
) -> None:
    payload: dict[str, object] = {
        "run_name": loaded.run_name,
        "run_dir": str(loaded.run_dir),
        "checkpoint_path": str(loaded.checkpoint_path),
        "used_ema": loaded.state.flow_ema is not None,
    }
    for key, value in vars(args).items():
        if isinstance(value, Path):
            payload[key] = str(value)
        else:
            payload[key] = value
    write_json(path, payload)
