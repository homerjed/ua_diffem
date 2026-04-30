"""Generate the figures used by the weak-lensing introduction notes.

The script intentionally uses the same synthetic dataset and Kaiser-Squires
helpers as the shear training code, so the plots describe the project data
rather than a separate toy example.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

PLOT_DEPS_AVAILABLE = True
PLOT_DEPS_MESSAGE = ""

try:
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.collections import LineCollection
except ModuleNotFoundError as exc:
    PLOT_DEPS_AVAILABLE = False
    PLOT_DEPS_MESSAGE = (
        "Missing plotting dependency. Run this script in the project scientific "
        "Python environment with numpy and matplotlib installed."
    )


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

if PLOT_DEPS_AVAILABLE:
    try:
        from ua_diffem.shear import (  # noqa: E402
            generate_lognormal_kappa_dataset,
            kaiser_squires_shear_numpy,
            ks_kernel_numpy,
        )
    except ModuleNotFoundError as exc:
        PLOT_DEPS_AVAILABLE = False
        PLOT_DEPS_MESSAGE = (
            "Missing project scientific dependency while importing ua_diffem.shear. "
            "Run this script in the full project Python environment."
        )


REPO_ROOT = Path(__file__).resolve().parents[2]
NOTES_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = NOTES_DIR / "images"

DEFAULT_DATASET_CONFIG = {
    "image_size": 64,
    "spectral_index": 2.5,
    "gaussian_sigma": 0.8,
    "noise_std": 0.50,
    "mask_fraction": 0.8,
    "mask_size": 16,
    "num_masks": 3,
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create weak-lensing note figures.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional path to a shear run config.json. If omitted, a local run config is auto-detected.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where PNG figures are written.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--observation_seed", type=int, default=11)
    parser.add_argument("--n_samples", type=int, default=8)
    return parser


def candidate_configs() -> list[Path]:
    return [
        REPO_ROOT / "runs" / "ua_diffem" / "my_shear_run2" / "config.json",
        REPO_ROOT / "runs" / "ua_diffem" / "my_shear_run3" / "config.json",
        REPO_ROOT / "runs" / "ua_diffem" / "my_shear_run" / "config.json",
        REPO_ROOT / "runs" / "ua_diffem" / "shear_basic" / "config.json",
    ]


def copy_existing_run_outputs(output_dir: Path) -> None:
    """Fallback for minimal environments without plotting dependencies."""
    output_dir.mkdir(parents=True, exist_ok=True)
    run_root = REPO_ROOT / "runs" / "ua_diffem" / "my_shear_run2"
    recon_dir = run_root / "reconstructions"

    def recon_step(path: Path) -> int:
        try:
            return int(path.stem.rsplit("_", 1)[-1])
        except ValueError:
            return -1

    recon_candidates = sorted(recon_dir.glob("reconstructions_em_*.png"), key=recon_step)
    preview_path = recon_candidates[-1] if recon_candidates else None
    sources = {
        "visual_examples.png": run_root / "visual_examples" / "visual_examples.png",
        "training_preview.png": preview_path,
        "weak_lensing_statistics.png": run_root
        / "weak_lensing_statistics"
        / "weak_lensing_statistics.png",
        "metric_distributions.png": run_root
        / "quantitative_reconstruction"
        / "metric_distributions.png",
    }

    copied: dict[str, str] = {}
    for target_name, source_path in sources.items():
        if source_path is None or not source_path.exists():
            continue
        target_path = output_dir / target_name
        shutil.copyfile(source_path, target_path)
        copied[target_name] = str(source_path)

    if not copied:
        raise SystemExit(
            f"{PLOT_DEPS_MESSAGE} No existing shear run output figures were found "
            f"under {run_root} to copy as a fallback."
        )

    metadata = {
        "mode": "copied_existing_run_outputs",
        "message": PLOT_DEPS_MESSAGE,
        "copied": copied,
    }
    (output_dir / "figure_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    print(PLOT_DEPS_MESSAGE)
    print(f"Copied existing shear run figures to {output_dir}")


def load_dataset_config(config_path: Path | None) -> tuple[dict[str, float], Path | None]:
    if config_path is not None:
        payload = json.loads(config_path.read_text())
        return dict(payload["dataset"]), config_path

    for candidate in candidate_configs():
        if candidate.exists():
            payload = json.loads(candidate.read_text())
            return dict(payload["dataset"]), candidate

    return dict(DEFAULT_DATASET_CONFIG), None


def make_rectangular_mask(
    *,
    n_samples: int,
    image_size: int,
    mask_fraction: float,
    mask_size: int,
    num_masks: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mask = np.ones((n_samples, 1, image_size, image_size), dtype=np.float32)
    if mask_fraction <= 0.0 or num_masks <= 0:
        return mask

    mask_size = max(1, min(image_size, int(mask_size)))
    max_offset = max(1, image_size - mask_size + 1)
    for _ in range(num_masks):
        top = rng.integers(0, max_offset, size=n_samples)
        left = rng.integers(0, max_offset, size=n_samples)
        apply_mask = rng.random(n_samples) < mask_fraction
        for idx in range(n_samples):
            if apply_mask[idx]:
                y0 = int(top[idx])
                x0 = int(left[idx])
                mask[idx, :, y0 : y0 + mask_size, x0 : x0 + mask_size] = 0.0
    return mask


def kaiser_squires_inverse_numpy(gamma: np.ndarray) -> np.ndarray:
    image_size = int(gamma.shape[-1])
    d1, d2 = ks_kernel_numpy(image_size)
    gamma1_hat = np.fft.fft2(gamma[:, 0], axes=(-2, -1))
    gamma2_hat = np.fft.fft2(gamma[:, 1], axes=(-2, -1))
    kappa_hat = d1 * gamma1_hat + d2 * gamma2_hat
    return np.fft.ifft2(kappa_hat, axes=(-2, -1)).real[:, None].astype(np.float32)


def quantile_limits(*arrays: np.ndarray, q_low: float = 0.01, q_high: float = 0.99) -> tuple[float, float]:
    values = np.concatenate([np.ravel(array) for array in arrays])
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin = float(np.quantile(finite, q_low))
    vmax = float(np.quantile(finite, q_high))
    if np.isclose(vmin, vmax):
        delta = max(abs(vmin), 1.0) * 0.1
        return vmin - delta, vmax + delta
    return vmin, vmax


def save_dataset_gallery(kappa: np.ndarray, output_dir: Path) -> None:
    n_samples = min(8, int(kappa.shape[0]))
    vmin, vmax = quantile_limits(kappa[:n_samples])
    fig, axes = plt.subplots(2, 4, figsize=(8.0, 4.2), dpi=220)
    for ax, image in zip(axes.ravel(), kappa[:n_samples, 0]):
        im = ax.imshow(image, cmap="magma", vmin=vmin, vmax=vmax)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.78, pad=0.015, label="kappa")
    fig.suptitle("Synthetic log-normal convergence samples", fontsize=12)
    fig.savefig(output_dir / "dataset_gallery.png", bbox_inches="tight")
    plt.close(fig)


def save_forward_model_panel(
    *,
    kappa: np.ndarray,
    gamma_true: np.ndarray,
    gamma_obs: np.ndarray,
    mask: np.ndarray,
    kappa_ks: np.ndarray,
    output_dir: Path,
) -> None:
    idx = 0
    gamma_mag = np.sqrt(np.sum(gamma_obs[idx] ** 2, axis=0))
    kappa_vmin, kappa_vmax = quantile_limits(kappa[idx], kappa_ks[idx])
    gamma_vmin, gamma_vmax = quantile_limits(gamma_true[idx])
    mag_vmax = float(np.quantile(gamma_mag, 0.99))
    if not np.isfinite(mag_vmax) or mag_vmax <= 0.0:
        mag_vmax = 1.0

    panels = [
        ("true kappa", kappa[idx, 0], "magma", kappa_vmin, kappa_vmax),
        ("true gamma1", gamma_true[idx, 0], "coolwarm", gamma_vmin, gamma_vmax),
        ("true gamma2", gamma_true[idx, 1], "coolwarm", gamma_vmin, gamma_vmax),
        ("observed mask", mask[idx, 0], "gray", 0.0, 1.0),
        ("observed |gamma|", gamma_mag, "viridis", 0.0, mag_vmax),
        ("Kaiser-Squires kappa", kappa_ks[idx, 0], "magma", kappa_vmin, kappa_vmax),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(9.0, 6.0), dpi=220)
    for ax, (title, image, cmap, vmin, vmax) in zip(axes.ravel(), panels):
        im = ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.025)
    fig.suptitle("Forward model used by the shear dataset", fontsize=12)
    fig.savefig(output_dir / "forward_model_panel.png", bbox_inches="tight")
    plt.close(fig)


def save_ks_kernel_plot(image_size: int, output_dir: Path) -> None:
    d1, d2 = ks_kernel_numpy(image_size)
    panels = [
        ("D1 = (ell_x^2 - ell_y^2) / ell^2", np.fft.fftshift(d1)),
        ("D2 = 2 ell_x ell_y / ell^2", np.fft.fftshift(d2)),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.7), dpi=220)
    for ax, (title, image) in zip(axes, panels):
        im = ax.imshow(image, cmap="coolwarm", vmin=-1.0, vmax=1.0)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.82, pad=0.02)
    fig.suptitle("Kaiser-Squires Fourier kernels", fontsize=12)
    fig.savefig(output_dir / "kaiser_squires_kernels.png", bbox_inches="tight")
    plt.close(fig)


def save_shear_sticks(
    *,
    kappa: np.ndarray,
    gamma_true: np.ndarray,
    output_dir: Path,
) -> None:
    image = kappa[0, 0]
    gamma1 = gamma_true[0, 0]
    gamma2 = gamma_true[0, 1]
    image_size = int(image.shape[-1])
    step = max(3, image_size // 18)
    ys = np.arange(step // 2, image_size, step)
    xs = np.arange(step // 2, image_size, step)
    xx, yy = np.meshgrid(xs, ys)

    g1 = gamma1[yy, xx]
    g2 = gamma2[yy, xx]
    mag = np.sqrt(g1**2 + g2**2)
    angle = 0.5 * np.arctan2(g2, g1)
    length = 0.35 * step * np.sqrt(mag / (np.quantile(mag, 0.95) + 1e-8))
    length = np.clip(length, 0.08 * step, 0.48 * step)

    dx = length * np.cos(angle)
    dy = length * np.sin(angle)
    segments = np.stack(
        [
            np.stack([xx - dx, yy - dy], axis=-1),
            np.stack([xx + dx, yy + dy], axis=-1),
        ],
        axis=2,
    ).reshape(-1, 2, 2)

    fig, ax = plt.subplots(figsize=(5.0, 5.0), dpi=220)
    vmin, vmax = quantile_limits(image)
    ax.imshow(image, cmap="magma", vmin=vmin, vmax=vmax)
    lines = LineCollection(segments, colors="white", linewidths=0.65, alpha=0.85)
    ax.add_collection(lines)
    ax.set_title("Shear as local spin-2 stretching directions", fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.savefig(output_dir / "shear_sticks.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = build_arg_parser().parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not PLOT_DEPS_AVAILABLE:
        copy_existing_run_outputs(output_dir)
        return

    dataset_cfg, source_config = load_dataset_config(args.config)
    image_size = int(dataset_cfg["image_size"])
    n_samples = max(1, int(args.n_samples))

    kappa = generate_lognormal_kappa_dataset(
        n_samples=n_samples,
        image_size=image_size,
        spectral_index=float(dataset_cfg["spectral_index"]),
        gaussian_sigma=float(dataset_cfg["gaussian_sigma"]),
        seed=args.seed,
    )
    gamma_true = kaiser_squires_shear_numpy(kappa[:, 0])
    rng = np.random.default_rng(args.observation_seed)
    noise = float(dataset_cfg["noise_std"]) * rng.normal(size=gamma_true.shape).astype(np.float32)
    mask = make_rectangular_mask(
        n_samples=n_samples,
        image_size=image_size,
        mask_fraction=float(dataset_cfg["mask_fraction"]),
        mask_size=int(dataset_cfg["mask_size"]),
        num_masks=int(dataset_cfg["num_masks"]),
        seed=args.observation_seed + 1,
    )
    gamma_obs = (gamma_true + noise) * mask
    kappa_ks = kaiser_squires_inverse_numpy(gamma_obs)

    save_dataset_gallery(kappa, output_dir)
    save_forward_model_panel(
        kappa=kappa,
        gamma_true=gamma_true,
        gamma_obs=gamma_obs,
        mask=mask,
        kappa_ks=kappa_ks,
        output_dir=output_dir,
    )
    save_ks_kernel_plot(image_size, output_dir)
    save_shear_sticks(kappa=kappa, gamma_true=gamma_true, output_dir=output_dir)

    metadata = {
        "source_config": str(source_config) if source_config is not None else None,
        "dataset": {
            key: dataset_cfg[key]
            for key in (
                "image_size",
                "spectral_index",
                "gaussian_sigma",
                "noise_std",
                "mask_fraction",
                "mask_size",
                "num_masks",
            )
        },
        "seed": args.seed,
        "observation_seed": args.observation_seed,
    }
    (output_dir / "figure_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    print(f"Wrote shear note figures to {output_dir}")
    if source_config is not None:
        print(f"Used dataset settings from {source_config}")


if __name__ == "__main__":
    main()
