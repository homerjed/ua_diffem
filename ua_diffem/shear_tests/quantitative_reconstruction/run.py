"""Quantitative reconstruction test for saved shear runs.

This test answers the question: "Does the saved posterior reconstruct held-out
convergence maps accurately, and is it consistent with the observed shear?"

What it does
------------
- regenerates a held-out synthetic dataset where ground-truth kappa is known
- corrupts it with the saved run's noise / masking forward model
- reconstructs kappa with the saved UA-DiffEM posterior
- compares against a Kaiser-Squires baseline
- measures both image-space accuracy and data-space forward residuals

What the metrics mean
---------------------
- `mse`, `rmse`, `mae`: direct pixelwise reconstruction error in kappa
- `psnr`: dynamic-range-aware reconstruction quality; higher is better
- `correlation`: whether recovered spatial structure tracks the truth; higher
  is better even if amplitudes are imperfect
- `data_mse`, `data_rmse`, `data_mae`, `data_nrmse`: residuals after pushing
  the reconstructed kappa back through the shear operator and comparing against
  the observed shear in measurement space

In short, lower error metrics and higher `psnr` / `correlation` are better.
The data-space metrics matter because a kappa image can look plausible while
still failing to explain the observed shear.

Important parameters
--------------------
All shared parameters come from `ua_diffem.shear_tests.common.add_shared_eval_args`.
The main ones to care about here are:

- `run_name`: saved run to evaluate
- `checkpoint_name`: optional checkpoint inside `states/`
- `test_size`: number of held-out examples
- `test_seed`: seed for clean synthetic fields
- `observation_seed`: seed for noisy / masked observations
- `posterior_batch_size`: reconstruction batch size
- `posterior_sample_steps`: number of posterior sampling steps
- `posterior_solver`: `euler` or `heun`
- `cfg_max_scale`, `ucg_scale`: optional guidance controls
- `use_raw_flow`: evaluate raw weights instead of EMA weights
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from ua_diffem.shear import kaiser_squires_shear_numpy
from ua_diffem.shear_tests.common import (
    add_shared_eval_args,
    generate_test_batch,
    load_shear_run,
    reconstruct_test_batch,
    write_eval_metadata,
    write_json,
)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the quantitative reconstruction test."""
    parser = argparse.ArgumentParser(
        description="Quantitative reconstruction metrics for a saved shear DiffEM run."
    )
    return add_shared_eval_args(parser)


def _mean_with_stats(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "median": float(np.median(values)),
    }


def _per_sample_correlation(truth: np.ndarray, pred: np.ndarray) -> np.ndarray:
    flat_truth = truth.reshape(truth.shape[0], -1)
    flat_pred = pred.reshape(pred.shape[0], -1)
    centered_truth = flat_truth - flat_truth.mean(axis=1, keepdims=True)
    centered_pred = flat_pred - flat_pred.mean(axis=1, keepdims=True)
    numerator = np.sum(centered_truth * centered_pred, axis=1)
    denominator = np.sqrt(np.sum(centered_truth**2, axis=1) * np.sum(centered_pred**2, axis=1)) + 1e-8
    return numerator / denominator


def _reconstruction_metrics(truth: np.ndarray, pred: np.ndarray) -> dict[str, np.ndarray]:
    diff = pred - truth
    mse = np.mean(diff**2, axis=(1, 2, 3))
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(diff), axis=(1, 2, 3))
    dynamic_range = np.ptp(truth, axis=(1, 2, 3)) + 1e-8
    psnr = 20.0 * np.log10(dynamic_range) - 10.0 * np.log10(mse + 1e-8)
    corr = _per_sample_correlation(truth, pred)
    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "psnr": psnr,
        "correlation": corr,
    }


def _forward_metrics(
    pred_kappa: np.ndarray,
    gamma_obs: np.ndarray,
    mask: np.ndarray,
) -> dict[str, np.ndarray]:
    pred_gamma = kaiser_squires_shear_numpy(pred_kappa[:, 0])
    residual = (pred_gamma - gamma_obs) * mask
    denom = np.sum(mask, axis=(1, 2, 3)) * gamma_obs.shape[1]
    denom = np.maximum(denom, 1.0)
    mse = np.sum(residual**2, axis=(1, 2, 3)) / denom
    rmse = np.sqrt(mse)
    mae = np.sum(np.abs(residual), axis=(1, 2, 3)) / denom
    obs_energy = np.sum((gamma_obs * mask) ** 2, axis=(1, 2, 3)) / denom
    normalized_rmse = rmse / (np.sqrt(obs_energy) + 1e-8)
    return {
        "data_mse": mse,
        "data_rmse": rmse,
        "data_mae": mae,
        "data_nrmse": normalized_rmse,
    }


def _merge_metrics(truth: np.ndarray, pred: np.ndarray, gamma_obs: np.ndarray, mask: np.ndarray) -> dict[str, np.ndarray]:
    return _reconstruction_metrics(truth, pred) | _forward_metrics(pred, gamma_obs, mask)


def _summary(metrics: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    return {name: _mean_with_stats(values) for name, values in metrics.items()}


def save_metric_plot(
    path,
    ua_metrics: dict[str, np.ndarray],
    ks_metrics: dict[str, np.ndarray],
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.8), dpi=220)
    panel_specs = [
        ("rmse", "Kappa RMSE"),
        ("psnr", "PSNR"),
        ("correlation", "Pixel Correlation"),
        ("data_rmse", "Observed-Shear RMSE"),
    ]

    for ax, (metric_name, title) in zip(axes.flat, panel_specs):
        ua_values = ua_metrics[metric_name]
        ks_values = ks_metrics[metric_name]
        bins = np.histogram_bin_edges(np.concatenate([ua_values, ks_values]), bins=24)
        ax.hist(ks_values, bins=bins, alpha=0.55, label="Kaiser-Squires", color="tab:gray")
        ax.hist(ua_values, bins=bins, alpha=0.65, label="UA-DiffEM", color="tab:blue")
        ax.set_title(title)
        ax.grid(alpha=0.2, linewidth=0.6)
        if metric_name == "correlation":
            ax.set_xlim(-1.0, 1.0)

    axes[0, 0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    """Execute the quantitative reconstruction test and save its artifacts.

    Outputs are written under:
    `runs/ua_diffem/<run_name>/quantitative_reconstruction/`

    Saved artifacts include:

    - `evaluation_config.json`: resolved evaluation settings
    - `metrics.npz`: per-example metric arrays for UA-DiffEM and Kaiser-Squires
    - `summary.json`: aggregate statistics and mean improvement deltas
    - `metric_distributions.png`: histogram comparison for a core metric subset
    """
    loaded = load_shear_run(
        run_name=args.run_name,
        output_name="quantitative_reconstruction",
        checkpoint_name=args.checkpoint_name,
        use_raw_flow=args.use_raw_flow,
    )
    write_eval_metadata(loaded, loaded.output_dir / "evaluation_config.json", args=args)

    batch = generate_test_batch(
        loaded,
        test_size=args.test_size,
        test_seed=args.test_seed,
        observation_seed=args.observation_seed,
    )
    recon = reconstruct_test_batch(
        loaded,
        batch,
        posterior_batch_size=args.posterior_batch_size,
        posterior_sample_steps=args.posterior_sample_steps,
        posterior_solver=args.posterior_solver,
        cfg_max_scale=args.cfg_max_scale,
        ucg_scale=args.ucg_scale,
    )

    truth = batch.kappa_true_phys.astype(np.float32)
    gamma_obs = batch.gamma_obs_phys.astype(np.float32)
    mask = np.asarray(batch.mask).astype(np.float32)

    ua_metrics = _merge_metrics(truth, recon.samples_phys, gamma_obs, mask)
    ks_metrics = _merge_metrics(truth, batch.kappa_ks_phys, gamma_obs, mask)

    np.savez(
        loaded.output_dir / "metrics.npz",
        **{f"ua_{name}": values.astype(np.float32) for name, values in ua_metrics.items()},
        **{f"ks_{name}": values.astype(np.float32) for name, values in ks_metrics.items()},
    )

    summary = {
        "run_name": loaded.run_name,
        "checkpoint_path": str(loaded.checkpoint_path),
        "n_examples": int(args.test_size),
        "ua_diffem": _summary(ua_metrics),
        "kaiser_squires": _summary(ks_metrics),
        "improvement_vs_kaiser_squires": {
            "rmse_delta": float(np.mean(ks_metrics["rmse"] - ua_metrics["rmse"])),
            "psnr_delta": float(np.mean(ua_metrics["psnr"] - ks_metrics["psnr"])),
            "correlation_delta": float(np.mean(ua_metrics["correlation"] - ks_metrics["correlation"])),
            "data_rmse_delta": float(np.mean(ks_metrics["data_rmse"] - ua_metrics["data_rmse"])),
        },
    }
    write_json(loaded.output_dir / "summary.json", summary)
    save_metric_plot(loaded.output_dir / "metric_distributions.png", ua_metrics, ks_metrics)


def main() -> None:
    args = build_arg_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
