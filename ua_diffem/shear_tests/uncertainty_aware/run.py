"""Uncertainty calibration diagnostics for saved shear runs.

This test answers the question: "Do the model's predicted uncertainties mean
what we hope they mean?"

What it does
------------
- reconstructs a held-out synthetic dataset with per-pixel predictive variance
- compares predicted standard deviation against realized reconstruction error
- measures empirical coverage for nominal Gaussian intervals
- computes a Gaussian negative log-likelihood surrogate
- builds reliability and risk-coverage plots

What the diagnostics mean
-------------------------
- `coverage`: empirical fraction of pixels whose true error falls inside the
  nominal 50 / 80 / 90 / 95 percent Gaussian intervals. Values close to the
  nominal target indicate better calibration.
- `gaussian_nll_mean`: lower is better; it rewards low error when uncertainty
  is small and penalizes overconfidence.
- `uncertainty_abs_error_correlation`: higher positive values mean the model
  tends to be uncertain exactly where it is wrong.
- reliability curve: compares predicted standard deviation to empirical RMSE.
  Points near the diagonal are better calibrated.
- risk-coverage curve: shows how error changes when we keep only the least
  uncertain pixels. Lower risk at low coverage means uncertainty is useful for
  filtering or triage.

Important parameters
--------------------
All shared evaluation parameters apply here. This test additionally exposes:

- `n_bins`: number of uncertainty bins used in the reliability plot
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from ua_diffem.shear_tests.common import (
    add_shared_eval_args,
    generate_test_batch,
    load_shear_run,
    reconstruct_test_batch,
    write_eval_metadata,
    write_json,
)


Z_BY_COVERAGE = {
    "50": 0.67448975,
    "80": 1.28155157,
    "90": 1.64485363,
    "95": 1.95996398,
}


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the uncertainty calibration test."""
    parser = argparse.ArgumentParser(
        description="Uncertainty calibration diagnostics for a saved shear DiffEM run."
    )
    parser = add_shared_eval_args(parser)
    parser.add_argument("--n_bins", type=int, default=10)
    return parser


def _quantile_bins(values: np.ndarray, n_bins: int) -> np.ndarray:
    edges = np.quantile(values, np.linspace(0.0, 1.0, n_bins + 1))
    edges[0] = min(edges[0], float(values.min()))
    edges[-1] = max(edges[-1], float(values.max()))
    for idx in range(1, len(edges)):
        if edges[idx] <= edges[idx - 1]:
            edges[idx] = edges[idx - 1] + 1e-6
    return edges


def run(args: argparse.Namespace) -> None:
    """Execute the uncertainty-aware evaluation and save calibration artifacts.

    Outputs are written under:
    `runs/ua_diffem/<run_name>/uncertainty_aware/`

    Saved artifacts include:

    - `evaluation_config.json`: resolved evaluation settings
    - `summary.json`: aggregate calibration statistics
    - `uncertainty_metrics.npz`: raw arrays for downstream analysis
    - `uncertainty_diagnostics.png`: reliability, coverage, and risk plots
    """
    loaded = load_shear_run(
        run_name=args.run_name,
        output_name="uncertainty_aware",
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
    pred = recon.samples_phys.astype(np.float32)
    pred_std = np.sqrt(np.maximum(recon.variance_phys.astype(np.float32), 1e-8))
    abs_error = np.abs(pred - truth)
    sq_error = (pred - truth) ** 2

    flat_std = pred_std.reshape(-1)
    flat_abs_error = abs_error.reshape(-1)
    flat_sq_error = sq_error.reshape(-1)

    coverage = {
        key: float(np.mean(flat_abs_error <= z_value * flat_std))
        for key, z_value in Z_BY_COVERAGE.items()
    }
    gaussian_nll = 0.5 * (
        np.log(2.0 * np.pi * np.maximum(pred_std**2, 1e-8)) + sq_error / np.maximum(pred_std**2, 1e-8)
    )
    corr = np.corrcoef(flat_std, flat_abs_error)[0, 1]

    edges = _quantile_bins(flat_std, args.n_bins)
    bin_ids = np.digitize(flat_std, edges[1:-1], right=False)
    reliability_pred = []
    reliability_err = []
    for bin_idx in range(args.n_bins):
        mask = bin_ids == bin_idx
        if not np.any(mask):
            continue
        reliability_pred.append(float(np.mean(flat_std[mask])))
        reliability_err.append(float(np.sqrt(np.mean(flat_sq_error[mask]))))

    order = np.argsort(flat_std)
    coverage_grid = np.linspace(0.1, 1.0, 10)
    risk = []
    for coverage_fraction in coverage_grid:
        n_keep = max(1, int(round(len(order) * coverage_fraction)))
        chosen = order[:n_keep]
        risk.append(float(np.sqrt(np.mean(flat_sq_error[chosen]))))

    summary = {
        "run_name": loaded.run_name,
        "checkpoint_path": str(loaded.checkpoint_path),
        "n_examples": int(args.test_size),
        "gaussian_nll_mean": float(np.mean(gaussian_nll)),
        "uncertainty_abs_error_correlation": float(corr),
        "coverage": coverage,
    }
    write_json(loaded.output_dir / "summary.json", summary)

    np.savez(
        loaded.output_dir / "uncertainty_metrics.npz",
        pred_std=pred_std.astype(np.float32),
        abs_error=abs_error.astype(np.float32),
        gaussian_nll=gaussian_nll.astype(np.float32),
        reliability_pred=np.asarray(reliability_pred, dtype=np.float32),
        reliability_err=np.asarray(reliability_err, dtype=np.float32),
        risk_coverage=coverage_grid.astype(np.float32),
        risk=np.asarray(risk, dtype=np.float32),
    )

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.6), dpi=220)

    axes[0].plot(reliability_pred, reliability_err, marker="o", color="tab:blue")
    limit = max(reliability_pred + reliability_err + [1e-6])
    axes[0].plot([0.0, limit], [0.0, limit], linestyle="--", color="0.3", linewidth=1.0)
    axes[0].set_xlabel("Predicted std")
    axes[0].set_ylabel("Empirical RMSE")
    axes[0].set_title("Reliability")
    axes[0].grid(alpha=0.2, linewidth=0.6)

    nominal = np.asarray([0.50, 0.80, 0.90, 0.95], dtype=np.float32)
    empirical = np.asarray([coverage["50"], coverage["80"], coverage["90"], coverage["95"]], dtype=np.float32)
    axes[1].plot(nominal, empirical, marker="o", color="tab:green")
    axes[1].plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="0.3", linewidth=1.0)
    axes[1].set_xlim(0.45, 1.0)
    axes[1].set_ylim(0.45, 1.0)
    axes[1].set_xlabel("Nominal coverage")
    axes[1].set_ylabel("Empirical coverage")
    axes[1].set_title("Interval Coverage")
    axes[1].grid(alpha=0.2, linewidth=0.6)

    axes[2].plot(coverage_grid, risk, marker="o", color="tab:red")
    axes[2].set_xlabel("Coverage fraction")
    axes[2].set_ylabel("RMSE")
    axes[2].set_title("Risk-Coverage")
    axes[2].grid(alpha=0.2, linewidth=0.6)

    fig.tight_layout()
    fig.savefig(loaded.output_dir / "uncertainty_diagnostics.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = build_arg_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
