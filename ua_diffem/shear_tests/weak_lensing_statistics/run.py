"""Weak-lensing summary-statistics test for saved shear runs.

This test answers the question: "Does the reconstruction preserve the
scientifically relevant structure, not just pixel-level appearance?"

What it does
------------
- reconstructs a held-out synthetic dataset
- computes mean convergence power spectra for truth, UA-DiffEM, and
  Kaiser-Squires
- computes simple peak-count histograms for the same three sets
- summarizes the relative L2 error of these statistics

What the statistics mean
------------------------
- power spectrum: measures how variance is distributed across spatial scales.
  Good recovery means the reconstruction preserves large- and small-scale
  structure in the right proportions.
- peak counts: probe non-Gaussian structure and extreme features that are often
  scientifically interesting in weak lensing.
- relative L2 errors: lower is better and indicates the recovered summary
  statistic is closer to the ground-truth one.

Important parameters
--------------------
All shared evaluation parameters apply here. This test additionally exposes:

- `n_ell_bins`: number of radial bins used for the power spectrum
- `n_peak_bins`: number of bins used for the peak histogram
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


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the weak-lensing statistics test."""
    parser = argparse.ArgumentParser(
        description="Weak-lensing summary statistics for a saved shear DiffEM run."
    )
    parser = add_shared_eval_args(parser)
    parser.add_argument("--n_ell_bins", type=int, default=16)
    parser.add_argument("--n_peak_bins", type=int, default=24)
    return parser


def _power_spectrum_batch(fields: np.ndarray, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    image_size = fields.shape[-1]
    freq = np.fft.fftfreq(image_size)
    ky, kx = np.meshgrid(freq, freq, indexing="ij")
    ell = np.sqrt(kx**2 + ky**2)
    ell_flat = ell.reshape(-1)
    bins = np.linspace(0.0, ell_flat.max(), n_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    bin_ids = np.digitize(ell_flat, bins[1:-1], right=False)

    spectra = []
    for field in fields[:, 0]:
        fft = np.fft.fft2(field)
        power = (np.abs(fft) ** 2 / float(image_size**2)).reshape(-1)
        binned = np.zeros(n_bins, dtype=np.float32)
        for bin_idx in range(n_bins):
            mask = bin_ids == bin_idx
            if np.any(mask):
                binned[bin_idx] = float(np.mean(power[mask]))
        spectra.append(binned)
    return centers.astype(np.float32), np.asarray(spectra, dtype=np.float32)


def _peak_values(field: np.ndarray) -> np.ndarray:
    padded = np.pad(field, 1, mode="edge")
    center = padded[1:-1, 1:-1]
    neighbors = [
        padded[1 + dy : 1 + dy + field.shape[0], 1 + dx : 1 + dx + field.shape[1]]
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if not (dy == 0 and dx == 0)
    ]
    neighbor_max = np.maximum.reduce(neighbors)
    peaks = center[center > neighbor_max]
    return peaks.astype(np.float32)


def _peak_value_lists(fields: np.ndarray) -> list[np.ndarray]:
    return [_peak_values(field[0]) for field in fields]


def _peak_bins_from_values(peak_lists: list[np.ndarray], n_bins: int) -> np.ndarray:
    non_empty = [values for values in peak_lists if values.size > 0]
    if not non_empty:
        return np.linspace(-1.0, 1.0, n_bins + 1, dtype=np.float32)

    pooled = np.concatenate(non_empty, axis=0)
    lo = float(np.quantile(pooled, 0.01))
    hi = float(np.quantile(pooled, 0.99))
    if hi <= lo:
        hi = lo + 1e-3
    return np.linspace(lo, hi, n_bins + 1, dtype=np.float32)


def _peak_histogram_from_lists(peak_lists: list[np.ndarray], bins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    histograms = []
    for peak_values in peak_lists:
        if peak_values.size == 0:
            hist = np.zeros(len(bins) - 1, dtype=np.float32)
        else:
            hist, _ = np.histogram(peak_values, bins=bins, density=True)
            hist = hist.astype(np.float32)
        histograms.append(hist)
    centers = 0.5 * (bins[:-1] + bins[1:])
    return centers.astype(np.float32), np.asarray(histograms, dtype=np.float32)


def run(args: argparse.Namespace) -> None:
    """Execute the weak-lensing summary-statistics evaluation.

    Outputs are written under:
    `runs/ua_diffem/<run_name>/weak_lensing_statistics/`

    Saved artifacts include:

    - `evaluation_config.json`: resolved evaluation settings
    - `summary.json`: aggregate relative-error statistics
    - `weak_lensing_statistics.npz`: raw spectra and peak histograms
    - `weak_lensing_statistics.png`: summary comparison figure
    """
    loaded = load_shear_run(
        run_name=args.run_name,
        output_name="weak_lensing_statistics",
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

    ell, truth_power = _power_spectrum_batch(batch.kappa_true_phys, args.n_ell_bins)
    _, pred_power = _power_spectrum_batch(recon.samples_phys, args.n_ell_bins)
    _, ks_power = _power_spectrum_batch(batch.kappa_ks_phys, args.n_ell_bins)

    truth_peak_lists = _peak_value_lists(batch.kappa_true_phys)
    pred_peak_lists = _peak_value_lists(recon.samples_phys)
    ks_peak_lists = _peak_value_lists(batch.kappa_ks_phys)
    peak_bins = _peak_bins_from_values(truth_peak_lists, args.n_peak_bins)
    peak_centers, truth_peaks = _peak_histogram_from_lists(truth_peak_lists, peak_bins)
    _, pred_peaks = _peak_histogram_from_lists(pred_peak_lists, peak_bins)
    _, ks_peaks = _peak_histogram_from_lists(ks_peak_lists, peak_bins)

    truth_power_mean = truth_power.mean(axis=0)
    pred_power_mean = pred_power.mean(axis=0)
    ks_power_mean = ks_power.mean(axis=0)
    truth_peaks_mean = truth_peaks.mean(axis=0)
    pred_peaks_mean = pred_peaks.mean(axis=0)
    ks_peaks_mean = ks_peaks.mean(axis=0)

    summary = {
        "run_name": loaded.run_name,
        "checkpoint_path": str(loaded.checkpoint_path),
        "power_spectrum_relative_l2_error": {
            "ua_diffem": float(np.linalg.norm(pred_power_mean - truth_power_mean) / (np.linalg.norm(truth_power_mean) + 1e-8)),
            "kaiser_squires": float(np.linalg.norm(ks_power_mean - truth_power_mean) / (np.linalg.norm(truth_power_mean) + 1e-8)),
        },
        "peak_histogram_relative_l2_error": {
            "ua_diffem": float(np.linalg.norm(pred_peaks_mean - truth_peaks_mean) / (np.linalg.norm(truth_peaks_mean) + 1e-8)),
            "kaiser_squires": float(np.linalg.norm(ks_peaks_mean - truth_peaks_mean) / (np.linalg.norm(truth_peaks_mean) + 1e-8)),
        },
    }
    write_json(loaded.output_dir / "summary.json", summary)

    np.savez(
        loaded.output_dir / "weak_lensing_statistics.npz",
        ell=ell.astype(np.float32),
        truth_power=truth_power.astype(np.float32),
        pred_power=pred_power.astype(np.float32),
        ks_power=ks_power.astype(np.float32),
        peak_centers=peak_centers.astype(np.float32),
        truth_peaks=truth_peaks.astype(np.float32),
        pred_peaks=pred_peaks.astype(np.float32),
        ks_peaks=ks_peaks.astype(np.float32),
    )

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), dpi=220)

    axes[0].plot(ell, truth_power_mean, label="truth", color="black", linewidth=1.5)
    axes[0].plot(ell, ks_power_mean, label="Kaiser-Squires", color="tab:gray")
    axes[0].plot(ell, pred_power_mean, label="UA-DiffEM", color="tab:blue")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Frequency radius")
    axes[0].set_ylabel("Mean power")
    axes[0].set_title("Convergence Power Spectrum")
    axes[0].grid(alpha=0.2, linewidth=0.6)
    axes[0].legend(frameon=False)

    axes[1].plot(peak_centers, truth_peaks_mean, label="truth", color="black", linewidth=1.5)
    axes[1].plot(peak_centers, ks_peaks_mean, label="Kaiser-Squires", color="tab:gray")
    axes[1].plot(peak_centers, pred_peaks_mean, label="UA-DiffEM", color="tab:blue")
    axes[1].set_xlabel("Peak amplitude")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Peak Counts")
    axes[1].grid(alpha=0.2, linewidth=0.6)

    fig.tight_layout()
    fig.savefig(loaded.output_dir / "weak_lensing_statistics.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = build_arg_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
