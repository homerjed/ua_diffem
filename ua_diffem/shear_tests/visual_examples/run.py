"""Qualitative visual sanity-checks for saved shear runs.

This test answers the question: "What do the reconstructions actually look
like, and do the uncertainty maps highlight the hard regions?"

What it does
------------
- regenerates a held-out synthetic dataset
- reconstructs each example with the saved posterior
- assembles a multi-row panel showing:
  truth kappa, observed shear magnitude, Kaiser-Squires baseline,
  posterior reconstruction, predicted uncertainty, and absolute error

What it means
-------------
This is the easiest test for spotting failure modes that summary numbers can
hide: oversmoothing, hallucinated structure, boundary artifacts, mismatch
between uncertainty and error, or cases where the baseline and posterior fail
in different ways. It is a qualitative inspection tool rather than a calibrated
performance metric.

Important parameters
--------------------
In addition to the shared evaluation parameters, this test adds:

- `n_examples`: how many examples to display in the saved panel

The effective number of generated examples is `max(test_size, n_examples)`, so
the plot always has enough samples to display.
"""

from __future__ import annotations

import argparse

from einops import rearrange
import matplotlib.pyplot as plt
import numpy as np

from ua_diffem.shear_tests.common import (
    add_shared_eval_args,
    generate_test_batch,
    load_shear_run,
    reconstruct_test_batch,
    write_eval_metadata,
)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the qualitative visual test."""
    parser = argparse.ArgumentParser(
        description="Save qualitative visual examples for a saved shear DiffEM run."
    )
    parser = add_shared_eval_args(parser, default_test_size=16)
    parser.add_argument("--n_examples", type=int, default=8)
    return parser


def _make_strip(images: np.ndarray) -> np.ndarray:
    return rearrange(images, "n c h w -> h (n w) c")


def _normalize(images: np.ndarray, *, vmin: float, vmax: float) -> np.ndarray:
    return np.clip((images - vmin) / (vmax - vmin + 1e-8), 0.0, 1.0)


def run(args: argparse.Namespace) -> None:
    """Execute the qualitative visual test and save its example panel.

    Outputs are written under:
    `runs/ua_diffem/<run_name>/visual_examples/`

    Saved artifacts include:

    - `evaluation_config.json`: resolved evaluation settings
    - `visual_examples.png`: panel for quick qualitative inspection
    - `examples.npz`: arrays used to make the panel
    """
    loaded = load_shear_run(
        run_name=args.run_name,
        output_name="visual_examples",
        checkpoint_name=args.checkpoint_name,
        use_raw_flow=args.use_raw_flow,
    )
    write_eval_metadata(loaded, loaded.output_dir / "evaluation_config.json", args=args)

    batch = generate_test_batch(
        loaded,
        test_size=max(args.test_size, args.n_examples),
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

    n_examples = min(args.n_examples, batch.kappa_true_phys.shape[0])
    truth = batch.kappa_true_phys[:n_examples]
    gamma_mag = np.sqrt(np.sum(batch.gamma_obs_phys[:n_examples] ** 2, axis=1, keepdims=True))
    ks = batch.kappa_ks_phys[:n_examples]
    pred = recon.samples_phys[:n_examples]
    uncertainty = np.sqrt(np.maximum(recon.variance_phys[:n_examples], 0.0)).mean(axis=1, keepdims=True)
    abs_error = np.abs(pred - truth)

    kappa_range = np.concatenate([truth, ks, pred], axis=0)
    kappa_vmin = float(np.quantile(kappa_range, 0.01))
    kappa_vmax = float(np.quantile(kappa_range, 0.99))
    gamma_vmax = float(np.quantile(gamma_mag, 0.99))
    uncertainty_vmax = float(np.quantile(uncertainty, 0.99))
    error_vmax = float(np.quantile(abs_error, 0.99))

    rows = [
        ("truth kappa", _make_strip(_normalize(truth, vmin=kappa_vmin, vmax=kappa_vmax)), "magma"),
        ("|gamma obs|", _make_strip(_normalize(gamma_mag, vmin=0.0, vmax=max(gamma_vmax, 1e-6))), "viridis"),
        ("Kaiser-Squires", _make_strip(_normalize(ks, vmin=kappa_vmin, vmax=kappa_vmax)), "magma"),
        ("reconstruction", _make_strip(_normalize(pred, vmin=kappa_vmin, vmax=kappa_vmax)), "magma"),
        ("uncertainty", _make_strip(_normalize(uncertainty, vmin=0.0, vmax=max(uncertainty_vmax, 1e-6))), "magma"),
        ("abs error", _make_strip(_normalize(abs_error, vmin=0.0, vmax=max(error_vmax, 1e-6))), "inferno"),
    ]

    strip_aspect = rows[0][1].shape[1] / rows[0][1].shape[0]
    fig_width = max(6.0, 0.85 * n_examples)
    row_height = fig_width / strip_aspect

    fig, axes = plt.subplots(
        len(rows),
        1,
        figsize=(fig_width, len(rows) * row_height + 0.12),
        dpi=220,
        gridspec_kw={"hspace": 0.0},
    )
    for ax, (label, grid, cmap) in zip(axes, rows):
        ax.imshow(grid[..., 0], vmin=0.0, vmax=1.0, cmap=cmap)
        ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.subplots_adjust(left=0.16, right=1.0, top=1.0, bottom=0.0, hspace=0.0)
    fig.savefig(loaded.output_dir / "visual_examples.png", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    np.savez(
        loaded.output_dir / "examples.npz",
        truth=truth.astype(np.float32),
        gamma_obs=batch.gamma_obs_phys[:n_examples].astype(np.float32),
        kaiser_squires=ks.astype(np.float32),
        reconstruction=pred.astype(np.float32),
        uncertainty=uncertainty.astype(np.float32),
        abs_error=abs_error.astype(np.float32),
    )


def main() -> None:
    args = build_arg_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
