"""Convenience wrapper for running the full saved-run shear evaluation suite.

This module does not define a new scientific test by itself. Instead, it lets
you launch all shear evaluation folders from one command while sharing the same
run selection and sampling controls.

What it does
------------
- parses one top-level set of shared evaluation arguments
- optionally restricts execution to a subset of tests
- forwards the resolved arguments into each test-specific CLI

Important parameters
--------------------
All shared evaluation parameters apply here. This wrapper additionally exposes:

- `tests`: subset of test folders to run
- `visual_n_examples`: number of examples to include in the qualitative panel
- `visual_test_size`: optional override for how many held-out examples the
  visual test generates
- `uncertainty_n_bins`: number of bins for the uncertainty reliability plot
- `weak_lensing_n_ell_bins`: number of power-spectrum bins
- `weak_lensing_n_peak_bins`: number of peak-histogram bins
"""

from __future__ import annotations

import argparse

from ua_diffem.shear_tests.common import add_shared_eval_args
from ua_diffem.shear_tests.quantitative_reconstruction.run import (
    build_arg_parser as build_quantitative_parser,
    run as run_quantitative,
)
from ua_diffem.shear_tests.uncertainty_aware.run import (
    build_arg_parser as build_uncertainty_parser,
    run as run_uncertainty,
)
from ua_diffem.shear_tests.visual_examples.run import (
    build_arg_parser as build_visual_parser,
    run as run_visual,
)
from ua_diffem.shear_tests.weak_lensing_statistics.run import (
    build_arg_parser as build_weak_lensing_parser,
    run as run_weak_lensing,
)


TEST_CHOICES = (
    "quantitative_reconstruction",
    "visual_examples",
    "uncertainty_aware",
    "weak_lensing_statistics",
)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the multi-test wrapper."""
    parser = argparse.ArgumentParser(
        description="Run all saved-run shear evaluation scripts from one command."
    )
    parser = add_shared_eval_args(
        parser,
        default_test_size=128,
        default_test_seed=123,
        default_observation_seed=456,
    )
    parser.add_argument(
        "--tests",
        nargs="+",
        choices=TEST_CHOICES,
        default=list(TEST_CHOICES),
        help="Subset of evaluation folders to run. Defaults to the full suite.",
    )
    parser.add_argument("--visual_n_examples", type=int, default=8)
    parser.add_argument(
        "--visual_test_size",
        type=int,
        default=None,
        help="Optional visual-only test size override. Defaults to max(16, visual_n_examples).",
    )
    parser.add_argument("--uncertainty_n_bins", type=int, default=10)
    parser.add_argument("--weak_lensing_n_ell_bins", type=int, default=16)
    parser.add_argument("--weak_lensing_n_peak_bins", type=int, default=24)
    return parser


def _append_arg(argv: list[str], flag: str, value) -> None:
    if value is None:
        return
    argv.extend([flag, str(value)])


def _shared_argv(args: argparse.Namespace, *, test_size: int) -> list[str]:
    argv = ["--run_name", args.run_name, "--test_size", str(test_size)]
    _append_arg(argv, "--checkpoint_name", args.checkpoint_name)
    _append_arg(argv, "--test_seed", args.test_seed)
    _append_arg(argv, "--observation_seed", args.observation_seed)
    _append_arg(argv, "--posterior_batch_size", args.posterior_batch_size)
    _append_arg(argv, "--posterior_sample_steps", args.posterior_sample_steps)
    _append_arg(argv, "--posterior_solver", args.posterior_solver)
    _append_arg(argv, "--cfg_max_scale", args.cfg_max_scale)
    _append_arg(argv, "--ucg_scale", args.ucg_scale)
    if args.use_raw_flow:
        argv.append("--use_raw_flow")
    return argv


def _run_test(name: str, argv: list[str]) -> None:
    if name == "quantitative_reconstruction":
        args = build_quantitative_parser().parse_args(argv)
        run_quantitative(args)
        return
    if name == "visual_examples":
        args = build_visual_parser().parse_args(argv)
        run_visual(args)
        return
    if name == "uncertainty_aware":
        args = build_uncertainty_parser().parse_args(argv)
        run_uncertainty(args)
        return
    if name == "weak_lensing_statistics":
        args = build_weak_lensing_parser().parse_args(argv)
        run_weak_lensing(args)
        return
    raise ValueError(f"Unknown test {name!r}.")


def main() -> None:
    """Run the requested subset of saved-run shear evaluation scripts."""
    args = build_arg_parser().parse_args()
    visual_test_size = (
        max(16, args.visual_n_examples)
        if args.visual_test_size is None
        else args.visual_test_size
    )

    test_argv_map = {
        "quantitative_reconstruction": _shared_argv(args, test_size=args.test_size),
        "visual_examples": _shared_argv(args, test_size=visual_test_size)
        + ["--n_examples", str(args.visual_n_examples)],
        "uncertainty_aware": _shared_argv(args, test_size=args.test_size)
        + ["--n_bins", str(args.uncertainty_n_bins)],
        "weak_lensing_statistics": _shared_argv(args, test_size=args.test_size)
        + [
            "--n_ell_bins",
            str(args.weak_lensing_n_ell_bins),
            "--n_peak_bins",
            str(args.weak_lensing_n_peak_bins),
        ],
    }

    for test_name in args.tests:
        print(f"[ua_diffem.shear_tests.run_all] Running {test_name} for run {args.run_name}.", flush=True)
        _run_test(test_name, test_argv_map[test_name])


if __name__ == "__main__":
    main()
