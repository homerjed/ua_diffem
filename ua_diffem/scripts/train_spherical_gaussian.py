from __future__ import annotations

from pathlib import Path
import sys

# Match the path bootstrap used by the main spherical trainer so this wrapper can
# also be executed directly from the command line.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_LINKED_FLOW_ROOT = _REPO_ROOT.parent / "linked_flow"
if __package__ in (None, ""):
    sys.path.insert(0, str(_REPO_ROOT))
if _LINKED_FLOW_ROOT.exists():
    sys.path.insert(0, str(_LINKED_FLOW_ROOT))

from ua_diffem.scripts.train_spherical import main as train_spherical_main


def main() -> None:
    """Run spherical training with Gaussian-dataset defaults."""
    # Reuse the shared training entrypoint, but choose defaults that make the
    # dataset and output directory explicitly Gaussian-focused.
    train_spherical_main(
        prior_family_default="gaussian",
        kappa_power_spectrum_default="camb",
        noise_std_default=0.02,
        run_dir_default=Path("runs/ua_diffem/spherical_gaussian"),
    )


if __name__ == "__main__":
    main()
