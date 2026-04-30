#!/usr/bin/env bash
# Build the spherical UA-DiffEM Singularity image.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRATCH_DIR="${SINGULARITY_BUILD_DIR:-/u/jhome}"
DEF_FILE="${REPO_ROOT}/singularity.def"
OUTPUT_SIF="${SINGULARITY_OUTPUT_SIF:-${SCRATCH_DIR}/ua-diffem-spherical.sif}"

mkdir -p "$SCRATCH_DIR"
export TMPDIR="$SCRATCH_DIR"
export TEMP="$SCRATCH_DIR"
export TMP="$SCRATCH_DIR"
export SINGULARITY_TMPDIR="$SCRATCH_DIR"
export APPTAINER_TMPDIR="$SCRATCH_DIR"
export APPTAINER_CACHEDIR="$SCRATCH_DIR"

AVAIL="$(df -B1 --output=avail "$SCRATCH_DIR" 2>/dev/null | tail -1)" || true
if [ -n "$AVAIL" ] && [ "$AVAIL" -lt $((20 * 1024 * 1024 * 1024)) ]; then
  echo "WARNING: Less than 20 GiB free on $SCRATCH_DIR. Build may fail with 'No space left on device'." >&2
  echo "         Free space: $((AVAIL / 1024 / 1024 / 1024)) GiB" >&2
fi

cd "$REPO_ROOT"
singularity build "$OUTPUT_SIF" "$DEF_FILE"
echo "Built: $OUTPUT_SIF"
