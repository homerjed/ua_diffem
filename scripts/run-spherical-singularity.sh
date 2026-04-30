#!/usr/bin/env bash
# Run the spherical UA-DiffEM Singularity image with this repo bind-mounted.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SIF="${SINGULARITY_SIF:-/u/jhome/ua-diffem-spherical.sif}"
BIND_ARGS=(-B "${REPO_ROOT}:/workspace")

if [ -d /u/jhome/linked_flow ]; then
  BIND_ARGS+=(-B /u/jhome/linked_flow:/linked_flow)
fi
if [ -d /scratch ]; then
  BIND_ARGS+=(-B /scratch:/scratch)
fi

if [ ! -f "$SIF" ]; then
  echo "SIF not found: $SIF" >&2
  echo "Build it from repo root: ./scripts/build-spherical-sif.sh" >&2
  echo "Or set SINGULARITY_SIF to your image path." >&2
  exit 1
fi

if [ $# -eq 0 ]; then
  singularity shell --nv --pwd /workspace "${BIND_ARGS[@]}" --env "PYTHONPATH=/workspace:/linked_flow" "$SIF"
else
  singularity exec --nv --pwd /workspace "${BIND_ARGS[@]}" --env "PYTHONPATH=/workspace:/linked_flow" "$SIF" "$@"
fi
