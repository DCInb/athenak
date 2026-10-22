#!/usr/bin/env bash
# Invoked inside the one-rank MPI launcher by solver_diagnostic.slurm.
set -euo pipefail
: "${ATHENA_REAL_EXE:?}"
: "${NSYS_OUTPUT:?}"
exec nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --force-overwrite=false --stats=true --output="$NSYS_OUTPUT" \
  "$ATHENA_REAL_EXE" "$@"
