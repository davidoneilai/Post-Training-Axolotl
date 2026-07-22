#!/usr/bin/env bash
set -e

if [[ -f /workspace/axolotl/scripts/cuda13_env.sh ]]; then
  # shellcheck source=/dev/null
  source /workspace/axolotl/scripts/cuda13_env.sh
fi

exec "$@"
