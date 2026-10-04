#!/usr/bin/env bash
set -euo pipefail
exec "$(cd "$(dirname "$0")" && pwd -P)/run_parallel_pymarl_env.sh" smacv2 "$@"
