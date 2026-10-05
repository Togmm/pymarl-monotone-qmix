#!/usr/bin/env bash
set -euo pipefail

# Matched S2Q/S2Q-PMIX comparison launcher for PyMARL + SMACv1.
# Both algorithms share the same environment, training budget, seeds, and launcher.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd "${SCRIPT_DIR}"

export ALGS="${ALGS:-s2q s2q_pmix_kan}"
export MAPS="${MAPS:-3s_vs_5z MMM2 6h_vs_8z}"
export SEEDS="${SEEDS:-1 41 141}"
export CUDA_DEVICES="${CUDA_DEVICES:-${CUDA_VISIBLE_DEVICES:-0}}"
export CUDA_DEVICES="${CUDA_DEVICES//,/ }"
export SLOTS_PER_GPU="${SLOTS_PER_GPU:-3}"
export T_MAX="${T_MAX:-2050000}"
export USE_CUDA="${USE_CUDA:-True}"
export LOG_DIR="${LOG_DIR:-parallel_logs/smacv1/s2q_vs_s2q_pmix}"

[[ "${SLOTS_PER_GPU}" =~ ^[1-3]$ ]] || {
  echo '[error] SLOTS_PER_GPU must be 1, 2, or 3.' >&2
  exit 1
}

exec bash "${SCRIPT_DIR}/run_parallel_pymarl_smacv1.sh"
