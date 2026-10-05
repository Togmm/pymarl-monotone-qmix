#!/usr/bin/env bash
set -euo pipefail

# PMIX versus S2Q-PMIX-KAN on PyMARL + SMACv1.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd "${SCRIPT_DIR}"

export ALGS="${ALGS:-monokan s2q_pmix_kan}"
export MAPS="${MAPS:-3s_vs_5z MMM2 6h_vs_8z}"
export SEEDS="${SEEDS:-1 41 141}"
export CUDA_DEVICES="${CUDA_DEVICES:-${CUDA_VISIBLE_DEVICES:-0}}"
export CUDA_DEVICES="${CUDA_DEVICES//,/ }"
export SLOTS_PER_GPU="${SLOTS_PER_GPU:-3}"
export T_MAX="${T_MAX:-2050000}"
export USE_CUDA="${USE_CUDA:-True}"
export LOG_DIR="${LOG_DIR:-parallel_logs/smacv1/s2q_pmix}"

exec bash "${SCRIPT_DIR}/run_parallel_pymarl_smacv1.sh"
