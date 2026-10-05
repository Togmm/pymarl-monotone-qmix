#!/usr/bin/env bash
set -euo pipefail

# QPLEX baseline launcher for PyMARL + SMACv1.
# The existing PyMARL runner creates three workers per GPU. Each worker runs
# independent map/seed jobs sequentially, leaving later jobs in its queue.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd "${SCRIPT_DIR}"

export ALGS="${ALGS:-qplex}"
# Keep the default matrix identical to submit_pymarl_smac_qplex.slurm.
export MAPS="${MAPS:-3s_vs_5z 5m_vs_6m MMM2}"
export SEEDS="${SEEDS:-1 41 141}"
export CUDA_DEVICES="${CUDA_DEVICES:-${CUDA_VISIBLE_DEVICES:-0}}"
export CUDA_DEVICES="${CUDA_DEVICES//,/ }"
export SLOTS_PER_GPU="${SLOTS_PER_GPU:-3}"
export T_MAX="${T_MAX:-2050000}"
export USE_CUDA="${USE_CUDA:-True}"
export LOG_DIR="${LOG_DIR:-parallel_logs/smac/qplex}"

[[ "${SLOTS_PER_GPU}" =~ ^[1-3]$ ]] || {
  echo '[error] SLOTS_PER_GPU must be 1, 2, or 3.' >&2
  exit 1
}

exec bash "${SCRIPT_DIR}/run_parallel_pymarl_smac.sh"
