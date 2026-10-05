#!/usr/bin/env bash
set -euo pipefail

# Matched QMIX/ICES-QMIX launcher for PyMARL + SMACv1.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd "${SCRIPT_DIR}"

export ALGS="${ALGS:-qmix ices}"
export MAPS="${MAPS:-3s_vs_5z 5m_vs_6m bane_vs_bane MMM2}"
export SEEDS="${SEEDS:-1 41 141}"
export CUDA_DEVICES="${CUDA_DEVICES:-${CUDA_VISIBLE_DEVICES:-0 1}}"
export CUDA_DEVICES="${CUDA_DEVICES//,/ }"
export SLOTS_PER_GPU="${SLOTS_PER_GPU:-1}"
export T_MAX="${T_MAX:-2050000}"
export LOG_DIR="${LOG_DIR:-parallel_logs/smac/ices}"

exec bash "${SCRIPT_DIR}/run_parallel_pymarl_smac.sh"
