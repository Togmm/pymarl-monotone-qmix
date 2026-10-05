#!/usr/bin/env bash
set -euo pipefail

# Re-run the final PyMARL MonoKAN configuration on SMACv1 MMM2:
#   V(s) + M(q, s)
#
# Defaults match the effective configuration of the final Sacred MMM2 runs.
# The existing ablation scripts are intentionally left unchanged.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

export CUDA_DEVICES="${CUDA_DEVICES:-1}"
export SLOTS_PER_GPU="${SLOTS_PER_GPU:-3}"
export ALGS="${ALGS:-monokan}"
export MAPS="${MAPS:-MMM2}"
export SEEDS="${SEEDS:-1 41 141}"
export T_MAX="${T_MAX:-2050000}"
export USE_CUDA="${USE_CUDA:-True}"
export USE_TENSORBOARD="${USE_TENSORBOARD:-True}"
export SAVE_MODEL="${SAVE_MODEL:-True}"
export LOG_DIR="${LOG_DIR:-parallel_logs/smacv1_final_mmm2}"

cd "${SCRIPT_DIR}"
exec "${SCRIPT_DIR}/run_parallel_pymarl_smac.sh"
