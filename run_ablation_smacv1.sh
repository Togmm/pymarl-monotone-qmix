#!/usr/bin/env bash
set -euo pipefail

# B1/B2/B3 SMACv1 ablation matrix:
#   1 map x 3 mixers x 3 seeds = 9 jobs.
# The existing runner assigns SLOTS_PER_GPU jobs to each GPU at a time.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

export CUDA_DEVICES="${CUDA_DEVICES:-1}"
export SLOTS_PER_GPU="${SLOTS_PER_GPU:-3}"
export ALGS="${ALGS:-${B1_CONFIG:-pmix_monokan_b1} ${B2_CONFIG:-pmix_monokan_b2} ${B3_CONFIG:-pmix_monokan_b3}}"
export MAPS="${MAPS:-5m_vs_6m}"
export SEEDS="${SEEDS:-1 41 141}"
export LOG_DIR="${LOG_DIR:-parallel_logs/smacv1_ablation}"

cd "${SCRIPT_DIR}"
exec "${SCRIPT_DIR}/run_parallel_pymarl_smac.sh"
