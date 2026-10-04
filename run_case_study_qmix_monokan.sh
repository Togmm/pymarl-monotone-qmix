#!/usr/bin/env bash
set -euo pipefail

# Run the basic QMIX and MonoKAN case study.
#
# Default workload: 2 algorithms x 3 seeds = 6 jobs, on two GPUs.
# The scheduler in run_parallel_pymarl_smac.sh keeps one job active on each
# GPU by default, so two jobs run concurrently and each GPU receives three
# jobs over the lifetime of this script.
#
# Examples:
#   bash run_case_study_qmix_monokan.sh
#   MAPS="5m_vs_6m" CUDA_DEVICES="0 1" bash run_case_study_qmix_monokan.sh
#   T_MAX=500000 SAVE_MODEL_INTERVAL=100000 bash run_case_study_qmix_monokan.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd "${SCRIPT_DIR}"

# Keep the baseline algorithm configs and only add checkpoint persistence.
export ALGS="${ALGS:-qmix monokan}"
export MAPS="${MAPS:-MMM2}"
export SEEDS="${SEEDS:-41}"
export CUDA_DEVICES="${CUDA_DEVICES:-0}"
export SLOTS_PER_GPU="${SLOTS_PER_GPU:-2}"

export T_MAX="${T_MAX:-2050000}"
export USE_CUDA="${USE_CUDA:-True}"
export USE_TENSORBOARD="${USE_TENSORBOARD:-True}"
export SAVE_MODEL="True"
export SAVE_MODEL_INTERVAL="${SAVE_MODEL_INTERVAL:-2000000}"
export LOG_DIR="${LOG_DIR:-parallel_logs/case_study_qmix_monokan}"

# Preserve any caller supplied overrides while forcing checkpoint interval to
# be explicit in the command line. `save_model=True` is passed by the shared
# scheduler through SAVE_MODEL above.
export EXTRA_OVERRIDES="save_model_interval=${SAVE_MODEL_INTERVAL}${EXTRA_OVERRIDES:+ ${EXTRA_OVERRIDES}}"

exec bash "${SCRIPT_DIR}/run_parallel_pymarl_smac.sh"
