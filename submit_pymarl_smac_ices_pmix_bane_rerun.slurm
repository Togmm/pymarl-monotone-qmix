#!/bin/bash
#SBATCH --job-name=pymarl_smac_ices_pmix_bane
#SBATCH --chdir=/labmount/users/202535331/MARL/pymarl-monotone-qmix
#SBATCH --output=/labmount/users/202535331/MARL/pymarl-monotone-qmix/slurm_logs/smac_ices_pmix_bane_rerun_out_%A_%a.txt
#SBATCH --error=/labmount/users/202535331/MARL/pymarl-monotone-qmix/slurm_logs/smac_ices_pmix_bane_rerun_err_%A_%a.txt
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=96:00:00
#SBATCH --partition=GPUNorm
#SBATCH --no-requeue

set -euo pipefail

REPO_DIR="/labmount/users/202535331/MARL/pymarl-monotone-qmix"
cd "${REPO_DIR}"
mkdir -p slurm_logs parallel_logs/smac/ices_pmix_bane_rerun

# Run one seed per array task so that each task only needs one GPU.
ARRAY_SEEDS=(1 41 141)
ARRAY_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
if [[ ! "${ARRAY_INDEX}" =~ ^[0-9]+$ || "${ARRAY_INDEX}" -ge "${#ARRAY_SEEDS[@]}" ]]; then
  echo "[error] SLURM_ARRAY_TASK_ID must be in the range 0-$(( ${#ARRAY_SEEDS[@]} - 1 )), got: ${ARRAY_INDEX}" >&2
  exit 1
fi
ARRAY_SEED="${ARRAY_SEEDS[ARRAY_INDEX]}"

export CONDA_ENV="${CONDA_ENV:-pymarl}"
export CONDA_ROOT="${CONDA_ROOT:-/labmount/users/202535331/miniconda3}"
export SC2_ROOT="${SC2_ROOT:-${REPO_DIR}/3rdparty/StarCraftII}"
export CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_DEVICES="${CUDA_DEVICES//,/ }"
# Each array task launches one Bane process for its selected seed. The lock
# also coordinates with other jobs on the same node.
export SLOTS_PER_GPU=1
export BANE_LOCK_SLOTS=1
export LOG_DIR="${LOG_DIR:-parallel_logs/smac/ices_pmix_bane_rerun/job_${SLURM_JOB_ID:-manual}}"
export T_MAX="${T_MAX:-2050000}"
export USE_CUDA="${USE_CUDA:-True}"
export USE_TENSORBOARD="${USE_TENSORBOARD:-True}"
export SAVE_MODEL="${SAVE_MODEL:-True}"
export DRY_RUN="${DRY_RUN:-False}"
export ALGS="ices_pmix_kan"
export MAPS="bane_vs_bane"
export SEEDS="${ARRAY_SEED}"
export EXPERIMENTS="ices_pmix_kan|bane_vs_bane|${ARRAY_SEED}"
# The previous runs reached 21.9--22.5 GiB before failing in the ICES world
# model. Reduce the learner batch while keeping the original replay buffer.
export EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-batch_size=24}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS=1

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY="localhost,127.0.0.1,::1"
export no_proxy="${NO_PROXY}"

echo "${SLURM_JOB_ID:-manual} array=${ARRAY_INDEX} seed=${ARRAY_SEED} $(hostname) ices-pmix bane rerun GPUs=${CUDA_DEVICES}"
echo "[experiments] ${EXPERIMENTS}"
echo "[overrides] ${EXTRA_OVERRIDES}"
bash "${REPO_DIR}/run_parallel_pymarl_smac.sh"
