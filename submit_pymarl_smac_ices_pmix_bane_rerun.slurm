#!/bin/bash
#SBATCH --job-name=pymarl_smac_ices_pmix_bane
#SBATCH --chdir=/labmount/users/202535331/MARL/pymarl-monotone-qmix
#SBATCH --output=/labmount/users/202535331/MARL/pymarl-monotone-qmix/slurm_logs/smac_ices_pmix_bane_rerun_out_%j.txt
#SBATCH --error=/labmount/users/202535331/MARL/pymarl-monotone-qmix/slurm_logs/smac_ices_pmix_bane_rerun_err_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:2
#SBATCH --nodelist=gn12,gn13
#SBATCH --time=96:00:00
#SBATCH --partition=GPUNorm
#SBATCH --no-requeue

set -euo pipefail

REPO_DIR="/labmount/users/202535331/MARL/pymarl-monotone-qmix"
cd "${REPO_DIR}"
mkdir -p slurm_logs parallel_logs/smac/ices_pmix_bane_rerun

export CONDA_ENV="${CONDA_ENV:-pymarl}"
export CONDA_ROOT="${CONDA_ROOT:-/labmount/users/202535331/miniconda3}"
export SC2_ROOT="${SC2_ROOT:-${REPO_DIR}/3rdparty/StarCraftII}"
export CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0 1}"
export CUDA_DEVICES="${CUDA_DEVICES//,/ }"
# Only one Bane process is launched, and it is run serially for the three
# failed seeds. The lock also coordinates with other jobs on the same node.
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
export SEEDS="1 41 141"
export EXPERIMENTS="ices_pmix_kan|bane_vs_bane|1;ices_pmix_kan|bane_vs_bane|41;ices_pmix_kan|bane_vs_bane|141"
# The previous runs reached 21.9--22.5 GiB before failing in the ICES world
# model. Reduce the learner batch while keeping the original replay buffer.
export EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-batch_size=32}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS=1

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY="localhost,127.0.0.1,::1"
export no_proxy="${NO_PROXY}"

echo "${SLURM_JOB_ID:-manual} $(hostname) ices-pmix bane rerun GPUs=${CUDA_DEVICES}"
echo "[experiments] ${EXPERIMENTS}"
echo "[overrides] ${EXTRA_OVERRIDES}"
bash "${REPO_DIR}/run_parallel_pymarl_smac.sh"
