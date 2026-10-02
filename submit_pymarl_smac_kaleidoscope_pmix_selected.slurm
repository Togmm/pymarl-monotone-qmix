#!/bin/bash
#SBATCH --job-name=pymarl_smac_kpmix_selected
#SBATCH --chdir=/labmount/users/202535331/MARL/pymarl-monotone-qmix
#SBATCH --output=/labmount/users/202535331/MARL/pymarl-monotone-qmix/slurm_logs/smac_kaleidoscope_pmix_selected_out_%j.txt
#SBATCH --error=/labmount/users/202535331/MARL/pymarl-monotone-qmix/slurm_logs/smac_kaleidoscope_pmix_selected_err_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:3
#SBATCH --time=96:00:00
#SBATCH --partition=GPUNorm
#SBATCH --no-requeue

set -euo pipefail

REPO_DIR="/labmount/users/202535331/MARL/pymarl-monotone-qmix"
cd "${REPO_DIR}"
mkdir -p slurm_logs parallel_logs/smac/kaleidoscope_pmix_selected

export CONDA_ENV="${CONDA_ENV:-pymarl}"
export CONDA_ROOT="${CONDA_ROOT:-/labmount/users/202535331/miniconda3}"
export SC2_ROOT="${SC2_ROOT:-${REPO_DIR}/3rdparty/StarCraftII}"
export CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0 1 2}"
export CUDA_DEVICES="${CUDA_DEVICES//,/ }"
# Keep non-bane maps concurrent, while the shared node-wide lock serializes
# bane_vs_bane. Two workers per GPU is safe for the ordinary maps.
export SLOTS_PER_GPU="${SLOTS_PER_GPU:-1}"
export BANE_LOCK_SLOTS=1
export LOG_DIR="${LOG_DIR:-parallel_logs/smac/kaleidoscope_pmix_selected/job_${SLURM_JOB_ID:-manual}}"
export T_MAX="${T_MAX:-2050000}"
export USE_CUDA="${USE_CUDA:-True}"
export USE_TENSORBOARD="${USE_TENSORBOARD:-True}"
export SAVE_MODEL="${SAVE_MODEL:-True}"
export DRY_RUN="${DRY_RUN:-False}"
export ALGS="kaleidoscope_pmix_kan"
export MAPS="3s_vs_5z MMM2 bane_vs_bane"
export SEEDS="1 41 141"
export EXPERIMENTS="kaleidoscope_pmix_kan|bane_vs_bane|1;kaleidoscope_pmix_kan|bane_vs_bane|141"
export OMP_NUM_THREADS=1

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY="localhost,127.0.0.1,::1"
export no_proxy="${NO_PROXY}"

echo "${SLURM_JOB_ID:-manual} $(hostname) kaleidoscope-pmix selected SMAC experiments"
echo "[experiments] ${EXPERIMENTS}"
bash "${REPO_DIR}/run_parallel_pymarl_smac.sh"
