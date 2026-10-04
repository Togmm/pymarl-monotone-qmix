#!/bin/bash
# Submit with:
#   sbatch submit_case_study_qmix_monokan.slurm

#SBATCH --job-name=pymarl_case_qmix_monokan
#SBATCH --chdir=/labmount/users/202535331/MARL/pymarl-monotone-qmix
#SBATCH --output=/labmount/users/202535331/MARL/pymarl-monotone-qmix/slurm_logs/case_study_qmix_monokan_out_%j.txt
#SBATCH --error=/labmount/users/202535331/MARL/pymarl-monotone-qmix/slurm_logs/case_study_qmix_monokan_err_%j.txt
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
mkdir -p slurm_logs parallel_logs/case_study_qmix_monokan

export CONDA_ENV="${CONDA_ENV:-pymarl}"
export CONDA_ROOT="${CONDA_ROOT:-/labmount/users/202535331/miniconda3}"
export SC2_ROOT="${SC2_ROOT:-${REPO_DIR}/3rdparty/StarCraftII}"

# Slurm may expose this as either "0,1" or "0 1".
export CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_DEVICES="${CUDA_DEVICES//,/ }"
export SLOTS_PER_GPU="${SLOTS_PER_GPU:-2}"

export T_MAX="${T_MAX:-2050000}"
export USE_CUDA="${USE_CUDA:-True}"
export USE_TENSORBOARD="${USE_TENSORBOARD:-True}"
export SAVE_MODEL="True"
export SAVE_MODEL_INTERVAL="${SAVE_MODEL_INTERVAL:-2000000}"
export DRY_RUN="${DRY_RUN:-False}"
export LOG_DIR="${LOG_DIR:-parallel_logs/case_study_qmix_monokan/job_${SLURM_JOB_ID:-manual}}"

# Keep the six-job case-study matrix explicit in the Slurm submission.
export ALGS="monokan qmix"
export MAPS="${MAPS:-MMM2}"
export SEEDS="${SEEDS:-41}"
export OMP_NUM_THREADS=1

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY="localhost,127.0.0.1,::1"
export no_proxy="${NO_PROXY}"

echo "${SLURM_JOB_ID:-manual} $(hostname) qmix+monokan GPUs=${CUDA_DEVICES} maps=${MAPS} seeds=${SEEDS}"
bash "${REPO_DIR}/run_case_study_qmix_monokan.sh"
