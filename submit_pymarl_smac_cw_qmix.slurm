#!/bin/bash
#SBATCH --job-name=pymarl_smac_cw
#SBATCH --chdir=/labmount/users/202535331/MARL/pymarl-monotone-qmix
#SBATCH --output=/labmount/users/202535331/MARL/pymarl-monotone-qmix/slurm_logs/smac_cw_qmix_out_%j.txt
#SBATCH --error=/labmount/users/202535331/MARL/pymarl-monotone-qmix/slurm_logs/smac_cw_qmix_err_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gres=gpu:2
#SBATCH --time=96:00:00
#SBATCH --partition=GPUNorm
#SBATCH --no-requeue

set -euo pipefail

REPO_DIR="/labmount/users/202535331/MARL/pymarl-monotone-qmix"
cd "${REPO_DIR}"
mkdir -p slurm_logs parallel_logs/smac/cw_qmix

export CONDA_ENV="${CONDA_ENV:-pymarl}"
export CONDA_ROOT="${CONDA_ROOT:-/labmount/users/202535331/miniconda3}"
export SC2_ROOT="${SC2_ROOT:-${REPO_DIR}/3rdparty/StarCraftII}"
export CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0 1}"
export CUDA_DEVICES="${CUDA_DEVICES//,/ }"
export SLOTS_PER_GPU="${SLOTS_PER_GPU:-2}"
export LOG_DIR="${LOG_DIR:-parallel_logs/smac/cw_qmix/job_${SLURM_JOB_ID:-manual}}"
export T_MAX="${T_MAX:-2050000}"
export USE_CUDA="${USE_CUDA:-True}"
export USE_TENSORBOARD="${USE_TENSORBOARD:-True}"
export SAVE_MODEL="${SAVE_MODEL:-True}"
export DRY_RUN="${DRY_RUN:-False}"
export ALGS="cw_qmix"
export MAPS="${MAPS:-3s_vs_5z 5m_vs_6m bane_vs_bane MMM2}"
export SEEDS="${SEEDS:-1 41 141}"
export OMP_NUM_THREADS=1

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY="localhost,127.0.0.1,::1"
export no_proxy="${NO_PROXY}"

echo "${SLURM_JOB_ID:-manual} $(hostname) cw_qmix smac GPUs=${CUDA_DEVICES} maps=${MAPS} seeds=${SEEDS}"
bash "${REPO_DIR}/run_parallel_pymarl_smac.sh"
