#!/usr/bin/env bash
#SBATCH --job-name=pymarl_smacv1
#SBATCH --chdir=/labmount/users/202535331/MARL/pymarl-monotone-qmix
#SBATCH --output=/labmount/users/202535331/MARL/pymarl-monotone-qmix/slurm_logs/smacv1_%j.out
#SBATCH --error=/labmount/users/202535331/MARL/pymarl-monotone-qmix/slurm_logs/smacv1_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=96:00:00
#SBATCH --partition=GPUNorm
#SBATCH --no-requeue
set -euo pipefail
REPO_DIR="/labmount/users/202535331/MARL/pymarl-monotone-qmix"; cd "${REPO_DIR}"
mkdir -p slurm_logs parallel_logs/smacv1/gpu
export CONDA_ENV="${CONDA_ENV:-pymarl}" CONDA_ROOT="${CONDA_ROOT:-/labmount/users/202535331/miniconda3}" SC2_ROOT="${SC2_ROOT:-${REPO_DIR}/3rdparty/StarCraftII}"
export ALGS="${ALGS:-qmix qplex}" MAPS="${MAPS:-3m 8m 2s3z 3s5z MMM2 6h_vs_8z}" SEEDS="${SEEDS:-1 41 141}"
export USE_CUDA=True MAX_CONCURRENT=1 T_MAX="${T_MAX:-2050000}" LOG_DIR="parallel_logs/smacv1/gpu/job_${SLURM_JOB_ID:-manual}" DRY_RUN="${DRY_RUN:-False}" OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
if [[ -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then source "${CONDA_ROOT}/etc/profile.d/conda.sh"; conda activate "${CONDA_ENV}" || true; fi
echo "job=${SLURM_JOB_ID:-manual} gpu=${CUDA_VISIBLE_DEVICES:-unset} maps=${MAPS} seeds=${SEEDS}"
bash "${REPO_DIR}/run_parallel_pymarl_smacv1.sh"
