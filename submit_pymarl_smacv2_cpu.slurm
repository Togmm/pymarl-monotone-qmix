#!/usr/bin/env bash
#SBATCH --job-name=pymarl_smacv2_cpu
#SBATCH --chdir=/labmount/users/202535331/MARL/pymarl-monotone-qmix
#SBATCH --output=/labmount/users/202535331/MARL/pymarl-monotone-qmix/slurm_logs/smacv2_cpu_%j.out
#SBATCH --error=/labmount/users/202535331/MARL/pymarl-monotone-qmix/slurm_logs/smacv2_cpu_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=96:00:00
#SBATCH --no-requeue
set -euo pipefail
REPO_DIR="/labmount/users/202535331/MARL/pymarl-monotone-qmix"; cd "${REPO_DIR}"
mkdir -p slurm_logs parallel_logs/smacv2/cpu
export CONDA_ENV="${CONDA_ENV:-pymarl}" CONDA_ROOT="${CONDA_ROOT:-/labmount/users/202535331/miniconda3}"
export ALGS="${ALGS:-qmix qplex}" SCENARIOS="${SCENARIOS:-protoss_5_vs_5 protoss_5_vs_6 terran_5_vs_5 terran_5_vs_6 zerg_5_vs_5 zerg_5_vs_6}" SEEDS="${SEEDS:-1 41 141}"
export USE_CUDA=False MAX_CONCURRENT="${MAX_CONCURRENT:-1}" T_MAX="${T_MAX:-5050000}" LOG_DIR="parallel_logs/smacv2/cpu/job_${SLURM_JOB_ID:-manual}" DRY_RUN="${DRY_RUN:-False}" OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
if [[ -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then source "${CONDA_ROOT}/etc/profile.d/conda.sh"; conda activate "${CONDA_ENV}" || true; fi
bash "${REPO_DIR}/run_parallel_pymarl_smacv2.sh"
