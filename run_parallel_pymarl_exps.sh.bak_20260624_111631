#!/usr/bin/env bash
set -euo pipefail

# Run from the PyMARL project root, e.g.:
#   cd ~/pymarl
#   bash run_parallel_pymarl_exps.sh
#
# Each experiment writes its terminal log to ./parallel_logs.

T_MAX="${T_MAX:-2050000}"
USE_TENSORBOARD="${USE_TENSORBOARD:-True}"
LOG_DIR="${LOG_DIR:-parallel_logs}"
CONDA_ENV="${CONDA_ENV:-marl}"

mkdir -p "${LOG_DIR}"

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV}"
else
  echo "[warn] conda command not found; assuming the correct Python env is already active."
fi

run_exp() {
  local name="$1"
  local config="$2"
  local map_name="$3"
  local seed="$4"
  local log_file="${LOG_DIR}/${name}_seed${seed}.log"

  echo "[launch] ${name} seed=${seed} map=${map_name} -> ${log_file}"
  python src/main.py \
    --config="${config}" \
    --env-config=sc2 \
    with \
    env_args.map_name="${map_name}" \
    use_tensorboard="${USE_TENSORBOARD}" \
    t_max="${T_MAX}" \
    seed="${seed}" \
    name="${name}_seed${seed}" \
    > "${log_file}" 2>&1 &
}

# Default seeds. Override by editing this array or running:
#   SEEDS="1 41 3407" bash run_parallel_pymarl_exps.sh
if [[ -n "${SEEDS:-}" ]]; then
  read -r -a SEED_LIST <<< "${SEEDS}"
else
  SEED_LIST=(1 41 3407)
fi

for seed in "${SEED_LIST[@]}"; do
  run_exp "hll_3s_vs_5z" "hll" "3s_vs_5z" "${seed}"
  run_exp "monokan_MMM2" "monokan" "MMM2" "${seed}"
  run_exp "amco_MMM2" "amco" "MMM2" "${seed}"
  run_exp "amco_3s_vs_5z" "amco" "3s_vs_5z" "${seed}"
done

echo
echo "Launched ${#SEED_LIST[@]} seed(s) x 4 experiments."
echo "Waiting for all jobs to finish..."

wait

echo "All experiments finished."
