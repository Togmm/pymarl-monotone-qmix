#!/usr/bin/env bash
set -euo pipefail

# Run from the PyMARL project root, e.g.:
#   cd ~/pymarl
#   bash run_parallel_pymarl_exps.sh
#
# Each experiment writes its terminal log to ./parallel_logs.
#
# CUDA selection:
#   CUDA_DEVICES="0" ./run_parallel_pymarl_exps.sh
#   CUDA_DEVICES="0 1" ./run_parallel_pymarl_exps.sh
# Multiple device ids are assigned round-robin by launch order.

T_MAX="${T_MAX:-2050000}"
USE_TENSORBOARD="${USE_TENSORBOARD:-True}"
LOG_DIR="${LOG_DIR:-parallel_logs}"
CONDA_ENV="${CONDA_ENV:-pymarl}"
CUDA_DEVICES="${CUDA_DEVICES:-0}"

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
  local cuda_device="$5"
  shift 5
  local log_file="${LOG_DIR}/${name}_seed${seed}.log"

  echo "[launch] ${name} seed=${seed} map=${map_name} cuda=${cuda_device} -> ${log_file}"
  (
    echo "[info] CUDA_VISIBLE_DEVICES=${cuda_device}"
    CUDA_VISIBLE_DEVICES="${cuda_device}" python src/main.py \
      --config="${config}" \
      --env-config=sc2 \
      with \
      env_args.map_name="${map_name}" \
      use_tensorboard="${USE_TENSORBOARD}" \
      t_max="${T_MAX}" \
      seed="${seed}" \
      name="${name}_seed${seed}" \
      "$@"
  ) > "${log_file}" 2>&1 &
  PIDS+=("$!")
  JOB_NAMES+=("${name}_seed${seed}")
}

# Default seeds. Override by editing this array or running:
#   SEEDS="1 41 338784093" bash run_parallel_pymarl_exps.sh
if [[ -n "${SEEDS:-}" ]]; then
  read -r -a SEED_LIST <<< "${SEEDS}"
else
  SEED_LIST=(1 41 141)
fi

read -r -a CUDA_DEVICE_LIST <<< "${CUDA_DEVICES}"
if [[ "${#CUDA_DEVICE_LIST[@]}" -eq 0 ]]; then
  echo "[error] CUDA_DEVICES is empty. Example: CUDA_DEVICES=\"0 1\""
  exit 1
fi

launch_index=0
PIDS=()
JOB_NAMES=()

next_cuda_device() {
  NEXT_CUDA_DEVICE="${CUDA_DEVICE_LIST[$((launch_index % ${#CUDA_DEVICE_LIST[@]}))]}"
  launch_index=$((launch_index + 1))
}

# Learned-calibrator HLL evaluation. With the default three seeds this
# launches six jobs concurrently, using each map's native baseline temperature
# and lattice size. The only changed mechanism is the shared Q calibrator.
for seed in "${SEED_LIST[@]}"; do
  next_cuda_device
  run_exp \
    "hll_calibrated_MMM2" "hll" "MMM2" "${seed}" "${NEXT_CUDA_DEVICE}" \
    hll_q_calibrator_enabled=True
  next_cuda_device
  run_exp \
    "hll_calibrated_3s_vs_5z" "hll" "3s_vs_5z" "${seed}" "${NEXT_CUDA_DEVICE}" \
    hll_q_calibrator_enabled=True
  next_cuda_device
  run_exp \
    "hll_calibrated_5m_vs_6m" "hll" "5m_vs_6m" "${seed}" "${NEXT_CUDA_DEVICE}" \
    hll_q_calibrator_enabled=True
done

echo
echo "Launched ${#PIDS[@]} jobs (${#SEED_LIST[@]} seeds x 2 experiments)."
echo "CUDA devices: ${CUDA_DEVICES}"
echo "Waiting for all jobs to finish..."

failed=0
for index in "${!PIDS[@]}"; do
  if ! wait "${PIDS[$index]}"; then
    echo "[error] ${JOB_NAMES[$index]} failed. Check its parallel log."
    failed=1
  fi
done

if [[ "${failed}" -ne 0 ]]; then
  echo "One or more experiments failed."
  exit 1
fi

echo "All experiments finished successfully."
