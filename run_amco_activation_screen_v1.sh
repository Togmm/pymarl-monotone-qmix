#!/usr/bin/env bash
set -euo pipefail

# One-seed activation screen for the AMCO 5m_vs_6m failure case.
# ReLU seed 141 already exists in the baseline results and is not rerun here.
SEED="${SEED:-141}"
T_MAX="${T_MAX:-2050000}"
USE_TENSORBOARD="${USE_TENSORBOARD:-True}"
LOG_DIR="${LOG_DIR:-parallel_logs/amco_activation_screen}"
CONDA_ENV="${CONDA_ENV:-pymarl}"
CUDA_DEVICES="${CUDA_DEVICES:-0}"

mkdir -p "${LOG_DIR}"

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV}"
else
  echo "[warn] conda command not found; assuming the correct environment is active."
fi

read -r -a CUDA_DEVICE_LIST <<< "${CUDA_DEVICES}"
if [[ "${#CUDA_DEVICE_LIST[@]}" -eq 0 ]]; then
  echo '[error] CUDA_DEVICES is empty. Example: CUDA_DEVICES="0 1 2"'
  exit 1
fi

launch_index=0

run_exp() {
  local label="$1"
  local activation="$2"
  local beta="$3"
  local cuda_device="${CUDA_DEVICE_LIST[$((launch_index % ${#CUDA_DEVICE_LIST[@]}))]}"
  local name="amco_${label}_5m_vs_6m_seed${SEED}"
  local log_file="${LOG_DIR}/${name}.log"
  launch_index=$((launch_index + 1))

  echo "[launch] ${name} cuda=${cuda_device} -> ${log_file}"
  (
    CUDA_VISIBLE_DEVICES="${cuda_device}" python src/main.py \
      --config=amco \
      --env-config=sc2 \
      with \
      env_args.map_name=5m_vs_6m \
      use_tensorboard="${USE_TENSORBOARD}" \
      t_max="${T_MAX}" \
      seed="${SEED}" \
      amco_mono_activation="${activation}" \
      amco_mono_softplus_beta="${beta}" \
      name="${name}"
  ) > "${log_file}" 2>&1 &
}

run_exp "softplus_b05" "softplus" "0.5"
run_exp "softplus_b10" "softplus" "1.0"
run_exp "elu" "elu" "1.0"

echo "Waiting for three AMCO activation-screen runs..."
wait
echo "AMCO activation screen finished."
