#!/usr/bin/env bash
set -euo pipefail

# Six-run diagnostic screen for Centered Softplus(beta=2.0).
# The two difficult maps test recovery of state-conditioned credit; 5m_vs_6m
# checks whether moving beta toward ReLU preserves the centered stability gain.
T_MAX="${T_MAX:-2050000}"
USE_TENSORBOARD="${USE_TENSORBOARD:-True}"
LOG_DIR="${LOG_DIR:-parallel_logs/amco_centered_softplus_b20_diagnostic_screen}"
CONDA_ENV="${CONDA_ENV:-pymarl}"
CUDA_DEVICES="${CUDA_DEVICES:-0 1 2 3 4 5}"

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
  local map_name="$1"
  local seed="$2"
  local cuda_device="${CUDA_DEVICE_LIST[$((launch_index % ${#CUDA_DEVICE_LIST[@]}))]}"
  local name="amco_centered_softplus_b20_${map_name}_seed${seed}"
  local log_file="${LOG_DIR}/${name}.log"
  launch_index=$((launch_index + 1))

  echo "[launch] ${name} cuda=${cuda_device} -> ${log_file}"
  (
    CUDA_VISIBLE_DEVICES="${cuda_device}" python src/main.py \
      --config=amco \
      --env-config=sc2 \
      with \
      env_args.map_name="${map_name}" \
      use_tensorboard="${USE_TENSORBOARD}" \
      t_max="${T_MAX}" \
      seed="${seed}" \
      amco_mono_activation="centered_softplus" \
      amco_mono_softplus_beta="2.0" \
      name="${name}"
  ) > "${log_file}" 2>&1 &
}

# Two reliable mechanism seeds per map; seed 1 is reserved for the follow-up
# only if this candidate passes the paired seed 41/141 screen.
run_exp "3s5z" "41"
run_exp "3s5z" "141"

run_exp "2c_vs_64zg" "41"
run_exp "2c_vs_64zg" "141"

run_exp "5m_vs_6m" "41"
run_exp "5m_vs_6m" "141"

echo "Waiting for six AMCO Centered Softplus beta=2 diagnostic runs..."
wait
echo "AMCO Centered Softplus beta=2 diagnostic batch finished."
