#!/usr/bin/env bash
set -euo pipefail

# Four-run validation for the unified AMCO state-input scale.
# Re-run the interrupted 3s5z seed and test scale=0.3 on all three 5m_vs_6m
# seeds while keeping Centered Softplus(beta=2.0) fixed.
T_MAX="${T_MAX:-2050000}"
USE_TENSORBOARD="${USE_TENSORBOARD:-True}"
LOG_DIR="${LOG_DIR:-parallel_logs/amco_centered_softplus_b20_scale030_validation}"
CONDA_ENV="${CONDA_ENV:-pymarl}"
CUDA_DEVICES="${CUDA_DEVICES:-0 1 2 3}"

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
  local name="amco_centered_softplus_b20_scale030_${map_name}_seed${seed}"
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
      amco_state_input_scale="0.3" \
      amco_state_input_scale_by_map."${map_name}"="0.3" \
      name="${name}"
  ) > "${log_file}" 2>&1 &
}

run_exp "1c3s5z" "1"
run_exp "1c3s5z" "41"
run_exp "1c3s5z" "141" 

run_exp "2s3z" "1"
run_exp "2s3z" "41"
run_exp "2s3z" "141"

run_exp "MMM2" "1"
run_exp "MMM2" "41"
run_exp "MMM2" "141"

echo "Waiting for four AMCO beta=2 scale=0.3 validation runs..."
wait
echo "AMCO beta=2 scale=0.3 validation batch finished."

run_exp "27m_vs_30m" "1"
run_exp "27m_vs_30m" "41"
run_exp "27m_vs_30m" "141"

echo "Waiting for four AMCO beta=2 scale=0.3 validation runs..."
wait
echo "AMCO beta=2 scale=0.3 validation batch finished."

run_exp "bane_vs_bane" "1"
run_exp "bane_vs_bane" "41"
run_exp "bane_vs_bane" "141"

echo "Waiting for four AMCO beta=2 scale=0.3 validation runs..."
wait
echo "AMCO beta=2 scale=0.3 validation batch finished."