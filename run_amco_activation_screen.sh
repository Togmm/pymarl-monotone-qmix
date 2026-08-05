#!/usr/bin/env bash
set -euo pipefail

# Fifteen-run validation for the five remaining standard maps.
# Use one common AMCO configuration: centered Softplus(beta=2.0) and
# state_input_scale=0.3 for every map and seed.
T_MAX="${T_MAX:-2050000}"
USE_TENSORBOARD="${USE_TENSORBOARD:-True}"
LOG_DIR="${LOG_DIR:-parallel_logs/amco_centered_softplus_b20_scale030_remaining_maps}"
CONDA_ENV="${CONDA_ENV:-pymarl}"
CUDA_DEVICES="${CUDA_DEVICES:-0 1 2 3 4 5 6 7 8 9 10 11}"

mkdir -p "${LOG_DIR}"

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV}"
else
  echo "[warn] conda command not found; assuming the correct environment is active."
fi

read -r -a CUDA_DEVICE_LIST <<< "${CUDA_DEVICES}"
if [[ "${#CUDA_DEVICE_LIST[@]}" -lt 12 ]]; then
  echo '[error] This batch script requires at least 12 CUDA devices.'
  echo '       Example: CUDA_DEVICES="0 1 2 3 4 5 6 7 8 9 10 11"'
  exit 1
fi

launch_index=0

run_exp() {
  local map_name="$1"
  local seed="$2"
  local cuda_device="${CUDA_DEVICE_LIST[$((launch_index % ${#CUDA_DEVICE_LIST[@]}))]}"
  local name="amco_centered_softplus_b20_scale030_remaining_${map_name}_seed${seed}"
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

run_batch() {
  local map_name
  for map_name in "$@"; do
    run_exp "${map_name}" "1"
    run_exp "${map_name}" "41"
    run_exp "${map_name}" "141"
  done
  echo "Waiting for $# maps ($# x 3 AMCO runs) in the current GPU batch..."
  wait
  echo "AMCO batch finished."
}

# Batch 1: 4 maps x 3 seeds = 12 simultaneous jobs, one per GPU.
run_batch "1c3s5z" "2s3z" "MMM2" "27m_vs_30m"

# Batch 2: the remaining map, using three of the now-free GPUs.
launch_index=0
run_batch "bane_vs_bane"

echo "All fifteen AMCO beta=2 scale=0.3 validation runs finished."
