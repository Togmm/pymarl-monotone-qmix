#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-pymarl}"
CONDA_ROOT="${CONDA_ROOT:-/labmount/users/202535331/miniconda3}"
CUDA_DEVICES="${CUDA_DEVICES:-${CUDA_VISIBLE_DEVICES:-0}}"
CUDA_DEVICES="${CUDA_DEVICES//,/ }"
SLOTS_PER_GPU="${SLOTS_PER_GPU:-1}"
LOG_DIR="${LOG_DIR:-parallel_logs/smac}"
T_MAX="${T_MAX:-2050000}"
USE_CUDA="${USE_CUDA:-True}"
USE_TENSORBOARD="${USE_TENSORBOARD:-True}"
SAVE_MODEL="${SAVE_MODEL:-True}"
DRY_RUN="${DRY_RUN:-False}"
EXPERIMENTS="${EXPERIMENTS:-}"
EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"
REPO_DIR="$(pwd -P)"
SC2_ROOT="${SC2_ROOT:-${REPO_DIR}/3rdparty/StarCraftII}"
BANE_LOCK_FILE="${BANE_LOCK_FILE:-/tmp/pymarl_bane_vs_bane_${USER:-unknown}.lock}"
# A Bane run needs the whole GPU because its peak CUDA memory is much higher
# than the other SMAC maps.  The admission lock prevents ordinary jobs from
# entering once a Bane job has reserved the GPU; the active lock lets ordinary
# jobs share the GPU while no Bane job is waiting or running.
GPU_LOCK_DIR="${GPU_LOCK_DIR:-/tmp/pymarl_gpu_locks_${USER:-unknown}}"
# bane_vs_bane allocates a large replay buffer and also starts an SC2 process.
# Keep one node-wide slot by default. Other maps still run in the remaining
# workers because only bane_vs_bane acquires this lock.
BANE_LOCK_SLOTS=1

DEFAULT_ALGS=(qmix ow_qmix cw_qmix qplex s2q)
DEFAULT_MAPS=(3s5z 2c_vs_64zg MMM2 3s_vs_5z corridor 6h_vs_8z)
DEFAULT_SEEDS=(1 41 141)

[[ -f "${REPO_DIR}/src/main.py" && -f "${REPO_DIR}/src/config/envs/sc2.yaml" ]] || {
  echo '[error] Run this script from the PyMARL repository root.' >&2
  exit 1
}
[[ -x "${SC2_ROOT}/Versions/Base75689/SC2_x64" ]] || {
  echo "[error] StarCraft II executable is missing: ${SC2_ROOT}/Versions/Base75689/SC2_x64" >&2
  exit 1
}

if [[ -n "${ALGS:-}" ]]; then read -r -a ALG_LIST <<< "${ALGS}"; else ALG_LIST=("${DEFAULT_ALGS[@]}"); fi
if [[ -n "${MAPS:-}" ]]; then read -r -a MAP_LIST <<< "${MAPS}"; else MAP_LIST=("${DEFAULT_MAPS[@]}"); fi
if [[ -n "${SEEDS:-}" ]]; then read -r -a SEED_LIST <<< "${SEEDS}"; else SEED_LIST=("${DEFAULT_SEEDS[@]}"); fi
read -r -a CUDA_DEVICE_LIST <<< "${CUDA_DEVICES}"

(( ${#CUDA_DEVICE_LIST[@]} && ${#ALG_LIST[@]} && ${#MAP_LIST[@]} && ${#SEED_LIST[@]} )) || {
  echo '[error] Empty algorithm/map/seed/GPU list.' >&2
  exit 1
}
[[ "${SLOTS_PER_GPU}" =~ ^[1-9][0-9]*$ ]] || { echo '[error] SLOTS_PER_GPU must be positive.' >&2; exit 1; }
[[ "${BANE_LOCK_SLOTS}" =~ ^[1-9][0-9]*$ ]] || { echo '[error] BANE_LOCK_SLOTS must be positive.' >&2; exit 1; }
[[ "${T_MAX}" =~ ^[1-9][0-9]*$ ]] || { echo '[error] T_MAX must be a positive integer.' >&2; exit 1; }
for alg in "${ALG_LIST[@]}"; do
  [[ -f "src/config/algs/${alg}.yaml" ]] || { echo "[error] Missing algorithm config: ${alg}" >&2; exit 1; }
done
for map in "${MAP_LIST[@]}"; do
  [[ "${map}" =~ ^[A-Za-z0-9_]+$ ]] || { echo "[error] Invalid map name: ${map}" >&2; exit 1; }
  [[ -f "${SC2_ROOT}/Maps/SMAC_Maps/${map}.SC2Map" ]] || {
    echo "[error] SMAC map is missing: ${SC2_ROOT}/Maps/SMAC_Maps/${map}.SC2Map" >&2
    exit 1
  }
done
for seed in "${SEED_LIST[@]}"; do
  [[ "${seed}" =~ ^[0-9]+$ ]] || { echo "[error] Invalid seed: ${seed}" >&2; exit 1; }
done
case "${DRY_RUN}" in
  True|true|TRUE|1|Yes|yes|YES) DRY_RUN_ENABLED=1 ;;
  False|false|FALSE|0|No|no|NO) DRY_RUN_ENABLED=0 ;;
  *) echo '[error] DRY_RUN must be True or False.' >&2; exit 1 ;;
esac

if (( ! DRY_RUN_ENABLED )); then
  # Do not allow the obsolete SC2PATH stored in the conda environment to be
  # confused with the repository-local installation selected above.
  unset SC2PATH
  # Keep an already activated PyMARL/virtual environment. This supports
  # path-based activation when the environment is not registered as "pymarl".
  current_env_path="${CONDA_PREFIX:-}"
  current_env_name="${current_env_path##*/}"
  current_python="$(command -v python 2>/dev/null || true)"
  keep_current_env=0
  if [[ "${current_env_name}" == "pymarl" || -n "${VIRTUAL_ENV:-}" || "${current_python}" == */pymarl/bin/python* ]]; then
    keep_current_env=1
  fi
  if (( ! keep_current_env )); then
    if ! command -v conda >/dev/null 2>&1 && [[ -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
      source "${CONDA_ROOT}/etc/profile.d/conda.sh"
    fi
    if command -v conda >/dev/null 2>&1; then
      eval "$(conda shell.bash hook)"
      conda activate "${CONDA_ENV}"
    fi
  fi
  # The pymarl conda environment has historically stored an obsolete
  # SC2PATH. Conda activation overwrites an earlier export, so force the
  # repository-local installation only after activation.
  export SC2PATH="${SC2_ROOT}"
  command -v python >/dev/null 2>&1 || { echo '[error] Python is unavailable.' >&2; exit 1; }
  python - <<'PY'
import os
from pathlib import Path
import torch, smac
sc2 = Path(os.environ["SC2PATH"]).resolve()
binary = sc2 / "Versions" / "Base75689" / "SC2_x64"
if not binary.is_file():
    raise FileNotFoundError(binary)
print("[env] torch and SMAC imports OK")
print("[env] SC2PATH={}".format(sc2))
PY
fi

mkdir -p "${LOG_DIR}" "${LOG_DIR}/.failures"
mkdir -p "${GPU_LOCK_DIR}"
JOBS=()
if [[ -n "${EXPERIMENTS}" ]]; then
  # Optional explicit list for reruns that are not a Cartesian product.
  # Format: alg|map|seed;alg|map|seed
  IFS=';' read -r -a EXPERIMENT_LIST <<< "${EXPERIMENTS}"
  for experiment in "${EXPERIMENT_LIST[@]}"; do
    IFS='|' read -r alg map seed extra <<< "${experiment}"
    [[ -n "${alg}" && -n "${map}" && -n "${seed}" && -z "${extra}" ]] || {
      echo "[error] Invalid experiment specification: ${experiment}" >&2
      exit 1
    }
    JOBS+=("${alg}|${map}|${seed}")
  done
else
  for map in "${MAP_LIST[@]}"; do
    for alg in "${ALG_LIST[@]}"; do
      for seed in "${SEED_LIST[@]}"; do
        JOBS+=("${alg}|${map}|${seed}")
      done
    done
  done
fi
(( ${#JOBS[@]} )) || { echo '[error] No experiments requested.' >&2; exit 1; }

# Validate the final job list as well as the Cartesian-product inputs above.
for job in "${JOBS[@]}"; do
  IFS='|' read -r alg map seed <<< "${job}"
  [[ -f "src/config/algs/${alg}.yaml" ]] || { echo "[error] Missing algorithm config: ${alg}" >&2; exit 1; }
  [[ "${map}" =~ ^[A-Za-z0-9_]+$ ]] || { echo "[error] Invalid map name: ${map}" >&2; exit 1; }
  [[ -f "${SC2_ROOT}/Maps/SMAC_Maps/${map}.SC2Map" ]] || {
    echo "[error] SMAC map is missing: ${map}" >&2
    exit 1
  }
  [[ "${seed}" =~ ^[0-9]+$ ]] || { echo "[error] Invalid seed: ${seed}" >&2; exit 1; }
done

echo "[matrix] algorithms: ${ALG_LIST[*]}"
echo "[matrix] maps: ${MAP_LIST[*]}"
echo "[matrix] seeds: ${SEED_LIST[*]}"
echo "[matrix] GPUs: ${CUDA_DEVICE_LIST[*]}, slots/GPU=${SLOTS_PER_GPU}, jobs=${#JOBS[@]}"

acquire_bane_lock() {
  local slot candidate_fd
  while true; do
    for (( slot=0; slot<BANE_LOCK_SLOTS; slot++ )); do
      exec {candidate_fd}>"${BANE_LOCK_FILE}.${slot}"
      if flock -n "${candidate_fd}"; then
        heavy_fd="${candidate_fd}"
        echo "[lock] acquired bane_vs_bane slot=${slot}"
        return 0
      fi
      eval "exec ${candidate_fd}>&-"
    done
    sleep 1
  done
}

run_job() {
  local worker_index="$1" gpu="$2" alg="$3" map="$4" seed="$5"
  local run_name="${alg}_${map}_seed${seed}"
  local log_file="${LOG_DIR}/${run_name}.log"
  local heavy_fd=""
  local gpu_admission_fd=""
  local gpu_active_fd=""
  local gpu_admission_lock="${GPU_LOCK_DIR}/gpu_${gpu}.admission.lock"
  local gpu_active_lock="${GPU_LOCK_DIR}/gpu_${gpu}.active.lock"
  local bane_job=0
  local cmd=(python src/main.py "--config=${alg}" --env-config=sc2 with
    "env_args.map_name=${map}" "t_max=${T_MAX}" "seed=${seed}"
    "use_cuda=${USE_CUDA}" "use_tensorboard=${USE_TENSORBOARD}"
    "save_model=${SAVE_MODEL}" "name=${run_name}")
  if [[ -n "${EXTRA_OVERRIDES}" ]]; then
    read -r -a extra_override_list <<< "${EXTRA_OVERRIDES}"
    cmd+=("${extra_override_list[@]}")
  fi

  if (( DRY_RUN_ENABLED )); then
    printf -v cmd_text ' %q' "${cmd[@]}"
    printf '[dry-run] CUDA_VISIBLE_DEVICES=%q%s > %q 2>&1\n' "${gpu}" "${cmd_text}" "${log_file}"
    return 0
  fi

  if [[ "${map}" == "bane_vs_bane" ]]; then
    bane_job=1
  fi

  # Every job briefly acquires the admission lock before taking the active
  # lock. A Bane job reserves admission before waiting for the node-wide Bane
  # slot, so its assigned GPU cannot be used by ordinary jobs while it waits.
  exec {gpu_admission_fd}>"${gpu_admission_lock}"
  flock -x "${gpu_admission_fd}"

  # With buffer_size=5000, bane_vs_bane reserves about 39.5 GiB for the
  # replay buffer before Python/SC2 overhead. Limit this high-memory map to one
  # concurrent job on the same node, even when multiple GPUs are used. Other
  # maps do not take this lock and can continue running concurrently.
  if [[ "${map}" == "bane_vs_bane" ]]; then
    # Use /tmp so separate Slurm jobs placed on the same node share the
    # node-wide slot. LOG_DIR is job-specific and cannot provide that.
    echo "[wait] ${run_name} waiting for the node-wide bane_vs_bane memory lock"
    acquire_bane_lock
  fi

  # A Bane job keeps admission exclusively while it waits for existing
  # ordinary jobs to leave the active lock, so later ordinary jobs cannot
  # overtake the waiting Bane job.
  exec {gpu_active_fd}>"${gpu_active_lock}"
  if (( bane_job )); then
    echo "[wait] ${run_name} waiting for exclusive GPU lock gpu=${gpu}"
    flock -x "${gpu_active_fd}"
  else
    flock -s "${gpu_active_fd}"
    # Ordinary jobs only need admission protection while acquiring the active
    # shared lock. Release it before running so other ordinary jobs can enter.
    flock -u "${gpu_admission_fd}"
    eval "exec ${gpu_admission_fd}>&-"
    gpu_admission_fd=""
  fi

  echo "[run] gpu=${gpu} alg=${alg} map=${map} seed=${seed} -> ${log_file}"
  if ! SC2PATH="${SC2_ROOT}" PYTHONHASHSEED="${seed}" OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="${gpu}" \
       PYTHONPATH="${REPO_DIR}/src" "${cmd[@]}" >"${log_file}" 2>&1; then
    echo "${alg} ${map} seed=${seed} log=${log_file}" >> "${LOG_DIR}/.failures/worker_${worker_index}.txt"
    flock -u "${gpu_active_fd}"
    eval "exec ${gpu_active_fd}>&-"
    if [[ -n "${gpu_admission_fd}" ]]; then
      flock -u "${gpu_admission_fd}"
      eval "exec ${gpu_admission_fd}>&-"
    fi
    if [[ -n "${heavy_fd}" ]]; then flock -u "${heavy_fd}"; eval "exec ${heavy_fd}>&-"; fi
    return 1
  fi
  flock -u "${gpu_active_fd}"
  eval "exec ${gpu_active_fd}>&-"
  if [[ -n "${gpu_admission_fd}" ]]; then
    flock -u "${gpu_admission_fd}"
    eval "exec ${gpu_admission_fd}>&-"
  fi
  if [[ -n "${heavy_fd}" ]]; then flock -u "${heavy_fd}"; eval "exec ${heavy_fd}>&-"; fi
}

worker() {
  local worker_index="$1"
  local gpu="${CUDA_DEVICE_LIST[$(( worker_index / SLOTS_PER_GPU ))]}"
  local had_failure=0
  for job_index in "${!JOBS[@]}"; do
    (( job_index % ( ${#CUDA_DEVICE_LIST[@]} * SLOTS_PER_GPU ) == worker_index )) || continue
    IFS='|' read -r alg map seed <<< "${JOBS[$job_index]}"
    run_job "${worker_index}" "${gpu}" "${alg}" "${map}" "${seed}" || had_failure=1
  done
  return "${had_failure}"
}

TOTAL_WORKERS=$(( ${#CUDA_DEVICE_LIST[@]} * SLOTS_PER_GPU ))
PIDS=()
for (( worker_index=0; worker_index<TOTAL_WORKERS; worker_index++ )); do
  worker "${worker_index}" &
  PIDS+=("$!")
done

failed=0
for pid in "${PIDS[@]}"; do wait "${pid}" || failed=1; done
if (( failed )); then
  echo '[error] One or more SMAC experiments failed.' >&2
  cat "${LOG_DIR}/.failures"/worker_*.txt 2>/dev/null >&2 || true
  exit 1
fi
echo "[done] All ${#JOBS[@]} SMAC experiments finished successfully."
