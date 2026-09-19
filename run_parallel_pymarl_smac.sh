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
REPO_DIR="$(pwd -P)"
SC2_ROOT="${SC2_ROOT:-${REPO_DIR}/3rdparty/StarCraftII}"
BANE_LOCK_FILE="${BANE_LOCK_FILE:-/tmp/pymarl_bane_vs_bane_${USER:-unknown}.lock}"

DEFAULT_ALGS=(qmix ow_qmix cw_qmix qplex)
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
  if ! command -v conda >/dev/null 2>&1 && [[ -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
    source "${CONDA_ROOT}/etc/profile.d/conda.sh"
  fi
  if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV}"
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
JOBS=()
for map in "${MAP_LIST[@]}"; do
  for alg in "${ALG_LIST[@]}"; do
    for seed in "${SEED_LIST[@]}"; do
      JOBS+=("${alg}|${map}|${seed}")
    done
  done
done

echo "[matrix] algorithms: ${ALG_LIST[*]}"
echo "[matrix] maps: ${MAP_LIST[*]}"
echo "[matrix] seeds: ${SEED_LIST[*]}"
echo "[matrix] GPUs: ${CUDA_DEVICE_LIST[*]}, slots/GPU=${SLOTS_PER_GPU}, jobs=${#JOBS[@]}"

run_job() {
  local worker_index="$1" gpu="$2" alg="$3" map="$4" seed="$5"
  local run_name="${alg}_${map}_seed${seed}"
  local log_file="${LOG_DIR}/${run_name}.log"
  local heavy_fd=""
  local cmd=(python src/main.py "--config=${alg}" --env-config=sc2 with
    "env_args.map_name=${map}" "t_max=${T_MAX}" "seed=${seed}"
    "use_cuda=${USE_CUDA}" "use_tensorboard=${USE_TENSORBOARD}"
    "save_model=${SAVE_MODEL}" "name=${run_name}")

  if (( DRY_RUN_ENABLED )); then
    printf -v cmd_text ' %q' "${cmd[@]}"
    printf '[dry-run] CUDA_VISIBLE_DEVICES=%q%s > %q 2>&1\n' "${gpu}" "${cmd_text}" "${log_file}"
    return 0
  fi

  # With buffer_size=5000, bane_vs_bane reserves about 39.5 GiB for the
  # replay buffer before Python/SC2 overhead. Never run two such jobs on the
  # same node concurrently, even when multiple GPUs are allocated.
  if [[ "${map}" == "bane_vs_bane" ]]; then
    # Use /tmp so separate Slurm jobs placed on the same node also serialize
    # this high-memory map. LOG_DIR is job-specific and cannot provide that.
    exec {heavy_fd}>"${BANE_LOCK_FILE}"
    echo "[wait] ${run_name} waiting for the node-wide bane_vs_bane memory lock"
    flock "${heavy_fd}"
  fi

  echo "[run] gpu=${gpu} alg=${alg} map=${map} seed=${seed} -> ${log_file}"
  if ! SC2PATH="${SC2_ROOT}" PYTHONHASHSEED="${seed}" OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="${gpu}" \
       PYTHONPATH="${REPO_DIR}/src" "${cmd[@]}" >"${log_file}" 2>&1; then
    echo "${alg} ${map} seed=${seed} log=${log_file}" >> "${LOG_DIR}/.failures/worker_${worker_index}.txt"
    if [[ -n "${heavy_fd}" ]]; then flock -u "${heavy_fd}"; eval "exec ${heavy_fd}>&-"; fi
    return 1
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
