#!/usr/bin/env bash
set -euo pipefail

# Shared launcher for the three environments exposed by this
# checkout. Thin wrappers named run_parallel_pymarl_{lbf,smacv1,smacv2}.sh
# pass the environment kind as the first argument.

ENV_KIND="${1:-}"
case "${ENV_KIND}" in
  lbf|smacv1|smacv2) ;;
  *) echo "usage: $0 {lbf|smacv1|smacv2}" >&2; exit 2 ;;
esac
shift || true

REPO_DIR="$(cd "$(dirname "$0")" && pwd -P)"
cd "${REPO_DIR}"
PYTHON="${PYTHON:-python}"
ALGS="${ALGS:-qmix qplex}"
SEEDS="${SEEDS:-1 41 141}"
if [[ "${ENV_KIND}" == smacv1 ]]; then
  T_MAX="${T_MAX:-2050000}"
else
  T_MAX="${T_MAX:-5050000}"
fi
USE_CUDA="${USE_CUDA:-False}"
DRY_RUN="${DRY_RUN:-False}"
MAX_CONCURRENT="${MAX_CONCURRENT:-1}"
LOG_DIR="${LOG_DIR:-parallel_logs/${ENV_KIND}}"

LBF_TASKS=(
  8x8_2p_2f_coop 8x8_2p_2f_2s_coop
  10x10_3p_3f 10x10_3p_3f_2s
  15x15_3p_4f 15x15_4p_3f
)
LBF_KEYS=(
  "lbforaging:Foraging-8x8-2p-2f-coop-v3"
  "lbforaging:Foraging-2s-8x8-2p-2f-coop-v3"
  "lbforaging:Foraging-10x10-3p-3f-v3"
  "lbforaging:Foraging-2s-10x10-3p-3f-v3"
  "lbforaging:Foraging-15x15-3p-4f-v3"
  "lbforaging:Foraging-15x15-4p-3f-v3"
)
SMACV1_MAPS=(3m 8m 2s3z 3s5z MMM2 6h_vs_8z)
SMACV2_SCENARIOS=(protoss_5_vs_5 protoss_5_vs_6 terran_5_vs_5 terran_5_vs_6 zerg_5_vs_5 zerg_5_vs_6)

read -r -a ALG_LIST <<< "${ALGS}"
read -r -a SEED_LIST <<< "${SEEDS}"
if [[ "${ENV_KIND}" == lbf ]]; then
  read -r -a ITEM_LIST <<< "${TASKS:-${LBF_TASKS[*]}}"
elif [[ "${ENV_KIND}" == smacv1 ]]; then
  read -r -a ITEM_LIST <<< "${MAPS:-${SMACV1_MAPS[*]}}"
else
  read -r -a ITEM_LIST <<< "${SCENARIOS:-${SMACV2_SCENARIOS[*]}}"
fi

(( ${#ALG_LIST[@]} && ${#SEED_LIST[@]} && ${#ITEM_LIST[@]} )) || {
  echo '[error] algorithm, task/map/scenario, and seed lists must be nonempty' >&2; exit 1;
}
[[ "${MAX_CONCURRENT}" =~ ^[1-9][0-9]*$ ]] || { echo '[error] MAX_CONCURRENT must be positive' >&2; exit 1; }
[[ "${T_MAX}" =~ ^[1-9][0-9]*$ ]] || { echo '[error] T_MAX must be positive' >&2; exit 1; }
for seed in "${SEED_LIST[@]}"; do
  [[ "${seed}" =~ ^[0-9]+$ ]] || { echo "[error] invalid seed: ${seed}" >&2; exit 1; }
done
case "${DRY_RUN}" in
  True|true|TRUE|1|yes|Yes|YES) DRY=1 ;;
  False|false|FALSE|0|no|No|NO) DRY=0 ;;
  *) echo '[error] DRY_RUN must be True or False' >&2; exit 1 ;;
esac

for alg in "${ALG_LIST[@]}"; do
  [[ -f "src/config/algs/${alg}.yaml" ]] || { echo "[error] missing algorithm config: ${alg}" >&2; exit 1; }
done
task_key() {
  local task="$1" i
  for i in "${!LBF_TASKS[@]}"; do
    [[ "${LBF_TASKS[$i]}" == "${task}" ]] && { echo "${LBF_KEYS[$i]}"; return 0; }
  done
  return 1
}
for item in "${ITEM_LIST[@]}"; do
  [[ "${item}" =~ ^[A-Za-z0-9_]+$ ]] || { echo "[error] invalid item name: ${item}" >&2; exit 1; }
  if [[ "${ENV_KIND}" == lbf ]]; then
    task_key "${item}" >/dev/null || { echo "[error] unsupported LBF task: ${item}" >&2; exit 1; }
  fi
done
if [[ "${ENV_KIND}" == smacv2 ]]; then
  for item in "${ITEM_LIST[@]}"; do
    [[ -f "src/config/envs/smacv2_configs/${item}.yaml" ]] || { echo "[error] missing SMACv2 scenario: ${item}" >&2; exit 1; }
  done
fi

if (( ! DRY )); then
  command -v "${PYTHON}" >/dev/null 2>&1 || { echo "[error] Python not found: ${PYTHON}" >&2; exit 1; }
  if [[ "${ENV_KIND}" != lbf ]]; then
    SC2_ROOT="${SC2_ROOT:-${REPO_DIR}/3rdparty/StarCraftII}"
    [[ -x "${SC2_ROOT}/Versions/Base75689/SC2_x64" ]] || { echo "[error] missing SC2 executable under ${SC2_ROOT}" >&2; exit 1; }
    if [[ "${ENV_KIND}" == smacv1 ]]; then
      for item in "${ITEM_LIST[@]}"; do
        [[ -f "${SC2_ROOT}/Maps/SMAC_Maps/${item}.SC2Map" ]] || { echo "[error] missing SMACv1 map: ${item}" >&2; exit 1; }
      done
    fi
    export SC2PATH="${SC2_ROOT}"
    unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
    export NO_PROXY="localhost,127.0.0.1,::1" no_proxy="localhost,127.0.0.1,::1"
  fi
  case "${ENV_KIND}" in
    lbf) "${PYTHON}" -c 'import gymnasium, lbforaging' ;;
    smacv1) "${PYTHON}" -c 'import smac' ;;
    smacv2) "${PYTHON}" -c 'import smacv2' ;;
  esac
fi

mkdir -p "${LOG_DIR}"
declare -a JOBS=()
for item in "${ITEM_LIST[@]}"; do
  for alg in "${ALG_LIST[@]}"; do
    for seed in "${SEED_LIST[@]}"; do JOBS+=("${alg}|${item}|${seed}"); done
  done
done
echo "[matrix] env=${ENV_KIND} algorithms=${ALG_LIST[*]} items=${ITEM_LIST[*]} seeds=${SEED_LIST[*]} jobs=${#JOBS[@]} use_cuda=${USE_CUDA}"

run_one() {
  local alg="$1" item="$2" seed="$3" log_file cmd_text
  local run_name="${alg}_${ENV_KIND}_${item}_seed${seed}"
  local -a cmd=("${PYTHON}" src/main.py "--config=${alg}")
  case "${ENV_KIND}" in
    lbf)
      cmd+=(--env-config=lbf "with" "env_args.key=$(task_key "${item}")" "env_args.time_limit=50" "env_args.common_reward=True" "env_args.reward_scalarisation=sum") ;;
    smacv1) cmd+=(--env-config=smacv1 "with" "env_args.map_name=${item}") ;;
    smacv2) cmd+=(--env-config=sc2v2 "with" "env_args.map_name=${item}") ;;
  esac
  cmd+=("t_max=${T_MAX}" "seed=${seed}" "use_cuda=${USE_CUDA}" "name=${run_name}")
  log_file="${LOG_DIR}/${run_name}.log"
  if (( DRY )); then
    printf -v cmd_text ' %q' "${cmd[@]}"
    printf '[dry-run]%s > %q 2>&1\n' "${cmd_text}" "${log_file}"
    return 0
  fi
  echo "[run] ${run_name}"
  PYTHONPATH="${REPO_DIR}/src" OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" "${cmd[@]}" >"${log_file}" 2>&1
}

declare -a PIDS=()
failed=0
for job in "${JOBS[@]}"; do
  IFS='|' read -r alg item seed <<< "${job}"
  run_one "${alg}" "${item}" "${seed}" & PIDS+=("$!")
  if (( ${#PIDS[@]} >= MAX_CONCURRENT )); then
    for pid in "${PIDS[@]}"; do wait "${pid}" || failed=1; done
    PIDS=()
  fi
done
for pid in "${PIDS[@]}"; do wait "${pid}" || failed=1; done
(( failed == 0 )) || { echo '[error] one or more jobs failed' >&2; exit 1; }
echo "[done] ${#JOBS[@]} ${ENV_KIND} jobs"
