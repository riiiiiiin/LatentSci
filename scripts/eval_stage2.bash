set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <exp_name> <stage1_exp_name> <dataset_name> [cuda_visible_devices]"
  exit 2
fi

# exp
EXP_NAME=<exp_name>
STAGE1_EXP_NAME=<stage_1_exp_name>
DATASET_NAME=ChemCoTBench
CUDA_DEVICES=0,1

# inference
BATCH_SIZE=1
MAX_NEW_TOKENS=2048
TEMPERATURE=0.7
TOP_P=0.8

SCRIPT_PATH="code_train_sft/train_sft_stage2.py"
OUTPUT_DIR="outputs/${EXP_NAME}"
STAGE1_DIR="outputs/${STAGE1_EXP_NAME}"
LORA_PATH="${STAGE1_DIR}/lora_weights"
PROJECTOR_PATH="${STAGE1_DIR}/projector.pt"
DATA_PATH="data/${DATASET_NAME}"

PYTHON_BIN="python"

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
LOG_NAME="${EXP_NAME}_${TIMESTAMP}"
INFERENCE_RESULTS_PATH="${OUTPUT_DIR}/inference_results_${TIMESTAMP}.json"

echo "========== Stage-2 Inference & Eval Runner =========="
echo "EXP_NAME:             ${EXP_NAME}"
echo "STAGE1_EXP_NAME:      ${STAGE1_EXP_NAME}"
echo "DATASET_NAME:         ${DATASET_NAME}"
echo "SCRIPT_PATH:          ${SCRIPT_PATH}"
echo "OUTPUT_DIR:           ${OUTPUT_DIR}"
echo "STAGE1_DIR:           ${STAGE1_DIR}"
echo "LORA_PATH:            ${LORA_PATH}"
echo "PROJECTOR_PATH:       ${PROJECTOR_PATH}"
echo "DATA_PATH:            ${DATA_PATH}"
echo "INFERENCE_RESULTS:    ${INFERENCE_RESULTS_PATH}"
echo "LOG_NAME (for eval):  ${LOG_NAME}"
if [[ -n "${CUDA_DEVICES}" ]]; then
  echo "CUDA_VISIBLE_DEVICES: ${CUDA_DEVICES}"
fi
echo "PYTHON:               ${PYTHON_BIN}"
echo "====================================================="

# 基本检查
if [[ ! -f "${SCRIPT_PATH}" ]]; then
  echo "ERROR: 推理脚本未找到: ${SCRIPT_PATH}"
  exit 3
fi

if [[ ! -d "${LORA_PATH}" ]]; then
  echo "ERROR: Stage-1 LoRA 目录未找到: ${LORA_PATH}"
  exit 4
fi

if [[ ! -f "${PROJECTOR_PATH}" ]]; then
  echo "ERROR: Stage-1 projector 文件未找到: ${PROJECTOR_PATH}"
  exit 5
fi

if [[ ! -e "${DATA_PATH}" ]]; then
  echo "ERROR: 数据路径未找到: ${DATA_PATH}"
  exit 6
fi

# 创建输出目录
mkdir -p "${OUTPUT_DIR}"
LOG_FILE="${OUTPUT_DIR}/stage2_inference_and_eval_${TIMESTAMP}.log"

# ---- 启动多进程推理（每张 GPU 一个进程） ----
echo "$(date +'%Y-%m-%d %H:%M:%S') - Launching inference per-GPU..." | tee -a "${LOG_FILE}"

# 解析 CUDA_DEVICES 列表为数组
IFS=',' read -ra GPU_ARRAY <<< "${CUDA_DEVICES}"
NUM_PROCS=${#GPU_ARRAY[@]}

PIDS=()
for idx in "${!GPU_ARRAY[@]}"; do
  GPU_ID="${GPU_ARRAY[idx]}"
  PROC_INDEX="${idx}"

  # per-process result & log file，避免覆盖
  RESULTS_PATH_PROC="${OUTPUT_DIR}/inference_results_${TIMESTAMP}.proc${PROC_INDEX}.json"
  LOG_FILE_PROC="${OUTPUT_DIR}/stage2_inference_and_eval_${TIMESTAMP}.proc${PROC_INDEX}.log"

  # 构造命令（在环境变量前缀里设置 CUDA_VISIBLE_DEVICES 以便只看到该卡）
  CMD_ENV=(CUDA_VISIBLE_DEVICES="${GPU_ID}")
  CMD=( "${PYTHON_BIN}" "${SCRIPT_PATH}"
        --mode inference
        --output_dir "${OUTPUT_DIR}"
        --data_path "${DATA_PATH}"
        --lora_path "${LORA_PATH}"
        --projector_path "${PROJECTOR_PATH}"
        --inference_results_path "${RESULTS_PATH_PROC}"
        --batch_size "${BATCH_SIZE}"
        --max_new_tokens "${MAX_NEW_TOKENS}"
        --temperature "${TEMPERATURE}"
        --top_p "${TOP_P}"
        --wandb_project "biolatentcot-stage2"
        --wandb_run_name "${LOG_NAME}.proc${PROC_INDEX}"
        --wandb_entity ""
        --proc_index "${PROC_INDEX}"
        --num_procs "${NUM_PROCS}"
        --gpu "${GPU_ID}"
  )

  # 打印并后台启动（日志重定向到 per-process log）
  echo "Launching proc ${PROC_INDEX} on GPU ${GPU_ID}: ${CMD_ENV[*]} ${CMD[*]}" | tee -a "${LOG_FILE}"
  ( "${CMD_ENV[@]}" "${CMD[@]}" 2>&1 | tee -a "${LOG_FILE_PROC}" ) &

  PIDS+=($!)
done

# 等待所有后台进程完成
for p in "${PIDS[@]}"; do
  wait "${p}" || {
    echo "$(date +'%Y-%m-%d %H:%M:%S') - ERROR: One of inference processes failed (pid=${p})." | tee -a "${LOG_FILE}"
    exit 10
  }
done

echo "$(date +'%Y-%m-%d %H:%M:%S') - All inference processes finished successfully." | tee -a "${LOG_FILE}"

# ---- 2) 进入 eval 目录并运行 eval_results.py ----
if [[ ! -d "eval" ]]; then
  echo "ERROR: eval 目录不存在，请确认 eval/eval_results.py 在 eval/ 下." | tee -a "${LOG_FILE}"
  exit 12
fi

# 使用绝对路径给 eval 脚本（防止相对路径问题）
ABS_RESULT_PATH="$(cd "$(dirname "${INFERENCE_RESULTS_PATH}")"; pwd)/$(basename "${INFERENCE_RESULTS_PATH}")"
echo "Preparing to run evaluation in ./eval with result_path = ${ABS_RESULT_PATH}" | tee -a "${LOG_FILE}"

pushd eval > /dev/null

EVAL_CMD=(
  "${PYTHON_BIN}" "eval_results.py"
  --result_path "${ABS_RESULT_PATH}"
  --log_name "${LOG_NAME}"
)

echo "Running evaluation command:" | tee -a "../${LOG_FILE}"
printf " %s " "${EVAL_CMD[@]}" | tee -a "../${LOG_FILE}"
echo "" | tee -a "../${LOG_FILE}"

if "${EVAL_CMD[@]}" 2>&1 | tee -a "../${LOG_FILE}"; then
  echo "$(date +'%Y-%m-%d %H:%M:%S') - Evaluation finished successfully." | tee -a "../${LOG_FILE}"
else
  echo "$(date +'%Y-%m-%d %H:%M:%S') - ERROR: Evaluation failed. See log: ${LOG_FILE}" | tee -a "../${LOG_FILE}"
  popd > /dev/null
  exit 20
fi

popd > /dev/null

echo "All done."
echo "Inference results: ${INFERENCE_RESULTS_PATH}"
echo "Eval log name: ${LOG_NAME}"
echo "Full log: ${LOG_FILE}"
echo "====================================================="