#!/bin/bash
# PPO Qwen RLHF 自动化训练脚本

# ==================== 配置 ====================
# 项目路径
PROJECT_DIR="/root/.openclaw/workspace/ppo_qwen_rlhf"

# Python环境
CONDA_ENV="your_env_name"  # 改成你的conda环境名

# 训练参数
MODEL_NAME="Qwen/Qwen2-VL-7B-Instruct"
BATCH_SIZE=4
LEARNING_RATE=1e-5
NUM_EPOCHS=3
KL_COEFF=0.1

# 数据路径
DATA_PATH="${PROJECT_DIR}/data/train.json"

# 输出路径
OUTPUT_DIR="./ppo_qwen_output"
CHECKPOINT_DIR="./checkpoints"

# 日志
LOG_FILE="${PROJECT_DIR}/train.log"

# ==================== 自动化功能 ====================
# 是否自动恢复训练（如果检查点存在）
AUTO_RESUME=true

# 监控间隔（秒）
MONITOR_INTERVAL=60

# ==================== 主程序 ====================

echo "========== PPO Qwen RLHF 自动化训练 =========="
echo "开始时间: $(date)"
echo "项目路径: ${PROJECT_DIR}"

# 激活conda环境
if [ -n "$CONDA_ENV" ]; then
    echo "激活环境: ${CONDA_ENV}"
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate $CONDA_ENV
fi

# 检查依赖
echo "检查依赖..."
cd ${PROJECT_DIR}
pip list | grep -E "torch|transformers|peft" || echo "缺失依赖，请先安装: pip install -r requirements.txt"

# 检查数据
if [ ! -f "$DATA_PATH" ]; then
    echo "错误: 训练数据不存在 $DATA_PATH"
    exit 1
fi

# 创建输出目录
mkdir -p ${OUTPUT_DIR}
mkdir -p ${CHECKPOINT_DIR}

# 检查是否有检查点需要恢复
LATEST_CKPT=$(ls -t ${CHECKPOINT_DIR}/checkpoint_* 2>/dev/null | head -1)

if [ "$AUTO_RESUME" = true ] && [ -n "$LATEST_CKPT" ]; then
    echo "发现检查点: ${LATEST_CKPT}"
    echo "将从此检查点恢复训练..."
    RESUME_ARG="--resume ${LATEST_CKPT}"
else
    RESUME_ARG=""
    echo "从头开始训练..."
fi

# 构建训练命令
CMD="python ${PROJECT_DIR}/train.py \
    --model_name ${MODEL_NAME} \
    --batch_size ${BATCH_SIZE} \
    --learning_rate ${LEARNING_RATE} \
    --num_epochs ${NUM_EPOCHS} \
    --kl_coef ${KL_COEFF} \
    --train_data_path ${DATA_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --checkpoint_dir ${CHECKPOINT_DIR} \
    ${RESUME_ARG}"

echo "执行命令: ${CMD}"

# 启动训练
nohup ${CMD} > ${LOG_FILE} 2>&1 &
TRAIN_PID=$!

echo "训练进程已启动, PID: ${TRAIN_PID}"
echo "日志文件: ${LOG_FILE}"
echo ""
echo "========== 训练监控 =========="

# 监控训练过程
while true; do
    if ! ps -p $TRAIN_PID > /dev/null 2>&1; then
        echo "训练进程已结束"
        break
    fi
    
    # 每分钟检查一次日志
    if [ -f "${LOG_FILE}" ]; then
        LAST_LINES=$(tail -n 5 ${LOG_FILE})
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${LAST_LINES}"
    fi
    
    sleep $MONITOR_INTERVAL
done

echo "========== 训练完成 =========="
echo "结束时间: $(date)"
echo "最终模型保存在: ${OUTPUT_DIR}/final_model"

# 显示训练统计
if [ -f "${LOG_FILE}" ]; then
    echo ""
    echo "========== 训练日志摘要 =========="
    grep -E "Epoch|Avg Reward| completed" ${LOG_FILE} | tail -n 20
fi