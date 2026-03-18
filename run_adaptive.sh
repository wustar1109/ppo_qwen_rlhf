#!/bin/bash
# PPO Qwen RLHF 自适应自动化训练脚本

# ==================== 配置 ====================
PROJECT_DIR="/root/.openclaw/workspace/ppo_qwen_rlhf"
CONDA_ENV="base"  # 改成你的conda环境名

# 训练参数
MODEL_NAME="Qwen/Qwen2-VL-7B-Instruct"
BATCH_SIZE=4
LEARNING_RATE=1e-5
NUM_EPOCHS=10  # 自适应训练通常需要更多epoch
KL_COEFF=0.1

# 自适应配置
ADAPTIVE=true
EARLY_STOP_REWARD=0.95
KL_TARGET=0.02

# 数据路径
DATA_PATH="${PROJECT_DIR}/data/train.json"

# 输出路径
OUTPUT_DIR="./ppo_qwen_output"
CHECKPOINT_DIR="./checkpoints"

# 日志
LOG_FILE="${PROJECT_DIR}/train.log"

echo "============================================"
echo "  PPO Qwen RLHF 自适应自动化训练"
echo "============================================"
echo "开始时间: $(date)"
echo "自适应训练: ${ADAPTIVE}"
echo "早停阈值: ${EARLY_STOP_REWARD}"
echo "KL目标: ${KL_TARGET}"
echo ""

# 激活conda环境
if [ -n "$CONDA_ENV" ]; then
    echo "激活环境: ${CONDA_ENV}"
    source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null
    conda activate $CONDA_ENV
fi

# 检查依赖
echo "检查Python环境..."
cd ${PROJECT_DIR}
python -c "import torch; print(f'PyTorch: {torch.__version__}')" || echo "请先安装依赖: pip install -r requirements.txt"

# 检查数据
if [ ! -f "$DATA_PATH" ]; then
    echo "警告: 训练数据不存在 $DATA_PATH"
    
    # 尝试创建示例数据
    mkdir -p ${PROJECT_DIR}/data
    echo '[
  {"prompt": "描述这张图片", "image_path": ""},
  {"prompt": "分析图像内容", "image_path": ""},
  {"prompt": "这张图片传达了什么？", "image_path": ""}
]' > ${DATA_PATH}
    echo "已创建示例数据文件"
fi

# 创建输出目录
mkdir -p ${OUTPUT_DIR}
mkdir -p ${CHECKPOINT_DIR}

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
    --save_interval 100 \
    --log_interval 5"

# 添加自适应参数
if [ "$ADAPTIVE" = true ]; then
    CMD="${CMD} --use_adaptive"
fi

echo "执行命令:"
echo "${CMD}"
echo ""
echo "开始训练..."
echo "日志文件: ${LOG_FILE}"
echo ""

# 启动训练（后台运行）
nohup ${CMD} > ${LOG_FILE} 2>&1 &
TRAIN_PID=$!

echo "训练进程已启动, PID: ${TRAIN_PID}"
echo "============================================"
echo "监控训练日志:"
echo "  tail -f ${LOG_FILE}"
echo ""
echo "查看自适应状态:"
echo "  grep -E 'Adaptive|LR:|KL:' ${LOG_FILE}"
echo ""
echo "停止训练:"
echo "  kill ${TRAIN_PID}"
echo "============================================"

# 等待一段时间，然后显示日志
sleep 5

if [ -f "${LOG_FILE}" ]; then
    echo ""
    echo "最新日志:"
    tail -n 20 ${LOG_FILE}
fi