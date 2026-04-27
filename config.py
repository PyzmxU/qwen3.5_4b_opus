import os

# ==========================================
# 1. Linux 原生路径与模型配置
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 使用 Unsloth 官方优化的预量化格式，配合 HF 镜像下载极快
MODEL_NAME_OR_PATH = "unsloth/Qwen3.5-4B" 
DATASET_ID = "Roman1111111/claude-opus-4.6-10000x"
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs_claude_cot")
OUTPUT_LORA = os.path.join(BASE_DIR, "lora_qwen3.5_4b_cot")

# ==========================================
# 2. 算力解封：3090 (24G) 满血参数
# ==========================================
MAX_SEQ_LENGTH = 16384
LOAD_IN_4BIT = True

# ==========================================
# 3. 智商解封：LoRA 全模块与高秩配置
# ==========================================
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# ==========================================
# 4. 训练节奏调度
# ==========================================
PER_DEVICE_TRAIN_BATCH_SIZE = 12
GRADIENT_ACCUMULATION_STEPS =1
NUM_TRAIN_EPOCHS = 2       # 替换 MAX_STEPS，跑完全量 1 万条数据
LEARNING_RATE = 2e-4
OPTIMIZER = "adamw_8bit"