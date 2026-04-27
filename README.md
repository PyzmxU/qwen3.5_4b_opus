# Qwen3.5-4B × Claude Opus 思维链微调

基于 [Unsloth](https://github.com/unslothai/unsloth) + LoRA 的全模块微调项目，让 Qwen3.5-4B 学会 Claude Opus 4.6 的深度推理能力。**单张 24GB 显卡即可训练。**

## 特性

| 特性 | 说明 |
|------|------|
| **全模块 LoRA** | 覆盖 `q_proj / k_proj / v_proj / o_proj / gate_proj / up_proj / down_proj`，r=32, alpha=64 |
| **Chain-of-Thought** | 注入 Claude Opus 4.6 推理轨迹，模型学会 `<｜begin_of_think｜>` / `<｜end_of_think｜>` 深度思考 |
| **24GB 显存优化** | 4-bit 量化加载 + Unsloth 梯度检查点 + adamw_8bit 优化器，RTX 3090 / 4090 即可跑满 |
| **完整工作流** | 训练 → 推理 → GGUF 量化 → llama.cpp / Ollama 部署，一条链路打通 |
| **性能评估** | 包含基线模型 vs LoRA 模型在 GSM8K 上的对比测试 |

## 项目结构

```
qwen3.5_4b_opus/
├── config.py                     # 训练参数配置（模型/数据/LoRA/调度）
├── train.py                      # 训练入口（Unsloth + SFTTrainer，含注释版与加压版两套参数）
├── data_handler.py               # 数据处理（HF 数据集加载 + 思维链模板映射 + 清洗过滤）
├── inference.py                  # 推理脚本（tokenizer.apply_chat_template 方式）
├── inferencecopy.py              # 推理脚本（手动构造 ChatML 格式，中文回复版）
├── export_gguf.py                # GGUF 导出（合并 LoRA → q4_k_m + q8_0）
├── export_base_gguf_origin.py    # 基础模型 GGUF 导出（不带 LoRA，用于对比实验）
└── eval_results/                 # 评估结果（基线 vs LoRA 在 GSM8K 上的表现）
    ├── base_gsm8k_test/          # 原始 Qwen3.5-4B 基线评测
    └── lora_gsm8k_test/          # LoRA 微调后评测
```

## 快速开始

### 环境

```
Python >= 3.10
PyTorch >= 2.1
CUDA 12.x
GPU: 24GB VRAM (RTX 3090 / 4090)
```

### 安装

```bash
pip install unsloth "transformers>=4.40.0" "trl>=0.9.0" "datasets>=2.18.0" peft accelerate bitsandbytes
```

### 训练

```bash
python train.py
```

训练流程：加载 4-bit 量化基础模型 → 挂载全模块 LoRA → 从 HuggingFace 拉取推理数据集 → 思维链模板格式化 → SFT 微调 → 保存 LoRA 权重到 `lora_qwen3.5_4b_cot/`。

### 推理

```bash
python inference.py
```

加载训练好的 LoRA 权重，输入示例问题，模型会输出带思维链的完整回答。

### 导出 GGUF

```bash
python export_gguf.py
```

将 LoRA 权重合并回基础模型，导出为 `q4_k_m`（4-bit，推荐）和 `q8_0`（8-bit，高质量）两种格式，可直接用于 llama.cpp 或转换 Ollama Modelfile。

## 配置参数

核心配置在 `config.py`：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `MODEL_NAME_OR_PATH` | `unsloth/Qwen3.5-4B` | 基础模型（HF 仓库名或本地路径） |
| `DATASET_ID` | `Roman1111111/claude-opus-4.6-10000x` | 训练数据集 |
| `MAX_SEQ_LENGTH` | 16384 | 最大序列长度 |
| `LORA_R` | 32 | LoRA 秩 |
| `LORA_ALPHA` | 64 | LoRA 缩放因子 |
| `PER_DEVICE_TRAIN_BATCH_SIZE` | 12 | 单卡批次大小 |
| `LEARNING_RATE` | 2e-4 | 学习率（余弦退火） |
| `NUM_TRAIN_EPOCHS` | 2 | 训练轮数 |

## 训练结果

### 损失下降趋势

| Step | Epoch | Loss | 说明 |
|------|-------|------|------|
| 1 | 0.00 | 1.0738 | 初始 Loss |
| 100 | 0.12 | 0.6367 | 快速下降 |
| 500 | 0.62 | 0.5262 | 稳步收敛 |
| 800 | 1.00 | 0.5246 | Epoch 1 完成 |
| 1000 | 1.25 | 0.3948 | Epoch 2 加速下降 |
| 1500 | 1.87 | 0.3256 | 最低点 |
| 1606 | 2.00 | 0.3505 | 最终 Loss |

### 关键指标

| 指标 | 值 |
|------|------|
| 总步数 | 1,606 |
| 总训练轮数 | 2 epoch |
| 最大 Loss | 1.8889（Step 3） |
| 最小 Loss | 0.2501（全局最低） |
| 最终 Loss | 0.3505 |
| 平均 Loss | 0.5395 |
| 学习率范围 | 2e-4 → 0（余弦退火） |

Loss 从 1.89 收敛至 0.35，下降 81.5%，训练曲线平滑无异常。

## 数据集

使用 [Claude Opus 4.6 推理数据集](https://huggingface.co/datasets/Roman1111111/claude-opus-4.6-10000x)，包含约 10,000 条 Claude Opus 4.6 的推理对话数据。`data_handler.py` 会在加载时自动将 `reasoning` 字段注入 `<｜begin_of_think｜>` / `<｜end_of_think｜>` 标签，同时过滤超出 `MAX_SEQ_LENGTH` 的超长样本。

## 部署

### llama.cpp

```bash
./llama-cli -m qwen3.5-4b-opus46-cot-Q4_K_M.gguf -p "你的问题" -n 2048 --temp 0.7
```

### Ollama

```bash
# 1. 创建 Modelfile
FROM ./qwen3.5-4b-opus46-cot-Q4_K_M.gguf
PARAMETER temperature 0.7
PARAMETER num_ctx 8192

# 2. 创建模型
ollama create qwen3.5-opus-cot -f Modelfile

# 3. 使用
ollama run qwen3.5-opus-cot
```

## HuggingFace 模型权重

| 模型 | 仓库 | 说明 |
|------|------|------|
| LoRA 适配器 | [Pyzmxu/qwen3.5-4b-opus46-cot-lora](https://huggingface.co/Pyzmxu/qwen3.5-4b-opus46-cot-lora) | QLoRA 微调权重（思维链推理） |
| GGUF q4_k_m | [Pyzmxu/qwen3.5-4b-opus46-cot-q4km](https://huggingface.co/Pyzmxu/qwen3.5-4b-opus46-cot-q4km) | 4-bit 量化模型 |
| GGUF q8_0 | [Pyzmxu/qwen3.5-4b-opus46-cot-q8](https://huggingface.co/Pyzmxu/qwen3.5-4b-opus46-cot-q8) | 8-bit 量化模型 |

## License

MIT
