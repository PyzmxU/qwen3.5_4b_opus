# Qwen3.5-4B 思维链微调 (Claude Opus CoT)

基于 [Unsloth](https://github.com/unslothai/unsloth) + LoRA 的全模块微调，让 Qwen3.5-4B 学会 Claude Opus 的深度推理能力。单张 24GB 显卡即可训练。

## 项目特点

- **全模块 LoRA** — 覆盖 `q_proj / k_proj / v_proj / o_proj / gate_proj / up_proj / down_proj`，r=32, alpha=64
- **Chain-of-Thought 思维链** — 使用 Claude Opus 4.6 的推理轨迹数据，注入 `<｜begin_of_think｜>` / `<｜end_of_think｜>` 思考标签
- **24GB 显卡友好** — 4-bit 量化加载 + Unsloth 梯度检查点，RTX 3090 / 4090 即可跑满
- **完整工作流** — 训练 → 推理 → GGUF 量化导出（llama.cpp / Ollama 可用）

## 目录结构

```
qwen3.5_4b_opus/
├── config.py              # 训练参数配置（模型/数据/LoRA/调度）
├── train.py               # 训练入口（Unsloth + SFTTrainer）
├── data_handler.py         # 数据处理（加载 HF 数据集 + CoT 模板映射）
├── inference.py           # 推理脚本（直接加载 LoRA 权重生成）
├── inferencecopy.py       # 推理脚本（手动 ChatML 格式版）
├── export_gguf.py         # GGUF 导出（合并 LoRA → q4_k_m + q8_0）
├── export_base_gguf_origin.py  # 基础模型 GGUF 导出（不带 LoRA）
└── .gitignore
```

## 快速开始

### 环境要求

```
Python >= 3.10
PyTorch >= 2.1
CUDA 12.x
GPU: 24GB VRAM (RTX 3090 / 4090)
```

### 安装依赖

```bash
pip install unsloth "transformers>=4.40.0" "trl>=0.9.0" "datasets>=2.18.0" peft accelerate bitsandbytes
```

### 训练

```bash
python train.py
```

训练完成后 LoRA 权重自动保存至 `lora_qwen3.5_4b_cot/`。

### 推理

```bash
python inference.py
```

### 导出 GGUF

```bash
python export_gguf.py
```

导出 `q4_k_m` 和 `q8_0` 两种量化格式，可直接用于 llama.cpp 或转换为 Ollama Modelfile 部署。

## 配置参数

编辑 `config.py` 调整训练参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `MODEL_NAME_OR_PATH` | `unsloth/Qwen3.5-4B` | 基础模型 |
| `DATASET_ID` | `Roman1111111/claude-opus-4.6-10000x` | 训练数据集 |
| `MAX_SEQ_LENGTH` | 16384 | 最大序列长度 |
| `LORA_R` | 32 | LoRA 秩 |
| `LORA_ALPHA` | 64 | LoRA 缩放因子 |
| `PER_DEVICE_TRAIN_BATCH_SIZE` | 12 | 单卡批次大小 |
| `LEARNING_RATE` | 2e-4 | 学习率 |
| `NUM_TRAIN_EPOCHS` | 2 | 训练轮数 |
| `LOAD_IN_4BIT` | True | 4-bit 量化加载 |

## 数据集

使用 [Claude Opus 4.6 推理数据集](https://huggingface.co/datasets/Roman1111111/claude-opus-4.6-10000x)，包含约 10,000 条 Claude Opus 的推理对话数据，经 `data_handler.py` 处理注入 `<｜begin_of_think｜>` 思考标签后用于 SFT 微调。

## License

MIT
