import os

# 插入这行：开启 PyTorch 动态显存碎片回收
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# ... 下面的代码保持不变 ...
# 激活 HuggingFace 国内镜像源 (极其关键)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth.chat_templates import get_chat_template

import config
from data_handler import load_and_prepare_dataset

def main():
    print(f"正在装载 4-bit 模型: {config.MODEL_NAME_OR_PATH}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = config.MODEL_NAME_OR_PATH,
        max_seq_length = config.MAX_SEQ_LENGTH,
        dtype = None,
        load_in_4bit = config.LOAD_IN_4BIT,
        # attn_implementation = "sdpa",
        attn_implementation = "flash_attention_2",
    )

    # 强制绑定 ChatML 模板
    tokenizer = get_chat_template(tokenizer, chat_template="chatml")

    # 配置全模块 LoRA
    print("正在挂载满血版 LoRA 适配器...")
    model = FastLanguageModel.get_peft_model(
        model,
        r = config.LORA_R,
        target_modules = config.TARGET_MODULES,
        lora_alpha = config.LORA_ALPHA,
        lora_dropout = config.LORA_DROPOUT,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
    )

    # 加载数据集
    dataset = load_and_prepare_dataset(config.DATASET_ID, tokenizer, config.MAX_SEQ_LENGTH)

    # 配置 SFTTrainer
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = config.MAX_SEQ_LENGTH,
        dataset_num_proc = 8,  # 利用云服务器的多核 CPU 加速处理
        packing = True,
        # args = TrainingArguments(
        #     output_dir = config.OUTPUT_DIR,
        #     per_device_train_batch_size = config.PER_DEVICE_TRAIN_BATCH_SIZE,
        #     gradient_accumulation_steps = config.GRADIENT_ACCUMULATION_STEPS,
        #     warmup_steps = 5,
        #     num_train_epochs = config.NUM_TRAIN_EPOCHS, # 跑满全量数据
        #     learning_rate = config.LEARNING_RATE,
        #     fp16 = False,
        #     bf16 = True, # 3090 原生支持并推荐使用 bf16
        #     logging_steps = 5,
        #     optim = config.OPTIMIZER,
        #     weight_decay = 0.01,
        #     lr_scheduler_type = "linear",
        #     max_grad_norm = 0.3,
        #     seed = 3407,
        #     dataloader_num_workers = 8, # 充分利用 12 核 CPU
        #     dataloader_pin_memory = True, # 锁定内存，加快数据拷贝到显存的速度
        #     gradient_checkpointing = True,
        #     save_steps = 100,
        #     save_total_limit = 3,
        # ),
        args = TrainingArguments(
        output_dir = config.OUTPUT_DIR,
    per_device_train_batch_size = config.PER_DEVICE_TRAIN_BATCH_SIZE,
    gradient_accumulation_steps = config.GRADIENT_ACCUMULATION_STEPS,
    warmup_ratio = 0.05,            # 使用比例代替步数，更科学
    num_train_epochs = config.NUM_TRAIN_EPOCHS,
    learning_rate = config.LEARNING_RATE,
    bf16 = True,                    # 保持 BF16
    fp16 = False,
    logging_steps = 1,              # 既然是高压训练，每步都监控更安全
    optim = "adamw_8bit",           # 明确指定 8bit 优化器
    weight_decay = 0.01,
    lr_scheduler_type = "cosine",   # 切换为余弦退火
    max_grad_norm = 1.0,            # 放宽梯度裁剪
    seed = 3407,
    # dataloader_num_workers = 8,      # 建议先回退到 4 观察
    dataloader_num_workers = 12,       # 压榨满你云服务器的所有核心
dataloader_prefetch_factor = 4,     # 预取的倍数拉满
    dataloader_pin_memory = True,
    gradient_checkpointing = True,
    save_steps = 100,
    save_total_limit = 3,
    report_to = "none",             # 减少网络 IO 干扰
    )
    )

    print("🚀 开始 3090 满血微调训练...")
    trainer.train()

    # 保存权重
    save_path = os.path.join(config.BASE_DIR, "lora_qwen3.5_4b_cot")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"✅ LoRA 权重已成功保存至 {save_path}！")

if __name__ == "__main__":
    main()