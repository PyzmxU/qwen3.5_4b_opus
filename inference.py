import os
from unsloth import FastLanguageModel
import config

# 定位刚训练好的权重目录
lora_dir = os.path.join(config.BASE_DIR, "lora_qwen3.5_4b_cot")

print(f"正在加载 LoRA 权重: {lora_dir}")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = lora_dir, 
    max_seq_length = config.MAX_SEQ_LENGTH,
    dtype = None,
    load_in_4bit = config.LOAD_IN_4BIT,
)

FastLanguageModel.for_inference(model)

messages = [
    {"role": "system", "content": "You are a helpful AI assistant that always thinks step-by-step."},
    {"role": "user", "content": "A farmer has 14 sheep. All but 8 die. How many are left? Please think step by step."},
]

inputs = tokenizer.apply_chat_template(
    messages,
    tokenize = True,
    add_generation_prompt = True,
    return_tensors = "pt",
).to("cuda")

# 释放生成长度至 2048，给予模型充足的 <think> 空间
print("\n🧠 模型正在思考中...\n" + "="*50)
outputs = model.generate(
    input_ids = inputs,
    max_new_tokens = 2048, 
    use_cache = True,
    temperature = 0.7
)

print(tokenizer.batch_decode(outputs, skip_special_tokens = True)[0])
print("="*50)