import os
import torch

# ==========================================
# 1. 环境与网络防线
# ==========================================
# 强制使用镜像源，解决 Hugging Face 墙的问题
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 关闭 Unsloth 默认的在线遥测统计，防止 TimeoutError
os.environ["UNSLOTH_DISABLE_TELEMETRY"] = "1"

from unsloth import FastLanguageModel
import config

# ==========================================
# 2. 模型与权重加载
# ==========================================
# 严格指向刚刚自动保存的 checkpoint-100 目录
lora_dir = os.path.join(config.OUTPUT_LORA) 

print(f"正在加载测试权重: {lora_dir}")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = lora_dir, 
    max_seq_length = config.MAX_SEQ_LENGTH,
    dtype = None,
    load_in_4bit = False,         # 【保命设置】必须 4bit，防止挤爆 3090 显存
    local_files_only = True,     # 【断网设置】严禁联网检查，只读本地缓存
)

# 切换为推理加速模式
FastLanguageModel.for_inference(model)

# ==========================================
# 3. Prompt 构造与预处理
# ==========================================
# 直接使用 Qwen 标准的 ChatML 格式，绕过 buggy 的 apply_chat_template
prompt = """<|im_start|>system
You are a helpful AI assistant that always thinks step-by-step 中文回复我.<|im_end|>
<|im_start|>user
10米的绳子把老虎拴住，老虎该怎样才能吃到20米外的草？ .<|im_end|>
<|im_start|>assistant
"""

# 【核心修复】：明确指定 text 参数，并关闭自动添加特殊符号（因为我们已经手动写了）
inputs = tokenizer(
    text = prompt, 
    add_special_tokens = False,
    return_tensors = "pt"
).to("cuda")

# ==========================================
# 4. 模型生成
# ==========================================
print("\n🧠 模型正在思考中...\n" + "="*50)
outputs = model.generate(
    **inputs, 
    max_new_tokens = 14048, 
    use_cache = True,
    temperature = 0.7
)

# 截取新生成的部分，去掉我们喂进去的 prompt 问题部分
prompt_length = inputs.input_ids.shape[1]
generated_tokens = outputs[0][prompt_length:]
answer = tokenizer.decode(generated_tokens, skip_special_tokens=True)

print(answer)
print("="*50)

# ==========================================
# 5. 显存回收与清理
# ==========================================
del model, tokenizer, inputs, outputs
import gc
gc.collect()
torch.cuda.empty_cache()
print("✅ 测试结束，显存已完美释放，全额还给训练进程。")