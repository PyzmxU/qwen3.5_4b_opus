# import os
# from unsloth import FastLanguageModel

# # ==========================================
# # 1. 路径配置
# # ==========================================
# # 指向你昨晚跑出来的最终 LoRA 文件夹
# lora_dir = "/root/qwen3.5_4b_opus/lora_qwen3.5_4b_cot"
# # GGUF 文件的输出目录
# export_dir = "/root/qwen3.5_4b_opus/gguf_exports"

# os.makedirs(export_dir, exist_ok=True)

# # ==========================================
# # 2. 加载模型与权重
# # ==========================================
# print(f"📥 正在加载基础模型与 LoRA 适配器: {lora_dir}")
# model, tokenizer = FastLanguageModel.from_pretrained(
#     model_name = lora_dir, 
#     max_seq_length = 16384,
#     dtype = None,
#     load_in_4bit = False, # 导出时必须关闭 4bit，以全精度读取
# )

# # ==========================================
# # 3. 导出 GGUF (llama.cpp 格式)
# # ==========================================
# print("\n⚙️ 开始编译并合并为 GGUF 格式...")

# # 导出选项 A：q4_k_m (最推荐的平衡量化方案)
# # 兼顾了推理速度和模型智商，体积约为 16bit 的四分之一
# model.save_pretrained_gguf(
#     export_dir, 
#     tokenizer, 
#     quantization_method = "q4_k_m",
# )

# # 导出选项 B：q8_0 (近乎无损的量化方案)
# # 如果你想最大程度保留 Claude 的思维链逻辑，推荐加上这个
# model.save_pretrained_gguf(
#     export_dir, 
#     tokenizer, 
#     quantization_method = "q8_0",
# )

# print(f"\n✅ 恭喜！GGUF 模型已成功导出至: {export_dir}")


import os

# ==========================================
# 0. 网络防线 (必须放在 unsloth 导入之前)
# ==========================================
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["UNSLOTH_DISABLE_TELEMETRY"] = "1"

from unsloth import FastLanguageModel

# ==========================================
# 1. 路径配置
# ==========================================
# lora_dir = "/root/qwen3.5_4b_opus/lora_qwen3.5_4b_cot"
# export_dir = "/root/qwen3.5_4b_opus/gguf_exports"
# ✅ 修改后：
lora_dir = "/root/blockdata/workspace/qwen3.5_4b_opus/lora_qwen3.5_4b_cot"
export_dir = "/root/blockdata/workspace/qwen3.5_4b_opus/gguf_exports"
os.makedirs(export_dir, exist_ok=True)

# ==========================================
# 2. 加载模型与权重
# ==========================================
print(f"📥 正在加载基础模型与 LoRA 适配器: {lora_dir}")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = lora_dir, 
    max_seq_length = 16384,
    dtype = None,
    load_in_4bit = False, 
    local_files_only = True, # 强制严禁联网检查基础模型
)

# ==========================================
# 3. 导出 GGUF (llama.cpp 格式)
# ==========================================
print("\n⚙️ 开始编译并合并为 GGUF 格式...")

# 导出 q4_k_m 格式
model.save_pretrained_gguf(
    export_dir, 
    tokenizer, 
    quantization_method = "q4_k_m",
)

# 导出 q8_0 格式
model.save_pretrained_gguf(
    export_dir, 
    tokenizer, 
    quantization_method = "q8_0",
)

print(f"\n✅ 恭喜！GGUF 模型已成功导出至: {export_dir}")