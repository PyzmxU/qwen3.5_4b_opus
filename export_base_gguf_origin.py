import os

# ==========================================
# 0. 网络防线
# ==========================================
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["UNSLOTH_DISABLE_TELEMETRY"] = "1"

from unsloth import FastLanguageModel

# ==========================================
# 1. 路径与名称配置 (核心修改点)
# ==========================================
# 【注意】这里必须填你训练最开始时，填在 config 里的那个 Hugging Face 基础模型 ID
# 比如 "Qwen/Qwen2.5-4B" 或 "Qwen/Qwen2.5-4B-Instruct"
base_model_name = "unsloth/Qwen3.5-4B" 

# 为了防止覆盖刚才的 LoRA 版，我们建一个 base_qwen 子文件夹
export_dir = "/root/blockdata/workspace/qwen3.5_4b_opus/gguf_exports/base_qwen"
os.makedirs(export_dir, exist_ok=True)

# ==========================================
# 2. 纯净加载 (不带 LoRA)
# ==========================================
print(f"📥 正在从 100G 数据盘的缓存中提取原始模型: {base_model_name}")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = base_model_name, 
    max_seq_length = 16384,
    dtype = None,
    load_in_4bit = False, 
    local_files_only = True, # 严禁联网，直接读取本地 100G 盘里的缓存
)

# ==========================================
# 3. 导出原味 GGUF
# ==========================================
print("\n⚙️ 开始编译并封装基础模型为 GGUF...")

# 同样导出 q4_k_m 格式，保证控制变量（只变了权重，没变量化精度）
model.save_pretrained_gguf(
    export_dir, 
    tokenizer, 
    quantization_method = "q4_k_m",
)

print(f"\n✅ 恭喜！纯净版基础模型 GGUF 已成功导出至: {export_dir}")