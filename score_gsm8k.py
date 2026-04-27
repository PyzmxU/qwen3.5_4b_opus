import json
import re
import sys

def get_target_number(target_str):
    """从目标字符串中提取标准答案（提取 #### 后面的数字）"""
    match = re.search(r'####\s*(-?[\d,]+(?:\.\d+)?)', target_str)
    if match:
        # 移除可能存在的逗号，例如 70,000 -> 70000
        return match.group(1).replace(',', '')
    return None

def get_all_numbers(text):
    """从文本中提取所有的数字（处理逗号和符号干扰）"""
    # 移除文本中的逗号，防止 70,000 被拆分成 70 和 000
    text = text.replace(',', '')
    # 提取所有整数和小数
    return re.findall(r'-?\d+(?:\.\d+)?', text)

def calculate_scores(file_path):
    total = 0
    correct_strict = 0    # 机器死板打分：必须有 #### 
    correct_last = 0      # 逻辑最终结论：文本最后一个数字是正确答案
    correct_anywhere = 0  # 松散匹配：只要推理过程中出现了这个数字就算对

    print(f"正在分析文件: {file_path} ...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                
                # 1. 获取标准答案
                target_num = get_target_number(data.get('target', ''))
                if not target_num:
                    continue
                    
                total += 1
                resps = data.get('resps', [])
                if not resps or not resps[0]:
                    continue
                    
                pred_text = resps[0][0]
                
                # 2. 提取模型回答中的所有数字
                all_nums = get_all_numbers(pred_text)
                
                # --- 评判维度 1: 严格匹配 (原版 lm-eval 逻辑) ---
                if re.search(r'####\s*' + re.escape(target_num) + r'\b', pred_text):
                    correct_strict += 1
                    
                # --- 评判维度 2: 结论匹配 (最后一个数字) ---
                if all_nums and all_nums[-1] == target_num:
                    correct_last += 1
                    
                # --- 评判维度 3: 过程匹配 (推理中出现即得分) ---
                if target_num in all_nums:
                    correct_anywhere += 1
                    
            except json.JSONDecodeError:
                continue

    if total == 0:
        print("未找到有效的评测数据，请检查文件格式。")
        return

    print("\n" + "="*40)
    print(f"📊 GSM8K 真实逻辑能力评估报告")
    print("="*40)
    print(f"总计有效题目: {total}")
    print("-" * 40)
    print(f"❌ 1. 严格格式得分 (含 ####)  : {correct_strict} / {total}  ({(correct_strict/total)*100:.2f}%)")
    print(f"⭐ 2. 最终结论得分 (末尾数字) : {correct_last} / {total}  ({(correct_last/total)*100:.2f}%)  <-- 最能反映真实能力的指标")
    print(f"🔍 3. 过程命中得分 (提及即算) : {correct_anywhere} / {total}  ({(correct_anywhere/total)*100:.2f}%)")
    print("="*40)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        calculate_scores(sys.argv[1])
    else:
        print("用法: python score_gsm8k.py <你的jsonl文件路径>")
        print("示例: python score_gsm8k.py samples_gsm8k_2026-04-27T22-30-20.755970.jsonl")