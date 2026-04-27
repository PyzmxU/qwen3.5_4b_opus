from datasets import load_dataset

def load_and_prepare_dataset(dataset_id, tokenizer, max_seq_length):
    print(f"正在拉取云端数据集: {dataset_id}...")
    dataset = load_dataset(dataset_id, split="train")

    def format_claude_reasoning(example):
        formatted_messages = []
        for msg in example.get("messages", []):
            raw_role = msg.get("role")
            raw_content = msg.get("content")
            
            role = str(raw_role) if raw_role is not None else "user"
            content = str(raw_content) if raw_content is not None else ""
            
            if role in ["system", "user"]:
                formatted_messages.append({"role": role, "content": content})
            elif role == "assistant":
                raw_reasoning = msg.get("reasoning")
                reasoning = str(raw_reasoning) if raw_reasoning is not None else ""
                
                # 注入 DeepSeek 风格的思考标签
                if reasoning.strip(): 
                    final_content = f"<think>\n{reasoning}\n</think>\n{content}"
                else:
                    final_content = content
                    
                formatted_messages.append({"role": "assistant", "content": final_content})

        try:
            text = tokenizer.apply_chat_template(formatted_messages, tokenize=False, add_generation_prompt=False)
        except Exception as e:
            text = ""
            
        estimated_tokens = len(text) // 3
        return {"text": text, "token_length": estimated_tokens}

    print("正在执行 8 核并行思维链模板映射...")
    # Linux 下可以毫无顾忌地拉高并发
    dataset = dataset.map(format_claude_reasoning, num_proc=8)

    initial_count = len(dataset)
    dataset = dataset.filter(lambda x: x["token_length"] < max_seq_length)
    print(f"清洗完毕：丢弃 {initial_count - len(dataset)} 条超长数据，保留 {len(dataset)} 条。")

    return dataset.shuffle(seed=3407)