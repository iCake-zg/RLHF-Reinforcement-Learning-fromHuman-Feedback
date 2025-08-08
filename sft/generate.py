from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 输入文本
input_text = "你是什么模型"
# 直接使用本地模型路径
model_path = "../configs/models/"

# 加载分词器（使用本地路径）
# 加载分词器
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-7B-Chat", 
                                          trust_remote_code=True,
                                          cache_dir=model_path)
# 编码输入
inputs = tokenizer(input_text, return_tensors="pt")


## 加载原始模型（使用本地路径）
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B-Chat", 
                                                  cache_dir=model_path,
                                                  trust_remote_code=True) # 只使用本地文件


## 生成和解码
base_output = base_model.generate(
    **inputs, max_new_tokens=100, pad_token_id=tokenizer.eos_token_id
)
base_result = tokenizer.decode(base_output[0], skip_special_tokens=True)

# 输出结果
print("🔹原始模型输出：")
print(base_result)