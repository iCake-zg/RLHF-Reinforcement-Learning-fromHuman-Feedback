

from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM


model_path = "./qwen-belle-attention-finetuned/checkpoint-5000-fixed"


input_text = "你好,你是什么模型？"

tokenizer = AutoTokenizer.from_pretrained(model_path,trust_remote_code=True)  # 只使用本地文件
# 编码输入
inputs = tokenizer(input_text, return_tensors="pt")

model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)



## 生成和解码
base_output = model.generate(
    **inputs, max_new_tokens=100, pad_token_id=tokenizer.eos_token_id
)
base_result = tokenizer.decode(base_output[0], skip_special_tokens=True)

# 输出结果
print("🔹原始模型输出：")
print(base_result)





