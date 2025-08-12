from transformers import AutoTokenizer, AutoModelForCausalLM,AutoConfig
import torch 
from safetensors.torch import safe_open,load_file
import glob, json, os

model_path = "./qwen-belle-attention-finetuned/checkpoint-5000"

cfg = AutoConfig.from_pretrained(model_path,trust_remote_code=True)
print(cfg.vocab_size)

cfg.vocab_size = 151936
cfg.save_pretrained(model_path)

model = AutoModelForCausalLM.from_pretrained(model_path,ignore_mismatched_sizes=True)

old_head = load_file(os.path.join(model_path, "model-00007-of-00007.safetensors"))["lm_head.weight"]

with torch.no_grad():
    # 兼容不同命名：优先 get_output_embeddings
    out = getattr(model, "get_output_embeddings", None)
    if callable(out) and out() is not None:
        w = out().weight
    else:
        w = model.lm_head.weight
    w[:old_head.shape[0]].copy_(old_head) 

tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
fixed_dir = model_path.rstrip("/")+ "-fixed"
model.save_pretrained(fixed_dir)
tok.save_pretrained(fixed_dir)
model.config.save_pretrained(fixed_dir)
print("saved to:", fixed_dir)

