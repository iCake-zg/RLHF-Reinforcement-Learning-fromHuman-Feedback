import glob, json, os
from safetensors.torch import safe_open
from transformers import AutoTokenizer, AutoConfig

model_path = "./qwen-belle-attention-finetuned/checkpoint-5000-fixed" # 改成你的目录

cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
print("config.vocab_size =", cfg.vocab_size, 
      "padded =", getattr(cfg, "padded_vocab_size", None))

tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
print("len(tokenizer) =", len(tok))

# 先看看有没有 index.json（能告诉每个权重在哪个分片）
idx_files = glob.glob(os.path.join(model_path, "*.index.json"))
if idx_files:
    with open(idx_files[0], "r") as f:
        idx = json.load(f)
    print("Total tensors in index:", len(idx.get("weight_map", {})))
    for k in ["model.embed_tokens.weight", "lm_head.weight",
              "transformer.wte.weight", "model.output_layer.weight"]:
        if k in idx.get("weight_map", {}):
            print(f"{k} -> {idx['weight_map'][k]}")

# 逐个分片查看 keys 与目标权重形状
for shard in sorted(glob.glob(os.path.join(model_path, "model-*.safetensors"))):
    with safe_open(shard, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        print(f"\n{os.path.basename(shard)} | tensors: {len(keys)}")
        for k in keys:
            if any(n in k for n in ["embed", "wte", "lm_head", "output_layer"]):
                t = f.get_tensor(k)   # 只取形状，开销很小
                print(f"  {k:35s} {tuple(t.shape)}")
