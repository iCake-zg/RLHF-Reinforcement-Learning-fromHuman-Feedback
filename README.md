
# Reinforcement Learning from Human Feedback (RLHF)

本仓库实现并记录了 **人类反馈强化学习（RLHF）** 的完整流程，涵盖三大核心阶段：

1. Supervised Fine-Tuning (SFT)
2. Reward Model Training (RM)
3. Policy Optimization (PPO)

![RLHF流程图](./assets/rlhf_diagram.png) <!-- 请确保图像路径正确 -->

---

## 📌 目录

- [1. Supervised Fine-Tuning (SFT)](#1-supervised-fine-tuning-sft)
- [2. Reward Model Training (RM)](#2-reward-model-training-rm)
- [3. Policy Optimization (PPO)](#3-policy-optimization-ppo)
- [📎 项目结构说明](#项目结构说明)
- [🚧 TODO](#todo)
- [📄 License](#license)

---

## 1. Supervised Fine-Tuning (SFT)

> **目标**：使用人类标注的高质量数据，让模型学会基础的“好行为”。

### ✅ 内容：

- **数据格式**：

```python
{
"prompt": "...",
"response": "..."
}
```

- **损失函数**：
- Cross Entropy Loss（交叉熵）

- **训练方式**：
- 通常基于预训练语言模型进行继续训练

### 📊数据集：[text](configs/datasets/Belle_open_source_0.5M.json)
- 数据模式：
源自 Stanford-Alpaca 最早开源的指令微调数据结构，后来被 Belle、Vicuna、Open-Assistant 等中文社区直接沿用

| 键名            | 作用              | 场景示例             |
| --------------- | --------------- | ---------------- |
| **instruction** | 描述“任务类型”或“高阶意图” | “将下列英文翻译成中文”|
| **input**       | 真正的“待处理内容”      | “I love apples.” |
| **output**      | 期望答案            | “我爱苹果。”          |

- 为什么input为空？
  - 简化标注：很多任务（开放式问答、创意写作、常识推理）本来就只有一句话指令，没必要再拆出第二个字段。
  - 兼容旧脚本：早期开源仓库（如 alpaca-lora、Chinese-LLaMA-Alpaca）的 collator 默认把 instruction 和 input 拼成一条 prompt




---

## 2. Reward Model Training (RM)

> **目标**：教会模型学会判断哪一个回答更好。

### ✅ 内容：

- **数据格式**：
```python
{
"prompt": "...",
"chosen": "...",
"rejected": "..."
}
```

- **结构设计**：
- 在语言模型顶部添加一个回归头（通常是一层线性层）

- **损失函数**：
- Pairwise Ranking Loss（成对排序损失）

---

## 3. Policy Optimization (PPO)

> **目标**：利用强化学习优化已有模型，使其输出更符合人类偏好。

### ✅ 内容：

- **强化学习算法**：
- PPO（Proximal Policy Optimization）

- **训练输入**：
- 模型生成多个回答 → 奖励模型打分 → 根据奖励进行策略更新

- **优化难点**：
- KL 约束与多步采样
- 显存优化（如 Memory-efficient PPO）

---

## 📎 项目结构说明（建议）

<pre lang="nohighlight">
  <code>## 📎 项目结构说明（建议） 
```bash rlhf/ 
              ├── sft/ # SFT训练脚本与数据 
              ├── reward_model/ # 奖励模型训练与评估 
              ├── ppo/ # RLHF主流程（PPO优化） 
              ├── configs/ # 配置文件（训练/模型/日志） 
              ├── assets/ # 图片/可视化内容（如rlhf_diagram.png） 
              └── README.md ``` 
  </code>
</pre>


---

## 🚧 TODO

- [ ] 支持QLoRA进行低资源训练
- [ ] 集成`trl`库快速构建训练流程
- [ ] 增加实验日志与TensorBoard支持
- [ ] 多模型支持（如 Qwen / LLaMA / Baichuan）


启动前请先设置hf镜像地址
```bash
Linux： export HF_ENDPOINT=https://hf-mirror.com
```
生成文字：在sft文件夹下
```bash
CUDA_VISIBLE_DEVICES=0 python generate.py
```

sft训练
```bash
CUDA_VISIBLE_DEVICES=0 python train.py icake-zg-train
```


对于sft训练保存的模型文件夹中，在模型文件中导入源模型文件的python文件
```bash
cp -f "$SRC/modeling_qwen.py" "$DST/"
cp -f "$SRC/configuration_qwen.py" "$DST/"
cp -f "$SRC/tokenization_qwen.py" "$DST/"
touch "$DST/__init__.py"
```

修改"config.json"中的指向
```json
  "auto_map": {
    "AutoConfig": "configuration_qwen.QWenConfig",
    "AutoModelForCausalLM": "modeling_qwen.QWenLMHeadModel",
    "AutoTokenizer":"tokenization_qwen.QWenTokenizer"
  },

  "vocab_size": 151851
```

## 🙋 Question
### Q1:
Qwen默认没有eos_token和pad_token,所以需要手动添加这两个值
```python
tokenizer.pad_token = tokenizer.eos_token = "<|endoftext|>"
model.config.pad_token_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
```

### Q2:



## 🖇️ KEYWORD

### model.vocab_size
- 来源：模型配置文件中的vocab_size（需要做64对齐）
- 作用：决定模型的Embedding层，以及LM HEAD的大小，即模型能直接处理的token ID 范围
- 注意：如果给tokenizer添加了新的token但是没有调用 model.resize_token_embeddings(len(tokenizer))，那么 model.vocab_size 不会变。

### tokenizer.vocab_size
- 来源：分词器的词表大小 tokenizer.get_vocab()
- 作用：反映当前分词器能识别的token总数
- 注意：常见的预训练模型加载 model.vocab_size和 tokenizer 一样大
- tokenizer中pad、bos、eos、unk token ID 是通过特殊值来区分的
  - bos_token="[BOS]" pad_token="[PAD]" eos_token="[EOS]" unk_token="[UNK]"


### DataCollatorForLanguageModeling（预训练时使用）

```python
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)
```
- MLM(Masked Language Modeling) 是BERT一类的双向模型训练方式，随机把输入的一部分token换成[mask]标签，然后再让模型还原这些被mask掉的token。
- CLM(Causal Language Modeling) 是GPT一类的单向模型训练方式，让模型预测第t个token时只能看到t-1个位置的信息，这样训练出来的模型只关注前后的相关性，不关注单词与单词之间的组合关系。CLM在推理时，每个token都是根据上文生成。但是BERT在推理时需要完整的上下文向量，然后才能计算每个token的mask logits。



---

## 📄 License

MIT License © 2025 [gezhou@usc.edu]
University of Southern California















