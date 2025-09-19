
# Reinforcement Learning from Human Feedback (RLHF)

本仓库实现并记录了 **人类反馈强化学习（RLHF）** 的完整流程，涵盖三大核心阶段：

1. Supervised Fine-Tuning (SFT)
2. Reward Model Training (RM)
3. Policy Optimization (PPO)

![RLHF流程图](./assets/rlhf_diagram.png) <!-- 请确保图像路径正确 -->

## 🐠飞书：
https://jslv90rgc1.feishu.cn/wiki/N8pkwxPDfi431nk6jiwcAvHUnBb?from=from_copylink

---

## 📌 目录

- [Reinforcement Learning from Human Feedback (RLHF)](#reinforcement-learning-from-human-feedback-rlhf)
  - [🐠飞书：](#飞书)
  - [📌 目录](#-目录)
  - [1. Supervised Fine-Tuning (SFT)](#1-supervised-fine-tuning-sft)
    - [✅ 内容：](#-内容)
    - [📊数据集：text](#数据集text)
  - [2. Reward Model Training (RM)](#2-reward-model-training-rm)
    - [✅ 内容：](#-内容-1)
  - [](#)
  - [3. Policy Optimization (PPO)](#3-policy-optimization-ppo)
    - [✅ 内容：](#-内容-2)
  - [📎 项目结构说明（建议）](#-项目结构说明建议)
  - [🙋 Question](#-question)
    - [Q1:](#q1)
    - [Q2:](#q2)
  - [🖇️ KEYWORD](#️-keyword)
    - [model.vocab\_size](#modelvocab_size)
    - [tokenizer.vocab\_size](#tokenizervocab_size)
    - [DataCollatorForLanguageModeling（预训练时使用）](#datacollatorforlanguagemodeling预训练时使用)
  - [📄 License](#-license)

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
- 数据集名称：HuggingFaceH4/ultrafeedback_binarized
- 数据链接：https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized
- 数据集实例：
```python
{
  "prompt": "how can i develop a habit of drawing daily",
  "prompt_id": "086b3e24f29b8956a01059f79c56db35d118a06fb6b844b095737d042795cd43",
  "chosen": [
    {
      "content": "how can i develop a habit of drawing daily",
      "role": "user"
    },
    {
      "content": "Developing a daily habit of drawing can be challenging but with consistent practice and a few tips, it can become an enjoyable and rewarding part of your daily routine. Here are some strategies to help you develop the habit of drawing daily:\n\n1. Set a specific time: Allocate a specific time of the day to draw. It could be in the morning, afternoon, or evening. Make drawing a part of your daily routine.\n2. Set a specific duration: Determine the amount of time you want to spend on drawing each day. It can be as little as 10 minutes or as long as an hour. Be consistent with the duration to help build the habit.\n3. Start small and simple: Don\\u2019t try to create a masterpiece every day, start with simple and easy-to-do sketches. Focus on improving your skills gradually.\n4. Use a variety of tools and mediums: Experiment with different tools like pencils, pens, markers, and different mediums like paper, canvas, or digital apps to keep your drawing practice interesting and engaging.\n5. Take breaks and rest: Taking breaks and resting after some time of drawing can help you avoid burnout and stay motivated.\n6. Challenge yourself: Set challenges like drawing objects from memory or a specific subject to improve your skills and keep your drawing practice interesting.\n7. Track your progress: Keep a record of your daily drawing practice and track your progress. This can be a source of motivation and help you see how far you\\u2019ve come.\n\nRemember, developing a habit takes time and patience. Stay consistent with your drawing practice, be flexible and open to trying new things, and with time, you\\u2019ll develop a habit of daily drawing that brings you joy and satisfaction.",
      "role": "assistant"
    }
  ],
  "rejected": [
    {
      "content": "how can i develop a habit of drawing daily",
      "role": "user"
    },
    {
      "content": "As an AI language model, I cannot personally develop habits for you. But, here are some tips for developing a habit of drawing daily:\n\n1. Start small: Start with simple drawings or doodles and gradually increase the complexity of your drawings.\n\n2. Set a schedule: Set a specific time every day to draw, and try to stick to it as much as possible.\n\n3. Make it fun: Don\\u2019t pressure yourself to create masterpieces every time you draw. Make it a relaxing and enjoyable experience.\n\n4. Use resources: There are many drawing tutorials available online. Use resources like YouTube or online drawing courses to help you improve your skills.\n\n5. Surround yourself with inspiration: Expose yourself to a variety of art forms, such as paintings, illustrations, and photographs, to inspire and motivate you.\n\nRemember, everyone has their own creative style and pace. Just keep practicing and enjoying the process of drawing.",
      "role": "assistant"
    }
  ],
  "messages": [
    {
      "content": "how can i develop a habit of drawing daily",
      "role": "user"
    },
    {
      "content": "Developing a daily habit of drawing can be challenging but with consistent practice and a few tips, it can become an enjoyable and rewarding part of your daily routine. Here are some strategies to help you develop the habit of drawing daily:\n\n1. Set a specific time: Allocate a specific time of the day to draw. It could be in the morning, afternoon, or evening. Make drawing a part of your daily routine.\n2. Set a specific duration: Determine the amount of time you want to spend on drawing each day. It can be as little as 10 minutes or as long as an hour. Be consistent with the duration to help build the habit.\n3. Start small and simple: Don\\u2019t try to create a masterpiece every day, start with simple and easy-to-do sketches. Focus on improving your skills gradually.\n4. Use a variety of tools and mediums: Experiment with different tools like pencils, pens, markers, and different mediums like paper, canvas, or digital apps to keep your drawing practice interesting and engaging.\n5. Take breaks and rest: Taking breaks and resting after some time of drawing can help you avoid burnout and stay motivated.\n6. Challenge yourself: Set challenges like drawing objects from memory or a specific subject to improve your skills and keep your drawing practice interesting.\n7. Track your progress: Keep a record of your daily drawing practice and track your progress. This can be a source of motivation and help you see how far you\\u2019ve come.\n\nRemember, developing a habit takes time and patience. Stay consistent with your drawing practice, be flexible and open to trying new things, and with time, you\\u2019ll develop a habit of daily drawing that brings you joy and satisfaction.",
      "role": "assistant"
    }
  ],
  "score_chosen": 8.5,
  "score_rejected": 8.5
}
```
- datasets[0]地址：RLHF-Reinforcement-Learning-fromHuman-Feedback/reward_model/data[0]view.json

- **结构设计**：
- 模型改动：只改动 transformer.h.30/31.c_atten/c_proj ,冻结其他层
- 数据设计：
```python
# CHOSEN
<|im_start|>user
how can i develop a habit of drawing daily<|im_end|>
<|im_start|>assistant
Developing a daily habit of drawing can be challenging but with consistent practice ...<|im_end|>

#REJECTED
<|im_start|>user
how can i develop a habit of drawing daily<|im_end|>
<|im_start|>assistant
As an AI language model, I cannot personally develop habits for you. But ...<|im_end|>
```


- **损失函数**：
- Pairwise Ranking Loss（成对排序损失）
```python
def compute_reward_loss(rewards):
    """
    计算reward model的preference loss
    rewards: shape (batch_size*2, 1) 或 (batch_size*2,)
    前半部分是chosen的rewards，后半部分是rejected的rewards
    """
    batch_size = rewards.shape[0] // 2
    chosen_rewards = rewards[:batch_size]      # 前半部分：chosen
    rejected_rewards = rewards[batch_size:]    # 后半部分：rejected
    
    # Preference loss: chosen应该比rejected得分更高
    loss = -torch.log(torch.sigmoid(chosen_rewards - rejected_rewards)).mean()
    return loss, chosen_rewards, rejected_rewards
```
- **代码结构**
<pre lang="nohighlight">
  <code>## 📎 项目结构说明（建议） 
```bash rlhf/reward_model
              ├── qwen-RW-finetuned/ # 奖励模型调整后的保存路径
              ├── data_process.py / # 对源数据进行处理
              ├── data[0]view_after_tokenizer.json / # tokenizer之后的第一条源数据
              ├── data[0]view.json / # 第一条源数据
              ├── dowanload_datasets.py / # 下载数据集
              ├── lora_set_train_parameters.py / # LoRA的参数设置（如果加入LoRA）
              ├── model_infor_check.py / # 模型基础信息检查
              ├── reward_data_collator_for_RW.py # 针对奖励模型的数据收集器 
              ├── set_train_parameters.py # 为 Transformer.Trainer 设置的训练参数 
              └── train.py  # 主训练流程
``` 
  </code>
</pre>
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















