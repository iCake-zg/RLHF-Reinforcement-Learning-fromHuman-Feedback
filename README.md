
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

<pre lang="nohighlight"><code>## 📎 项目结构说明（建议） ```bash rlhf/ ├── sft/ # SFT训练脚本与数据 ├── reward_model/ # 奖励模型训练与评估 ├── ppo/ # RLHF主流程（PPO优化） ├── configs/ # 配置文件（训练/模型/日志） ├── assets/ # 图片/可视化内容（如rlhf_diagram.png） └── README.md ``` </code></pre>


---

## 🚧 TODO

- [ ] 支持QLoRA进行低资源训练
- [ ] 集成`trl`库快速构建训练流程
- [ ] 增加实验日志与TensorBoard支持
- [ ] 多模型支持（如 Qwen / LLaMA / Baichuan）

---

## 📄 License

MIT License © 2025 [gezhou@usc.edu]
University of Southern California















