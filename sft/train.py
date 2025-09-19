# Qwen-7B-Chat 注意力层微调 - BELLE数据集
# 只微调 c_attn 和 c_proj 层

import json
import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
import os

# ===============================
# 1. 模型和分词器初始化
# ===============================
def setup_qwen_model_and_tokenizer(model_path="../configs/models/"):
    """初始化Qwen模型和分词器，只微调注意力层"""
    
    print(f"正在加载模型: {model_path}")
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen-7B-Chat",
        cache_dir=model_path,
        trust_remote_code=True,
        local_files_only=True
    )
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen-7B-Chat",
        cache_dir=model_path,
        torch_dtype=torch.float32,
        bf16 = False,
        fp32 = True,
        trust_remote_code=True,
        local_files_only=True
    )

    model.config.use_cache = False  # 禁用缓存以支持梯度检查点
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})  # 启用梯度检查点以节省显存

    print("model.config.vocab_size =", model.config.vocab_size)
    print("tokenizer.vocab_size =", tokenizer.vocab_size)
    print("eos_token:", tokenizer.eos_token)
    print("eos_token_id:", tokenizer.eos_token_id)
    print("pad_token:", tokenizer.pad_token)
    print("pad_token_id:", tokenizer.pad_token_id)

    print("id->token[151643]:", tokenizer.convert_ids_to_tokens(151643))
    print("added_vocab keys:", list(tokenizer.get_added_vocab().keys())[:10])

    print("eos_token:", tokenizer.eos_token)
    print("eos_token_id:", tokenizer.eos_token_id)

    # 设置特殊token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = "<|endoftext|>"
        model.config.pad_token_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")

    print("eos_token:", tokenizer.eos_token)
    print("eos_token_id:", tokenizer.eos_token_id)
    print("pad_token:", tokenizer.pad_token)
    print("pad_token_id:", tokenizer.pad_token_id)

    
    if len(tokenizer) > model.config.vocab_size:
        model.resize_token_embeddings(len(tokenizer))
        print(f"调整模型embedding大小至: {model.config.vocab_size}")
    
    # 冻结所有参数
    for param in model.parameters():
        param.requires_grad = False

    
    # 只解冻 c_attn 和 c_proj 层
    trainable_params = 0
    total_params = 0
    
    # for name, param in model.named_parameters():
    #     total_params += param.numel()
    #     # 检查是否是目标层
    #     if "c_attn" in name or "c_proj" in name:
    #         param.requires_grad = True
    #         trainable_params += param.numel()
    #         # print(f"解冻参数: {name}")

    for name, param in model.named_parameters():
        total_params += param.numel()
        # 只关心最后两层
        if name.startswith("transformer.h.30.") or name.startswith("transformer.h.31."):
            # 再细化到 c_attn / c_proj
            if "c_attn" in name or "c_proj" in name:
                param.requires_grad = True
                trainable_params += param.numel()
            else:
                param.requires_grad = False    # 其余保持冻结
        else:
            param.requires_grad = False        # 其他层全部冻结
    
    print(f"可训练参数: {trainable_params:,}")
    print(f"总参数: {total_params:,}")
    print(f"可训练参数占比: {100 * trainable_params / total_params:.2f}%")
    print(model.dtype)
    
    return model, tokenizer



# ===============================
# 2. 数据处理（针对Qwen格式）
# ===============================
def format_qwen_chat_data(example):
    """
    将BELLE数据格式化为Qwen对话格式
    Qwen格式: <|im_start|>system\n你是一个有用的助手。<|im_end|>\n<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>
    """
    instruction = example['instruction']
    output = example['output']
    
    # Qwen对话格式
    conversation = f"<|im_start|>system\n你是一个有用的助手。<|im_end|>\n<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"
    
    return {"text": conversation}

def preprocess_qwen_dataset(dataset, tokenizer, max_length=1024):
    """预处理数据集为Qwen格式"""
    
    # 格式化数据
    formatted_dataset = dataset.map(format_qwen_chat_data)
    
    def tokenize_function(examples):
        # 分词
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_overflowing_tokens=False,
        )
        
        # 设置labels
        tokenized["labels"] = tokenized["input_ids"].copy()


        # print("\n--- 检查 tokenized 示例 ---")
        # print(f"input_ids: {tokenized['input_ids'][0]}")
        # print(f"labels: {tokenized['labels'][0]}")
        # print(f"labels 类型: {type(tokenized['labels'][0])}")
        
        return tokenized
    
    # 应用分词
    tokenized_dataset = formatted_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=formatted_dataset.column_names,
        num_proc=4  # 多进程加速
    )
    
    return tokenized_dataset

# ===============================
# 3. 数据加载
# ===============================
def load_and_prepare_belle_data(tokenizer, sample_size=None):
    """加载并准备本地BELLE数据"""
    import json
    from datasets import Dataset
    
    print("正在加载本地BELLE数据集...")
    
    # 从本地JSONL文件加载数据（每行一个JSON对象）
    data_path = "../configs/datasets/Belle_open_source_0.5M.json"
    
    try:
        raw_data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:  # 跳过空行
                    try:
                        json_obj = json.loads(line)
                        raw_data.append(json_obj)
                    except json.JSONDecodeError as e:
                        print(f"跳过无效的JSON行: {line[:50]}... 错误: {e}")
                        continue
            
        print(f"成功加载 {len(raw_data)} 条数据")
        
    except FileNotFoundError:
        print(f"文件未找到: {data_path}")
        print("请确认文件路径是否正确")
        return None
    except Exception as e:
        print(f"加载数据时出错: {e}")
        return None
    
    # 如果指定了样本大小，进行采样
    if sample_size and sample_size < len(raw_data):
        raw_data = raw_data[:sample_size]
        print("源数据:", raw_data[:3])
        print(f"使用 {sample_size} 个样本进行训练")
    
    # 数据质量过滤
    def filter_function(example):
        # 过滤空输出或过短输出
        if not example.get('output') or len(example['output'].strip()) < 5:
            return False
        # 过滤过长的样本  
        instruction = example.get('instruction', '')
        output = example.get('output', '')
        if len(instruction + output) > 2000:
            return False
        return True
    
    # 过滤数据
    filtered_data = [item for item in raw_data if filter_function(item)]
    print(f"过滤后数据量: {len(filtered_data)}")
    print(f"示例数据: {filtered_data[:3]}")  # 打印前3个样本
    # 转换为Dataset对象
    dataset = Dataset.from_list(filtered_data)
    
    # 预处理
    processed_dataset = preprocess_qwen_dataset(dataset, tokenizer)

    print("检查处理后的数据...")
    sample_data = processed_dataset[0]  

    print(f"样本数据键: {list(sample_data.keys())}")
    print("input_ids 示例:", sample_data['input_ids'][:50])
    print("labels 示例:", sample_data['labels'][:50])
    print("attention_mask 示例:", sample_data['attention_mask'][:50])
    print("token_type_ids 示例:", sample_data['token_type_ids'][:50] if 'token_type_ids' in sample_data else "无 token_type_ids")
    
    if 'input_ids' in sample_data:
        input_ids = sample_data['input_ids']
        print(f"input_ids 范围: {min(input_ids)} 到 {max(input_ids)}")
        print(f"input_ids 中超出词汇表的token: {[id for id in input_ids if id >= tokenizer.vocab_size]}")
    
    return processed_dataset


# ===============================
# 4. 训练配置
# ===============================
def create_qwen_training_args(output_dir="./qwen-belle-attention-finetuned"):
    """创建针对Qwen的训练参数"""
    return TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        
        # 训练轮数和步数
        num_train_epochs=2,
        max_steps=5000,
        
        # 批次大小
        per_device_train_batch_size=2,  # 根据显存调整
        dataloader_num_workers=0,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,   # 有效batch_size = 2*8 = 16
        
        # 学习率设置
        learning_rate=1e-4,  # 只微调部分层，可以用稍大的学习率
        weight_decay=0.01,
        warmup_steps=200,
        
        # 精度和优化
        fp16=True,
        bf16=False,
        dataloader_pin_memory=False,
        gradient_checkpointing=True,  # 节省显存
        
        # 日志和保存
        logging_steps=1,
        save_steps=500,
        eval_steps=500,
        disable_tqdm=False,
        save_total_limit=3,
        
        # 评估设置
        eval_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        
        # 其他
        remove_unused_columns=False,
        report_to=None,  # 不使用wandb等
    )

def check_all_labels(dataset, vocab_size):
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=1)
    for idx, batch in enumerate(dataloader):
        labels = batch["labels"]
        if isinstance(labels, list):
            labels = torch.tensor(labels)
        if labels.max() >= vocab_size or labels.min() < -100:
            print(f"❌ 样本 {idx} 标签越界！max={labels.max().item()}, min={labels.min().item()}")
            return False
    print("✅ 所有标签合法")
    return True

# ===============================
# 5. 主训练流程
# ===============================
def main():
    # 1. 设置随机种子
    torch.manual_seed(42)
    
    # 2. 初始化模型和分词器
    model, tokenizer = setup_qwen_model_and_tokenizer()

    
    #3. 加载和预处理数据
    processed_dataset = load_and_prepare_belle_data(
        tokenizer, 
        sample_size=10000  # 使用1万个样本进行测试，设为None使用全部数据
    )
    
    # # 4. 分割数据集
    train_test_split = processed_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = train_test_split['train']
    eval_dataset = train_test_split['test']
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(eval_dataset)}")

    
    # # 5. 创建数据收集器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # #6. 设置训练参数
    training_args = create_qwen_training_args()

    

    print("lm_head out_features =", model.lm_head.weight.shape[0])
    print("tokenizer.vocab_size =", tokenizer.vocab_size)

        ## 裁剪头
    old_lm_head = model.lm_head
    new_lm_head = torch.nn.Linear(
        old_lm_head.in_features,
        tokenizer.vocab_size,           # 151851
        bias=old_lm_head.bias is not None
    )
    # 把前 151851 个权重拷过来
    with torch.no_grad():
        new_lm_head.weight[:151851] = old_lm_head.weight[:151851]
        if old_lm_head.bias is not None:
            new_lm_head.bias[:151851] = old_lm_head.bias[:151851]

    new_lm_head = new_lm_head.to(model.dtype)   
    model.lm_head = new_lm_head
    model.config.vocab_size = tokenizer.vocab_size
    print(model.dtype)
    print("new lm_head out_features =", model.lm_head.weight.shape[0])
    print("tokenizer.vocab_size     =", len(tokenizer))



    # #7. 创建训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    print(">>> 裸跑一步，看是否卡住...")
    print(train_dataset[0])
    batch = data_collator([train_dataset[0]] * 2)   # 模拟一个 batch
    batch = {k: v.to(model.device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)
    print("裸跑 loss =", outputs.loss.item())


    # # 8. 开始训练
    # print("开始训练...")
    # trainer.train()
    
    # #9. 保存模型
    # print("保存模型...")
    # trainer.save_model()
    # tokenizer.save_pretrained(training_args.output_dir)
    
    # # 10. 保存训练信息
    # with open(os.path.join(training_args.output_dir, "training_info.json"), "w") as f:
    #     json.dump({
    #         "model_path": "../configs/models/Qwen/Qwen-7B-Chat",
    #         "fine_tuned_layers": ["c_attn", "c_proj"],
    #         "dataset": "BelleGroup/train_0.5M_CN",
    #         "training_args": training_args.to_dict()
    #     }, f, indent=2, ensure_ascii=False)
    
    # print("训练完成！")




# ===============================
# 6. 推理测试
# ===============================
def test_qwen_model(model_path="./qwen-belle-attention-finetuned", test_instructions=None):
    """测试微调后的Qwen模型"""
    
    if test_instructions is None:
        test_instructions = [
            "请解释什么是人工智能",
            "写一首关于春天的诗",
            "1+1等于多少？请详细解释",
        ]
    
    print("正在加载微调后的模型...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    print("开始测试...")
    for i, instruction in enumerate(test_instructions):
        print(f"\n{'='*50}")
        print(f"测试 {i+1}: {instruction}")
        print(f"{'='*50}")
        
        # 构建Qwen格式的输入
        prompt = f"<|im_start|>system\n你是一个有用的助手。<|im_end|>\n<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
        
        # 编码
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # 生成
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                do_sample=True,
                top_p=0.8,
                repetition_penalty=1.1,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id
            )
        
        # 解码并提取回答
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # 提取assistant部分
        if "<|im_start|>assistant\n" in response:
            answer = response.split("<|im_start|>assistant\n")[-1]
            if "<|im_end|>" in answer:
                answer = answer.split("<|im_end|>")[0]
            print(f"回答: {answer}")
        else:
            print(f"完整回答: {response}")




# ===============================
# 7. 使用示例
# ===============================

if __name__ == "__main__":
    # main()
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "icake-zg-train":
            main()
        elif sys.argv[1] == "test":
            model_path = sys.argv[2] if len(sys.argv) > 2 else "./qwen-belle-attention-finetuned"
            test_qwen_model(model_path)
        else:
            print("使用方法:")
            print("python script.py train  # 开始训练")
            print("python script.py test [model_path]  # 测试模型")
    else:
        print("数据格式示例:")
        sample = {
            "instruction": "给定一个英文句子，翻译成中文。\nI love to learn new things every day.",
            "input": "",
            "output": "我每天喜欢学习新事物。"
        }
        print(json.dumps(sample, ensure_ascii=False, indent=2))
        
        print("\n转换后的Qwen格式:")
        qwen_format = "<|im_start|>system\n你是一个有用的助手。<|im_end|>\n<|im_start|>user\n给定一个英文句子，翻译成中文。\nI love to learn new things every day.<|im_end|>\n<|im_start|>assistant\n我每天喜欢学习新事物。<|im_end|>"
        print(qwen_format)