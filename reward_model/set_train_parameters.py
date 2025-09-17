



from transformers import TrainingArguments


def set_train_parameters(output_dir):
    '''
    """设置训练args"""
    Args:
        Return: 
            TrainingArguments
    '''
    
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