




from transformers import(
    DataCollatorForLanguageModeling,
    Trainer
)
import torch
from torch.utils.data import DataLoader
from data_process import DataParse
from model_infor_check import model_tokenizer_load
from set_train_parameters import set_train_parameters
from reward_data_collator_for_RW import RewardDataCollator,RewardDataCollatorConcatenated
from tqdm import tqdm
from typing import Any

def train(model_name,model_path,datasets_name,datasets_path):

    torch.manual_seed(42)
    # ===============================
    # 1. 模型和分词器初始化,冻结非训练层，只解冻训练层
    # ===============================
    model,tokenizer = model_tokenizer_load(model_path=model_path,model_name=model_name)
    tokenizer.pad_token = tokenizer.eos_token = "<|endoftext|>"
    model.config.pad_token_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")


    # ===============================
    # 2.冻结非训练层，只解冻训练层，打印训练参数
    # ===============================
    # initalize the trainable_params and total_params
    trainable_params = 0
    total_params = 0

    # set all param.requires_grad = False
    for param in model.parameters():
         param.requires_grad = False

    # statistical training params
    for name,param in model.named_parameters():
         total_params += param.numel()
         # set last 2 layers
         if name.startswith("transformer.h.30.") or name.startswith("transformer.h.31."):
            if "c_attn" in name or "c_proj" in name:
                param.requires_grad = True
                trainable_params += param.numel()
            else:
                param.requires_grad = False
         else:
            param.requires_grad = False

    # print parameters training
    print(f"可训练参数: {trainable_params:,}")
    print(f"总参数: {total_params:,}")
    print(f"可训练参数占比: {100 * trainable_params / total_params:.2f}%")
    print(model.dtype)


    # ===============================
    # 3. 数据处理和加载（针对Qwen格式）
    # ===============================
    dataparser = DataParse(
        tokenizer=tokenizer,
        sample_size=1000,
        datasets_name=datasets_name,
        datasets_path =datasets_path,
        max_length=2048
    )
    raw_dataset = dataparser.load_and_parse_data()
    processed_dataset = dataparser.process_dataset_for_reward_model(raw_dataset)
    # give up the dataloader there


    # ===============================
    # 4. 分割数据集
    # ===============================
    train_test_split = processed_dataset.train_test_split(test_size=0.1,seed = 42)
    train_dataset = train_test_split['train']
    eval_dataset = train_test_split['test']
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(eval_dataset)}")

    # ===============================
    # 4.数据集收集器
    # ===============================
    '''
        这个 collator 是为普通的语言模型训练设计的，它期望标准的 input_ids 字段，而不是我们 reward model 的配对数据格式(input_ids_chosen, input_ids_rejected)等
    '''
    # data_collator = DataCollatorForLanguageModeling(
    #     tokenizer=tokenizer,
    #     mlm = False,
    # )

    '''
        自定义数据收集器类： RewardModelDataCollator(CLASS)
    '''
    data_collator = RewardDataCollatorConcatenated(tokenizer = tokenizer)
    

    # ===============================
    # 5.设置训练参数
    # ===============================
    training_args = set_train_parameters("./qwen-RW-finetuned")


    # ===============================
    # 6.创建训练器
    # ===============================
    '''传统Trainer 训练器'''
    # trainer = Trainer(
    #     model = model,
    #     args = training_args,
    #     train_dataset = train_dataset,
    #     eval_dataset = eval_dataset,
    #     processing_class = tokenizer,
    #     data_collator = data_collator
    # )

    '''自定义训练器 优化器 前向传播和损失'''
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=data_collator
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=data_collator
    )
    
    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr = 1e-5, 
        weight_decay= 0.01
        )

    # Foward
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


    # ===============================
    # 7.裸跑
    # ===============================
    # print("=====================裸跑========================>>>>>>>>>>")
    # batch = data_collator([train_dataset[0]] * 4)
    # batch = {k: v.to(model.device) for k,v in batch.items()}
    # with torch.no_grad():
    #         outputs = model(**batch)
    # print("裸跑输出:",outputs)


    # ===============================
    # 8.训练
    # ===============================
    '''
        自定义训练流程
    '''
    print('============================== Self Train Begin=======================')
    model.train()
    for epoch in range(3):
        print(f"\nEpoch {epoch + 1}/3")

        for setp,batch in enumerate(tqdm(train_dataloader)):
            batch = {k:v.to(model.device) if isinstance(v,torch.tensor) else v 
                     for k,v in batch.item()}
            
            # Forward
            outputs = model(**batch)
            rewards = outputs.logits.squeeze(-1) if hasattr(outputs,'logits') else outputs.squeeze(-1)

            # Loss
            loss,chosen_rewards,rejected_rewards = compute_reward_loss(rewards=rewards)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 50 == 0:
                print(f"Step {step}: Loss={loss.item():.4f}, "
                    f"Chosen={chosen_rewards.mean().item():.4f}, "
                    f"Rejected={rejected_rewards.mean().item():.4f}")

    # ===============================
    # 9.保存模型和训练信息
    # ===============================



if __name__ == "__main__":
    datasets_name = "HuggingFaceH4/ultrafeedback_binarized"
    datasets_path = "../configs/datasets/"
    model_path = "../configs/models/"
    model_name = "Qwen/Qwen-7B-Chat"
    train(
        model_name=model_name,
        model_path=model_path,
        datasets_name=datasets_name,
        datasets_path=datasets_path
    )