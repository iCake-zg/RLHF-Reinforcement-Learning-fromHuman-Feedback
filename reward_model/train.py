




from transformers import(
    DataCollatorForLanguageModeling,
    Trainer
)
import torch
from data_process import DataParse
from model_infor_check import model_tokenizer_load
from set_train_parameters import set_train_parameters



def train():

    torch.manual_seed(42)
    # ===============================
    # 1. 模型和分词器初始化
    # ===============================
    model,tokenizer = model_tokenizer_load(model_path=model_path,model_name=model_name)
    tokenizer.pad_token = tokenizer.eos_token = "<|endoftext|>"
    model.config.pad_token_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")

    # ===============================
    # 2. 数据处理和加载（针对Qwen格式）
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
    dataloader = dataparser.create_DataLoader(processed_dataset, batch_size=4)

    # ===============================
    # 3. 分割数据集
    # ===============================
    train_test_split = processed_dataset.train_test_split(test_size=0.1,seed = 42)
    train_dataset = train_test_split['train']
    eval_dataset = train_test_split['test']
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(eval_dataset)}")

    # ===============================
    # 4.数据集收集器
    # ===============================
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm = False,
    )

    # ===============================
    # 5.设置训练参数
    # ===============================
    training_args = set_train_parameters("./qwen-RW-finetuned")

    # ===============================
    # 6.创建训练器
    # ===============================
    trainer = Trainer(
        model = model,
        args = training_args,
        train_dataset = train_dataset,
        eval_dataset = eval_dataset,
        processing_class = tokenizer,
        data_collator = data_collator
    )

    # ===============================
    # 7.裸跑
    # ===============================


    # ===============================
    # 8.训练
    # ===============================


    # ===============================
    # 9.保存模型和训练信息
    # ===============================



