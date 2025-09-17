




from transformers import(
    AutoModelForCausalLM,
    AutoTokenizer
)
import torch
from data_process import DataParse
from model_infor_check import model_tokenizer_load




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
    # 3. 训练配置
    # ===============================

