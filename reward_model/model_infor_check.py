import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, PeftModel
from datasets import load_dataset


def model_tokenizer_load(model_path,model_name):

    """加载模型和分词器"""

    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                cache_dir=model_path,
                                                 trust_remote_code=True)


    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                cache_dir = model_path,
                                                trust_remote_code=True)
    
    return model,tokenizer
                            


def check_model_base_parameters(model,tokenizer):
    """
    检查模型和分词器的参数
    """
    print("模型名称:", model.name_or_path)
    print("模型数据类型:", model.dtype)
    print("分词器名称:", tokenizer.name_or_path)
    print("模型参数数量:", model.num_parameters())
    print("分词器词汇表大小(tokenizer.vocab_size):", tokenizer.vocab_size)
    print("模型词汇表大小(model.config.vocab_size):", model.config.vocab_size)



if __name__ == "__main__":
    model_path = "../configs/models/"
    model_name = "Qwen/Qwen-7B-Chat"
    model,tokenizer = model_tokenizer_load(
            model_path=model_path,
            model_name=model_name)
    
    check_model_base_parameters(model,tokenizer)