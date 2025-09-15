



import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, PeftModel
from datasets import load_dataset

'''
set model_path and data_path
'''
datasets_name = "HuggingFaceH4/ultrafeedback_binarized"
datasets_path = "../configs/datasets/"
model_name = "Qwen/Qwen-7B-Chat"
model_path = "../configs/model/"



def model_tokenizer_load(model_path:str):
    '''
    """加载模型和分词器"""
    Args:
        Input: 
            model_path(str)

        Return: 
            model(object)
            tokenizer(object)
    '''

    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                cache_dir=model_path,
                                                 trust_remote_code=True,
                                                 local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                cache_dir = model_path,
                                                local_files_only=True,
                                                trust_remote_code=True)
    
    return model,tokenizer



def set_train_parameters(model:object,tarin_name:str):
    '''
    """设置训练参数"""
    Args:
        Input: 
            model(object)
            tarin_name(str): layers need to train

        Return: 
            PASS
    '''
    total_params = 0
    train_params = 0
    for name,param in model.named_parameters():
        total_params += param.numel()   
        if name.startswith(tarin_name):
            param.requires_grad = True
            train_params += param.numel()
        else:
            param.requires_grad = False
    print(f"总参数量: {total_params}, 训练参数量: {train_params}",f"可训练参数占比: {100 * train_params / total_params:.2f}%")

    pass



# def lora_set_train_parameters(model):

#     lora_config = LoraConfig(
#         r=8,
#         lora_alpha=16,
#         lora_dropout=0.05,
#         bias="none",
#         task_type="CAUSAL_LM",
#         target_modules=["c_attn","c_proj"],
#     )
#     model = get_peft_model(model, lora_config)
#     print("LoRA 参数量:", model.print_trainable_parameters())
#     return model


class DataParse(object):

    def __init__(self,tokenizer,simple_size:int):
        self.tokenizer = tokenizer
        self.simple_size = simple_size

    def load_and_prepare_data(tokenizer,simeple_size:int):
        datasets = load_dataset(datasets_name,split="train_prefs",
                            cache_dir=datasets_path)
        print(f"数据集 {datasets_name} 加载完成，数据集大小: {datasets.num_rows}")

        return datasets

    @classmethod
    def chat_text(self,messages,add_generation_prompt = False,ensure_eos = True):
        s = self.tokenizer.apply_chat_template(messages,tokenizer = False,add_generation_prompt = add_generation_prompt)
        if ensure_eos and self.tokenizer.eos_token and not s.endswith(self.tokenizer.eos_token):
            s += self.tokenizer.eos_token
        return s
    
    @classmethod
    def to_pair_text(self,example):
        chosen_text = self.chat_text(example["chosen"])
        rejected_text = self.chat_text(example["rejected"])
        
        return {"chosen_text":chosen_text,"rejected_text":rejected_text}
    
    




