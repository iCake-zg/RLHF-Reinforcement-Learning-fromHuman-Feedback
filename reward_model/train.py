



import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, PeftModel
from datasets import load_dataset
from typing import List,Dict,Any
'''
set model_path and data_path
'''
datasets_name = "HuggingFaceH4/ultrafeedback_binarized"
datasets_path = "../configs/datasets/"
model_path = "../configs/models/"
model_name = "Qwen/Qwen-7B-Chat"



def model_tokenizer_load(model_path,model_name):

    '''
    """加载模型和分词器"""
    Args:
        Input: 
            object:DataParse

        Return: 
            datasets(object):load_dataset
    '''

    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                cache_dir=model_path,
                                                 trust_remote_code=True)


    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                cache_dir = model_path,
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



def lora_set_train_parameters(model):

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["c_attn","c_proj"],
    )
    model = get_peft_model(model, lora_config)
    print("LoRA 参数量:", model.print_trainable_parameters())
    return model




class DataParse(object):

    def __init__(self,
                 tokenizer:None,
                 sample_size:int,
                 datasets_name:str,
                 datasets_path:str):
        super(DataParse).__init__()

        self.tokenizer = tokenizer
        self.sample_size = sample_size
        self.datasets_name = datasets_name
        self.datasets_path = datasets_path

    def load_and_parse_data(self):
        '''
        """加载和处理数据"""
        Args:
            Input: 
                object:DataParse

            Return: 
                datasets(object):load_dataset
        '''
        
        # load datesets
        datasets = load_dataset(self.datasets_name,split="train_prefs",cache_dir=self.datasets_path)
        print(f"数据集{self.datasets_name}加载完成，数据集大小{datasets.num_rows}")

        # Sample is Need
        if self.sample_size and self.sample_size < datasets.num_rows :
            datasets = datasets.select(range(self.sample_size))
            print(f"采样后大小{datasets.num_rows}")

        return datasets
    





if __name__ == "__main__":

    model,tokenizer = model_tokenizer_load(model_path=model_path,model_name=model_name)
    dataparser = DataParse(sample_size=100,datasets_name=datasets_name,datasets_path = datasets_path,tokenizer=tokenizer)
    datasets = dataparser.load_and_parse_data()


