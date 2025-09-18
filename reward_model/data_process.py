



import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, PeftModel
from datasets import load_dataset
from typing import List,Dict,Any
from model_infor_check import model_tokenizer_load



'''
set model_path and data_path
'''
datasets_name = "HuggingFaceH4/ultrafeedback_binarized"
datasets_path = "../configs/datasets/"
model_path = "../configs/models/"
model_name = "Qwen/Qwen-7B-Chat"




class DataParse(object):

    def __init__(self,
                 tokenizer:None,
                 sample_size:int,
                 datasets_name:str,
                 datasets_path:str,
                 max_length = 2048):
        super(DataParse).__init__()

        # set pad_token == eos_token
        # if self.tokenizer.pad_token is None:
        #     self.tokenizer.pad_token = self.tokenizer.eos_token

        self.tokenizer = tokenizer
        self.sample_size = sample_size
        self.datasets_name = datasets_name
        self.datasets_path = datasets_path
        self.max_length = max_length

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
    
    def format_reward_pairs(self,messages: List[Dict]) -> str:
        '''
        """加载和处理数据"""
        Args:
            Input: 
                messages: List[Dict]: messages
            Return: 
                formatted_message
        '''
        formatted_message = ""

        for message in messages:
            role = message["role"]
            content = message["content"]
            if role == "user":
                formatted_message += f"<|im_start|>user\n{content}<|im_end|>\n"
            elif role == "assistant":
                formatted_message += f"<|im_start|>assistant\n{content}<|im_end|>\n"

            return formatted_message.strip()

    def creat_reward_pairs(self,example:Dict)  -> Dict:
        '''
        """把数据转换为reward model 需要的配对格式"""
        Args:
            Input: 
                messages: List[Dict]: messages
            Return: 
                reward_pair:Dict
        '''

        # get basic message
        prompt = example.get("prompt")
        chosen_messages = example.get("chosen",[])
        rejected_messages = example.get("rejected",[])

        # construct chosen conversation
        chosen_conversation = self.format_reward_pairs(chosen_messages)
        rejected_conversation = self.format_reward_pairs(rejected_messages)

        return {
            "prompt":prompt,
            "chosen":chosen_conversation,
            "rejected":rejected_conversation,
            "chosen_score":example.get("score_chosen"),
            "rejected_score":example.get("score_rejected")
        }

    def tokenizer_pair(self,example:Dict) -> Dict:
        '''
        """把reward model的配对格式 进行tokenizer化"""
        Args:
            Input: 
                reward_pair: Dict
            Return: 
                reward_pair: Dict (After tokenizered)
        '''

        # Tokenizer chosen response
        chosen_encodings = self.tokenizer(
            example["chosen"],
            truncation = True,       # 截断
            max_length = self.max_length,
            padding = "max_length",
            return_tensor = None
        )

        # Tokenizer rejected response
        rejected_encodings = self.tokenizer(
            example["rejected"],
            truncation = True,
            max_length = self.max_length,
            padding = "max_length",
            return_tensor = None
        )

        return {
            "input_ids_chosen":chosen_encodings["input_ids"],
            "attention_mask_chosen":chosen_encodings["attention_mask"],
            "input_ids_rejected":rejected_encodings["input_ids"],
            "attention_mask_rejected":rejected_encodings["attention_mask"],
            "chosen_score":example["chosen_score"],
            "rejected_score":example["rejected_score"]
        }

    def process_dataset_for_reward_model(self,datasets):
        '''
        """完整的数据处理流程"""
        Args:
            Input: 
                datasets
            Return: 
                datasets (After tokenizered)
        '''

        # Convert reward model format
        print("开始转换为 Reward Model 格式")
        reward_datasets = datasets.map(
            self.creat_reward_pairs,
            desc = "Creating reward pairs"
        )

        # Tokenizer
        tokenized_dataset = reward_datasets.map(
            self.tokenizer_pair,
            desc = "Toknizer pairs",
            remove_columns = reward_datasets.column_names
        )

        return tokenized_dataset
    
    def save_processed_data(self,
                            processed_date,
                            output_path:str):
        '''
        """保存处理好的数据"""
        Args:
            Input: 
                datasets
                output_path(str):  path which save the data
            Return: 
                pass
        '''

        processed_date.save_to_disk(output_path)
        print(f"处理后的数据集已保存到: {output_path}")

        pass

    def create_DataLoader(
            self,
            precessed_dataset,
            batch_size:int = 8,
            shuffle:bool = True,
    ):
        '''
        """创建dataloader 用于训练"""
        Args:
            Input: 
            precessed_data()
            batch_size(int)
            shuffle(bool = True),
        Return: 
            Dataloader
        '''

        precessed_dataset.set_format(
            type = 'torch',
            columns = ['input_ids_chosen', 'attention_mask_chosen',
                                           'input_ids_rejected', 'attention_mask_rejected']
        )

        from torch.utils.data import DataLoader

        return DataLoader(
            precessed_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self.collate_fn
        )
    
    def collate_fn(self, batch) -> Dict:
        '''
        """自定义 collate_fn 函数"""
        Args:
            Input: 
            batch
        Return: 
            Dict
        '''

        return {
            'input_ids_chosen': torch.stack([torch.tensor(item['input_ids_chosen']) for item in batch]),
            'attention_mask_chosen': torch.stack([torch.tensor(item['attention_mask_chosen']) for item in batch]),
            'input_ids_rejected': torch.stack([torch.tensor(item['input_ids_rejected']) for item in batch]),
            'attention_mask_rejected': torch.stack([torch.tensor(item['attention_mask_rejected']) for item in batch]),
        }

        

        








if __name__ == "__main__":

    model,tokenizer = model_tokenizer_load(model_path=model_path,model_name=model_name)
    tokenizer.pad_token = tokenizer.eos_token = "<|endoftext|>"
    model.config.pad_token_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    dataparser = DataParse(
        tokenizer=tokenizer,
        sample_size=1000,
        datasets_name=datasets_name,
        datasets_path =datasets_path,
        max_length=2048
    )
    raw_dataset = dataparser.load_and_parse_data()
    processed_dataset = dataparser.process_dataset_for_reward_model(raw_dataset)
    # dataparser.save_processed_data(processed_date=processed_dataset,output_path='../configs/processed_data/')

    dataloader = dataparser.create_DataLoader(processed_dataset, batch_size=4)
    for batch in dataloader:
        print("Batch shapes:")
        for key, value in batch.items():
            print(f"{key}: {value.shape}")
        break
    print("数据处理完成！")

