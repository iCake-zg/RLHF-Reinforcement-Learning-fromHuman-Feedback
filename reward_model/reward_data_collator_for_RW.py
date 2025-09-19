


import torch
from dataclasses import dataclass
from transformers import PreTrainedTokenizer
from typing import Dict,Any,List

@dataclass
class RewardDataCollator:
    tokenizer:PreTrainedTokenizer
    padding: bool = True
    max_length: int = None
    pad_to_multiple_of:int = None

    def __call__ (
        self,
        features:List[Dict[str,Any]],
    ) -> Dict[str,torch.tensor]:

        '''
        分离 chosen 和 rejected 
        '''

        chosen_features = []
        rejected_features = []

        # split the feature
        for feature in features:
            chosen_features.append({
                "input_ids": feature['input_ids_chosen'],
                "attention_mask":feature['attention_mask_chosen']
            })
            rejected_features.append({
                "input_ids": feature['input_ids_rejected'],
                "attention_mask":feature['attention_mask_rejected']
            })

        # toknerizer and padding 

        chosen_batch = self.tokenizer.pad(
            chosen_features,
            padding = self.padding,
            max_length = self.max_length,
            pad_to_multiple_of = self.pad_to_multiple_of,
            return_tensors = "pt"
        )
        rejected_batch = self.tokenizer.pad(
            rejected_features,
            padding = self.padding,
            max_length = self.max_length,
            pad_to_multiple_of = self.pad_to_multiple_of,
            return_tensors = "pt"
        )

        return {
            'input_ids_chosen': chosen_batch['input_ids'],
            'attention_mask_chosen': chosen_batch['attention_mask'],
            'input_ids_rejected': rejected_batch['input_ids'],
            'attention_mask_rejected': rejected_batch['attention_mask'],
        }



@dataclass
class RewardDataCollatorConcatenated:

    def __init__(self,tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self,features):
        
        # collect all data_chosen and data_rejected
        all_input_ids = []
        all_attention_masks = []

        for feature in features:
            all_input_ids.extend([
                feature['input_ids_chosen'],
                feature['input_ids_rejected']
            ])

            all_attention_masks.extend([
                feature['attention_mask_chosen'],
                feature['attention_mask_rejected']
            ])

        padded = self.tokenizer.pad(
            [{'input_ids':ids,'attention_mask':attention_mask}
             for ids,attention_mask in zip(all_input_ids,all_attention_masks)],
             return_tensors = 'pt'
        )

        return {
            'input_ids':padded['input_ids'],
            'attention_mask':padded['attention_mask'],
        }


        

