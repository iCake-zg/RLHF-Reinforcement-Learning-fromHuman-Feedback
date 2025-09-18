


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



        

