


import torch





class RewardModelDataCollator:

    def __init__(
        self,
        tokenizer,
        pad_to_multiple_of = None
    ):
        super(RewardModelDataCollator).__init__
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self,features):
        '''
        分离 chosen 和 rejected 数据
        Args:
            Input:
                features

            Output:
                batch
        '''
        chosen_featrues = []
        rejected_featrues = []

        # split chosen and rejected features
        for feature in features:
            chosen_featrues.append(
                {
                    'input_ids':feature['input_ids_chosen'],
                    'attention_mask':feature['attention_mask_chosen']
                }
            )

            rejected_featrues.append(
                {
                    'input_ids':feature['input_ids_rejected'],
                    'attention_mask':feature['attention_mask_rejected']
                }
            )
        
        # proecess chosen and rejected features individually
        chosen_batch = self._collate_batch(chosen_featrues)
        rejected_batch = self._collate_batch(rejected_featrues)

        return {
            'input_ids_chosen': chosen_batch['input_ids'],
            'attention_mask_chosen': chosen_batch['attention_mask'],
            'input_ids_rejected': rejected_batch['input_ids'],
            'attention_mask_rejected': rejected_batch['attention_mask']
        }


    def _collate_batch(self,features):
        '''
        处理单个batch
        Args:
            Input:
                features

            Output:
                batch
        '''

        batch = {}

        if 'input_ids' in features[0]:
            input_ids = [torch.tensor(f['input_ids'] for f in features)]
            batch['input_ids'] = torch.nn.utils.rnn.pad_sequence(
                input_ids,
                batch_first=True,
                padding_value = self.tokenizer.pad_token_id
            )

        if 'attention_mask' in features[0]:
            attention_mask = [torch.tensor(f['attention_mask'] for f in features)]
            batch['attention_mask'] = torch.nn.utils.rnn.pad_sequence(
                attention_mask,
                batch_first=True,
                padding_value = self.tokenizer.pad_token_id
            )
        
        return batch
        

