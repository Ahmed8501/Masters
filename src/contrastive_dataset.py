import json
import torch
from torch.utils.data import Dataset

class ContrastiveNL2SQLDataset(Dataset):
    def __init__(self, path, tokenizer, max_len=128):
        with open(path, 'r', encoding='utf-8') as f:
            self.data = [json.loads(line) for line in f]
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, idx):
        item = self.data[idx]
        encoding = self.tokenizer(
            text=item['query'],
            text_pair=item['schema_element'],
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(item['label'], dtype=torch.float)
        }

    def __len__(self):
        return len(self.data)
