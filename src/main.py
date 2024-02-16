from transformers import T5Tokenizer, T5Model
import json
from SQP1Dataset import initialize_datasets, SQP1Dataset
from torch.utils.data import Dataset, DataLoader

# Load tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-base")

datasets = initialize_datasets('data/train.json', 'data/dev.json', tokenizer)

validation_dataloader = DataLoader(datasets['dev'],
                                   batch_size=64,
                                   shuffle=False,
                                   collate_fn=SQP1Dataset.collate_fn)

batch = next(iter(validation_dataloader))
print(batch)

print(f"{len(datasets['dev'])}")
