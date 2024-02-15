from transformers import BertTokenizer
from strategyqa_dataset import StrategyQADataset
import json

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Define file path to your JSON data
json_file = 'data/train.json'
lines = []
with open(json_file, encoding="utf8") as f:
  lines = f.readlines()

# print(lines)

# Create dataset instance
dataset = StrategyQADataset(json_file, tokenizer)

# Access a sample from the dataset
print(f"{dataset[0]['encoded_evidence']}")
# print(f"{dataset.data[0]['evidence'][0]}")
# print(f"{dataset.data[0]['evidence'][0][1][0]}")
