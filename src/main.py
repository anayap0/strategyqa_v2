from transformers import BertTokenizer
import json
from SQP1Dataset import initialize_datasets

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Define file path to your JSON data
json_file = 'data/train.json'
lines = []
with open(json_file, encoding="utf8") as f:
  lines = f.readlines()

data = json.load(open(json_file, encoding="utf8"))
datasets = initialize_datasets('data/train.json', 'data/dev.json', tokenizer)

print(f"{datasets['train']}")
