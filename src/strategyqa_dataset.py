import json
from torch.utils.data import Dataset

class StrategyQADataset(Dataset):
    def __init__(self, json_file, tokenizer, max_length=512):
        self.data = json.loads(open(json_file, encoding="utf8"))
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        question = item['question']
        decomposition = item['decomposition']
        evidence = item['evidence']

        # Tokenize question and decomposition
        encoded_inputs = self.tokenizer(
            question, decomposition, return_tensors='pt', padding=True, truncation=True, max_length=self.max_length)

        # Process evidence
        encoded_evidence = []
        for evidences in evidence:
            encoded_facts = []
            for fact in evidences:
                if isinstance(fact, list):  # If fact is a list of strings
                    encoded_fact = self.tokenizer.encode_plus(
                        fact[0], return_tensors='pt', padding=True, truncation=True, max_length=self.max_length)
                    encoded_facts.append(encoded_fact)
                else:  # If fact is a string indicating operation
                    encoded_facts.append(fact[0])
            encoded_evidence.append(encoded_facts)

        return {
            'input_ids': encoded_inputs['input_ids'],
            'attention_mask': encoded_inputs['attention_mask'],
            'encoded_evidence': encoded_evidence
        }
