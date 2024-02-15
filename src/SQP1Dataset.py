import json
from torch.utils.data import Dataset
from SQP1 import SQP1Example
from transformers import BertTokenizer

class SQP1Dataset(Dataset):
    def __init__(self, raw_data, tokenizer, max_length=512):
        self.data = [SQP1Example.from_dict(data) for data in raw_data]
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

def initialize_datasets(json_file_train, json_file_dev, tokenizer: BertTokenizer) -> dict:
    """
    Initialize the dataset objects for all splits based on the raw data.
    :param tokenizer: A tokenizer used to prepare the inputs for a model (see details in https://huggingface.co/docs/transformers/main_classes/tokenizer).
    :return: A dictionary of the dataset splits.
    """
    raw_data_train = json.load(open(json_file_train, encoding="utf8"))
    raw_data_dev = json.load(open(json_file_dev, encoding="utf8"))
    split_datasets = {}
    split_datasets['train'] = SQP1Dataset(raw_data_train, tokenizer)
    split_datasets['dev'] = SQP1Dataset(raw_data_dev, tokenizer)


    print(split_datasets['train'].data)
    return split_datasets