import json
from torch.utils.data import Dataset
from SQP1 import SQP1Example
from transformers import AutoTokenizer
import torch
from typing import List, Tuple
from transformers import T5Tokenizer, T5Model


class SQP1Dataset(Dataset):
    tokenizer: AutoTokenizer = None

    def __init__(self, raw_data, tokenizer, max_length=512):
        self.data = [SQP1Example.from_dict(data) for data in raw_data]
        SQP1Dataset.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def __iter__(self):
        """
        Get an iterator for the dataset.
        """
        return iter(self.data)

    @staticmethod
    def collate_fn(batched_samples: List[SQP1Example]) -> dict:
        questions = [sample.question for sample in batched_samples]
        decomps = ["<SEP>".join(sample.decompositions)
                   for sample in batched_samples]

        input_ids = SQP1Dataset.tokenizer(
            questions, return_tensors="pt", padding=True, truncation=True)
        target_ids = SQP1Dataset.tokenizer(
            decomps, return_tensors="pt", padding=True, truncation=True)

        return {'input_ids': input_ids, 'target_ids': target_ids}


def convert_to_t5_format(tokenizer, question, decomposition_questions):
    # Tokenize the input question and decomposition questions
    inputs = tokenizer(question)
    outputs = tokenizer("<SEP>".join(decomposition_questions))

    return inputs, outputs


def initialize_datasets(json_file_train, json_file_dev, tokenizer) -> dict:
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

    return split_datasets
