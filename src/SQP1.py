from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerFast, AutoTokenizer

@dataclass
class SQP1Example: 
    question: str
    decompositions: List[str]

    @staticmethod
    def from_dict(data: dict):
        question = data['question']
        decompositions = data['decompositions']

        return SQP1Example(
            question=question,
            decompositions=decompositions,
    )

def initialize_datasets(tokenizer: PreTrainedTokenizerFast) -> dict:
    """
    Initialize the dataset objects for all splits based on the raw data.
    :param tokenizer: A tokenizer used to prepare the inputs for a model (see details in https://huggingface.co/docs/transformers/main_classes/tokenizer).
    :return: A dictionary of the dataset splits.
    """
    raw_data = load_dataset("gpt3mix/sst2")
    split_datasets = {}

    for split_name in raw_data.keys():
        split_data = list(raw_data[split_name])

        split_datasets[split_name] = SST2Dataset(tokenizer, split_data)

    return split_datasets