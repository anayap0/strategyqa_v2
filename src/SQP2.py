from dataclasses import dataclass
from typing import List, Tuple
import json

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerFast, AutoTokenizer

@dataclass
class SQP2Example: 
    decompositions: List[str]
    answres: List[str]

    @staticmethod
    def from_dict(data: dict):
        decompositions = data['decompositions']
        answers = data['answers']

        return SQP2Example(
            decompositions=decompositions,
            answers=answers,
    )