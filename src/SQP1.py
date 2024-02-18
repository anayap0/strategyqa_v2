from dataclasses import dataclass
from typing import List, Tuple
import json

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerFast, AutoTokenizer

@dataclass
class SQP1Example: 
    question: str
    decompositions: List[str]
    qid: str

    @staticmethod
    def from_dict(data: dict):
        question = data['question']
        decompositions = data['decomposition']
        qid = data['qid']

        return SQP1Example(
            question=question,
            decompositions=decompositions,
            qid=qid
    )