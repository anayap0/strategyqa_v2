import json
from torch.utils.data import Dataset
from SQP2 import SQP2Example
from transformers import BertTokenizer
import torch
from typing import List, Tuple
from transformers import T5Tokenizer, T5Model


class SQP2Dataset(Dataset):
    tokenizer: T5Tokenizer = None

    def __init__(self, raw_data, tokenizer, max_length=512):
        self.data = [SQP2Example.from_dict(data) for data in raw_data]
        SQP2Dataset.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
        # question = item['question']
        # decomposition = item['decomposition']
        # evidence = item['evidence']

        # # Tokenize question and decomposition
        # encoded_inputs = self.tokenizer(
        #     question, decomposition, return_tensors='pt', padding=True, truncation=True, max_length=self.max_length)

        # # Process evidence
        # encoded_evidence = []
        # for evidences in evidence:
        #     encoded_facts = []
        #     for fact in evidences:
        #         if isinstance(fact, list):  # If fact is a list of strings
        #             encoded_fact = self.tokenizer.encode_plus(
        #                 fact[0], return_tensors='pt', padding=True, truncation=True, max_length=self.max_length)
        #             encoded_facts.append(encoded_fact)
        #         else:  # If fact is a string indicating operation
        #             encoded_facts.append(fact[0])
        #     encoded_evidence.append(encoded_facts)

        # return {
        #     'input_ids': encoded_inputs['input_ids'],
        #     'attention_mask': encoded_inputs['attention_mask'],
        #     'encoded_evidence': encoded_evidence
        # }

    def __iter__(self):
        """
        Get an iterator for the dataset.
        """
        # TODO: return an iterator for sample_list.
        return iter(self.data)

    @staticmethod
    def collate_fn(samples: List[SQP2Example]) -> dict:
        input_ids_list = []
        target_ids_list = []

        questions = [sample.question for sample in samples]
        decomps = [sample.decompositions for sample in samples]

        for question, decomposition_questions in zip(questions, decomps):
            input_ids, target_ids = convert_to_t5_format(
                SQP2Example .tokenizer, question, decomposition_questions)
            input_ids_list.append(input_ids)
            target_ids_list.append(target_ids)

        return {'input_ids': input_ids_list, 'target_ids': target_ids_list}


def convert_to_t5_format(tokenizer, question, decomposition_questions):
    # Tokenize the input question and decomposition questions
    tokenized_question = tokenizer.tokenize(question)
    tokenized_decomposition_questions = [
        tokenizer.tokenize(q) for q in decomposition_questions]

    # Define special tokens
    bos_token = tokenizer.bos_token
    eos_token = tokenizer.eos_token
    pad_token_id = tokenizer.pad_token_id

    # Concatenate input and output sequences
    inputs = [bos_token] + tokenized_question + [eos_token]
    targets = [[bos_token] + q + [eos_token]
               for q in tokenized_decomposition_questions]

    # Calculate max length
    max_length = max(len(inputs), max(len(target) for target in targets))

    # Pad sequences to the same length
    # inputs = inputs + [pad_token_id] * (max_length - len(inputs))
    # targets = [target + [pad_token_id] * (max_length - len(target)) for target in targets]

    # Convert tokens to token IDs
    input_ids = tokenizer.convert_tokens_to_ids(inputs)
    target_ids = [tokenizer.convert_tokens_to_ids(
        target) for target in targets]

    return input_ids, target_ids


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
