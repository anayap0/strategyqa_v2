# StrategyQA Dataset NLP Final Project
In this repository, we attempt to solve the problem posed by the StrategyQA Dataset - how can we train Language Models to answer implicit reasoning questions?

For a detailed explanation on the task, please refer to the original dataset paper: https://arxiv.org/abs/2101.02235

## Using our Repo
The majority of our code is located in [src/Experiment_Part_1.ipynb](https://github.com/anayap0/strategyqa_v2/blob/main/src/Experiment_Part_1.ipynb) and src/Experiment_Part_2.ipynb files. Please follow the comments and code in these files to recreate our code. Note that some of the paths in the notebooks may need to be changed to match where the data is stored in your file system.

## Building corpus
In order for Experiment_Part_2.ipynb to work, an index to search over Wikipedia must be created. Please follow [src/generate_index.ipynb](https://github.com/anayap0/strategyqa_v2/blob/main/src/generate_index.ipynb) to generate this index.
