# StrategyQA Dataset NLP Final Project
In this repository, we attempt to solve the problem posed by the StrategyQA Dataset - how can we train Language Models to answer implicit reasoning questions?

We chose four models for each step in the pipeline:
1. **Question decompositions**: T5-break-finetuned (https://huggingface.co/mrm8488/t5-base-finetuned-break_data). This model was additionally finetuned, and we trained on train.json for 15 epochs with learning rate of 0.0001 and batch size of 8.
2. **Document retrieval**: BAAI/bge-reranker-large (https://huggingface.co/BAAI/bge-reranker-large) in conjunction with pyserini indexer and Lucene Searcher to retrive documents for a decomposition question. 20 docs are retrieved in the initial search step with lucene, then the model reranks and retrieves the top 5 search results for a particular decomposition question.
3. **QA model**: Google Flan T5 Large (https://huggingface.co/google/flan-t5-large) Answers the first 1...n-1 decomposition questions with context from document retrieval.
4. **BooleanQA model**: RoBERTa base BoolQ (https://huggingface.co/shahrukhx01/roberta-base-boolq) Answers the final decomposition question. Classified probabilities are converted to true (1) or false (0) based on whichever was classified as having a higher probability.

### Additional Code
- decomposition models and experiments: https://colab.research.google.com/drive/1fiBRFlLwLvnWYdLrQoVB99p1Mbvd4obx
- QA and fact retrieval models and experiments: https://colab.research.google.com/drive/1RQHA8Ik05mDQ4ewqhZt--Zsn7ZFYi7AQ
- Final Submission Notebook: https://colab.research.google.com/drive/1jj_r0qt_eqEH1Yr-QtCS3GFW-IfxVotp?usp=sharing


For a detailed explanation on the task, please refer to the original dataset paper: https://arxiv.org/abs/2101.02235

## Using our Repo
The majority of our code is located in ```src/Experiment_Part_1.ipynb``` and ```src/Experiment_Part_2.ipynb``` files. Please follow the comments and code in these files to recreate our code. Note that some of the paths in the notebooks may need to be changed to match where the data is stored in your file system.

## Building corpus
In order for ```Experiment_Part_2.ipynb``` to work, an index to search over Wikipedia must be created. Please follow ```src/generate_index.ipynb``` to generate this index.
