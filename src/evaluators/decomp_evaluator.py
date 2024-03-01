import argparse
import json
import jsonlines
import sys

from sari import SARI


class AnswerEvaluator:
    def __init__(self):
        self._correct = 0.0
        self._total = 0.0

    def __call__(self, gold, prediction):
        self._correct += int(gold["answer"] == prediction)
        self._total += 1

    def get_metrics(self):
        return {"Accuracy": self._correct / self._total}


class DecompositionEvaluator:
    def __init__(self):
        self._sari = SARI()

    def __call__(self, gold, prediction):
        sources = [gold["question"].replace("?", " ?").split()]
        predictions = [" ".join(prediction).replace("?", " ?").split()]
        targets = [[" ".join(gold["decomposition"]).replace("?", " ?").split()]]

        self._sari(sources, predictions, targets)

    def get_metrics(self):
        return {"SARI": self._sari.get_metric()["SARI"]}


class ParagraphsEvaluator:
    def __init__(self):
        self._scores = []
        self._retrieval_limit = 10

    @staticmethod
    def _recall(relevant_paragraphs, retrieved_paragraphs):
        result = len(set(relevant_paragraphs).intersection(retrieved_paragraphs)) / len(
            relevant_paragraphs
        )
        return result

    def __call__(self, gold, prediction):
        evidence_per_annotator = []
        for annotator in gold["evidence"]:
            evidence_per_annotator.append(
                set(
                    evidence_id
                    for step in annotator
                    for x in step
                    if isinstance(x, list)
                    for evidence_id in x
                )
            )
        retrieved_paragraphs = prediction[: self._retrieval_limit]

        score_per_annotator = []
        for evidence in evidence_per_annotator:
            score = self._recall(evidence, retrieved_paragraphs) if len(evidence) > 0 else 0
            score_per_annotator.append(score)

        annotator_maximum = max(score_per_annotator)
        self._scores.append(annotator_maximum)

    def get_metrics(self):
        return {f"Recall@{self._retrieval_limit}": float(sum(self._scores)) / len(self._scores)}


class EvaluatorWrapper:
    def __init__(self, eval_keys):
        self._evaluators = {eval_key: self._get_evaluator(eval_key) for eval_key in eval_keys}
        self._retrieval_limit = 10
    @staticmethod
    def _get_evaluator(eval_key):
        evaluator = {
            "answer": AnswerEvaluator(),
            "decomposition": DecompositionEvaluator(),
            "paragraphs": ParagraphsEvaluator(),
        }[eval_key]

        return evaluator

    def __getitem__(self, eval_key):
        return self._evaluators[eval_key]

    def get_metrics(self):
        metrics = {
            "Accuracy": 0.0,
            "SARI": 0.0,
            f"Recall@{self._retrieval_limit}": 0.0,
        }
        for evaluator in self._evaluators.values():
            metrics.update(evaluator.get_metrics())
        return metrics


def evaluate(gold_annotations, all_predictions):
    evaluator = EvaluatorWrapper(all_predictions.keys())
    for gold_instance in gold_annotations:
        qid = gold_instance["qid"]
        for predictions_key, predictions in all_predictions.items():
            evaluator[predictions_key](gold_instance, predictions)
    return evaluator.get_metrics()


def main(golds_file, predictions_file):
    # golds_file = # "data/dev.json"
    # predictions_file = "data/generated/t5predictions.jsonl"

    with open(golds_file, encoding="utf8") as infile:
        gold_annotations = json.load(infile)

    predictions = []

    with jsonlines.open(predictions_file) as reader:
        for obj in reader:
            predictions.append(obj)

    if len(gold_annotations) != len(predictions):
        raise Exception(
            f"The predictions file does not contain the same number of instances as the "
            "number of test instances."
        )

    all_predictions = {}
    for prediction in predictions:
        if len(all_predictions) == 0:
            all_predictions['decomposition'] = {}
            # for key_option in ["answer", "decomposition", "paragraphs"]:
            #     if key_option in prediction.keys():
            #         all_predictions[key_option] = {}
        # else:
        #     error_message = f"There is a difference betweeen prediction types provided for instances: {all_predictions.keys()} != {prediction.keys()}"
        #     for prediction_key in all_predictions.keys():
        #         assert prediction_key in prediction, error_message
        #     assert len(all_predictions.keys()) == len(prediction.keys()), error_message

        for prediction_key in all_predictions.keys():
            all_predictions[prediction_key][prediction['qid']] = prediction['predicted_decomposition']

    results = evaluate(gold_annotations, all_predictions)
    print(results)


if __name__ == "__main__":
  main(sys.argv[1], sys.argv[2])