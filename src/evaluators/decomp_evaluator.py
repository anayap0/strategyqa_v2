import json
import jsonlines
import sys
from sari import SARI

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

def evaluate(gold_annotations, all_predictions):
    evaluator = DecompositionEvaluator()
    for i in range(len(gold_annotations)):
        evaluator(gold_annotations[i], all_predictions[i])

    return evaluator.get_metrics()

def main(golds_file, predictions_file):
    with open(golds_file, encoding="utf8") as infile:
        gold_annotations = json.load(infile)

    predictions = []

    with jsonlines.open(predictions_file) as reader:
        for obj in reader:
            predictions.append(obj['predicted_decomposition'])

    if len(gold_annotations) != len(predictions):
        raise Exception(
            f"The predictions file does not contain the same number of instances as the "
            "number of test instances."
        )

    results = evaluate(gold_annotations, predictions)
    print(results)

if __name__ == "__main__":
  main(sys.argv[1], sys.argv[2])