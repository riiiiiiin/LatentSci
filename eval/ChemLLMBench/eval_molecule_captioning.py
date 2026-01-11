import os, json
from core.utils import extract_answer
import logging
import os
import statistics

from core.task_evaluator import TextSimiliarityTaskEvaluator

def evaluate_molecule_captioning_score(model_name, sample_count, gt_path, logs_dir, results_dir):
    mol_captioning_evaluator = TextSimiliarityTaskEvaluator()
    result = mol_captioning_evaluator.run(model_name, sample_count, gt_path, logs_dir, "molecule_captioning")
    os.makedirs(os.path.join(results_dir, 'molecule_captioning'), exist_ok=True)
    with open(os.path.join(results_dir, 'molecule_captioning', f'{model_name}_{sample_count}.json'), 'w') as f:
        json.dump(result, f, indent=4)
    return result