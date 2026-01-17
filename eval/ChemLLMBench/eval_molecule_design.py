import os, json
from core.utils import extract_answer
import logging
import os
from core.task_evaluator import MolSimiliarityTaskEvaluator

logger = logging.getLogger(__name__)

def evaluate_molecule_design_score(model_name, sample_count, gt_path, logs_dir, results_dir):
    mol_design_evaluator = MolSimiliarityTaskEvaluator()
    result = mol_design_evaluator.evaluate_score(model_name, sample_count, gt_path, logs_dir, results_dir, "molecule_design")
    os.makedirs(os.path.join(results_dir, 'molecule_design'), exist_ok=True)
    with open(os.path.join(results_dir, 'molecule_design', f'{model_name}_{sample_count}.json'), 'w') as f:
        json.dump(result, f, indent=4)
    return result