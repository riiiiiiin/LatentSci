from ChemLLMBench.core.metrics import exact_match
from core.utils import extract_answer
import logging
import json
import os

logger = logging.getLogger(__name__)

from core.task_evaluator import MolSimiliarityTaskEvaluator

def evaluate_reaction_prediction_score(model_name,  gt_path, logs_dir, results_dir,sample_count=1):
    mol_design_evaluator = MolSimiliarityTaskEvaluator()
    result = mol_design_evaluator.evaluate_score(model_name, sample_count, gt_path, logs_dir, "reaction_prediction")
    os.makedirs(os.path.join(results_dir, "ChemLLMBench", 'reaction_prediction'), exist_ok=True)
    with open(os.path.join(results_dir, "ChemLLMBench", 'reaction_prediction', f'{model_name}_{sample_count}.json'), 'w') as f:
        json.dump(result, f, indent=4)
    return result

def record_reaction_prediction_results(model_name, gt_path, logs_dir, results_dir, sample_count = 1):
    evaluator = MolSimiliarityTaskEvaluator()
        
    dataframe = evaluator.record_results(model_name, sample_count, gt_path, logs_dir, "reaction_prediction")
        
    os.makedirs(f"{results_dir}/ChemLLMBench/reaction_prediction", exist_ok=True)
    dataframe.to_csv(f"{results_dir}/ChemLLMBench/reaction_prediction/eval_results_{model_name}.csv", index=False)