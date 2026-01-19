import sys, re, os, json
from ChemCoTBench.rxn.rxnutils import read_json, is_valid_smiles
from core.utils import extract_answer
import logging
import os

logger = logging.getLogger(__name__)

subtask_to_result_key = {
    "rcr": "SMILES",
    "nepp": "pred_smi",
    "mechsel": "choice",
    "major_product": "Major Product",
    "byproduct": "Byproduct(s)",
    "retro": "Reactants"
}

from core.task_evaluator import MolSimiliarityTaskEvaluator
class RxnEvaluator(MolSimiliarityTaskEvaluator):
    def extract_answer(self, pred, task):
        ans = super().extract_answer(pred, task)
        if ans is None:
            return ""
        # Note: original ChemCoTBench code does not skip invalid answers
        return ans
        
    def extract_gt(self, gt_raw_item, task):
        gt = gt_raw_item['gt']
        if task in ['major_product', 'byproduct']:
            gt = json.loads(gt)
            gt = gt.get(subtask_to_result_key[task], '')
        return gt   
    def prepare_metadata(self, sample):
        return None

from core.task_evaluator import TextExactMatchTaskEvaluator

class MechSelEvaluator(TextExactMatchTaskEvaluator):
    def extract_gt(self, gt_raw_item, task):
        gt = gt_raw_item['gt'].lower()
        return gt
    
    def extract_answer(self, pred, task):
        pred = pred['result'].lower()
        if not pred.isalpha():
            return None
        return pred
    
    def prepare_metadata(self, sample):
        return None

def evaluate_rxn_score(model_name: str, gt_path: str, logs_dir: str, results_dir, sample_count):
    all_results = {}
    subtasks = subtask_to_result_key.keys()
    rxn_evaluator = RxnEvaluator()
    mechsel_evaluator = MechSelEvaluator()
    for subtask in subtasks:
        logger.info(f'evaluating {subtask} for model {model_name}')
        if subtask == 'MechSel' or subtask == 'mechsel':
            all_results[subtask] = mechsel_evaluator.evaluate_score(model_name, sample_count, gt_path, logs_dir, subtask)
        else:
            all_results[subtask] = rxn_evaluator.evaluate_score(model_name, sample_count, gt_path, logs_dir, subtask)
    logger.info(f"eval_score_{model_name}_rxn:\n\r{all_results}")
    os.makedirs(f"{results_dir}/rxn", exist_ok=True)
    json.dump(all_results, open(f"{results_dir}/rxn/eval_score_{model_name}.json", "w"), indent=4)

    return all_results
        
def record_rxn_score(model_name, gt_path, logs_dir, results_dir, sample_count = 1):
    rxn_evaluator = RxnEvaluator()
    mechsel_evaluator = MechSelEvaluator()
    for task in subtask_to_result_key.keys():
        logger.info(f'recording {task} for model {model_name}')
    
        if task == 'MechSel' or task == 'mechsel':
            dataframe = mechsel_evaluator.record_results(model_name, sample_count, gt_path, logs_dir, task)
        else:
            dataframe = rxn_evaluator.record_results(model_name, sample_count, gt_path, logs_dir, task)

        os.makedirs(f"{results_dir}/rxn/{task}", exist_ok=True)
        dataframe.to_csv(f"{results_dir}/rxn/{task}/eval_results_{model_name}.csv", index=False)