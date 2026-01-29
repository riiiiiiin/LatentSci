
import json
import os
import logging
from ChemCoTBench.core.eval_metric import mol_opt_evaluater
from core.utils import extract_answer
import regex as re

logger = logging.getLogger(__name__)

def eval_molund_from_list(gt_list, pred_list, total_number, task):
    # this_function input: 
    #   gt_list for gt_molecules
    #   pred_list for pred_molecules
    #   total_number: len(gt_molecules)+len(cases that cannot extract SMILES)
    score = None
    if task in ["ring_system", "ring_system_scaffold", "permutated"]: # "ring_system_scaffold", "ring_system" as the same
        count = sum(1 for item in pred_list if str(item).lower() == "yes")
        if len(pred_list) == 0: score = None
        else: score = count / len(pred_list)
    elif task == "mutated":
        count = sum(1 for item in pred_list if str(item).lower() == "no")
        if len(pred_list) == 0: score = None
        else: score = count / len(pred_list)
    elif task == 'equivalence': # "equivalence" = "mutated" + "permutated"
        count = 0
        for i in range(len(pred_list)):
            if str(pred_list[i]).lower() == str(gt_list[i]).lower():
                count += 1
        if len(pred_list) == 0: score = None
        else: score = count / len(pred_list)
    elif task in ["ring_count", "fg_samples", "fg_count"]:
        assert len(gt_list) == len(pred_list)
        if len(gt_list) == 0: score = None
        else: score = sum([abs(int(pred_list[i])-int(gt_list[i])) for i in range(len(pred_list))]) / len(gt_list)
    elif task in ["murcko", "Murcko_scaffold"]: # "murcko" "Murcko_scaffold" are the same
        assert len(gt_list) == len(pred_list)
        prop_evaluater = mol_opt_evaluater(prop='qed')
        scaffold_hard, scaffold_soft = prop_evaluater.scaffold_consistency(src_mol_list=gt_list, tgt_mol_list=pred_list)
        if len(gt_list) == 0: score = None
        else: score = scaffold_soft / len(pred_list)
    
    my_dict = {
        "score": score,
        f"{task}-valid-rate": len(pred_list)/total_number
    }
    
    return my_dict

tasks = [
    "fg_count",
    'ring_count',
    'Murcko_scaffold',
    'ring_system_scaffold'
]

from core.task_evaluator import BaseTaskEvaluator
class MolUndEvaluator(BaseTaskEvaluator):
    def extract_gt(self, gt_raw_item, task):
        return gt_raw_item['gt']
    
    def extract_answer(self, pred, task):
        answer = extract_answer(pred)
        if task in ["ring_count", "fg_samples", "fg_count"]:
            try:
                answer = abs(int(re.sub(r'\D', '', answer)))
            except:
                return None
        return answer
    
    def prepare_metadata(self, sample):
        return None
    
    def evaluate_predictions(self, preds, gts, total_len, metadata = None, task_name = None):
        return eval_molund_from_list(gt_list=gts, pred_list=preds, total_number=total_len, task=task_name)

def evaluate_molund_score(model_name, gt_path, logs_dir, results_dir, sample_count):
    result_dict = dict()
    evaluator = MolUndEvaluator()
    
    for task in tasks:
        logger.info(f'evaluating {task} for model {model_name}')
        
        result_dict[task] = evaluator.evaluate_score(model_name, sample_count, gt_path, logs_dir, task)
    
    logger.info(f"eval_score_{model_name}_molund:\n\r{result_dict}")
    os.makedirs(f"{results_dir}/molund", exist_ok=True)
    json.dump(result_dict, open(f"{results_dir}/molund/eval_score_{model_name}.json", "w"), indent=4)
    
    return result_dict      
        
def record_molund_results(model_name, gt_path, logs_dir, results_dir, sample_count = 1):
    evaluator = MolUndEvaluator()
    for task in tasks:
        logger.info(f'recording {task} for model {model_name}')

        dataframe = evaluator.record_results(model_name, sample_count, gt_path, logs_dir, task)

        os.makedirs(f"{results_dir}/molund/{task}", exist_ok=True)
        dataframe.to_csv(f"{results_dir}/molund/{task}/eval_results_{model_name}.csv", index=False)