import json
from eval.utils import extract_answer
from eval.eval_molund import check_string_type
from eval.eval_molund import eval_molund_from_list
import logging
import os

logger = logging.getLogger(__name__)

def evaluate_molund_score(model_name):
    task_dict = dict(
        fg_samples="fg_samples", murcko='Murcko_scaffold', ring_count='ring_count',
        ring_system='ring_system_scaffold', equivalence = 'equivalence'
    )
    gt_key_dict = dict(
        fg_samples="fg_num", murcko='largest_scaffold', ring_count='count',
        ring_system='', equivalence = 'gt'
    )
    result_dict = dict()
    
    for task in task_dict.keys():
        logger.info(f'evaluating {task} for model {model_name}')
        if 'llama' not in model_name:
            file_name = f"logs/{task_dict[task]}/{model_name}.json"
            
        pred_results = json.load(open(file_name, "r"))
        invalid_number = 0
        
        pred_list, gt_list = list(), list()
        for pred in pred_results:
            answer = extract_answer(pred['result'])
            if answer is None:
                invalid_number += 1
                continue
            pred_list.append(answer)
            if gt_key_dict[task] != "":
                gt_list.append(pred[gt_key_dict[task]])
        
        assert len(pred_results) == invalid_number+len(pred_list)
        result_dict[task] = eval_molund_from_list(gt_list=gt_list, pred_list=pred_list, total_number=len(pred_results), task=task)
        logger.debug(model_name, task, result_dict[task])
    
    logger.info(f"eval_score_{model_name}_molund:\n\r{result_dict}")
    os.makedirs("results/molund", exist_ok=True)
    json.dump(result_dict, open(f"results/molund/eval_score_{model_name}.json", "w"), indent=4)
    
    return result_dict