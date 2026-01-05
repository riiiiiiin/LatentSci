import json
from eval.eval_moledit import eval_moledit_from_list
from eval.utils import extract_answer
import logging
import os

logger = logging.getLogger(__name__)

def evaluate_moledit_score(model_name, gt_path): 
    result_dict = dict()
    
    for task in ['add', 'delete', 'sub']:
        logger.info(f'evaluating {task} for model {model_name}')
        file_name = f"logs/{task}/{model_name}.json" 
        pred_results = json.load(open(file_name, "r"))
        
        gt_name = f"{gt_path}/{task}.json"
        gts = json.load(open(gt_name, "r"))
        
        invalid_number = 0
        pred_list, src_list = list(), list()
        group_a, group_b = list(), list()
        
        for i, pred in enumerate(pred_results):
            answer = extract_answer(pred['result'])
            if answer is None:
                invalid_number += 1
                continue
            pred_list.append(answer)
            gt = gts[i]
            meta = json.loads(gt['meta'])
            src_list.append(meta['molecule'])
            if task == 'add': group_a.append(meta['added_group'])
            elif task == 'delete': group_a.append(meta['removed_group'])
            elif task == 'sub':
                group_a.append(meta['added_group']); group_b.append(meta['removed_group'])
        
        assert len(src_list) == len(pred_list)
        assert len(src_list) == len(group_a)
        
        result_dict[task] = eval_moledit_from_list(src_list=src_list, pred_list=pred_list, group_a=group_a, group_b=group_b, task=task, total_number=len(pred_results)) 
    
    logger.info(f"eval_score_{model_name}_moledit:\n\r{result_dict}")
    os.makedirs("results/moledit", exist_ok=True)
    json.dump(result_dict, open(f"results/moledit/eval_score_{model_name}.json", "w"), indent=4)
    
    return result_dict
