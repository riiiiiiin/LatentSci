import json
import os
import logging
from tqdm import tqdm
from ChemCoTBench.core.eval_metric import mol_opt_evaluater
from core.utils import extract_answer
from core.task_evaluator import BaseTaskEvaluator
import logging

logger = logging.getLogger(__name__)

prop_dict = {
    'logp': 'logp',
    'solubility': 'solubility',
    'qed': 'qed',
    'drd': 'drd2',
    'gsk': 'gsk3b',
    'jnk': 'jnk3'
}
class MolOptEvaluator(BaseTaskEvaluator):
    def extract_gt(self, gt_raw_item, task):
        meta = json.loads(gt_raw_item['meta'])
        return meta['molecule']
    
    def prepare_metadata(self, sample):
        return None
    
    def evaluate_predictions(self, preds, gts, total_len, metadata = None, task_name = None):
        prop_evaluater = mol_opt_evaluater(prop=prop_dict[task_name])
        
        improve_statistic = prop_evaluater.property_improvement(src_mol_list=gts, tgt_mol_list=preds, total_num=total_len)
        
        scaffold_hard, scaffold_soft = prop_evaluater.scaffold_consistency(src_mol_list=gts, tgt_mol_list=preds)
    
        return improve_statistic
    
def evaluate_molopt_score(model_name, gt_path, logs_dir, results_dir, sample_count):
    result_dict = dict()
    evaluator = MolOptEvaluator()
    
    for task in prop_dict.keys():
        logger.info(f'evaluating {task} for model {model_name}')
        
        result_dict[task] = evaluator.evaluate_score(model_name, sample_count, gt_path, logs_dir, task)
    
    
    logger.info(f"eval_score_{model_name}_molopt:\n\r{result_dict}")
    os.makedirs(f"{results_dir}/molopt", exist_ok=True)
    json.dump(result_dict, open(f"{results_dir}/molopt/eval_score_{model_name}.json", "w"), indent=4)
    
    return result_dict

def record_molopt_results(model_name, gt_path, logs_dir, results_dir, sample_count = 1):
    evaluator = MolOptEvaluator()
    for task in prop_dict.keys():
        logger.info(f'recording {task} for model {model_name}')

        dataframe = evaluator.record_results(model_name, sample_count, gt_path, logs_dir, task)

        os.makedirs(f"{results_dir}/molopt/{task}", exist_ok=True)
        dataframe.to_csv(f"{results_dir}/molopt/{task}/eval_results_{model_name}.csv", index=False)