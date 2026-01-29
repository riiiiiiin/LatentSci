from core.task_evaluator import TextSimiliarityTaskEvaluator, MolSimiliarityTaskEvaluator
import logging
import os
import json
import pandas as pd

logger = logging.getLogger(__name__)
text_evaluator = TextSimiliarityTaskEvaluator()
mol_evaluator = MolSimiliarityTaskEvaluator()

def evaluate_captioning_score(model_name, gt_path, logs_dir, results_dir, task_name, sample_count = 1):
    results = text_evaluator.evaluate_score(model_name, sample_count, gt_path, logs_dir, task_name)
    
    logger.info(f"eval_score_{model_name}_{task_name}:\n\r{results}")
    os.makedirs(f"{results_dir}/InstructMol/{task_name}", exist_ok=True)
    json.dump(results, open(f"{results_dir}/InstructMol/{task_name}/eval_score_{model_name}.json", "w"), indent=4)
    
    return results

def record_captioning_score(model_name, gt_path, logs_dir, results_dir, task_name, sample_count = 1):
    dataframe = text_evaluator.record_results(model_name, sample_count, gt_path, logs_dir, task_name)

    os.makedirs(f"{results_dir}/InstructMol/{task_name}", exist_ok=True)
    dataframe.to_csv(f"{results_dir}/InstructMol/{task_name}/eval_results_{model_name}.csv", index=False)

def evaluate_mol_similarity_score(model_name, gt_path, logs_dir, results_dir, task_name, sample_count = 1):
    results = mol_evaluator.evaluate_score(model_name, sample_count, gt_path, logs_dir, task_name)

    logger.info(f"eval_score_{model_name}_{task_name}:\n\r{results}")
    os.makedirs(f"{results_dir}/InstructMol/{task_name}", exist_ok=True)
    json.dump(results, open(f"{results_dir}/InstructMol/{task_name}/eval_score_{model_name}.json", "w"), indent=4)
    
    return results

def record_mol_similarity_score(model_name, gt_path, logs_dir, results_dir, task_name, sample_count = 1):
    dataframe = mol_evaluator.record_results(model_name, sample_count, gt_path, logs_dir, task_name)
    
    os.makedirs(f"{results_dir}/InstructMol/{task_name}", exist_ok=True)
    dataframe.to_csv(f"{results_dir}/InstructMol/{task_name}/eval_results_{model_name}.csv", index=False)

def eval_all_InstructMol(log_name, dataset_path, logs_dir, results_dir, num_samples = 1):
    tasks = ['molecular description generation', 'forward reaction prediction', 'reagent prediction', 'retrosynthesis']
    
    for task in tasks:
        if task == 'molecular description generation':
            evaluate_captioning_score(log_name, f"{dataset_path}/{task}", logs_dir, results_dir, task, num_samples)
        else:
            evaluate_mol_similarity_score(log_name, f"{dataset_path}/{task}", logs_dir, results_dir, task, num_samples)
            
def record_all_InstructMol(log_name, dataset_path, logs_dir, results_dir, num_samples = 1):
    tasks = ['molecular description generation', 'forward reaction prediction', 'reagent prediction', 'retrosynthesis']

    for task in tasks:
        if task == 'molecular description generation':
            record_captioning_score(log_name, f"{dataset_path}/{task}", logs_dir, results_dir, task, num_samples)
        else:
            record_mol_similarity_score(log_name, f"{dataset_path}/{task}", logs_dir, results_dir, task, num_samples)