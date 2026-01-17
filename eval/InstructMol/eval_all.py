from core.task_evaluator import TextSimiliarityTaskEvaluator, MolSimiliarityTaskEvaluator
import logging
import os
import json

logger = logging.getLogger(__name__)
text_evaluator = TextSimiliarityTaskEvaluator()
mol_evaluator = MolSimiliarityTaskEvaluator()

def evaluate_captioning_score(model_name, gt_path, logs_dir, results_dir, task_name, sample_count = 1):
    results = text_evaluator.evaluate_score(model_name, sample_count, gt_path, logs_dir, task_name)
    
    logger.info(f"eval_score_{model_name}_{task_name}:\n\r{results}")
    os.makedirs(f"{results_dir}/InstructMol/{task_name}", exist_ok=True)
    json.dump(results, open(f"{results_dir}/InstructMol/{task_name}/eval_score_{model_name}.json", "w"), indent=4)
    
    return results

def evaluate_mol_similarity_score(model_name, gt_path, logs_dir, results_dir, task_name, sample_count = 1):
    results = mol_evaluator.evaluate_score(model_name, sample_count, gt_path, logs_dir, task_name)

    logger.info(f"eval_score_{model_name}_{task_name}:\n\r{results}")
    os.makedirs(f"{results_dir}/InstructMol/{task_name}", exist_ok=True)
    json.dump(results, open(f"{results_dir}/InstructMol/{task_name}/eval_score_{model_name}.json", "w"), indent=4)
    
    return results

def eval_all_InstructMol(log_name, dataset_path, logs_dir, results_dir, num_samples = 1):
    tasks = ['molecular description generation', 'forward reaction prediction', 'description-guided molecule design', 'reagent prediction', 'retrosynthesis']
    
    for task in tasks:
        if task == 'molecular description generation':
            evaluate_captioning_score(log_name, f"{dataset_path}/{task}", logs_dir, results_dir, task, num_samples)
        else:
            evaluate_mol_similarity_score(log_name, f"{dataset_path}/{task}", logs_dir, results_dir, task, num_samples)