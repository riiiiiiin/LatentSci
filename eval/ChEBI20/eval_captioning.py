from core.task_evaluator import TextSimiliarityTaskEvaluator
import logging
import os
import json

logger = logging.getLogger(__name__)

def evaluate_captioning_score(model_name, gt_path, logs_dir, results_dir, sample_count = 1):
    evaluator = TextSimiliarityTaskEvaluator()
    results = evaluator.evaluate_score(model_name, sample_count, gt_path, logs_dir, 'molecule_description_generation')
    
    logger.info(f"eval_score_{model_name}_molecule_description_generation:\n\r{results}")
    os.makedirs(f"{results_dir}/ChEBI20/molecule_description_generation", exist_ok=True)
    json.dump(results, open(f"{results_dir}/ChEBI20/molecule_description_generation/eval_score_{model_name}.json", "w"), indent=4)
    
    return results