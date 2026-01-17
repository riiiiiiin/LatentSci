from core.task_evaluator import AllCoTTextSimiliarityTaskEvaluator

import os
import glob
from pathlib import Path
import json

evaluator = AllCoTTextSimiliarityTaskEvaluator()

def eval_all_ChemCoTDataset_textwise(log_name, dataset_path, logs_dir, results_dir):
    all_task_files = glob.glob(os.path.join(dataset_path, "**/*.json"), recursive=True)
    all_results = {}
    for task_file in all_task_files:
        P = Path(task_file)
        task_name = P.stem
        
        gt_dir = os.path.join(dataset_path, P.parent.name)
        all_results[task_name] = evaluator.evaluate_score(log_name, 1, gt_dir, logs_dir, task_name)

    os.makedirs(os.path.join(results_dir, "ChemCoTDataset-textwise"), exist_ok=True)
    with open(os.path.join(results_dir, "ChemCoTDataset-textwise", f"{log_name}.json"), "w") as f:
        json.dump(all_results, f, indent=4)