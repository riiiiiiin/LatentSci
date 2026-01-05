import os
import json
from collections import defaultdict
from typing import Any, Dict, List
import logging

logger = logging.getLogger(__name__)

def build_grouped_save_data(
    raw_results_path: str,
    save_results_dir: str,
    log_name: str
) -> Dict[str, Any]:

    with open(raw_results_path, 'r') as f:
        results = json.load(f)['test_results']
    
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in results:
        task_key = r.get("task", "__unknown__")
        grouped[task_key].append(r)

    for task, group in grouped.items():
        task_dir = os.path.join(save_results_dir, task)
        os.makedirs(task_dir, exist_ok=True)
        task_path = os.path.join(task_dir, f'{log_name}.json')
        with open(task_path, 'w') as f:
            json.dump(group, f, ensure_ascii=False, indent=4)