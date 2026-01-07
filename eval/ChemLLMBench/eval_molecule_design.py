import os, json
from core.utils import extract_answer
import logging
import os

from core.evaluator import MoleculeSMILESEvaluator
evaluator = MoleculeSMILESEvaluator()
logger = logging.getLogger(__name__)

def evaluate_molecule_design_score(model_name, gt_path, logs_dir, results_dir):
    log_dir = f"{logs_dir}/molecule_design"
    
    if not os.path.exists(log_dir):
        raise ValueError(f"logs_dir {log_dir} is not correct")
    
    with open(f"{log_dir}/{model_name}.json", 'r') as f:
        samples = json.load(f)
    with open(f'{gt_path}/molecule_design.json') as f:
        gt_raw = json.load(f)
    
    preds = []
    gts = []
    for i, sample in enumerate(samples):
        meta = gt_raw[i]['meta']
        meta = json.loads(meta)
        gts.append(meta['reference'])
        
        pred = extract_answer(sample['result'])
        preds.append(pred)
        
    res = evaluator.evaluate(preds, gts)
    fts = (res['rdk_sims'] + res['maccs_sims'] + res['morgan_sims']) / 3
    res['fts'] = fts
    
    os.makedirs(f"{results_dir}/molecule_design", exist_ok=True)
    json.dump(res, open(f"{results_dir}/molecule_design/eval_score_{model_name}.json", "w"), indent=4)
        
    return res