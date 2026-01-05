import sys, re, os, json
from .rxnutils import read_json, is_valid_smiles
from eval.utils import extract_answer
import logging
import os

from .evaluator import MoleculeSMILESEvaluator
evaluator = MoleculeSMILESEvaluator()
logger = logging.getLogger(__name__)

subtask_to_result_key = {
    "rcr": "SMILES",
    "nepp": "pred_smi",
    "mechsel": "choice",
    "major_product": "Major Product",
    "byproduct": "Byproduct(s)",
    "retro": "Reactants"
}

def _combine_list(raw):
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                raw = '.'.join(parsed)
            # 如果解析成功但不是 list，什么都不做
        except (json.JSONDecodeError, TypeError):
            # 不能解析为 JSON，什么都不做
            pass
    elif isinstance(raw, list):
        raw = '.'.join(raw)
    return raw

def evaluate_mol(model_name: str, subtask: str, log_dir: str = None):
    if log_dir is None:
        log_dir = f"logs/{subtask}"
    
    if not os.path.exists(log_dir):
        raise ValueError(f"logs_dir {log_dir} is not correct")
    samples = read_json(f"{log_dir}/{model_name}.json")
    preds = []
    gts = []
    for sample in samples:
        gt = sample['gt']
        if subtask in ['major_product', 'byproduct']:
            gt = json.loads(gt)
            gts.append(gt.get(subtask_to_result_key[subtask], ''))
        elif subtask == 'retro':
            if len(gt) == 0:
                continue
            gt = _combine_list(gt)
            gts.append(gt)
        else:
            gts.append(gt)

        try:
            pred_smiles = extract_answer(sample['result'])
            preds.append(pred_smiles)
        except Exception as e:
            logger.debug(f'error parsing {sample['result']}: {e}')
            preds.append('')
        
    res = evaluator.evaluate(preds, gts)
    fts = (res['rdk_sims'] + res['maccs_sims'] + res['morgan_sims']) / 3
    res['fts'] = fts
        
    return res

def evaluate_MechSel(model_name: str, logs_dir: str = 'logs/mechsel'):
    """
    Evaluate the reaction mechanism selection prediction.

    Args:
        logs_dir (str): The directory where the logs are stored.
        model_name (str): The name of the model.

    Returns:
        None
    """
    if not os.path.exists(logs_dir):
        raise ValueError(f"logs_dir {logs_dir} is not correct")
    samples = read_json(f"{logs_dir}/{model_name}.json")

    preds = []
    gts = []
    for sample in samples:
        pred_choice = extract_answer(sample['result'])
        preds.append(pred_choice)
        # if pred_choice is not a valid choice, we treat it as empty
        if not pred_choice.lower().isalpha():
            pred_choice = ""

        pred_choice = pred_choice.lower()
        gt = sample['gt'].lower()
        preds.append(pred_choice)
        gts.append(gt)

    accuracy = sum(1 for pred, gt in zip(preds, gts) if pred == gt) / len(gts)
    return {"MCQ Accuracy (mean)": accuracy}

def evaluate_rxn_score(model_name: str, logs_dir: str = 'logs'):
    all_results = {}
    subtasks = subtask_to_result_key.keys()
    for subtask in subtasks:
        logger.info(f'evaluating {subtask} for model {model_name}')
        try:
            if subtask == 'MechSel' or subtask == 'mechsel':
                all_results[subtask] = evaluate_MechSel(model_name)
            elif subtask in ['major_product', 'byproduct']:
                all_results[subtask] = evaluate_mol(model_name, subtask, f"{logs_dir}/fs")
            else:
                all_results[subtask] = evaluate_mol(model_name, subtask)
        except Exception as e:
            logger.error(f"Error evaluating {subtask} for {model_name}: {e}")
            all_results[subtask] = None
    logger.info(f"eval_score_{model_name}_rxn:\n\r{all_results}")
    os.makedirs("results/rxn", exist_ok=True)
    json.dump(all_results, open(f"results/rxn/eval_score_{model_name}.json", "w"), indent=4)

    return all_results
