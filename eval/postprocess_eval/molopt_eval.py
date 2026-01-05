import json
from eval.eval_molopt import eval_molopt_from_list
from eval.utils import extract_answer
import logging
import os

logger = logging.getLogger(__name__)

def evaluate_molopt_score(model_name, gt_path):
    ## 在get_molopt_cot中得到test结果, 我们评测这些test结果
    prop_dict = dict(logp='logp', solubility='solubility', qed="qed",  drd='drd2', jnk='jnk3', gsk='gsk3b')
    # prop_dict = dict(logp='logp', solubility='solubility', qed="qed",  drd='drd2', gsk='gsk3b')
    
    result_final = dict()
    
    for prop in prop_dict.keys():
        logger.info(f'evaluating {prop} for model {model_name}')
        file_name = f"logs/{prop}/{model_name}.json"
        pred_results = json.load(open(file_name, "r"))
        
                
        gt_name = f"{gt_path}/{prop}.json"
        gts = json.load(open(gt_name, "r"))
        
        tgt_smiles_list, src_smiles_list = list(), list()
        
        invalid_number = 0
        
        for i, pred in enumerate(pred_results):
            answer = extract_answer(pred['result'])
            if answer is None:
                invalid_number += 1
                continue
            tgt_smiles_list.append(answer)
            gt = gts[i]
            meta = json.loads(gt['meta'])
            src_smiles_list.append(meta['molecule'])
        
        logger.debug(len(pred_results), invalid_number, len(src_smiles_list))
        assert len(src_smiles_list) == len(tgt_smiles_list)
        assert len(pred_results) == invalid_number + len(src_smiles_list)
        
        result_dict = eval_molopt_from_list(optimized_prop=prop, gt_list=src_smiles_list, pred_list=tgt_smiles_list, total_number=len(pred_results))
        result_final[prop] = result_dict
    
    logger.info(f"eval_score_{model_name}_molopt:\n\r{result_final}")
    os.makedirs("results/molopt", exist_ok=True)
    json.dump(result_final, open(f"results/molopt/eval_score_{model_name}.json", "w"), indent=4)
    
    return result_final