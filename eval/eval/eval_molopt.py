import json
from tqdm import tqdm
from eval.eval_metric import mol_opt_evaluater

def tranform_str_to_json(str_input):
    ## 假如LLM输出的是类似json的字符串, 我需要设定一个逻辑, 把字符串重新转换成json
    ## o1-mini的感觉, 是要移除字符串里面的\n，并且把所有的\"都改成 "
    if "</think>\n\n" in str_input:
        str_input = str_input.split("</think>\n\n")[-1]
        
    if "```json\n" in str_input:
        str_input = str_input.split("```json\n")[1]
        str_input = str_input.replace("\n```", '')
    
    unescaped_str = str_input.replace('\n    ', '').replace('\n', '').replace('\"', '"')
    try:
        json_obj = json.loads(unescaped_str)
        return json_obj
    except json.JSONDecodeError as e:
        return None

def eval_molopt_from_list(optimized_prop, gt_list, pred_list, total_number):
    # this_function input: 
    #   gt_list for gt_molecules
    #   pred_list for pred_molecules
    #   total_number: len(gt_molecules)+len(cases that cannot extract SMILES)
    
    prop_dict = dict(logp='logp', solubility='solubility', qed="qed",  drd='drd2', jnk='jnk3', gsk='gsk3b')
    prop_evaluater = mol_opt_evaluater(prop=prop_dict[optimized_prop])
    
    improve_statistic = prop_evaluater.property_improvement(src_mol_list=gt_list, tgt_mol_list=pred_list, total_num=total_number)
    scaffold_hard, scaffold_soft = prop_evaluater.scaffold_consistency(src_mol_list=gt_list, tgt_mol_list=pred_list)
    
    result_dict = dict(
        improvement=improve_statistic,
        scaffold=dict(hard=scaffold_hard/total_number, soft=scaffold_soft/total_number),
    )
    
    return result_dict
    