from postprocess_eval.moledit_eval import evaluate_moledit_score
from postprocess_eval.molopt_eval import evaluate_molopt_score
from postprocess_eval.molund_eval import evaluate_molund_score
from postprocess_eval.rxn_eval import evaluate_rxn_score
from group_results import build_grouped_save_data
from argparse import ArgumentParser
import os
import json

parser = ArgumentParser()
parser.add_argument('--result_path', type=str, required=True)
parser.add_argument('--log_name', type=str, required=True)
args = parser.parse_args()

result_path = args.result_path
log_name = args.log_name

os.makedirs("./logs", exist_ok=True)
build_grouped_save_data(result_path, "./logs", log_name)

moledit_results = evaluate_moledit_score(log_name)
molopt_results = evaluate_molopt_score(log_name)
molund_results = evaluate_molund_score(log_name)
rxn_results = evaluate_rxn_score(log_name)

all_results = {
    'moledit': moledit_results,
    'molopt': molopt_results,
    'molund': molund_results,
    'rxn': rxn_results
}

os.makedirs('results/all_results', exist_ok=True)
json.dump(all_results, open(f'results/all_results/{log_name}.json', 'w'), indent=4)