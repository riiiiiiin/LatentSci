from ChemLLMBench.eval_molecule_captioning import evaluate_molecule_captioning_score, record_molecule_captioning_score
from ChemLLMBench.eval_reaction_prediction import evaluate_reaction_prediction_score, record_reaction_prediction_results
from ChemLLMBench.eval_reagent_selection import evaluate_reagent_selection_score, record_reagent_selection_results
from ChemLLMBench.eval_retro import evaluate_retro_score, record_retro_results

def eval_all_ChemLLMBench(log_name, dataset_path, logs_dir, results_dir, num_samples):
    evaluate_molecule_captioning_score(log_name, f"{dataset_path}/molecule_captioning", logs_dir, results_dir, num_samples)
    evaluate_reaction_prediction_score(log_name, f"{dataset_path}/reaction_prediction", logs_dir, results_dir, num_samples)
    evaluate_reagent_selection_score(log_name, f"{dataset_path}/reagent_selection", logs_dir, results_dir, num_samples)
    evaluate_retro_score(log_name, f"{dataset_path}/retro", logs_dir, results_dir, num_samples)
    
def record_all_ChemLLMBench(log_name, dataset_path, logs_dir, results_dir, num_samples):
    record_molecule_captioning_score(log_name, f"{dataset_path}/molecule_captioning", logs_dir, results_dir, num_samples)
    record_reaction_prediction_results(log_name, f"{dataset_path}/reaction_prediction", logs_dir, results_dir, num_samples)
    record_reagent_selection_results(log_name, f"{dataset_path}/reagent_selection", logs_dir, results_dir, num_samples)
    record_retro_results(log_name, f"{dataset_path}/retro", logs_dir, results_dir, num_samples)