from ChEBI20.eval_captioning import evaluate_captioning_score

def eval_all_ChEBI20(log_name, dataset_path, logs_dir, results_dir, num_samples = 1):
    evaluate_captioning_score(log_name, f'{dataset_path}/molecule_description_generation', logs_dir, results_dir, num_samples)