import sys
import os
import glob
import json
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

# 将 code_train_sft 路径加入 sys.path 以便导入 dataloader
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
train_code_path = os.path.join(project_root, "code_train_sft")
sys.path.append(train_code_path)

# 现在可以从 dataloader 导入 extract_fields
from dataloader import extract_fields

def analyze_dataset_stats(data_path):
    print(f"Scanning data from: {data_path}")
    
    # 扫描所有 JSON 文件并排除 rxn/rcr.json
    all_json_files = glob.glob(os.path.join(data_path, "**/*.json"), recursive=True)
    data_files = [f for f in all_json_files if not f.endswith("rcr.json")]
    
    print(f"Found {len(data_files)} JSON files.")
    
    # 使用 datasets 加载（不进行 tokenization，只做字段提取）
    ds = load_dataset("json", data_files=data_files)["train"]
    
    step_counts = []
    char_lengths = []
    
    max_steps = 0
    max_example_id = None

    print("Analyzing examples...")
    for example in tqdm(ds):
        try:
            processed = extract_fields(example)
            
            # 1. 步骤数分析
            steps = processed.get("cot_steps", [])
            num_steps = len(steps)
            step_counts.append(num_steps)
            
            if num_steps > max_steps:
                max_steps = num_steps
                max_example_id = example.get("id")
                
            # 2. CoT 文本长度分析 (字符数)
            cot_text = processed.get("cot", "")
            char_lengths.append(len(cot_text))
            
        except Exception:
            continue

    if not step_counts:
        print("No valid data found.")
        return

    def print_dist(name, data):
        data = np.array(data)
        print(f"\n--- {name} Distribution ---")
        print(f"Min:    {np.min(data)}")
        print(f"Max:    {np.max(data)}")
        print(f"Mean:   {np.mean(data):.2f}")
        print(f"Std:    {np.std(data):.2f}")
        print(f"Quantiles:")
        for q in [25, 50, 75, 90, 95, 99]:
            print(f"  {q}%: {np.percentile(data, q)}")

    print("\n" + "="*40)
    print("📊 DATASET ANALYSIS REPORT")
    print("="*40)
    print(f"Total valid examples: {len(step_counts)}")
    
    print_dist("CoT Step Count", step_counts)
    print_dist("CoT Char Length", char_lengths)
    
    print("\n" + "="*40)
    print(f"Maximum steps example ID: {max_example_id}")
    print("="*40)
    
    # 给出一个建议的 max_latent_stage
    print(f"\n💡 Suggested max_latent_stage for Coconut: {max_steps}")
    print("Note: You might want to set it slightly lower if the max is an outlier.")

if __name__ == "__main__":
    DATA_PATH = "/mnt/afs/L202500070/Bio-LatentCOT/ChemCotDataset/chemcotbench-cot"
    analyze_dataset_stats(DATA_PATH)

