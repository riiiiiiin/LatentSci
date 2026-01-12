from sklearn.model_selection import train_test_split
import glob
import os
import json
from pathlib import Path

path = 'data/chemcotbench-cot'
test_path = 'data/chemcotbench-cot-test'
all_json_files = glob.glob(os.path.join(path, "**/*.json"), recursive=True)

for file in all_json_files:
    with open(file, 'r') as f:
        data = json.load(f)
    train_data, test_data = train_test_split(data, test_size=0.05, random_state=42)

    for sample in test_data:
        try:
            cot_string = sample.get("struct_cot") or sample.get('cot_result')
            if cot_string.startswith("```json"):
                cot_string = cot_string[7:-3].strip()
            cot_content = json.loads(cot_string)
        except json.JSONDecodeError as e:
            print(f"\n[CRITICAL DATA ERROR] JSON is malformed in example ID: {sample.get('id')}")
            print(f"[ERROR DETAILS]: {e}")
            print(f"[RAW CONTENT]: {cot_string}")
            print(f'[FILE]: {file}')
            print(f'[SAMPLE]: {sample}')
            raise
        except:
            print(file)
            print(sample)
            raise

        if isinstance(cot_content, str):
            cleaned = cot_content.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:-3].strip()
            try:
                cot_dict = json.loads(cleaned)
            except json.JSONDecodeError as e:
                print(f"\n[CRITICAL DATA ERROR] Secondary JSON parsing failed for example ID: {sample.get('id')}")
                print(f"[ERROR DETAILS]: {e}")
                print(f"[CLEANED CONTENT]: {repr(cleaned)}")
                raise
        else:
            cot_dict = cot_content

        cot_steps = []
        for i, (k, v) in enumerate(cot_dict.items()):
            if k == "output":
                continue
            cot_steps.append(f"Step {i+1}:\n{k}: {v}")

        cot_value = "\n\n".join(cot_steps)
        sample['cot_reference'] = cot_value
    
    P = Path(file)
    parent_name = P.parent.name
    os.makedirs(os.path.join(test_path, parent_name), exist_ok=True)
    with open(os.path.join(test_path, parent_name, P.name), 'w') as f:
        json.dump(test_data, f, indent=4)