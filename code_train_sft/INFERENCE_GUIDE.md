# Inference Guide for train_sft_stage2.py

## Summary of Changes

The toy test code has been removed and replaced with a robust inference system that can:
- Load test data from JSON files (same format as training data)
- Run inference on all test samples
- Save detailed results with ground truth labels for evaluation

## Key Changes

### 1. Removed Functions
- ❌ `test_lora_inference()` - Removed toy examples with hardcoded SMILES
- ❌ `batch_test_lora_inference()` - Removed hardcoded test cases

### 2. New Functions

#### `load_test_data(test_data_path)`
- Loads test data from JSON files (file or directory)
- Uses the same data format as training data
- Supports the `extract_fields()` function from `dataloader.py`
- Returns a list of test cases with `smiles`, `query`, `label`, and `cot`

#### `run_inference_on_test_data(...)`
- Runs inference on real test data
- Supports batch processing (currently batch_size=1)
- Saves detailed results including:
  - Generated responses
  - Ground truth labels (if available)
  - Ground truth CoT (if available)
  - Error information (if any)
- Outputs results in JSON format with metadata

### 3. New Command-Line Arguments

```bash
--mode inference                    # Set to inference mode
--test_data_path PATH              # Path to test data (file or directory)
--max_new_tokens N                 # Maximum tokens to generate (default: 2048)
--temperature T                    # Sampling temperature (default: 0.7)
--top_p P                         # Top-p sampling (default: 0.9)
--max_test_samples N              # Limit number of test samples (default: None = all)
--inference_results_path PATH     # Where to save results (default: auto-generated)
```

## Usage Examples

### Basic Inference

Run inference on test data with default settings:

```bash
python train_sft_stage2.py \
  --mode inference \
  --output_dir ./trained_model \
  --test_data_path /path/to/test/data \
  --lora_path ./trained_model/lora_weights \
  --projector_path ./trained_model/projector.pt
```

### Advanced Inference with Custom Parameters

```bash
python train_sft_stage2.py \
  --mode inference \
  --output_dir ./trained_model \
  --test_data_path /path/to/test/data \
  --lora_path ./trained_model/lora_weights \
  --projector_path ./trained_model/projector.pt \
  --max_new_tokens 4096 \
  --temperature 0.5 \
  --top_p 0.95 \
  --max_test_samples 100 \
  --inference_results_path ./results/my_inference_results.json
```

### Test on Specific JSON File

```bash
python train_sft_stage2.py \
  --mode inference \
  --output_dir ./trained_model \
  --test_data_path /path/to/single_test_file.json \
  --lora_path ./trained_model/lora_weights \
  --projector_path ./trained_model/projector.pt
```

### Test on Directory of JSON Files

```bash
python train_sft_stage2.py \
  --mode inference \
  --output_dir ./trained_model \
  --test_data_path /path/to/test/directory/ \
  --lora_path ./trained_model/lora_weights \
  --projector_path ./trained_model/projector.pt
```

## Output Format

The inference results are saved in JSON format with the following structure:

```json
{
  "timestamp": "2026-01-01T12:00:00.000000",
  "test_data_path": "/path/to/test/data",
  "model_info": {
    "device": "cuda",
    "total_parameters": 123456789,
    "trainable_parameters": 12345678
  },
  "generation_config": {
    "max_new_tokens": 2048,
    "temperature": 0.7,
    "top_p": 0.9
  },
  "num_samples": 100,
  "test_results": [
    {
      "sample_id": 0,
      "smiles": ["CC1[NH2+]CCC1C(=O)Nc1cc(C(N)=O)ccc1Cl"],
      "query": "Modify the molecule...",
      "generated_response": "<answer> ... </answer>",
      "ground_truth_label": "<answer> ... </answer>",
      "ground_truth_cot": "Step 1:\n..."
    },
    ...
  ]
}
```

## Test Data Format

The test data should be in the same JSON format as the training data (ChemCot format):
- Each JSON file contains a list of examples
- Each example should have `query`, `meta`, and `struct_cot` fields
- The `meta` field should contain `molecule` or `reactants`, `gt`, and/or `reference`

## Notes

1. **Memory Management**: The inference function automatically clears GPU cache after each sample
2. **Error Handling**: If a sample fails, the error is logged and saved in the results
3. **Progress Tracking**: Progress is logged for each sample during inference
4. **Automatic Results Path**: If no results path is specified, a timestamped file is created automatically
5. **Ground Truth Comparison**: Results include ground truth labels for easy evaluation

## Next Steps for Evaluation

After running inference, you can:
1. Load the results JSON file
2. Compare `generated_response` with `ground_truth_label`
3. Calculate metrics like:
   - Exact match accuracy
   - BLEU/ROUGE scores
   - Chemical validity (for SMILES outputs)
   - Task-specific metrics

Example evaluation script structure:

```python
import json

# Load results
with open('inference_results.json', 'r') as f:
    data = json.load(f)

# Evaluate
for result in data['test_results']:
    generated = result['generated_response']
    ground_truth = result['ground_truth_label']
    # Compare and compute metrics...
```




