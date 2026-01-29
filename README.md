# LatentChem

🛠️ Environment Configuration

1️⃣ Clone the repository

git clone https://github.com/xinwuye/LatentChem.git

cd LatentChem


2️⃣ Create conda environment

```
conda env create -f env.yml

conda activate latentchem_dev

pip install trl==0.26.2 pytorch-fast-transformers==0.4.0 rdkit peft==0.17.1  plotext wandb liger-kernel vllm==0.11.2

# The following installation is only needed for small moleculae GRPO training. During the two following cmds, there might be some compatibility errors. Ignore them.

pip install PyTDC

pip install transformers==4.57.3 accelerate==1.10.1
```

## Training

📊 Data Preparation

1️⃣ Download dataset

···
huggingface-cli download --repo-type dataset anonymousssss22321/latentchem ChemCotDataset.tar.gz --local-dir .
···

Run the following commands to extract the files into the required directory structure:

···
mkdir -p ChemCotDataset# Extract the content# This results in the structure: ChemCotDataset/chemcotbench-cot/

tar -xzvf ChemCotDataset.tar.gz -C ChemCotDataset
···

2️⃣Data clearning

```
cd code_train_sft 

python xiufu.py
```

🧠 Base Model Download (under dir LatentChem)

Download pretrained model from Hugging Face

```
huggingface-cli download --resume-download Qwen/Qwen3-8B-Base --local-dir ./models/Qwen3-8B-Base
```

Download SMI-TED from Hugging Face

```
huggingface-cli download --resume-download ibm-research/materials.smi-ted --local-dir ./models/smi-ted
```

🏋️ Training 
```
accelerate launch --multi_gpu --num_processes 8 train_stage3.py \
  --training_stage 1 \
  --epochs_per_stage 3 \
  --output_dir ./outputs/stage1 \
  --batch_size 4 \
  --grad_accum 1 \
  --lr 2e-4 \
  --cf_lambda 0.2 --cf_margin 0.1 \
  --cf_prob 1.0

accelerate launch --multi_gpu --num_processes 8 train_stage3.py \
  --training_stage 2 \
  --epochs_per_stage 3 \
  --lora_path ./outputs/stage1/stage1/lora_weights \
  --projector_path ./outputs/stage1/stage1/mm_projector.pt \
  --output_dir ./outputs/stage2 \
  --batch_size 4 \
  --grad_accum 1 \
  --lr 2e-4 \
  --cf_lambda 0.2 --cf_margin 0.1 \
  --cf_prob 1.0

accelerate launch --multi_gpu --num_processes 8 train_stage3.py \
  --training_stage 3 \
  --is_coconut false \
  --is_both_latent false \
  --is_taskthinker true \
  --is_bioupdater true \
  --epochs_per_stage 3 \
  --lora_path ./outputs/stage2/stage2/lora_weights \
  --projector_path ./outputs/stage2/stage2/mm_projector.pt \
  --output_dir ./outputs/stage3 \
  --batch_size 4 \
  --grad_accum 1 \
  --lr 2e-4 \
  --cf_lambda 0.2 --cf_margin 0.1 \
  --cf_prob 1.0 \
  --freeze_llm true --freeze_projector true
  
accelerate launch --multi_gpu --num_processes 8 train_grpo_try2.py \
  --run_name stage4 \
  --lora_path ./outputs/stage3/stage3/lora_weights \
  --projector_path ./outputs/stage3/stage3/mm_projector.pt \
  --output_dir ./outputs/stage4 \
  --is_both_latent false \
  --is_taskthinker true \
  --is_bioupdater true \
  --use_reward_answer_tag true \
  --use_reward_answer_type_validity true \
  --use_reward_answer_correctness_bench true \
  --batch_size 16 \
  --grad_accum 1 \
  --lr 1e-5 \
  --epochs 1 \
  --max_prompt_length 2048 \
  --max_completion_length 2048 \
  --num_generations 8 \
  --num_iterations 1 \
  --beta 0.0 \
  --use_liger \
  --use_vllm \
  --vllm_mode colocate \
  --vllm_gpu_memory_utilization 0.3 \
  --vllm_max_model_len 4096 \
  --temperature 1.5 \
  --gradient_checkpointing \
  --freeze_bio_updater true --freeze_task_thinker true
  ```

## Testing


## License
This code is released under the MIT license (see [LICENSE](LICENSE)).


