# BioLatentCOT

🛠️ Environment Installation
1️⃣ Clone the repository

git clone https://github.com/xinwuye/BioLatentCOT.git

cd BioLatentCOT


2️⃣ Create conda environment (recommended)

```
conda env create -f biolatent_environment2.yml

conda activate biolatenecot_dev

pip install trl==0.26.2 pytorch-fast-transformers==0.4.0 rdkit peft==0.17.1  plotext wandb liger-kernel vllm==0.11.2

# During the two following cmds, there might be some compatibility errors. Ignore them.
pip install PyTDC

pip install transformers==4.57.3 accelerate==1.10.1
```

📊 Data Preparation

1️⃣ Download dataset

The `OpenMol/ChemCoTDataset` is a gated dataset on Hugging Face, which requires you to accept the access conditions before downloading.

Follow these steps:

1. Open the dataset page in your browser:  
   https://huggingface.co/datasets/OpenMol/ChemCoTDataset

2. Make sure you are logged in to your Hugging Face account (top-right corner).

3. You will see a prompt:  
   *"You need to agree to share your contact information to access this dataset."*  
   There will be a checkbox and an **Agree** or **Access repository** button.

4. Check the box and click the button to submit (approval is automatic; no need to wait for manual review).

5. After agreeing, the page will refresh and the full file list will become visible, meaning you now have access.

6. Return to your terminal and run the download command (keep the mirror and acceleration settings for faster download in China):

   ```bash
   export HF_ENDPOINT=https://hf-mirror.com   # Use HF mirror (if not already set) if you need it in China

   huggingface-cli download --resume-download OpenMol/ChemCoTDataset --repo-type dataset --local-dir ./ChemCotDataset
   ```                                    

2️⃣Dataset repair,To fill in missing fields in the dataset, use single quotes (''). And we will not use the data in rxn for the time being.

cd code_train_sft 

python xiufu.py

🧠 Model Download (under dir BioLatentCOT)

Download pretrained model from Hugging Face

huggingface-cli download --resume-download Qwen/Qwen3-8B-Base --local-dir ./models/Qwen3-8B-Base

Download small molecule foundation model from Hugging Face

huggingface-cli download --resume-download ibm-research/materials.smi-ted --local-dir ./models/smi-ted

🏋️ Training 
```
accelerate launch --multi_gpu --num_processes 8 train_stage3.py \
  --training_stage 1 \
  --epochs_per_stage 3 \
  --output_dir ./outputs/stage1-lr2e-4-cf_margin01 \
  --batch_size 4 \
  --grad_accum 1 \
  --lr 2e-4 \
  --cf_lambda 0.2 --cf_margin 0.1 \
  --cf_prob 1.0

accelerate launch --multi_gpu --num_processes 8 train_stage3.py \
  --training_stage 2 \
  --epochs_per_stage 3 \
  --lora_path ./outputs/stage1-lr2e-4-cf_margin01/stage1/lora_weights \
  --projector_path ./outputs/stage1-lr2e-4-cf_margin01/stage1/mm_projector.pt \
  --output_dir ./outputs/stage2-lr2e-4-cf_margin01 \
  --batch_size 4 \
  --grad_accum 1 \
  --lr 2e-4 \
  --cf_lambda 0.2 --cf_margin 0.1 \
  --cf_prob 1.0

accelerate launch --multi_gpu --num_processes 8 train_stage3.py \
  --training_stage 3 \
  --is_coconut false \
  --is_both_latent true \
  --epochs_per_stage 3 \
  --lora_path ./outputs/stage2-lr2e-4-cf_margin01/stage2/lora_weights \
  --projector_path ./outputs/stage2-lr2e-4-cf_margin01/stage2/mm_projector.pt \
  --output_dir ./outputs/stage3-lr2e-4-cf_margin01 \
  --batch_size 4 \
  --grad_accum 1 \
  --lr 2e-4 \
  --cf_lambda 0.2 --cf_margin 0.1 \
  --cf_prob 1.0 
  
accelerate launch --multi_gpu --num_processes 8 train_grpo_try2.py \
  --run_name stage4 \
  --lora_path ./outputs/stage3-lr2e-4-cf_margin01-freeze_llm-freeze_projector/stage3/lora_weights \
  --projector_path ./outputs/stage3-lr2e-4-cf_margin01-freeze_llm-freeze_projector/stage3/mm_projector.pt \
  --output_dir ./outputs/stage4-lr2e-4-cf_margin01-freeze_llm-freeze_projector-124-lr1e-5-temp15 \
  --is_both_latent true \
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
  --gradient_checkpointing
  ```




## License
This code is released under the MIT license (see [LICENSE](LICENSE)).