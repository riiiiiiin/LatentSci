# BioLatentCOT

🛠️ Environment Installation
1️⃣ Clone the repository

git clone https://github.com/xinwuye/BioLatentCOT.git

cd BioLatentCOT


2️⃣ Create conda environment (recommended)

conda env create -f biolatent_environment1.yml

conda activate biolatenecot_dev

pip install trl==0.15.2 pytorch-fast-transformers==0.4.0 rdkit==2024.3.6024.3.6 peft==0.17.1


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

🧠 Model Download

Download pretrained model from Hugging Face

huggingface-cli download --resume-download Qwen/Qwen3-8B-Base --local-dir ./models/Qwen3-8B-Base

Download small molecule foundation model from Hugging Face

huggingface-cli download --resume-download ibm-research/materials.smi-ted --local-dir ./models/smi-ted

🏋️ Training sft

注意要将model_new.py中QueryAttentionProjector输出的维度，与train中collate_fn维度对齐，collate_fn为（QueryAttentionProjector输出的维度+2）*1

python train_sft_stage2.py \
  --mode train \
  --data_path chemcotbench-cot \
  --model_path ./qwen3_mol_sft_lora_results \
  --batch_size 2 \
  --max_seq_length 512 \
  --epochs 3

python train_sft_stage2.py   --mode train   --data_path ../ChemCotDataset/chemcotbench-cot   --model_path ./qwen3_mol_sft_lora_results   --batch_size 2   --max_seq_length 512   --epochs 3


🏋️ Training sft_cot

python train_sft_stage2.py \
  --mode cotinue \
  --data_path chemcotbench-cot \
  --model_path ./qwen3_mol_sft_lora_results \
  --batch_size 2 \
  --max_seq_length 1024 \
  --epochs 3





🔍 Inference
python train_sft_stage2.py \
  --mode inference \
  --model_path ./qwen3_mol_sft_lora_results


Output example：

INFO:__main__:
==================================================
INFO:__main__:Input SMILES: ['CC1[NH2+]CCC1C(=O)Nc1cc(C(N)=O)ccc1Cl']
INFO:__main__:Input prompt: Modify the molecule CC1[NH2+]CCC1C(=O)Nc1cc(C(N)=O)ccc1Cl by adding a carboxyl.
INFO:__main__:Model device: cuda:0
INFO:__main__:Input IDs device: cuda:0
INFO:__main__:Attention mask device: cuda:0
tensor([[ 2691,   279,  1803,  ..., 19238,     8,  3706]], device='cuda:0')
generate_text:  Add the carbox to the end of the molecule. CC1[NH2CCC1[NH] to the C(=O] to the molecule. CC1[NH2CCC1CC(C(=OCC1[NH2CCC1[NH]CC1]C(=O]C(=OCC[NH]CO1[NH2C(=CCC1[NH2CCC2CC[NH]C(C(=O]C(=O]C(=O]C(=OCC1)C(O)C(=CC(O)C(O)C(C(=O]C(O)C(=O)C(=CC(C(=O)CCC(C(=CC1)C(=O)C(=O)N1)C(=O)C(=O)CCC1)C(=O)C1)C(C)CCC1C(=O]C1C(=CCC(C(=O)N(C(=O)C1)N(C(=O)C(O)C(=O)C(=CC(C)C(=OCCO)C(C(=CC2C(O)C2)CC[NH2O)N(C(=O2)C(C(=CC1)C1)C(C(=OCCO)CC2[NH2)C(=CC1)C(C(=O)C(=CC1C2)N(C)CC2)C(O)C(O)N(C)C(O)C(O)N2C(O)C2CC(C)CCO]CC1[NH2)C2C2C(O)C(O)C2C2CO(C)C2C(O)C1CC1C(O)C(C)N2C2C2CC2C1)C(=CC1CCO2C2CC2C2CO2CCO2CC(O)C(=CC1)C(O)C2)C2OCCO(O)N2C1CCO2CC2(C)C(O)C(O)CC2C(O)C1C(O)CC(O)CO2CC(C)C(C)C(C)C(COCCO2COCC2CO2CCC2C(O)C2C(=CC2C(O)C(C)C2CC1C1C12C2C2)CC1C(O)C2C(O)C(O)C2C2CCO2C2)C(O)C(O)C(C)CO2CC2C1CC(O)C2C1(O)C(C)CO(C)C(O)C(C(=O2C(O)C2COO2CO2C(O)C(O)C2C(C)C2C1CC1C2C(O)C(O)C(O)CC2CC2C1)C(O)C(O)C(O)C(O)C(O)CC2C(O)C(O)C(C)C2CO(CO2CO(C)C(O)C(C)CCO1C1CC2CCO1C(O)C2CO2CCC2C1C(O)CO(O)C(O)C(O)CO2CCOCC1C1C2C(O)C(O)C1C(O)CC2CO(O)C(O)C(C)C(C)C(O)C(O)C(O)C1CCOCCOCC1C(O)C(O)C2CCC(O)CC(O)C(O)C2C(O)C(O)CCO1OCCO2C(O)CO(O)C(O)C(O)CC(O)C(O)C(O)CC(O)C(O)CC1CO2CO(O)C(O)C(O)C2CCC1CC2C(O)C(O)C2C(O)C1C(O)C(C)C(O)CO(O)C(O)C(O)C2CCC2C(O)CC2CO2O2CC(O)C1CC2C(O)C(O)C(O)C(O)C(O)C(O)C2C(O)C(O)CC(O)C1CC(O)C(O)C2C(O)C(O)C(O)C(O)C1CCO(C)C(O)C(O)C(O)C1C2C(O)C(O)C(O)C(O)C2(O)C(O)C2C(O)CC(O)C(O)CCO2C(O)C(O)C(O)C(O)C(O)C(O)CO(C)C(O)C(O)C(O)C(O)C(O)CC
INFO:__main__:Generated response: O)C(C(=O]C(O)C(=O)C(=CC(C(=O)CCC(C(=CC1)C(=O)C(=O)N1)C(=O)C(=O)CCC1)C(=O)C1)C(C)CCC1C(=O]C1C(=CCC(C(=O)N(C(=O)C1)N(C(=O)C(O)C(=O)C(=CC(C)C(=OCCO)C(C(=CC2C(O)C2)CC[NH2O)N(C(=O2)C(C(=CC1)C1)C(C(=OCCO)CC2[NH2)C(=CC1)C(C(=O)C(=CC1C2)N(C)CC2)C(O)C(O)N(C)C(O)C(O)N2C(O)C2CC(C)CCO]CC1[NH2)C2C2C(O)C(O)C2C2CO(C)C2C(O)C1CC1C(O)C(C)N2C2C2CC2C1)C(=CC1CCO2C2CC2C2CO2CCO2CC(O)C(=CC1)C(O)C2)C2OCCO(O)N2C1CCO2CC2(C)C(O)C(O)CC2C(O)C1C(O)CC(O)CO2CC(C)C(C)C(C)C(COCCO2COCC2CO2CCC2C(O)C2C(=CC2C(O)C(C)C2CC1C1C12C2C2)CC1C(O)C2C(O)C(O)C2C2CCO2C2)C(O)C(O)C(C)CO2CC2C1CC(O)C2C1(O)C(C)CO(C)C(O)C(C(=O2C(O)C2COO2CO2C(O)C(O)C2C(C)C2C1CC1C2C(O)C(O)C(O)CC2CC2C1)C(O)C(O)C(O)C(O)C(O)CC2C(O)C(O)C(C)C2CO(CO2CO(C)C(O)C(C)CCO1C1CC2CCO1C(O)C2CO2CCC2C1C(O)CO(O)C(O)C(O)CO2CCOCC1C1C2C(O)C(O)C1C(O)CC2CO(O)C(O)C(C)C(C)C(O)C(O)C(O)C1CCOCCOCC1C(O)C(O)C2CCC(O)CC(O)C(O)C2C(O)C(O)CCO1OCCO2C(O)CO(O)C(O)C(O)CC(O)C(O)C(O)CC(O)C(O)CC1CO2CO(O)C(O)C(O)C2CCC1CC2C(O)C(O)C2C(O)C1C(O)C(C)C(O)CO(O)C(O)C(O)C2CCC2C(O)CC2CO2O2CC(O)C1CC2C(O)C(O)C(O)C(O)C(O)C(O)C2C(O)C(O)CC(O)C1CC(O)C(O)C2C(O)C(O)C(O)C(O)C1CCO(C)C(O)C(O)C(O)C1C2C(O)C(O)C(O)C(O)C2(O)C(O)C2C(O)CC(O)C(O)CCO2C(O)C(O)C(O)C(O)C(O)C(O)CO(C)C(O)C(O)C(O)C(O)C(O)CC
INFO:__main__:==================================================




# Coconut

The code base is the official implementation of [Training Large Language Models to Reason in a Continuous Latent Space](https://arxiv.org/abs/2412.06769).

![coconut](assets/coconut.png)

## Getting Started
Clone repo:
```
git clone git@github.com:facebookresearch/coconut.git
cd coconut
```

Setup environment:
```
conda create --name coconut python=3.12
conda activate coconut
pip install -r requirements.txt
```

The code relies on [wandb](https://wandb.ai/site/) for logging. Please log in your wandb account following this [document](https://docs.wandb.ai/ref/cli/wandb-login/) before running any experiments.

## Data

The data for training and evaluation should be presented as a json file like below:

```python
[
  {
    "question": "...",
    "answer": "...",
    "steps": ["...", "...", ...]
  },
  ...
]
```

The file should contain a list of data points. Each data point is composed of a question (str), an answer (str), and a list of steps (str), where each of them is a string.

For example, you can download and process the [GSM8K](https://arxiv.org/abs/2110.14168) dataset (with [augmented training and validation sets](https://github.com/da03/Internalize_CoT_Step_by_Step/tree/e06a32ee5e4cd117171daeb4755d2a97ece62761/data/gsm8k)) by running:

```bash
bash preprocessing/gsm_icot.bash
```

## Arguments

The configuration of a run should be specified in a yaml file (an example can be found [here](args/gsm_coconut.yaml)).

- **General settings**

  - **project**: Project name for wandb
  - **save_path**: Your path to store the checkpoints
  - **only_eval**: If true, only load a model and test on the data from `val_path` (must used along with `load_model_path`). Otherwise, train the model on `train_path` and test on `val_path` after every epoch.

- **Method**
  - **coconut**: Train coconut model
  - **cot**: Train cot model
  - **no_thoughts**: Train coconut (w/o thought) model
  - **no_cot**: Train no-cot model

- **Training settings**

  - **c_thought**: Number of continuous thoughts for each reasoning step
  - **epochs_per_stage**: Number of epochs for every training stage
  - **max_latent_stage**: The maximum number of training stages (in addition to the initial stage)
  - **pad_latent_to_max**: If the number of reasoning steps is fewer than the index of current training stage, pad the number of continuous thoughts.
  - **save_only_improve**: Save the model only when there the best validation accuracy is updated. Recommended to set `False` for Coconut model training, because otherwise the checkpoints in the last stage might now get saved.
  - **uniform_prob**: The probability to mix data from other stages. 0 for standard experiment, 0.3 for analysis experiment.
  - **model_id**: Huggingface model id to load as the initialization, e.g., `openai-community/gpt2`
  - **load_model_path**: The path to a checkpoint to load. Used in two cases: (1) for evaluation (2) to initialize coconut from a CoT-tuned model.
  - **seed**: Random seed.
  - **resume**: The epoch to resume. Can be used when we want to skip the initial training stages.
  - **bf16**: Whether to use bf16 training.
  - **train_path**: Path to the training set.
  - **val_path**: Path to the validation or test set (depending on `only_eval`)
  - **reset_optimizer**: Whether to reset the optimizer when swtiching training stages.
  - **batch_size_training**: Batch size to train the model per GPU.
  - **debug**: If true, there is no wandb and model saving. A subset of data will be used.
  - **gradient_accumulation_steps**: Gradient accumulation steps
  - **num_epochs**: Maximum training epoches.
  - **lr**: Learning rate
  - **weight_decay**: Weight decay


## Training

Run the following commands (replacing `N_GPUS` and `PATH_TO_ARGS`):

```
torchrun --nnodes 1 --nproc_per_node N_GPUS run.py PATH_TO_ARGS
```

## Reproducing Experiments

Here we provide instructions to reproduce our experiments in the paper.

All the commands below assume 4 * A100 (80GB) GPUs. You may change the corresponding arguments in the config file (`batch_size_training`, `gradient_accumulation_steps`) and `nproc_per_node` when launching the run, to adapt your resources.


### GSM8K

Preprocessing data:

```bash
bash preprocessing/gsm_icot.bash
```

First train the model with CoT (as the stage 0 training)

```bash
torchrun --nnodes 1 --nproc_per_node 4 run.py args/gsm_cot.yaml
```

Select a checkpoint as the initialization of Coconut (the validation accuracy is expected to be around 40%). Replace the `load_model_path` in the [args/gsm_coconut.yaml](args/gsm_coconut.yaml) with your selected checkpoint, and run:

```bash
torchrun --nnodes 1 --nproc_per_node 4 run.py args/gsm_coconut.yaml
```

Find the checkpoint with best validation accuracy, and put the path as `load_model_path` in [args/gsm_coconut_eval.yaml](args/gsm_coconut_eval.yaml). To evaluate:

```bash
torchrun --nnodes 1 --nproc_per_node 4 run.py args/gsm_coconut_eval.yaml
```

### ProntoQA

Please clone the official [github repo](https://github.com/asaparov/prontoqa/tree/f0145b867b3c106285ec9ea1941a3f6eb7c6162d) of [ProntoQA](https://arxiv.org/pdf/2210.01240) and generate a raw dataset with:

```bash
cd prontoqa
python run_experiment.py --model-name json --model-size dummy --ordering random --num-trials 10000 --few-shot-examples 0 --ontology fictional --min-hops 5 --max-hops 5 --hops-skip 1
```

Then copy the generated `5hop_0shot_random.json` file to `data` directory, and preprocess the dataset with:

```bash
python preprocessing/prontoqa.py
```


Then run the following to train the model:
```bash
torchrun --nnodes 1 --nproc_per_node 4 run.py args/prontoqa_coconut.yaml
```

Find the checkpoint with best validation accuracy, and put the path as `load_model_path` in [args/prosqa_coconut_eval.yaml](args/prosqa_coconut_eval.yaml). To evaluate:

```bash
torchrun --nnodes 1 --nproc_per_node 4 run.py args/prosqa_coconut_eval.yaml
```


### ProsQA

The ProsQA dataset is at [data/prosqa_*.json](data).

Then run the following to train the model:
```bash
torchrun --nnodes 1 --nproc_per_node 4 run.py args/prosqa_coconut.yaml
```

Find the checkpoint with best validation accuracy, and put the path as `load_model_path` in [args/prosqa_coconut_eval.yaml](args/prosqa_coconut_eval.yaml). To evaluate:

```bash
torchrun --nnodes 1 --nproc_per_node 4 run.py args/prosqa_coconut_eval.yaml
```




## Citation
If you use this code base in your research, please cite our paper with the following BibTex entry:
```bibtex
@article{hao2024training,
  title={Training Large Language Models to Reason in a Continuous Latent Space},
  author={Hao, Shibo and Sukhbaatar, Sainbayar and Su, DiJia and Li, Xian and Hu, Zhiting and Weston, Jason and Tian, Yuandong},
  journal={arXiv preprint arXiv:2412.06769},
  year={2024}
}
```

## License
This code is released under the MIT license (see [LICENSE](LICENSE)).