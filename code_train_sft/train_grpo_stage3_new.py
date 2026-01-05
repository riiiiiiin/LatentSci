import os
import re
import argparse
import logging
from datetime import datetime

import torch
import wandb
from datasets import load_dataset
from transformers import TrainerCallback

from typing import Optional

from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from trl import GRPOConfig

# Local project imports (same style as train_stage3.py)
from config import ModelConfig
from dataloader import extract_fields  # query formatting + smiles extraction
from model_stage3 import Qwen3MoleculeLLM

# Stage-3 GRPO trainer wrapper (specialized for smiles-conditioned multimodal model)
from grpo_trainer_trl0152 import BioLatentCOTGRPOTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LoraTrainingMonitorCallback(TrainerCallback):
    """LoRA training monitor callback (prints key metrics)."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        if "loss" in logs:
            logger.info(f"Step {state.global_step}: loss = {logs['loss']:.4f}")
        if "learning_rate" in logs:
            logger.info(f"Step {state.global_step}: lr = {logs['learning_rate']:.6f}")


def load_grpo_data_with_labels(path: str):
    """
    Build a GRPO dataset with:
      - prompt: str
      - input_smiles: list[str]
      - label: str   (formatted like '<answer> ... </answer>')
    """
    import glob

    all_json_files = glob.glob(os.path.join(path, "**/*.json"), recursive=True)
    data_files = [f for f in all_json_files if not f.endswith("rcr.json")]
    ds = load_dataset("json", data_files=data_files)["train"]

    bad_ids = [
        "f7e567a6-47de-4c77-8c1f-9049689322e8",
        "bedfe3e8-ab07-4b8e-b872-ae281e5f55af",
        "9cb0a77d-6203-4686-9c8b-45fd3fc770f2",
    ]
    ds = ds.filter(lambda x: x["id"] not in bad_ids)

    dataset = ds.map(extract_fields, batched=False, remove_columns=ds.column_names)
    dataset = dataset.rename_column("query", "prompt")
    dataset = dataset.remove_columns(
        [c for c in dataset.column_names if c not in ("prompt", "input_smiles", "label")]
    )
    return dataset


_ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", flags=re.IGNORECASE | re.DOTALL)


def _extract_answer(text: str) -> Optional[str]:
    m = _ANSWER_RE.search(text or "")
    if not m:
        return None
    ans = m.group(1).strip()
    ans = re.sub(r"\s+", " ", ans)
    return ans


def accuracy_reward(prompts, completions, label=None, **kwargs):
    """1.0 if extracted <answer> matches label's answer, else 0.0."""
    if label is None:
        return [0.0 for _ in completions]
    out = []
    for c, lab in zip(completions, label):
        pred = _extract_answer(c)
        gold = _extract_answer(lab)
        out.append(1.0 if (pred is not None and gold is not None and pred == gold) else 0.0)
    return out


def format_reward(prompts, completions, **kwargs):
    """0.2 if completion contains <answer>...</answer>, else 0.0."""
    out = []
    for c in completions:
        out.append(0.2 if _extract_answer(c) is not None else 0.0)
    return out


def load_trained_components_stage3(model, lora_weights_path=None, mm_projector_path=None):
    """
    Stage-3 loader: loads LoRA weights and the unified projector+bio_updater checkpoint.
    Checkpoint format matches `train_stage3.py`.
    """
    if lora_weights_path and os.path.exists(lora_weights_path):
        logger.info("Loading LoRA weights from: %s", lora_weights_path)
        model.model = PeftModel.from_pretrained(model.model, lora_weights_path, is_trainable=True)

    if mm_projector_path and os.path.exists(mm_projector_path):
        logger.info("Loading unified multi-modal weights from: %s", mm_projector_path)
        device = next(model.parameters()).device
        checkpoint = torch.load(mm_projector_path, map_location=device)
        if "projector" in checkpoint:
            model.projector.load_state_dict(checkpoint["projector"])
        if "bio_updater" in checkpoint:
            model.bio_updater.load_state_dict(checkpoint["bio_updater"])
        logger.info("Successfully loaded projector and bio_updater.")

    return model


def _ensure_lora_and_trainables(model: Qwen3MoleculeLLM):
    """
    Mirrors `train_stage3.py` intent:
    - LoRA on the base LLM is trainable
    - projector + bio_updater are trainable
    - everything else frozen
    """
    for p in model.parameters():
        p.requires_grad = False

    if not hasattr(model.model, "peft_config") or model.model.peft_config is None:
        logger.info("Configuring LoRA from scratch...")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
        )
        model.model = get_peft_model(model.model, lora_config)

    for p in model.projector.parameters():
        p.requires_grad = True
    for p in model.bio_updater.parameters():
        p.requires_grad = True


def main():
    parser = argparse.ArgumentParser(description="Stage-3 GRPO training (NEW) for Bio-LatentCOT")

    # Data / init weights (match README Stage-3 SFT launch args)
    parser.add_argument("--data_path", type=str, default=ModelConfig.DEFAULT_DATA_PATH)
    parser.add_argument("--lora_path", type=str, default=None, help="Stage-3 starting LoRA folder (optional)")
    parser.add_argument("--projector_path", type=str, default=None, help="Stage-3 starting mm_projector.pt (optional)")
    parser.add_argument("--output_dir", type=str, default="./outputs/grpo_stage3_new")

    # Training
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=42)

    # GRPO
    parser.add_argument("--max_prompt_length", type=int, default=2048)
    parser.add_argument("--max_completion_length", type=int, default=256)
    parser.add_argument("--num_generations", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--logging_steps", type=int, default=10)

    # Output
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--save_full_model", type=lambda x: (str(x).lower() == "true"), default=False)

    # WandB
    parser.add_argument("--wandb_project", type=str, default="qwen3-molecule-unified")
    parser.add_argument("--wandb_mode", type=str, default="offline", choices=["offline", "online", "disabled"])

    args = parser.parse_args()

    run_name = args.run_name or f"Stage3-GRPO-{datetime.now().strftime('%m%d-%H%M')}"
    out_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)

    # Build model (same defaults as train_stage3.py)
    mol_config = {
        "num_queries": ModelConfig.NUM_QUERIES,
        "input_dim": ModelConfig.INPUT_DIM,
        "num_heads": ModelConfig.NUM_HEADS,
    }
    model = Qwen3MoleculeLLM(qwen_model_name=ModelConfig.DEFAULT_QWEN_PATH, mol_config=mol_config)
    tokenizer = model.tokenizer

    # Load Stage-3 weights (LoRA + unified projector/bio_updater)
    if args.lora_path or args.projector_path:
        model = load_trained_components_stage3(model, args.lora_path, args.projector_path)

    _ensure_lora_and_trainables(model)
    model.train()

    # Dataset
    train_dataset = load_grpo_data_with_labels(args.data_path)

    # Training args
    grpo_args = GRPOConfig(
        output_dir=out_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        seed=args.seed,
        bf16=True,
        remove_unused_columns=False,
        logging_steps=args.logging_steps,
        save_strategy="no",
        report_to="wandb" if args.wandb_mode != "disabled" else None,
        # GRPO-specific
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        num_generations=args.num_generations,
        temperature=args.temperature,
        top_p=args.top_p,
        use_vllm=False,
        beta=0.0,  # required: our Stage-3 trainer disables ref-model KL
        log_completions=True,
        ddp_find_unused_parameters=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_8bit",
        lr_scheduler_type="cosine",
        weight_decay=0.01,
    )

    # WandB
    if args.wandb_mode != "disabled":
        if wandb.run is not None:
            wandb.finish()
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            mode=args.wandb_mode,
            config=vars(args),
        )

    trainer = BioLatentCOTGRPOTrainer(
        model=model,
        reward_funcs=[accuracy_reward, format_reward],
        args=grpo_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        callbacks=[LoraTrainingMonitorCallback()],
    )

    trainer.train()

    # Save
    if args.save_full_model:
        logger.info("Saving full model to %s", out_dir)
        trainer.save_model(out_dir)

    # Always save LoRA + multimodal weights + tokenizer (stage3 style)
    lora_out = os.path.join(out_dir, "lora_weights")
    os.makedirs(lora_out, exist_ok=True)
    model.model.save_pretrained(lora_out)

    mm_out = os.path.join(out_dir, "mm_projector.pt")
    mm_weights = {"projector": model.projector.state_dict(), "bio_updater": model.bio_updater.state_dict()}
    torch.save(mm_weights, mm_out)
    tokenizer.save_pretrained(out_dir)

    logger.info("✅ Stage3-GRPO completed. Saved to %s", out_dir)


if __name__ == "__main__":
    main()


