import argparse
import logging
import os

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

from config import ModelConfig
from reflection_factory import get_domain_specific_func
load_data = get_domain_specific_func("load_data")
from model_text import load_qwen3_text_model


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextOnlyDataCollator(DataCollatorForSeq2Seq):
    """
    `load_data()` keeps extra columns like `smiles` and `cot_len`.
    A plain text-only CausalLM doesn't accept them, so drop them before padding.
    """

    def __call__(self, features):
        for f in features:
            f.pop("smiles", None)
            f.pop("cot_len", None)
        return super().__call__(features)


def train_text():
    parser = argparse.ArgumentParser(description="Text-only LoRA SFT for Qwen3-8B on ChemCoTBench-style data.")
    parser.add_argument("--data_path", type=str, default=ModelConfig.DEFAULT_DATA_PATH)
    parser.add_argument(
        "--training_stage",
        type=int,
        default=2,
        choices=[1, 2],
        help="1: no-CoT (answer only), 2: with-CoT (cot + answer).",
    )
    parser.add_argument("--output_dir", type=str, default="./outputs/text_sft")

    # Training
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_seq_length", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=42)

    # Efficiency / precision
    parser.add_argument("--bf16", type=lambda x: (str(x).lower() == "true"), default=True)
    parser.add_argument(
        "--gradient_checkpointing",
        type=lambda x: (str(x).lower() == "true"),
        default=True,
        help="Match `train_stage3.py` default (True).",
    )

    # LoRA
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)

    # Logging / saving
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--save_total_limit", type=int, default=1)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)

    args = parser.parse_args()

    include_cot = bool(args.training_stage == 2)
    stage_name = "stage2_withcot" if include_cot else "stage1_nocot"
    run_dir = os.path.join(args.output_dir, stage_name)
    os.makedirs(run_dir, exist_ok=True)

    logger.info("Loading dataset from %s (include_cot=%s)", args.data_path, include_cot)
    train_dataset = load_data(
        args.data_path,
        include_cot=include_cot,
        max_len=int(args.max_seq_length),
        is_coconut=False,
    )

    logger.info("Loading base text model from %s", ModelConfig.DEFAULT_QWEN_PATH)
    model, tokenizer = load_qwen3_text_model(
        ModelConfig.DEFAULT_QWEN_PATH,
        torch_dtype=(torch.bfloat16 if args.bf16 else torch.float32),
        device_map=None,
    )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.config.use_cache = False

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(args.lora_r),
        lora_alpha=int(args.lora_alpha),
        lora_dropout=float(args.lora_dropout),
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

    # Match `train_stage3.py` default: wandb enabled (offline by default).
    os.environ.setdefault("WANDB_MODE", "offline")
    os.environ.setdefault("WANDB_PROJECT", "qwen3-text-sft")

    training_args = TrainingArguments(
        output_dir=run_dir,
        num_train_epochs=float(args.epochs),
        max_steps=int(args.max_steps),
        per_device_train_batch_size=int(args.batch_size),
        gradient_accumulation_steps=int(args.grad_accum),
        learning_rate=float(args.lr),
        bf16=bool(args.bf16),
        seed=int(args.seed),
        logging_steps=int(args.logging_steps),
        save_strategy="no",
        save_steps=int(args.save_steps),
        save_total_limit=int(args.save_total_limit),
        gradient_checkpointing=bool(args.gradient_checkpointing),
        gradient_checkpointing_kwargs={"use_reentrant": False} if bool(args.gradient_checkpointing) else None,
        ddp_find_unused_parameters=True,
        optim="adamw_8bit",
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        remove_unused_columns=False,
        report_to="wandb",
    )

    data_collator = TextOnlyDataCollator(tokenizer=tokenizer, model=model, padding=True, label_pad_token_id=-100)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # Save LoRA + tokenizer
    final_dir = trainer.args.output_dir
    lora_dir = os.path.join(final_dir, "lora_weights")
    os.makedirs(lora_dir, exist_ok=True)
    trainer.model.save_pretrained(lora_dir)
    tokenizer.save_pretrained(final_dir)
    trainer.save_state()
    logger.info("Saved LoRA to %s and tokenizer to %s", lora_dir, final_dir)


if __name__ == "__main__":
    train_text()
