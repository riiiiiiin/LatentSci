import os
import argparse
import logging
from datetime import datetime
from typing import List

import torch
from peft import LoraConfig, TaskType, get_peft_model, PeftModel

from config import ModelConfig
from dataloader import load_grpo_data
from model_stage3 import Qwen3MoleculeLLM
from trainer.grpo_trainer import QwenMoleculeGRPOTrainer
from trainer.grpo_config import GRPOConfig


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _ensure_lora_and_trainables(model: Qwen3MoleculeLLM):
    """
    Make sure LoRA is enabled on the underlying text model, and that the multimodal heads are trainable.
    Mirrors the training intent of `train_stage3.py`, but for GRPO.
    """
    # Freeze everything first
    for p in model.parameters():
        p.requires_grad = False

    # Enable / create LoRA on the base LLM
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

    # Multimodal heads trainable
    for p in model.projector.parameters():
        p.requires_grad = True
    for p in model.bio_updater.parameters():
        p.requires_grad = True


def load_trained_components_stage3(model, lora_weights_path=None, mm_projector_path=None):
    """
    Same checkpoint format as `train_stage3.py`:
    - LoRA weights folder
    - Combined projector + bio_updater file (mm_projector.pt)
    """
    if lora_weights_path and os.path.exists(lora_weights_path):
        logger.info(f"Loading LoRA weights from: {lora_weights_path}")
        model.model = PeftModel.from_pretrained(model.model, lora_weights_path, is_trainable=True)

    if mm_projector_path and os.path.exists(mm_projector_path):
        logger.info(f"Loading unified multi-modal weights from: {mm_projector_path}")
        device = next(model.parameters()).device
        checkpoint = torch.load(mm_projector_path, map_location=device)
        model.projector.load_state_dict(checkpoint["projector"])
        model.bio_updater.load_state_dict(checkpoint.get("bio_updater", {}), strict=False)
        logger.info("Loaded projector (+ bio_updater if present).")

    return model


def format_reward_answer_tag(prompts: List[str], completions: List[str], completion_ids=None, **kwargs):
    """
    Minimal "try1" reward:
    - reward 1.0 if model outputs a non-empty `<answer> ... </answer>` span
    - else 0.0
    """
    rewards = []
    for c in completions:
        c = c or ""
        lo = c.lower()
        has = ("<answer>" in lo) and ("</answer>" in lo)
        if not has:
            rewards.append(0.0)
            continue
        try:
            inner = lo.split("<answer>", 1)[1].split("</answer>", 1)[0].strip()
            rewards.append(1.0 if len(inner) > 0 else 0.0)
        except Exception:
            rewards.append(0.0)
    return rewards


def main():
    parser = argparse.ArgumentParser(description="GRPO try1 training for Bio-LatentCOT (smiles-aware, optional vLLM).")
    parser.add_argument("--data_path", type=str, default=ModelConfig.DEFAULT_DATA_PATH)

    # Load starting weights (optional)
    parser.add_argument("--lora_path", type=str, default=None, help="Initial LoRA weights folder (optional)")
    parser.add_argument("--projector_path", type=str, default=None, help="Initial mm_projector.pt (optional)")

    # Output
    parser.add_argument("--output_dir", type=str, default="./outputs/grpo_try1")
    parser.add_argument("--run_name", type=str, default=None)

    # Training
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_prompt_length", type=int, default=2048)
    parser.add_argument("--max_completion_length", type=int, default=256)

    # GRPO
    parser.add_argument("--num_generations", type=int, default=2)
    parser.add_argument("--num_iterations", type=int, default=1)
    parser.add_argument("--steps_per_generation", type=int, default=1)
    parser.add_argument("--beta", type=float, default=0.0, help="KL beta (0 disables ref model).")
    parser.add_argument("--epsilon", type=float, default=0.2)

    # Sampling
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)

    # Efficiency
    parser.add_argument("--use_liger", action="store_true", help="Use Liger Kernel for memory efficient training.")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing to save memory.")

    # vLLM
    parser.add_argument("--use_vllm", action="store_true")
    parser.add_argument("--vllm_mode", type=str, default="colocate", choices=["colocate", "server"])
    parser.add_argument("--vllm_tensor_parallel_size", type=int, default=1)
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--vllm_ckpt", type=str, default=None, help="Optional vLLM model path/name (defaults to model path).")
    parser.add_argument("--vllm_max_model_len", type=int, default=4096, help="Maximum model length for vLLM engine.")

    args = parser.parse_args()

    run_name = args.run_name or f"grpo_try1-{datetime.now().strftime('%m%d-%H%M')}"
    os.makedirs(args.output_dir, exist_ok=True)

    # 1) Build model
    mol_config = {
        "num_queries": ModelConfig.NUM_QUERIES,
        "input_dim": ModelConfig.INPUT_DIM,
        "num_heads": ModelConfig.NUM_HEADS,
    }
    model = Qwen3MoleculeLLM(qwen_model_name=ModelConfig.DEFAULT_QWEN_PATH, mol_config=mol_config)

    # 2) Load weights if provided
    if args.lora_path or args.projector_path:
        model = load_trained_components_stage3(model, args.lora_path, args.projector_path)

    # 3) Ensure trainables
    _ensure_lora_and_trainables(model)

    # 4) Dataset
    train_dataset = load_grpo_data(args.data_path)

    # 5) GRPO config
    grpo_args = GRPOConfig(
        output_dir=os.path.join(args.output_dir, run_name),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if args.gradient_checkpointing else None,
        ddp_find_unused_parameters=True,
        remove_unused_columns=False,
        report_to="none",
        seed=args.seed,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        num_generations=args.num_generations,
        num_iterations=args.num_iterations,
        steps_per_generation=args.steps_per_generation,
        beta=args.beta,
        epsilon=args.epsilon,
        epsilon_high=args.epsilon,
        loss_type="grpo",
        temperature=args.temperature,
        top_p=args.top_p,
        use_vllm=args.use_vllm,
        vllm_mode=args.vllm_mode,
        vllm_ckpt=args.vllm_ckpt,
        vllm_max_model_length=args.vllm_max_model_len,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_tensor_parallel_size=args.vllm_tensor_parallel_size,
        use_liger_kernel=False,
        use_liger_manual=args.use_liger,
    )

    trainer = QwenMoleculeGRPOTrainer(
        model=model,
        args=grpo_args,
        reward_funcs=format_reward_answer_tag,
        train_dataset=train_dataset,
        processing_class=model.tokenizer,
    )

    trainer.train()

    # Save final LoRA + multimodal heads (compatible with stage checkpoints)
    final_dir = grpo_args.output_dir
    lora_dir = os.path.join(final_dir, "lora_weights")
    os.makedirs(lora_dir, exist_ok=True)
    model.model.save_pretrained(lora_dir)
    mm_path = os.path.join(final_dir, "mm_projector.pt")
    torch.save({"projector": model.projector.state_dict(), "bio_updater": model.bio_updater.state_dict()}, mm_path)
    model.tokenizer.save_pretrained(final_dir)
    logger.info(f"Saved LoRA to {lora_dir} and mm weights to {mm_path}")


if __name__ == "__main__":
    main()

