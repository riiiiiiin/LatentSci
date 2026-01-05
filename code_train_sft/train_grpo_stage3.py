import os
import re
import argparse
import logging
import inspect
from datetime import datetime
from typing import Any, Optional, Union

import torch
import torch.nn as nn
import wandb
from datasets import load_dataset
from transformers import TrainerCallback

from trl import GRPOConfig, GRPOTrainer

# Local project imports (same style as train_stage3.py)
from config import ModelConfig
from dataloader import extract_fields  # reuse your ChemCot parsing (includes label + input_smiles)
from model_stage3 import Qwen3MoleculeLLM
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LoraTrainingMonitorCallback(TrainerCallback):
    """LoRA training monitor callback (prints key metrics)."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
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
    dataset = dataset.remove_columns([c for c in dataset.column_names if c not in ("prompt", "input_smiles", "label")])
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
    """
    if lora_weights_path and os.path.exists(lora_weights_path):
        logger.info(f"Loading LoRA weights from: {lora_weights_path}")
        model.model = PeftModel.from_pretrained(model.model, lora_weights_path, is_trainable=True)

    if mm_projector_path and os.path.exists(mm_projector_path):
        logger.info(f"Loading unified multi-modal weights from: {mm_projector_path}")
        device = next(model.parameters()).device
        checkpoint = torch.load(mm_projector_path, map_location=device)
        if "projector" in checkpoint:
            model.projector.load_state_dict(checkpoint["projector"])
        if "bio_updater" in checkpoint:
            model.bio_updater.load_state_dict(checkpoint["bio_updater"])
        logger.info("Successfully loaded projector and bio_updater.")

    return model


class BioLatentCOTGRPOTrainer(GRPOTrainer):
    """
    TRL(0.15.x) GRPOTrainer wrapper to support Bio-LatentCOT Stage-3 `Qwen3MoleculeLLM`.

    Key differences vs vanilla TRL:
      - generation must call `model.generate(smiles_list=..., input_ids=..., attention_mask=...)`
      - logprob alignment must account for molecule-prefix length: n_mols*(num_queries+2)
      - reference model KL is disabled (beta must be 0.0)
    """

    def _extract_smiles(self, rows: list[dict[str, Any]]) -> list[list[str]]:
        if "input_smiles" in rows[0]:
            raw = [r.get("input_smiles") for r in rows]
        elif "smiles" in rows[0]:
            raw = [r.get("smiles") for r in rows]
        else:
            raise ValueError("Dataset rows must include `input_smiles` (or `smiles`) for Stage-3 GRPO.")

        smiles: list[list[str]] = []
        for s in raw:
            if s is None:
                smiles.append([])
            elif isinstance(s, list):
                smiles.append([str(x) for x in s])
            else:
                smiles.append([str(s)])
        return smiles

    # Compatibility: some `transformers.Trainer` versions call `_get_train_sampler(dataset)` (with dataset arg),
    # while TRL 0.15.x defines `_get_train_sampler(self)` (no dataset arg). Accept the extra argument and delegate.
    def _get_train_sampler(self, dataset=None):
        return super()._get_train_sampler()

    def _get_per_token_logps_smiles(self, model, input_ids, attention_mask, logits_to_keep, smiles):
        """
        Compute per-token logprobs for completion tokens when the model prepends an implicit molecule prefix.
        Returns a (B, T) tensor aligned to the last `logits_to_keep` tokens of `input_ids`.
        """
        # If `transformers.Trainer` wrapped the model in `torch.nn.DataParallel` (common in single-process multi-GPU),
        # calling `model(...)` will replicate and may break non-standard submodules. For logprob computation we only
        # need one device, so we unwrap and run on the current device.
        if isinstance(model, torch.nn.DataParallel):
            model = model.module

        # Forward through wrapper model (it will build fused embeddings internally)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, smiles=smiles, return_dict=True)
        logits_full = outputs.logits  # (B, L_fused, V)
        # next-token alignment; if sequence length <= 1, there are no next-token logits to score
        if logits_full.size(1) <= 1:
            return torch.zeros(
                (input_ids.size(0), logits_to_keep), device=logits_full.device, dtype=torch.float32
            )
        logits_full = logits_full[:, :-1, :]  # (B, L_fused-1, V)

        completion_ids = input_ids[:, -logits_to_keep:]
        completion_attn = attention_mask[:, -logits_to_keep:]
        completion_lens = completion_attn.sum(dim=1).to(torch.long)
        text_lens = attention_mask.sum(dim=1).to(torch.long)
        prompt_lens = (text_lens - completion_lens).clamp(min=0)

        mol_prefix_lens = torch.tensor(
            [(len(s) if isinstance(s, list) else 0) * (int(getattr(model, "num_queries")) + 2) for s in smiles],
            device=logits_full.device,
            dtype=torch.long,
        )

        # Important: `logits_to_keep` includes right-padding, but the wrapper model only has logits for real tokens.
        # We keep a fixed (B, logits_to_keep) tensor for downstream GRPO, so we compute logits only for valid
        # completion positions and fill the rest with zeros.
        j = torch.arange(logits_to_keep, device=logits_full.device, dtype=torch.long).unsqueeze(0)
        pos = (mol_prefix_lens.unsqueeze(1) + prompt_lens.unsqueeze(1) + j - 1).clamp(min=0)
        # Ensure we don't index past the available fused length (can happen when completion is shorter than padding).
        max_pos = logits_full.size(1) - 1
        pos = pos.clamp(max=max_pos)
        batch_idx = torch.arange(logits_full.size(0), device=logits_full.device, dtype=torch.long).unsqueeze(1)
        logits = logits_full[batch_idx, pos]  # (B, T, V)

        # Temperature scaling consistent with GRPO args (TRL 0.15.x GRPOConfig exposes `temperature`)
        logits = logits / float(getattr(self.args, "temperature", 1.0))

        from trl.trainer.utils import selective_log_softmax

        # Defensive shape handling: in rare edge cases the fused sequence can be too short, leading to empty logits.
        # Keep the GRPO contract by returning a (B, logits_to_keep) tensor.
        if logits.size(1) == 0:
            return torch.zeros((completion_ids.size(0), completion_ids.size(1)), device=logits.device, dtype=torch.float32)

        t = completion_ids.size(1)
        lt = logits.size(1)
        use_t = min(t, lt)
        logps_part = selective_log_softmax(logits[:, :use_t], completion_ids[:, :use_t])
        logps_part = logps_part * completion_attn[:, :use_t].to(torch.float32)
        if use_t == t:
            logps = logps_part
        else:
            logps = torch.zeros((completion_ids.size(0), t), device=logits.device, dtype=torch.float32)
            logps[:, :use_t] = logps_part
        return logps

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")

        if getattr(self, "beta", 0.0) != 0.0:
            raise ValueError("BioLatentCOTGRPOTrainer requires beta=0.0 (no reference-model KL).")

        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        smiles = inputs["smiles"]

        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        per_token_logps = self._get_per_token_logps_smiles(model, input_ids, attention_mask, logits_to_keep, smiles)

        advantages = inputs["advantages"]
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        per_token_loss = -per_token_loss
        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1).clamp(min=1.0)).mean()
        return loss

    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        # NOTE: TRL's GRPOTrainer uses a no-op collator, so `inputs` is actually a list[dict].
        device = self.accelerator.device
        rows = inputs
        prompts = [x["prompt"] for x in rows]
        smiles = self._extract_smiles(rows)

        # Tokenize prompts (text only). Wrapper model will add the molecule prefix embeddings itself.
        prompt_inputs = self.processing_class(
            prompts, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
        )
        prompt_inputs = super(GRPOTrainer, self)._prepare_inputs(prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        if self.args.use_vllm:
            raise ValueError("Stage-3 GRPO does not support vLLM (smiles-conditioned multimodal generation).")

        # Generate one completion per row (the sampler arranges num_generations copies of each prompt).
        nonpad_text_lens = prompt_mask.sum(dim=1).to(torch.long)
        mol_prefix_lens = torch.tensor(
            [(len(s) if isinstance(s, list) else 0) * (int(getattr(self.model, "num_queries")) + 2) for s in smiles],
            device=nonpad_text_lens.device,
            dtype=torch.long,
        )
        fused_prompt_len = int((mol_prefix_lens + nonpad_text_lens).max().item())

        # Use unwrap_model_for_generation for deepspeed/fsdp safety
        from trl.models import unwrap_model_for_generation

        with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
            seqs = unwrapped_model.generate(
                smiles_list=smiles,
                input_ids=prompt_ids,
                attention_mask=prompt_mask,
                max_new_tokens=int(self.max_completion_length),
                do_sample=True,
                temperature=float(getattr(self.args, "temperature", 1.0)),
                top_p=float(getattr(self, "_gen_top_p", 0.95)),
                # Do NOT pass pad/eos here: Qwen3MoleculeLLM.generate() already forwards them to the base LM,
                # and passing them twice triggers: "got multiple values for keyword argument 'pad_token_id'".
                repetition_penalty=float(self.generation_config.repetition_penalty),
            )

        completion_ids = seqs[:, fused_prompt_len:]

        # Mask everything after EOS (handle the edge case where generation returns 0 new tokens)
        if completion_ids.size(1) == 0:
            completion_mask = torch.zeros(
                (completion_ids.size(0), 0), dtype=torch.long, device=completion_ids.device
            )
        else:
            is_eos = completion_ids == self.processing_class.eos_token_id
            eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
            eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
            sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
            completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Decode completions for reward computation
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)

        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, nn.Module):
                texts = [p + c for p, c in zip(prompts, completions_text)]
                reward_inputs = reward_processing_class(
                    texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                )
                reward_inputs = super(GRPOTrainer, self)._prepare_inputs(reward_inputs)
                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]
            else:
                keys = [key for key in rows[0] if key not in ["prompt", "completion"]]
                reward_kwargs = {key: [example[key] for example in rows] for key in keys}
                out = reward_func(prompts=prompts, completions=completions_text, **reward_kwargs)
                rewards_per_func[:, i] = torch.tensor(out, dtype=torch.float32, device=device)

        from accelerate.utils import gather

        rewards_per_func = gather(rewards_per_func)
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).sum(dim=1)

        if self.num_generations is None or int(self.num_generations) < 2:
            raise ValueError("GRPO requires num_generations >= 2 (std over a single sample is undefined).")

        mean_grouped = rewards.view(-1, int(self.num_generations)).mean(dim=1)
        std_grouped = rewards.view(-1, int(self.num_generations)).std(dim=1)
        mean_grouped = mean_grouped.repeat_interleave(int(self.num_generations), dim=0)
        std_grouped = std_grouped.repeat_interleave(int(self.num_generations), dim=0)
        advantages = (rewards - mean_grouped) / (std_grouped + 1e-4)

        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]

        # Return tensors used by compute_loss
        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "smiles": smiles,
        }


def train_grpo_stage3():
    parser = argparse.ArgumentParser(description="Stage 3 GRPO Training for Bio-LatentCOT")
    parser.add_argument("--data_path", type=str, default=ModelConfig.DEFAULT_DATA_PATH)
    parser.add_argument("--lora_path", type=str, default=None, help="LoRA weights to start from (optional)")
    parser.add_argument(
        "--projector_path", type=str, default=None, help="Unified projector+bio_updater weights to start from (optional)"
    )
    parser.add_argument("--output_dir", type=str, default="./outputs/grpo_stage3")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--max_prompt_length", type=int, default=2048)
    parser.add_argument("--max_completion_length", type=int, default=256)
    parser.add_argument("--num_generations", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_full_model", type=lambda x: (str(x).lower() == "true"), default=False)
    parser.add_argument("--wandb_project", type=str, default="qwen3-molecule-unified")
    parser.add_argument("--wandb_mode", type=str, default="offline", choices=["offline", "online", "disabled"])
    args = parser.parse_args()

    # Ensure W&B is truly disabled when requested (Transformers may still attach WandbCallback if W&B is installed).
    if args.wandb_mode == "disabled":
        os.environ["WANDB_DISABLED"] = "true"
        os.environ["WANDB_MODE"] = "disabled"

    # Model init (same mol_config defaults as train_stage3.py)
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

    # Ensure LoRA exists
    if not hasattr(model.model, "peft_config") or model.model.peft_config is None:
        logger.info("Configuring LoRA from scratch...")
        for p in model.parameters():
            p.requires_grad = False
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
        )
        model.model = get_peft_model(model.model, lora_config)

    # Make projector + bio_updater trainable (and keep LoRA trainable)
    for p in model.projector.parameters():
        p.requires_grad = True
    for p in model.bio_updater.parameters():
        p.requires_grad = True

    model.train()

    # Dataset for GRPO
    train_dataset = load_grpo_data_with_labels(args.data_path)

    # Training args
    stage_output_dir = os.path.join(args.output_dir, "grpo_stage3")
    report_to = [] if args.wandb_mode == "disabled" else "wandb"

    # 兼容不同版本的 TRL (0.15 vs 0.24+)
    grpo_config_kwargs = {
        "output_dir": stage_output_dir,
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "learning_rate": args.lr,
        "bf16": True,
        "remove_unused_columns": False,
        "logging_steps": args.logging_steps,
        "save_strategy": "no",
        "report_to": report_to,
        "max_prompt_length": args.max_prompt_length,
        "max_completion_length": args.max_completion_length,
        "num_generations": args.num_generations,
        "temperature": args.temperature,
        "use_vllm": False,
        "beta": 0.0,  # required for this smiles-aware trainer
        "log_completions": True,
        "ddp_find_unused_parameters": True,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "optim": "adamw_8bit",
        "lr_scheduler_type": "cosine",
        "weight_decay": 0.01,
    }

    # TRL 0.24+ 支持在 GRPOConfig 中设置 top_p/top_k
    grpo_params = inspect.signature(GRPOConfig.__init__).parameters
    if "top_p" in grpo_params:
        grpo_config_kwargs["top_p"] = args.top_p

    grpo_args = GRPOConfig(**grpo_config_kwargs)

    # WandB
    if args.wandb_mode != "disabled":
        if wandb.run is not None:
            wandb.finish()
        wandb.init(
            project=args.wandb_project,
            name=f"Stage3-GRPO-{datetime.now().strftime('%m%d-%H%M')}",
            mode=args.wandb_mode,
            config=vars(args),
        )

    # Important: TRL 0.15.x would normally create a reference model unconditionally.
    # For this Stage-3 multimodal wrapper, we disable ref-model usage and KL by forcing beta=0.0 and by ensuring
    # create_reference_model returns None at init time.
    import trl.trainer.grpo_trainer as grpo_mod

    grpo_mod.create_reference_model = lambda _model: None

    trainer = BioLatentCOTGRPOTrainer(
        model=model,
        reward_funcs=[accuracy_reward, format_reward],
        args=grpo_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        callbacks=[LoraTrainingMonitorCallback()],
    )
    # TRL 0.15.x GRPOConfig does not include top_p/top_k; keep them as explicit generation knobs.
    trainer._gen_top_p = float(args.top_p)

    trainer.train()

    # Save
    os.makedirs(stage_output_dir, exist_ok=True)
    if args.save_full_model:
        logger.info("Saving full model to %s", stage_output_dir)
        trainer.save_model(stage_output_dir)

    # Always save LoRA + multimodal weights + tokenizer (stage3 style)
    lora_out = os.path.join(stage_output_dir, "lora_weights")
    os.makedirs(lora_out, exist_ok=True)
    model.model.save_pretrained(lora_out)

    mm_out = os.path.join(stage_output_dir, "mm_projector.pt")
    mm_weights = {"projector": model.projector.state_dict(), "bio_updater": model.bio_updater.state_dict()}
    torch.save(mm_weights, mm_out)
    tokenizer.save_pretrained(stage_output_dir)

    logger.info("✅ Stage3-GRPO completed. Saved to %s", stage_output_dir)


if __name__ == "__main__":
    train_grpo_stage3()


