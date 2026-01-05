"""
Stage-3 (Bio-LatentCOT) GRPO trainer.

This file started as a copy of TRL's `GRPOTrainer`, but has been specialized for the project's
multi-modal Stage-3 model (`Qwen3MoleculeLLM` in `model_stage3.py`).

Why this specialization is needed:
- The model requires an extra `smiles` argument during forward pass.
- The model *implicitly prepends* a molecule prefix (length = n_mols*(num_queries+2)) before the text tokens.
  So GRPO's per-token logprob slicing must be offset by that prefix length.
- We disable reference-model KL for now (beta must be 0.0), because cloning a full multimodal wrapper is expensive.

Expected dataset columns:
- `prompt` (str)
- `input_smiles` or `smiles` (list[str] or str)
Any extra columns are passed through to custom reward functions.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Union

import torch
from torch import nn

from accelerate.utils import gather
from trl import GRPOTrainer
from trl.models import unwrap_model_for_generation

try:
    # TRL versions differ; some do not export all helpers.
    from trl.trainer.utils import selective_log_softmax  # type: ignore
except Exception:  # pragma: no cover
    def selective_log_softmax(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Fallback: gather per-token log-probabilities of `input_ids` from `logits`.
        logits: (B, T, V), input_ids: (B, T)
        returns: (B, T)
        """
        return torch.log_softmax(logits, dim=-1).gather(-1, input_ids.unsqueeze(-1)).squeeze(-1)

# Reward function: callable(prompts, completions, ...) -> list[float]
RewardFunc = Union[str, nn.Module, Callable[..., list[float]]]


def _ensure_list_of_smiles(x: Any) -> list[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(s) for s in x]
    return [str(x)]


class BioLatentCOTGRPOTrainer(GRPOTrainer):
    """
    TRL GRPOTrainer wrapper to support Bio-LatentCOT Stage-3 `Qwen3MoleculeLLM`.

    Notes:
    - vLLM is not supported in this path.
    - Reference-model KL is disabled: set `beta=0.0`.
    """

    def __init__(self, *args, **kwargs):
        # TRL 0.15.x may create a reference model even when beta=0.
        # We temporarily patch that inside the module to avoid a costly deepcopy.
        import trl.trainer.grpo_trainer as grpo_mod  # TRL internal module

        orig_create_ref = getattr(grpo_mod, "create_reference_model", None)
        if orig_create_ref is not None:
            grpo_mod.create_reference_model = lambda _model: None  # type: ignore[assignment]
        try:
            super().__init__(*args, **kwargs)
        finally:
            if orig_create_ref is not None:
                grpo_mod.create_reference_model = orig_create_ref  # type: ignore[assignment]

    def _extract_smiles(self, rows: list[dict[str, Any]]) -> list[list[str]]:
        if not rows:
            return []
        if "input_smiles" in rows[0]:
            raw = [r.get("input_smiles") for r in rows]
        elif "smiles" in rows[0]:
            raw = [r.get("smiles") for r in rows]
        else:
            raise ValueError("Dataset rows must include `input_smiles` (or `smiles`) for Stage-3 GRPO.")
        return [_ensure_list_of_smiles(s) for s in raw]

    def _mol_prefix_lens(self, model, smiles: list[list[str]], device) -> torch.Tensor:
        # Stage-3 molecule prefix length for each sample:
        # num_mols * (num_queries + 2) where +2 accounts for <mol_start>/<mol_end>.
        num_queries = int(getattr(model, "num_queries"))
        return torch.tensor(
            [len(s) * (num_queries + 2) for s in smiles],
            device=device,
            dtype=torch.long,
        )

    def _get_per_token_logps_smiles(
        self,
        model,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        logits_to_keep: int,
        smiles: list[list[str]],
    ) -> torch.Tensor:
        """
        Compute per-token logprobs for completion tokens when the model prepends an implicit molecule prefix.
        Returns a (B, T) tensor aligned to the last `logits_to_keep` tokens of `input_ids`.
        """
        # Forward through wrapper model (it builds fused embeddings internally).
        # IMPORTANT: model_stage3 forward doesn't support `logits_to_keep`.
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, smiles=smiles, return_dict=True)
        logits_full = outputs.logits  # (B, L_fused, V)
        logits_full = logits_full[:, :-1, :]  # next-token alignment

        completion_ids = input_ids[:, -logits_to_keep:]
        completion_attn = attention_mask[:, -logits_to_keep:]
        completion_lens = completion_attn.sum(dim=1).to(torch.long)
        text_lens = attention_mask.sum(dim=1).to(torch.long)
        prompt_lens = (text_lens - completion_lens).clamp(min=0)

        mol_prefix_lens = self._mol_prefix_lens(model, smiles, device=logits_full.device)

        # For each completion token j, pick the corresponding fused position:
        # fused_pos = mol_prefix_len + prompt_len + j - 1  (because logits are shifted by 1)
        j = torch.arange(logits_to_keep, device=logits_full.device, dtype=torch.long).unsqueeze(0)
        pos = (mol_prefix_lens.unsqueeze(1) + prompt_lens.unsqueeze(1) + j - 1).clamp(min=0)
        batch_idx = torch.arange(logits_full.size(0), device=logits_full.device, dtype=torch.long).unsqueeze(1)
        logits = logits_full[batch_idx, pos]  # (B, T, V)

        # Keep consistency with sampling temperature.
        if getattr(self, "generation_config", None) is not None and getattr(self.generation_config, "temperature", None):
            logits = logits / float(self.generation_config.temperature)

        logps = selective_log_softmax(logits, completion_ids) * completion_attn.to(torch.float32)
        return logps

    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        """
        TRL's GRPOTrainer uses a no-op collator, so `inputs` here is actually `list[dict]` (a micro-batch).
        We:
        - tokenize prompt text
        - generate completion conditioned on smiles
        - compute rewards + advantages
        - return tensors used by compute_loss
        """
        device = self.accelerator.device
        rows = inputs  # type: ignore[assignment]

        prompts = [x["prompt"] for x in rows]
        smiles = self._extract_smiles(rows)

        prompt_inputs = self.processing_class(
            prompts,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        # Bypass GRPOTrainer._prepare_inputs and call Trainer._prepare_inputs
        prompt_inputs = super(GRPOTrainer, self)._prepare_inputs(prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        if getattr(self.args, "use_vllm", False):
            raise ValueError("Stage-3 GRPO does not support vLLM (smiles-conditioned multimodal generation).")

        # Generation returns token IDs including a dummy prompt portion of length = fused_prompt_len.
        nonpad_text_lens = prompt_mask.sum(dim=1).to(torch.long)
        mol_prefix_lens = self._mol_prefix_lens(self.model, smiles, device=nonpad_text_lens.device)
        fused_prompt_len = int((mol_prefix_lens + nonpad_text_lens).max().item())

        with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
            seqs = unwrapped_model.generate(
                smiles_list=smiles,
                input_ids=prompt_ids,
                attention_mask=prompt_mask,
                max_new_tokens=int(self.max_completion_length),
                do_sample=True,
                temperature=float(getattr(self.generation_config, "temperature", 1.0)),
                top_p=float(getattr(self.generation_config, "top_p", 1.0)),
                repetition_penalty=float(getattr(self.generation_config, "repetition_penalty", 1.0)),
                pad_token_id=self.processing_class.pad_token_id,
                eos_token_id=self.processing_class.eos_token_id,
            )

        completion_ids = seqs[:, fused_prompt_len:]

        # Mask everything after EOS
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
                    texts,
                    return_tensors="pt",
                    padding=True,
                    padding_side="right",
                    add_special_tokens=False,
                )
                reward_inputs = super(GRPOTrainer, self)._prepare_inputs(reward_inputs)
                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]
            else:
                keys = [key for key in rows[0] if key not in ["prompt", "completion"]]
                reward_kwargs = {key: [example[key] for example in rows] for key in keys}
                out = reward_func(prompts=prompts, completions=completions_text, **reward_kwargs)
                rewards_per_func[:, i] = torch.tensor(out, dtype=torch.float32, device=device)

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

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "smiles": smiles,
        }

    def compute_loss(self, model, inputs, return_outputs: bool = False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")

        if float(getattr(self, "beta", 0.0)) != 0.0:
            raise ValueError("BioLatentCOTGRPOTrainer currently requires beta=0.0 (no reference-model KL).")

        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        smiles = inputs["smiles"]

        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        per_token_logps = self._get_per_token_logps_smiles(
            model, input_ids, attention_mask, logits_to_keep, smiles
        )
        advantages = inputs["advantages"]

        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        per_token_loss = -per_token_loss
        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1).clamp(min=1.0)).mean()
        return loss