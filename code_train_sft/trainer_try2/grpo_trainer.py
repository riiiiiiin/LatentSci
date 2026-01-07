"""
Thin adapter around TRL's official GRPOTrainer to support Bio-LatentCOT's molecule-conditioned model.

We keep TRL's implementation intact by:
- importing the installed `trl` package
- subclassing `trl.trainer.grpo_trainer.GRPOTrainer`
- overriding only the parts that must be molecule-aware:
  * generation (needs fused prompt embeddings from `model_stage3.Qwen3MoleculeLLM.get_prompt_embeddings`)
  * per-token log-prob extraction (model prepends molecule tokens, so logits must be indexed with a prefix offset)
  * passing `smiles` through the buffered batches so `compute_loss` can run
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import torch

from trl.models import unwrap_model_for_generation
from trl.trainer.grpo_trainer import GRPOTrainer as _TRL_GRPOTrainer
from trl.trainer.utils import entropy_from_logits, selective_log_softmax


def _normalize_smiles_batch(smiles_batch: list[Any] | None) -> list[list[str]] | None:
    if smiles_batch is None:
        return None
    out: list[list[str]] = []
    for item in smiles_batch:
        if item is None:
            out.append([])
        elif isinstance(item, str):
            out.append([item])
        else:
            out.append(list(item))
    return out


class QwenMoleculeGRPOTrainer(_TRL_GRPOTrainer):
    """
    GRPOTrainer adapter for `model_stage3.Qwen3MoleculeLLM`.

    Requirements on the model:
    - `forward(..., smiles=List[List[str]])` must be supported
    - `get_prompt_embeddings(smiles_list, input_ids, attention_mask, ...) -> (prompt_embeds, prompt_attn_mask)` must exist
    - If `beta != 0`, LoRA/PEFT is expected on the *inner* text model, exposing `.disable_adapter()`.
    """

    def __init__(
        self,
        model,
        reward_funcs,
        args=None,
        train_dataset=None,
        eval_dataset=None,
        processing_class=None,
        reward_processing_classes=None,
        callbacks=None,
        optimizers=(None, None),
        peft_config=None,
        tools=None,
        rollout_func=None,
    ):
        inner = getattr(model, "model", None)
        if inner is not None and hasattr(inner, "disable_adapter") and not hasattr(model, "disable_adapter"):
            # Make TRL's `with unwrap_model(model).disable_adapter():` work for a wrapper model.
            model.disable_adapter = inner.disable_adapter  # type: ignore[attr-defined]

        orig_beta = float(getattr(args, "beta", 0.0) or 0.0) if args is not None else 0.0
        can_disable_adapter = hasattr(model, "disable_adapter")

        # Avoid allocating a separate reference model for wrapper models when we can disable the inner adapter instead.
        if args is not None and orig_beta != 0.0 and can_disable_adapter:
            args.beta = 0.0
            super().__init__(
                model=model,
                reward_funcs=reward_funcs,
                args=args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                processing_class=processing_class,
                reward_processing_classes=reward_processing_classes,
                callbacks=callbacks,
                optimizers=optimizers,
                peft_config=peft_config,
                tools=tools,
                rollout_func=rollout_func,
            )
            args.beta = orig_beta
            self.beta = orig_beta
        else:
            super().__init__(
                model=model,
                reward_funcs=reward_funcs,
                args=args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                processing_class=processing_class,
                reward_processing_classes=reward_processing_classes,
                callbacks=callbacks,
                optimizers=optimizers,
                peft_config=peft_config,
                tools=tools,
                rollout_func=rollout_func,
            )

        if getattr(self, "use_vllm", False):
            raise NotImplementedError(
                "This molecule-conditioned GRPO trainer does not support vLLM yet because generation needs "
                "`prompt_embeds` built from `smiles`. Please run with `use_vllm=False`."
            )
        if getattr(self, "tools", None):
            raise NotImplementedError("Tools/tool-calling is not supported in this molecule GRPO trainer.")

        self._current_smiles: list[list[str]] | None = None

    def _extract_smiles_from_inputs(self, inputs: list[dict[str, Any]]) -> list[list[str]]:
        if not inputs:
            return []
        if "smiles" in inputs[0]:
            smiles = [ex.get("smiles") for ex in inputs]
        else:
            smiles = [ex.get("input_smiles") for ex in inputs]
        smiles_norm = _normalize_smiles_batch(smiles)
        if smiles_norm is None:
            raise ValueError("Batch is missing `input_smiles` (or `smiles`) required by the molecule model.")
        return smiles_norm

    def _generate_and_score_completions(self, inputs: list[dict[str, Any]]):
        self._current_smiles = self._extract_smiles_from_inputs(inputs)
        output = super()._generate_and_score_completions(inputs)
        output["smiles"] = self._current_smiles
        return output

    def _generate_single_turn(self, prompts: list):
        if getattr(self, "use_vllm", False):
            raise NotImplementedError("vLLM is not supported for smiles-based generation.")

        if prompts and not isinstance(prompts[0], str):
            raise NotImplementedError("Conversational prompts are not supported for the molecule GRPO trainer.")

        generate_inputs = self.processing_class(text=prompts, padding=True, padding_side="left", return_tensors="pt")
        generate_inputs = super()._prepare_inputs(generate_inputs)
        prompt_ids, prompt_mask = generate_inputs["input_ids"], generate_inputs["attention_mask"]

        smiles = self._current_smiles
        if smiles is None or len(smiles) != prompt_ids.size(0):
            raise RuntimeError(
                "Internal error: missing/size-mismatched smiles for generation. "
                f"smiles={None if smiles is None else len(smiles)}, batch={prompt_ids.size(0)}."
            )

        with (
            unwrap_model_for_generation(
                self.model_wrapped,
                self.accelerator,
                gather_deepspeed3_params=self.args.ds3_gather_for_generation,
                generation_kwargs=self.generation_kwargs,
            ) as unwrapped_model,
            torch.no_grad(),
        ):
            if not hasattr(unwrapped_model, "get_prompt_embeddings"):
                raise AttributeError(
                    "Model must implement `get_prompt_embeddings(smiles_list, input_ids, attention_mask, ...)` "
                    "for molecule-conditioned GRPO generation."
                )

            prompt_embeds, prompt_attn_mask = unwrapped_model.get_prompt_embeddings(
                smiles_list=smiles,
                input_ids=prompt_ids,
                attention_mask=prompt_mask,
                refine_bio_tokens=True,
            )
            fused_prompt_len = prompt_embeds.size(1)

            out = unwrapped_model.model.generate(
                inputs_embeds=prompt_embeds,
                attention_mask=prompt_attn_mask,
                generation_config=self.generation_config,
                return_dict_in_generate=True,
                output_scores=True,
            )

        sequences = out.sequences  # (B, ?)
        gen_len = len(out.scores)
        if sequences.size(1) == gen_len:
            completion_ids = sequences
        elif sequences.size(1) == fused_prompt_len + gen_len:
            completion_ids = sequences[:, fused_prompt_len:]
        else:
            # Fallback: keep the generated tail.
            completion_ids = sequences[:, -gen_len:] if gen_len > 0 else sequences[:, 0:0]

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.eos_token_id
        if completion_ids.size(1) > 0:
            eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=completion_ids.device)
            has_eos = is_eos.any(dim=1)
            eos_idx[has_eos] = is_eos.int().argmax(dim=1)[has_eos]
            sequence_indices = torch.arange(is_eos.size(1), device=completion_ids.device).expand(is_eos.size(0), -1)
            completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        else:
            completion_mask = torch.zeros_like(completion_ids, dtype=torch.int)

        prompt_ids_list = [p[m].tolist() for p, m in zip(prompt_ids, prompt_mask.bool())]
        completion_ids_list = [c[m].tolist() for c, m in zip(completion_ids, completion_mask.bool())]
        logprobs = None
        extra_fields = {}
        return prompt_ids_list, completion_ids_list, logprobs, extra_fields

    def _get_per_token_logps_and_entropies(
        self,
        model,
        input_ids,
        attention_mask,
        logits_to_keep,
        batch_size=None,
        compute_entropy=False,
        pixel_values=None,
        image_grid_thw=None,
        num_images=None,
        pixel_attention_mask=None,
        image_sizes=None,
        token_type_ids=None,
        smiles: Optional[list[list[str]]] = None,
    ):
        smiles_to_use = smiles if smiles is not None else self._current_smiles
        if smiles_to_use is None:
            return super()._get_per_token_logps_and_entropies(
                model,
                input_ids,
                attention_mask,
                logits_to_keep,
                batch_size=batch_size,
                compute_entropy=compute_entropy,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                num_images=num_images,
                pixel_attention_mask=pixel_attention_mask,
                image_sizes=image_sizes,
                token_type_ids=token_type_ids,
            )

        batch_size = batch_size or input_ids.size(0)
        all_logps = []
        all_entropies = []

        model_to_check = self.accelerator.unwrap_model(model)
        num_queries = int(getattr(model_to_check, "num_queries", 0))
        if num_queries <= 0:
            # Unexpected: fall back to TRL behavior.
            return super()._get_per_token_logps_and_entropies(
                model,
                input_ids,
                attention_mask,
                logits_to_keep,
                batch_size=batch_size,
                compute_entropy=compute_entropy,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                num_images=num_images,
                pixel_attention_mask=pixel_attention_mask,
                image_sizes=image_sizes,
                token_type_ids=token_type_ids,
            )

        for start in range(0, input_ids.size(0), batch_size):
            end = start + batch_size
            input_ids_batch = input_ids[start:end]
            attention_mask_batch = attention_mask[start:end]
            smiles_batch = smiles_to_use[start:end]

            model_inputs = {
                "input_ids": input_ids_batch,
                "attention_mask": attention_mask_batch,
                "use_cache": False,
                "smiles": smiles_batch,
            }
            logits_full = model(**model_inputs).logits  # (B, L_fused, V)
            logits_full = logits_full[:, :-1, :]  # next-token shift

            completion_ids = input_ids_batch[:, -logits_to_keep:]
            completion_attn = attention_mask_batch[:, -logits_to_keep:]

            completion_lens = completion_attn.sum(dim=1).to(torch.long)
            text_lens = attention_mask_batch.sum(dim=1).to(torch.long)
            prompt_lens = (text_lens - completion_lens).clamp(min=0)

            mol_prefix_lens = torch.tensor(
                [(len(s) if isinstance(s, list) else 0) * (num_queries + 2) for s in smiles_batch],
                device=logits_full.device,
                dtype=torch.long,
            )

            j = torch.arange(logits_to_keep, device=logits_full.device, dtype=torch.long).unsqueeze(0)
            pos = mol_prefix_lens.unsqueeze(1) + prompt_lens.unsqueeze(1) + j - 1
            pos = pos.clamp(min=0, max=logits_full.size(1) - 1)
            batch_idx = torch.arange(logits_full.size(0), device=logits_full.device, dtype=torch.long).unsqueeze(1)
            logits = logits_full[batch_idx, pos]  # (B, T, V)
            logits = logits / self.temperature

            logps = selective_log_softmax(logits, completion_ids) * completion_attn.to(torch.float32)
            all_logps.append(logps)

            if compute_entropy:
                with torch.no_grad():
                    entropies = entropy_from_logits(logits) * completion_attn.to(torch.float32)
                all_entropies.append(entropies)

        logps = torch.cat(all_logps, dim=0)
        entropies = torch.cat(all_entropies, dim=0) if compute_entropy else None
        return logps, entropies

    def _compute_loss(self, model, inputs):
        # Ensure the current batch's `smiles` is visible to `_get_per_token_logps_and_entropies` when the parent
        # implementation calls it (it doesn't thread `smiles` explicitly).
        if "smiles" in inputs:
            self._current_smiles = inputs["smiles"]
        return super()._compute_loss(model, inputs)
