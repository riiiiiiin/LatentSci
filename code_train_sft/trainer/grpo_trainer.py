"""
Molecule-specific GRPO trainer for Bio-LatentCOT.

This file is adapted from TRL's `GRPOTrainer` and a prior domain fork, but is specialized for
the Bio-LatentCOT multi-modal model `Qwen3MoleculeLLM` (see `code_train_sft/model_stage3.py`).

Key differences vs a text-only GRPO trainer:
- The model requires an extra `smiles` argument in `forward()`:
  `smiles` is a python object: List[List[str]] of length batch_size.
- We keep `smiles` aligned with examples through:
  - shuffling
  - splitting `steps_per_generation`
  - log-prob computation on prompt+completion sequences

Dataset expectations (for training/eval):
- Each example must have:
  - `prompt`: str (or `query` as fallback)
  - `smiles`: List[str] (or `input_smiles` as fallback)
"""

from __future__ import annotations

import inspect
import os
import random
import time
from collections import defaultdict, deque
from contextlib import nullcontext
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.utils.data
import transformers
from accelerate import logging
from accelerate.utils import gather, gather_object, is_peft_model, set_seed
try:
    from datasets import Dataset, IterableDataset
except Exception:  # pragma: no cover
    # Some environments ship a broken `datasets` install (e.g., `soundfile` present but system `libsndfile` missing).
    # We keep import-time robustness; training will still require a working `datasets`.
    Dataset = object  # type: ignore
    IterableDataset = object  # type: ignore
from packaging import version
from torch import nn
from torch.utils.data import DataLoader, Sampler
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
)
from transformers.trainer_utils import seed_worker
from transformers.utils import is_datasets_available, is_peft_available

from .grpo_config import GRPOConfig


def identity(x):
    return x


def pad(tensors: list[torch.Tensor], padding_value: int | float = 0, padding_side: str = "right") -> torch.Tensor:
    """Pad a list of tensors to the same length."""
    if not tensors:
        return torch.tensor([])
    max_len = max(t.size(0) for t in tensors)
    out = torch.full((len(tensors), max_len), padding_value, device=tensors[0].device, dtype=tensors[0].dtype)
    for i, t in enumerate(tensors):
        if padding_side == "right":
            out[i, : t.size(0)] = t
        else:
            out[i, -t.size(0) :] = t
    return out


def selective_log_softmax(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Compute log_softmax and select values at label indices."""
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    return torch.gather(log_probs, -1, labels.unsqueeze(-1)).squeeze(-1)


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Compute entropy from logits."""
    probs = torch.nn.functional.softmax(logits, dim=-1)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    return -torch.sum(probs * log_probs, dim=-1)


def nanstd(x: torch.Tensor, dim: int | None = None, keepdim: bool = False):
    """Standard deviation ignoring NaNs."""
    mask = ~torch.isnan(x)
    if dim is None:
        mean = x[mask].mean()
        return torch.sqrt(((x[mask] - mean) ** 2).mean())
    else:
        # Simplified per-dim nanstd
        mean = torch.nanmean(x, dim=dim, keepdim=True)
        sq_diff = (x - mean) ** 2
        return torch.sqrt(torch.nanmean(sq_diff, dim=dim, keepdim=keepdim))


logger = logging.get_logger(__name__)

if is_peft_available():
    from peft import PeftConfig, get_peft_model


class SyncRefModelCallback(TrainerCallback):
    """
    Minimal replacement for `trl.trainer.callbacks.SyncRefModelCallback`.

    Syncs `ref_model` towards the current policy model every `ref_model_sync_steps` steps:
      `ref <- alpha * policy + (1 - alpha) * ref`
    """

    def __init__(self, ref_model: nn.Module, accelerator, alpha: float = 0.6, sync_steps: int = 512):
        self.ref_model = ref_model
        self.accelerator = accelerator
        self.alpha = float(alpha)
        self.sync_steps = int(sync_steps)

    def on_step_end(self, args, state, control, **kwargs):
        if self.ref_model is None:
            return control
        if self.sync_steps <= 0:
            return control
        if state.global_step == 0 or (state.global_step % self.sync_steps) != 0:
            return control

        model = kwargs.get("model", None)
        if model is None:
            return control

        with torch.no_grad():
            policy = self.accelerator.unwrap_model(model)
            ref = self.accelerator.unwrap_model(self.ref_model)
            for ref_param, policy_param in zip(ref.parameters(), policy.parameters()):
                if ref_param.data.shape != policy_param.data.shape:
                    continue
                ref_param.data.mul_(1.0 - self.alpha).add_(policy_param.data, alpha=self.alpha)

        return control

# Reward function: callable(prompts, completions, ...) -> list[float]
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], List[float]]]


class RepeatSampler(Sampler):
    """
    Sampler that repeats the indices of a dataset in a structured manner.

    Note: this sampler is intentionally rank-agnostic (matches TRL's behavior). In distributed training, the
    `accelerate`/`transformers.Trainer` dataloader wrapper shards batches across processes. If we shard here too, we risk
    double-sharding and breaking the `[idx] * num_generations` grouping required by `rewards.view(-1, G)`.
    """

    def __init__(
        self,
        data_source,
        mini_repeat_count: int,
        batch_size: int = 1,
        repeat_count: int = 1,
        shuffle: bool = True,
        seed: Optional[int] = None,
    ):
        self.data_source = data_source
        self.mini_repeat_count = mini_repeat_count
        self.batch_size = batch_size
        self.repeat_count = repeat_count
        self.num_samples = len(data_source)
        self.shuffle = shuffle
        self.seed = seed

        if shuffle:
            # Create a local random generator so we don't depend on global RNG state.
            self.generator = torch.Generator()
            if seed is not None:
                self.generator.manual_seed(seed)

    def __iter__(self):
        if self.shuffle:
            indexes = torch.randperm(self.num_samples, generator=self.generator).tolist()
        else:
            indexes = list(range(self.num_samples))

        # Group into batches of `batch_size` unique indices, and drop partials.
        chunks = [indexes[i : i + self.batch_size] for i in range(0, len(indexes), self.batch_size)]
        chunks = [chunk for chunk in chunks if len(chunk) == self.batch_size]

        for chunk in chunks:
            for _ in range(self.repeat_count):
                for index in chunk:
                    for _ in range(self.mini_repeat_count):
                        yield index

    def __len__(self) -> int:
        return (self.num_samples // self.batch_size) * self.batch_size * self.mini_repeat_count * self.repeat_count


# vLLM is optional.


def qwen_mol_grpo_collate_fn(
    batch: List[Dict[str, Any]],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    add_prompt_suffix: str = "\n\n",
) -> Dict[str, Any]:
    prompts: List[str] = []
    smiles: List[List[str]] = []

    for ex in batch:
        p = ex.get("prompt", None)
        if p is None:
            p = ex.get("query", None)
        if p is None:
            raise KeyError("Each example must have `prompt` (or `query`).")
        prompts.append(str(p))

        s = ex.get("smiles", None)
        if s is None:
            s = ex.get("input_smiles", None)
        if s is None:
            raise KeyError("Each example must have `smiles` (or `input_smiles`).")
        if isinstance(s, str):
            s = [s]
        smiles.append(list(s))

    # Tokenize prompt-only (GRPO generates completion)
    enc = tokenizer(
        [p + add_prompt_suffix for p in prompts],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )

    return {
        "prompt": prompts,
        "original_prompts": prompts,
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "smiles": smiles,
    }


def _split_batch_dict(d: Dict[str, Any], num_splits: int) -> List[Dict[str, Any]]:
    """Split dict of tensors/lists along batch dimension into `num_splits` equal chunks."""
    if num_splits <= 1:
        return [d]

    batch_size: Optional[int] = None
    for v in d.values():
        if isinstance(v, torch.Tensor):
            batch_size = int(v.size(0))
            break
        if isinstance(v, list):
            batch_size = len(v)
            break
    if batch_size is None:
        raise ValueError("Cannot split batch: no tensor/list found.")
    if batch_size % num_splits != 0:
        raise ValueError(f"Batch size {batch_size} not divisible by num_splits={num_splits}.")

    chunk = batch_size // num_splits
    outs: List[Dict[str, Any]] = []
    for i in range(num_splits):
        s, e = i * chunk, (i + 1) * chunk
        sub: Dict[str, Any] = {}
        for k, v in d.items():
            if isinstance(v, torch.Tensor):
                sub[k] = v[s:e]
            elif isinstance(v, list):
                sub[k] = v[s:e]
            elif isinstance(v, dict):
                # Handle nested dicts like multimodal_inputs
                sub[k] = {kk: (vv[s:e] if isinstance(vv, (torch.Tensor, list)) else vv) for kk, vv in v.items()}
            else:
                sub[k] = v
        outs.append(sub)
    return outs


class QwenMoleculeGRPOTrainer(Trainer):
    """
    GRPO trainer specialized for Bio-LatentCOT's `Qwen3MoleculeLLM`.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        reward_funcs: Union[RewardFunc, List[RewardFunc]],
        args: Optional[GRPOConfig] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, Dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, List[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[List] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        **kwargs,
    ):
        if args is None:
            model_name = model.config._name_or_path.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        model_init_kwargs = args.model_init_kwargs or {}

        if isinstance(model, str):
            raise TypeError("`model` must be an instantiated model (e.g., Qwen3MoleculeLLM), not a string.")

        # Inspect forward signature (some models don't support logits_to_keep)
        self.model_kwarg_keys = (
            inspect.signature(model.forward).parameters.keys()
            if not hasattr(model, "get_base_model")
            else inspect.signature(model.get_base_model().forward).parameters.keys()
        )

        model_init_kwargs["use_cache"] = False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")

        if peft_config is not None:
            if not is_peft_available():
                raise ImportError("PEFT is required to use `peft_config`. Run `pip install peft`.")
            model = get_peft_model(model, peft_config)

        # Tokenizer / processing class
        if processing_class is None:
            if hasattr(model, "tokenizer"):
                processing_class = model.tokenizer
            else:
                processing_class = AutoTokenizer.from_pretrained(model.config._name_or_path)
        tokenizer = processing_class
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        self.pad_token = tokenizer.pad_token
        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id
        # Keep a stable attribute name across transformers versions.
        self.processing_class = processing_class

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        self.reward_func_names = []
        for i, rf in enumerate(reward_funcs):
            if isinstance(rf, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(rf, num_labels=1, **model_init_kwargs)
            if isinstance(reward_funcs[i], nn.Module):
                self.reward_func_names.append(reward_funcs[i].config._name_or_path.split("/")[-1])
            else:
                self.reward_func_names.append(reward_funcs[i].__name__)
        self.reward_funcs = reward_funcs

        # Reward weights
        if args.reward_weights is not None:
            if len(args.reward_weights) != len(reward_funcs):
                raise ValueError("Number of reward_weights must match number of reward_funcs.")
            self.reward_weights = torch.tensor(args.reward_weights, dtype=torch.float32)
        else:
            self.reward_weights = torch.ones(len(reward_funcs), dtype=torch.float32)

        # Reward processing classes
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        if len(reward_processing_classes) != len(reward_funcs):
            raise ValueError("reward_processing_classes must have same length as reward_funcs.")
        for i, (rpc, rf) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(rf, PreTrainedModel):
                if rpc is None:
                    rpc = AutoTokenizer.from_pretrained(rf.config._name_or_path)
                if rpc.pad_token_id is None:
                    rpc.pad_token = rpc.eos_token
                rf.config.pad_token_id = rpc.pad_token_id
                reward_processing_classes[i] = rpc
        self.reward_processing_classes = reward_processing_classes

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length
        self.num_generations = args.num_generations
        self.temperature = args.temperature
        self.top_p = getattr(args, "top_p", 0.9)
        self.top_k = getattr(args, "top_k", 50)
        self.min_p = getattr(args, "min_p", None)
        self.repetition_penalty = getattr(args, "repetition_penalty", 1.0)
        self.use_vllm = bool(getattr(args, "use_vllm", False))
        self.vllm_mode = getattr(args, "vllm_mode", "colocate")
        self.vllm_tensor_parallel_size = int(getattr(args, "vllm_tensor_parallel_size", 1) or 1)
        self.vllm_gpu_memory_utilization = float(getattr(args, "vllm_gpu_memory_utilization", 0.9))
        self.vllm_enable_sleep_mode = bool(getattr(args, "vllm_enable_sleep_mode", False))
        self.vllm_server_host = getattr(args, "vllm_server_host", "localhost")
        self.vllm_server_port = int(getattr(args, "vllm_server_port", 8000))
        self.vllm_server_base_url = getattr(args, "vllm_server_base_url", None)
        self.vllm_server_timeout = float(getattr(args, "vllm_server_timeout", 60.0))
        self.vllm_ckpt = getattr(args, "vllm_ckpt", None)
        self.vllm_max_model_length = int(getattr(args, "vllm_max_model_length", 4096))
        self.scale_rewards = getattr(args, "scale_rewards", "group")
        self.importance_sampling_level = getattr(args, "importance_sampling_level", "token")
        self.mask_truncated_completions = getattr(args, "mask_truncated_completions", False)
        self.top_entropy_quantile = getattr(args, "top_entropy_quantile", 1.0)

        if self.top_entropy_quantile < 1.0:
            # This fork doesn't implement entropy masking; keep guard consistent with the upstream options.
            raise NotImplementedError("Entropy quantile masking is not supported in this molecule GRPO trainer.")

        # Multi-step
        self.num_iterations = getattr(args, "num_iterations", 1)
        self._step = 0
        self._buffered_inputs = None

        # Generation config
        generation_kwargs = {
            "max_new_tokens": self.max_completion_length,
            "do_sample": True,
            "pad_token_id": tokenizer.pad_token_id,
            "bos_token_id": getattr(tokenizer, "bos_token_id", None),
            "eos_token_id": tokenizer.eos_token_id,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "min_p": self.min_p,
            "repetition_penalty": self.repetition_penalty,
        }
        extra_gen_kwargs = getattr(args, "generation_kwargs", None)
        if extra_gen_kwargs is not None:
            generation_kwargs.update(extra_gen_kwargs)
        self.generation_config = GenerationConfig(**generation_kwargs)

        self.loss_type = getattr(args, "loss_type", "grpo") or "grpo"
        if self.loss_type != "grpo":
            raise NotImplementedError(
                f"Only loss_type='grpo' is implemented in this extracted trainer, got {self.loss_type!r}. "
                "Pass `loss_type='grpo'` when constructing GRPOConfig."
            )

        # Trainer init (keep compatibility across transformers versions).
        trainer_init_kwargs = dict(
            model=model,
            args=args,
            data_collator=identity,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            callbacks=callbacks,
            optimizers=optimizers,
        )
        trainer_sig = inspect.signature(Trainer.__init__).parameters
        if "processing_class" in trainer_sig:
            trainer_init_kwargs["processing_class"] = processing_class
        elif "tokenizer" in trainer_sig:
            trainer_init_kwargs["tokenizer"] = processing_class
        super().__init__(**trainer_init_kwargs)

        # Reference model (KL)
        self.beta = args.beta
        self.ref_model = None
        if self.beta != 0.0:
            # This trainer computes KL either by disabling PEFT adapters on the policy model (preferred), or by using an
            # explicitly provided `ref_model` (not implemented here for the multimodal wrapper).
            wrapper = model
            inner = getattr(wrapper, "model", None)
            if inner is None or not hasattr(inner, "disable_adapter"):
                raise NotImplementedError(
                    "KL beta != 0 requires a PEFT-wrapped inner model exposing `.disable_adapter()` in this trainer. "
                    "Either use LoRA/PEFT (recommended) or set `beta=0.0`."
                )

        # Metrics/log buffers
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        gen_bs = getattr(args, "generation_batch_size", args.per_device_train_batch_size)
        self._logs = {
            "prompt": deque(maxlen=gen_bs),
            "completion": deque(maxlen=gen_bs),
            "rewards": defaultdict(lambda: deque(maxlen=gen_bs)),
            "advantages": deque(maxlen=gen_bs),
        }

        set_seed(args.seed, device_specific=True)

        self.model_accepts_loss_kwargs = False
        self.current_gradient_accumulation_steps = int(getattr(self.args, "gradient_accumulation_steps", 1)) or 1

        if self.ref_model is not None:
            # Keep it simple: let accelerator place it
            self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        if getattr(args, "sync_ref_model", False):
            self.add_callback(
                SyncRefModelCallback(
                    ref_model=self.ref_model,
                    accelerator=self.accelerator,
                    alpha=getattr(args, "ref_model_mixup_alpha", 0.6),
                    sync_steps=getattr(args, "ref_model_sync_steps", 512),
                )
            )

        # ---------------------------------------------------------
        # Optional Liger Kernel init
        # ---------------------------------------------------------
        if getattr(args, "use_liger_manual", False):
            try:
                # Try new Liger Kernel API first
                try:
                    from liger_kernel.transformers import patch_liger as patch_func
                except ImportError:
                    from liger_kernel.transformers import _apply_liger_kernel_to_instance as patch_func

                # We patch the underlying LLM, not the wrapper, to avoid breaking 
                # the wrapper's custom forward (molecule fusion) logic.
                wrapper = self.accelerator.unwrap_model(self.model)
                
                # If it's our custom wrapper, use the helper to get the real ForCausalLM.
                # Otherwise fall back to .model or the wrapper itself.
                if hasattr(wrapper, "_get_actual_llm"):
                    inner_model = wrapper._get_actual_llm()
                else:
                    inner_model = getattr(wrapper, "model", wrapper)

                # IMPORTANT: If inner_model is a PeftModel, we MUST patch its base_model (the ForCausalLM).
                # Otherwise Liger's lce_forward will call PeftModel.model(...) which returns logits, 
                # causing shape mismatch when lce_forward then calls lm_head(logits).
                if hasattr(inner_model, "base_model") and hasattr(inner_model.base_model, "model"):
                    target_to_patch = inner_model.base_model.model
                else:
                    target_to_patch = inner_model

                logger.info(f"Applying Liger Kernel to: {type(target_to_patch).__name__}")
                try:
                    patch_func(model=target_to_patch)
                except TypeError:
                    patch_func(target_to_patch)
                logger.info("Liger Kernel applied successfully.")
            except Exception as e:
                logger.warning(f"Failed to apply Liger Kernel: {e}")

        # ---------------------------------------------------------
        # Optional vLLM init
        # ---------------------------------------------------------
        self.llm = None
        self.vllm_client = None
        self._last_loaded_step = -1
        self.tp_group = None

        if self.use_vllm:
            # Import vLLM only when needed to show the real error if it's broken
            from vllm import LLM

            # Server mode uses a vLLM HTTP server; colocate mode runs vLLM inside the training workers.
            if self.vllm_mode == "server":
                try:
                    from trl.extras.vllm_client import VLLMClient  # type: ignore
                except Exception as e:
                    raise ModuleNotFoundError(
                        "Your installed `trl` does not provide `trl.extras.vllm_client` required for vLLM server mode. "
                        "Either upgrade TRL or use `--vllm_mode colocate`."
                    ) from e
                if self.accelerator.is_main_process:
                    base_url = self.vllm_server_base_url or f"http://{self.vllm_server_host}:{self.vllm_server_port}"
                    self.vllm_client = VLLMClient(base_url=base_url, connection_timeout=self.vllm_server_timeout)
                    self.vllm_client.init_communicator(device=torch.cuda.current_device())
            elif self.vllm_mode == "colocate":
                if self.vllm_tensor_parallel_size > 1:
                    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
                        raise RuntimeError("vLLM TP>1 requires torch.distributed initialized.")
                    if not self.accelerator.num_processes % self.vllm_tensor_parallel_size == 0:
                        raise ValueError(
                            f"vllm_tensor_parallel_size ({self.vllm_tensor_parallel_size}) must divide world size "
                            f"({self.accelerator.num_processes}) evenly."
                        )
                    self.tp_group, _ = torch.distributed.new_subgroups_by_enumeration(
                        [
                            list(
                                range(
                                    i * self.vllm_tensor_parallel_size,
                                    (i + 1) * self.vllm_tensor_parallel_size,
                                )
                            )
                            for i in range(self.accelerator.num_processes // self.vllm_tensor_parallel_size)
                        ]
                    )

                # vLLM will be loaded from a checkpoint path (or model name).
                vllm_model_id = self.vllm_ckpt or self.model.config._name_or_path
                vllm_max_num_seqs = (
                    args.per_device_train_batch_size
                    * self.vllm_tensor_parallel_size
                    * getattr(args, "steps_per_generation", 1)
                )
                self.llm = LLM(
                    model=vllm_model_id,
                    tensor_parallel_size=self.vllm_tensor_parallel_size,
                    gpu_memory_utilization=self.vllm_gpu_memory_utilization,
                    max_num_seqs=vllm_max_num_seqs,
                    max_model_len=self.vllm_max_model_length,
                    distributed_executor_backend="external_launcher",
                    seed=self.accelerator.process_index // self.vllm_tensor_parallel_size,
                    enable_prompt_embeds=True,
                )
                if self.vllm_enable_sleep_mode:
                    self.llm.sleep(level=1)
            else:
                raise ValueError(f"vllm_mode must be 'server' or 'colocate', got {self.vllm_mode!r}.")

            # Avoid process desync
            self.accelerator.wait_for_everyone()

    def _fix_param_name_to_vllm(self, name: str, extra_prefixes: Optional[List[str]] = None) -> str:
        extra_prefixes = extra_prefixes or []
        prefixes = ["_checkpoint_wrapped_module."] + extra_prefixes
        for p in prefixes:
            name = name.replace(p, "")
        return name

    def _move_model_to_vllm(self):
        """
        Sync HF model weights to vLLM.

        Important: for Bio-LatentCOT, we only sync the underlying text LLM (`Qwen3MoleculeLLM.model`),
        not the projector/mol encoder. Molecule fusion happens on the HF side when building prompt embeds.
        """
        if not self.use_vllm:
            return

        # Select underlying LLM module (AutoModelForCausalLM or PEFT wrapper)
        wrapper = self.accelerator.unwrap_model(self.model)
        llm_module = getattr(wrapper, "model", wrapper)

        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        zero_stage_3 = deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3
        if zero_stage_3:
            import deepspeed

            gather_if_zero3 = deepspeed.zero.GatheredParameters
        else:
            from contextlib import nullcontext

            gather_if_zero3 = nullcontext

        # PEFT: merge adapters before syncing, then unmerge.
        if is_peft_model(llm_module):
            with gather_if_zero3(list(llm_module.parameters())):
                if hasattr(llm_module, "merge_adapter"):
                    llm_module.merge_adapter()

                for name, param in llm_module.named_parameters():
                    name = name.removeprefix("base_model.model.").replace(".base_layer", "")
                    if getattr(llm_module, "prefix", None) and llm_module.prefix in name:
                        continue
                    if "original_module" in name:
                        continue
                    name = self._fix_param_name_to_vllm(name, extra_prefixes=["modules_to_save.default."])

                    with gather_if_zero3([param]):
                        if self.vllm_mode == "server" and self.accelerator.is_main_process:
                            self.vllm_client.update_named_param(name, param.data)
                        elif self.vllm_mode == "colocate":
                            llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                            llm_model.load_weights([(name, param.data)])

                if hasattr(llm_module, "unmerge_adapter"):
                    llm_module.unmerge_adapter()
        else:
            for name, param in llm_module.named_parameters():
                name = self._fix_param_name_to_vllm(name)
                with gather_if_zero3([param]):
                    if self.vllm_mode == "server" and self.accelerator.is_main_process:
                        self.vllm_client.update_named_param(name, param.data)
                    elif self.vllm_mode == "colocate":
                        llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                        llm_model.load_weights([(name, param.data)])

        if self.vllm_mode == "server" and self.accelerator.is_main_process:
            self.vllm_client.reset_prefix_cache()
        elif self.vllm_mode == "colocate":
            self.llm.reset_prefix_cache()

    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        max_len = self.max_prompt_length or 2048
        data_collator = partial(qwen_mol_grpo_collate_fn, tokenizer=self.processing_class, max_length=max_len)

        if is_datasets_available() and isinstance(train_dataset, Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size * getattr(self.args, "steps_per_generation", 1),
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = partial(
                seed_worker, num_workers=self.args.dataloader_num_workers, rank=self.args.process_index
            )
            if self.args.dataloader_num_workers > 0:
                dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

    def _get_train_sampler(self, dataset: Optional[Dataset] = None) -> Sampler:
        if dataset is None:
            dataset = self.train_dataset

        steps_per_gen = getattr(self.args, "steps_per_generation", 1)
        total_repeats = self.num_iterations * steps_per_gen

        # `generation_batch_size` is global (across all ranks). We sample `generation_batch_size // G` unique prompts,
        # then repeat each prompt `G` times contiguously to form full groups.
        generation_batch_size = getattr(self.args, "generation_batch_size", None)
        if generation_batch_size is None:
            generation_batch_size = self.args.per_device_train_batch_size * self.accelerator.num_processes * steps_per_gen

        return RepeatSampler(
            data_source=dataset,
            mini_repeat_count=self.num_generations,
            batch_size=generation_batch_size // self.num_generations,
            repeat_count=total_repeats,
            shuffle=getattr(self.args, "shuffle_dataset", True),
            seed=self.args.seed,
        )

    def _compute_logps_single_batch(
        self,
        model,
        input_ids,
        attention_mask,
        logits_to_keep,
        compute_entropy,
        **custom_multimodal_inputs,
    ):
        if logits_to_keep == 0:
            device = input_ids.device
            logps = torch.zeros(input_ids.size(0), 0, device=device)
            entropies = torch.zeros(input_ids.size(0), 0, device=device) if compute_entropy else None
            return logps, entropies

        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "use_cache": False,
            **custom_multimodal_inputs,
        }
        if "logits_to_keep" in self.model_kwarg_keys:
            model_inputs["logits_to_keep"] = logits_to_keep + 1

        logits = model(**model_inputs).logits  # (B, L, V)
        logits = logits[:, :-1, :]
        logits = logits[:, -logits_to_keep:, :]
        logits = logits / self.temperature

        completion_ids = input_ids[:, -logits_to_keep:]
        logps = selective_log_softmax(logits, completion_ids)

        entropies = None
        if compute_entropy:
            with torch.no_grad():
                entropies = entropy_from_logits(logits)
        return logps, entropies

    def _get_per_token_logps_and_entropies(
        self,
        model,
        input_ids,
        attention_mask,
        logits_to_keep,
        compute_entropy,
        batch_size=None,
        **custom_multimodal_inputs,
    ):
        batch_size = batch_size or input_ids.size(0)
        all_logps = []
        all_entropies = []
        for start in range(0, input_ids.size(0), batch_size):
            end = start + batch_size
            sliced_multimodal_inputs = {}
            for k, v in custom_multimodal_inputs.items():
                if v is None:
                    sliced_multimodal_inputs[k] = None
                elif isinstance(v, torch.Tensor):
                    sliced_multimodal_inputs[k] = v[start:end]
                elif isinstance(v, list):
                    sliced_multimodal_inputs[k] = v[start:end]
                elif isinstance(v, dict):
                    sliced_multimodal_inputs[k] = {kk: (vv[start:end] if isinstance(vv, torch.Tensor) else vv) for kk, vv in v.items()}
                else:
                    sliced_multimodal_inputs[k] = v

            # Bio-LatentCOT wrapper models prepend molecule embeddings, so completion logits must be indexed
            # with per-example prompt length + molecule prefix length (prompts may be variable length after padding).
            smiles = sliced_multimodal_inputs.get("smiles", None)
            model_to_check = self.accelerator.unwrap_model(model)
            has_num_queries = hasattr(model_to_check, "num_queries") or hasattr(model, "num_queries")

            if logits_to_keep == 0:
                logps = torch.zeros(input_ids[start:end].size(0), 0, device=input_ids.device)
                entropies = torch.zeros(input_ids[start:end].size(0), 0, device=input_ids.device) if compute_entropy else None
            elif smiles is not None and has_num_queries:
                input_ids_batch = input_ids[start:end]
                attention_mask_batch = attention_mask[start:end]

                model_inputs = {
                    "input_ids": input_ids_batch,
                    "attention_mask": attention_mask_batch,
                    "use_cache": False,
                    "smiles": smiles,
                }
                logits_full = model(**model_inputs).logits  # (B, L_fused, V)
                logits_full = logits_full[:, :-1, :]  # next-token shift

                completion_ids = input_ids_batch[:, -logits_to_keep:]  # (B, T)
                completion_attn = attention_mask_batch[:, -logits_to_keep:]  # (B, T)

                completion_lens = completion_attn.sum(dim=1).to(torch.long)
                text_lens = attention_mask_batch.sum(dim=1).to(torch.long)
                prompt_lens = (text_lens - completion_lens).clamp(min=0)

                num_queries = int(getattr(model_to_check, "num_queries", getattr(model, "num_queries", 0)))
                mol_prefix_lens = torch.tensor(
                    [(len(s) if isinstance(s, list) else 0) * (num_queries + 2) for s in smiles],
                    device=logits_full.device,
                    dtype=torch.long,
                )

                # Defensive: ensure mol_prefix_lens matches the current mini-batch size
                if mol_prefix_lens.size(0) > logits_full.size(0):
                    mol_prefix_lens = mol_prefix_lens[: logits_full.size(0)]

                j = torch.arange(logits_to_keep, device=logits_full.device, dtype=torch.long).unsqueeze(0)
                pos = mol_prefix_lens.unsqueeze(1) + prompt_lens.unsqueeze(1) + j - 1
                pos = pos.clamp(min=0, max=logits_full.size(1) - 1)
                batch_idx = torch.arange(logits_full.size(0), device=logits_full.device, dtype=torch.long).unsqueeze(1)
                logits = logits_full[batch_idx, pos]  # (B, T, V)
                logits = logits / self.temperature

                logps = selective_log_softmax(logits, completion_ids) * completion_attn.to(torch.float32)
                entropies = None
                if compute_entropy:
                    with torch.no_grad():
                        entropies = entropy_from_logits(logits) * completion_attn.to(torch.float32)
            else:
                logps, entropies = self._compute_logps_single_batch(
                    model,
                    input_ids[start:end],
                    attention_mask[start:end],
                    logits_to_keep,
                    compute_entropy,
                    **sliced_multimodal_inputs,
                )
            all_logps.append(logps)
            if compute_entropy:
                all_entropies.append(entropies)
        logps = torch.cat(all_logps, dim=0)
        entropies = torch.cat(all_entropies, dim=0) if compute_entropy else None
        return logps, entropies

    def _get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep, **custom_multimodal_inputs):
        logps, _ = self._get_per_token_logps_and_entropies(
            model,
            input_ids,
            attention_mask,
            logits_to_keep=logits_to_keep,
            compute_entropy=False,
            **custom_multimodal_inputs,
        )
        return logps

    def _prepare_inputs(self, generation_batch: Dict[str, Union[torch.Tensor, Any]]) -> Dict[str, Union[torch.Tensor, Any]]:
        mode = "train" if self.model.training else "eval"
        if mode == "train":
            steps_per_gen = getattr(self.args, "steps_per_generation", 1)
            # Generate every (steps_per_gen * num_iterations) steps
            # - steps_per_gen: split large generation batch into smaller gradient steps
            # - num_iterations: reuse same rollouts for multiple policy updates (PPO-style)
            generate_every = steps_per_gen * self.num_iterations
            
            if self._step % generate_every == 0 or self._buffered_inputs is None:
                # Generate and score completions for the incoming batch
                generation_batch = self._generate_and_score_completions(generation_batch)

                # Shuffle for better gradient variance
                batch_size = len(generation_batch["advantages"])
                perm = list(range(batch_size))
                random.shuffle(perm)
                for k, v in generation_batch.items():
                    if isinstance(v, torch.Tensor):
                        generation_batch[k] = v[perm]
                    elif isinstance(v, list):
                        generation_batch[k] = [v[i] for i in perm]

                # Split the batch into steps_per_gen smaller chunks for gradient computation
                # Each chunk will be used num_iterations times
                generation_batches = _split_batch_dict(generation_batch, steps_per_gen)

                # Detach tensors to save memory and prevent backprop through old computation graphs
                self._buffered_inputs = []
                for b in generation_batches:
                    detached = {}
                    for k, v in b.items():
                        detached[k] = v.detach() if isinstance(v, torch.Tensor) else v
                    self._buffered_inputs.append(detached)

            # Return the appropriate buffered chunk
            # Cycles through chunks (steps_per_gen times) and repeats (num_iterations times)
            inputs = self._buffered_inputs[self._step % steps_per_gen]
            self._step += 1
        else:
            inputs = self._generate_and_score_completions(generation_batch)
        return inputs

    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        device = self.accelerator.device
        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)

        # Repeat other columns to match num generations already handled upstream; keep generic.
        keys = [k for k in inputs.keys() if k not in ["prompt", "completion", "completion_ids"]]
        reward_kwargs = {k: inputs[k] for k in keys}
        reward_kwargs["trainer_state"] = self.state

        for i, (rf, rpc, name) in enumerate(zip(self.reward_funcs, self.reward_processing_classes, self.reward_func_names)):
            if isinstance(rf, nn.Module):
                texts = [p + c for p, c in zip(prompts, completions)]
                reward_inputs = rpc(text=texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=True)
                reward_inputs = super()._prepare_inputs(reward_inputs)
                with torch.inference_mode():
                    rewards_per_func[:, i] = rf(**reward_inputs).logits[:, 0]
            else:
                out = rf(prompts=prompts, completions=completions, completion_ids=completion_ids_list, **reward_kwargs)
                out = [r if r is not None else torch.nan for r in out]
                rewards_per_func[:, i] = torch.tensor(out, dtype=torch.float32, device=device)

        rewards_per_func = gather(rewards_per_func)
        return rewards_per_func

    def _generate_and_score_completions(self, inputs: Dict[str, Union[torch.Tensor, Any]]) -> Dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device

        prompts_text = inputs["prompt"]
        original_prompts = inputs.get("original_prompts", prompts_text)
        smiles = inputs.get("smiles")

        prompt_ids = inputs["input_ids"].to(device)
        prompt_mask = inputs["attention_mask"].to(device)

        if self.use_vllm:
            if smiles is None:
                raise ValueError("use_vllm=True requires `smiles` in the batch.")
            if self.vllm_mode == "server":
                # NOTE: server mode would require a VLLMClient endpoint that supports prompt_embeds.
                # We keep this explicit to avoid silent wrong behavior.
                raise NotImplementedError(
                    "vLLM server mode is not implemented for smiles-based prompt_embeds. Use vllm_mode='colocate'."
                )

            # Sync weights if needed
            if self.state.global_step != self._last_loaded_step:
                self._move_model_to_vllm()
                self._last_loaded_step = self.state.global_step

            # Build prompt embeddings (including molecule fusion + latent feedback) on HF side.
            wrapper = self.accelerator.unwrap_model(self.model)
            if not hasattr(wrapper, "get_prompt_embeddings"):
                raise AttributeError(
                    "Model must implement `get_prompt_embeddings(smiles_list, input_ids, attention_mask, ...)` "
                    "to support vLLM prompt_embeds generation."
                )

            # TP gather: vLLM TP groups need all prompts for the group.
            if self.vllm_tensor_parallel_size > 1:
                if self.tp_group is None:
                    raise RuntimeError("TP group not initialized.")
                gathered_prompt_ids = [torch.empty_like(prompt_ids) for _ in range(self.vllm_tensor_parallel_size)]
                gathered_prompt_mask = [torch.empty_like(prompt_mask) for _ in range(self.vllm_tensor_parallel_size)]
                torch.distributed.all_gather(gathered_prompt_ids, prompt_ids, group=self.tp_group)
                torch.distributed.all_gather(gathered_prompt_mask, prompt_mask, group=self.tp_group)

                gathered_smiles = [None for _ in range(self.vllm_tensor_parallel_size)]
                torch.distributed.all_gather_object(gathered_smiles, smiles, group=self.tp_group)
                all_smiles = [s for sub in gathered_smiles for s in sub]

                all_prompt_ids = torch.cat(gathered_prompt_ids, dim=0)
                all_prompt_mask = torch.cat(gathered_prompt_mask, dim=0)
                orig_size = len(prompts_text)
            else:
                all_prompt_ids = prompt_ids
                all_prompt_mask = prompt_mask
                all_smiles = smiles
                orig_size = len(prompts_text)

            if self.vllm_enable_sleep_mode:
                torch.cuda.empty_cache()
                self.llm.wake_up(level=1)

            with torch.inference_mode():
                prompt_embeds, prompt_attn = wrapper.get_prompt_embeddings(
                    smiles_list=all_smiles,
                    input_ids=all_prompt_ids,
                    attention_mask=all_prompt_mask,
                    refine_bio_tokens=True,
                )
            # Trim padding per sample and feed vLLM prompt_embeds
            embed_list = [prompt_embeds[i][prompt_attn[i].bool()].contiguous() for i in range(prompt_embeds.size(0))]

            from vllm import SamplingParams

            sampling_params = SamplingParams(
                n=1,
                repetition_penalty=self.repetition_penalty,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                min_p=0.0 if self.min_p is None else self.min_p,
                max_tokens=self.max_completion_length,
                logprobs=0,
            )

            start = time.time()
            all_outputs = self.llm.generate(
                [{"prompt_embeds": e} for e in embed_list],
                sampling_params=sampling_params,
                use_tqdm=False,
            )
            end = time.time()
            if self.accelerator.is_main_process:
                print(f"vLLM generation time: {end - start:.6f} seconds")

            all_completion_ids = [output.token_ids for outputs in all_outputs for output in outputs.outputs]

            if self.vllm_tensor_parallel_size > 1:
                local_rank_in_group = torch.distributed.get_rank(group=self.tp_group)
                tp_slice = slice(local_rank_in_group * orig_size, (local_rank_in_group + 1) * orig_size)
                completion_ids_list = all_completion_ids[tp_slice]
            else:
                completion_ids_list = all_completion_ids

            completion_ids = pad(
                [torch.tensor(ids, device=device, dtype=torch.long) for ids in completion_ids_list],
                padding_value=self.pad_token_id,
            )
            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        else:
            # Build prompt embeddings (including molecule fusion + latent feedback) on HF side.
            wrapper = self.accelerator.unwrap_model(self.model)
            if not hasattr(wrapper, "get_prompt_embeddings"):
                raise AttributeError(
                    "Model must implement `get_prompt_embeddings(smiles_list, input_ids, attention_mask, ...)` "
                    "to support GRPO generation."
                )

            with torch.no_grad():
                prompt_embeds, prompt_attn_fused = wrapper.get_prompt_embeddings(
                    smiles_list=smiles,
                    input_ids=prompt_ids,
                    attention_mask=prompt_mask,
                    refine_bio_tokens=True,
                )
                fused_prompt_len = prompt_embeds.size(1)

                start = time.time()
                # Use return_dict_in_generate=True to disambiguate completion-only vs prompt+completion
                out = wrapper.model.generate(
                    inputs_embeds=prompt_embeds,
                    attention_mask=prompt_attn_fused,
                    generation_config=self.generation_config,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
                end = time.time()
                if self.accelerator.is_main_process:
                    print(f"Generation time: {end - start:.6f} seconds")

            prompt_completion_ids = out.sequences
            gen_len = len(out.scores)

            # Robust disambiguation:
            if prompt_completion_ids.size(1) == gen_len:
                # sequences is completion-only
                completion_ids = prompt_completion_ids
            elif prompt_completion_ids.size(1) == fused_prompt_len + gen_len:
                # sequences is fused_prompt + completion
                completion_ids = prompt_completion_ids[:, fused_prompt_len:]
            else:
                raise RuntimeError(
                    f"Unexpected generate() output shape: seqs={prompt_completion_ids.size(1)}, "
                    f"fused_prompt={fused_prompt_len}, gen_len={gen_len}. "
                    "This usually means the model's generate() logic is inconsistent with inputs_embeds."
                )

            # We always catenate with the original TEXT prompt IDs for the rest of the trainer logic
            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)

        # Mask everything after first EOS
        is_eos = completion_ids == self.eos_token_id
        if is_eos.size(1) > 0:
            eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
            mask_has_eos = is_eos.any(dim=1)
            eos_idx[mask_has_eos] = is_eos.int().argmax(dim=1)[mask_has_eos]
            seq_idx = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
            completion_mask = (seq_idx <= eos_idx.unsqueeze(1)).int()
        else:
            completion_mask = torch.zeros_like(completion_ids, dtype=torch.int)

        completion_ids_list = [row[mask_row].tolist() for row, mask_row in zip(completion_ids, completion_mask.bool())]

        completion_lengths = completion_mask.sum(1)
        agg_completion_lengths = self.accelerator.gather(completion_lengths)

        if self.mask_truncated_completions:
            truncated = ~is_eos.any(dim=1)
            completion_mask = completion_mask * (~truncated).unsqueeze(1).int()

        # Full sequence attention mask for logp computation
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        mode = "train" if self.model.training else "eval"
        batch_size = self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size

        with torch.no_grad():
            # If num_iterations > 1, we MUST store the logprobs from the sampling policy
            # to maintain a correct PPO/GRPO ratio as the model weights update.
            must_store_old_logps = self.num_iterations > 1
            generate_every = self.args.steps_per_generation * self.num_iterations

            if self.use_vllm or must_store_old_logps or (self.args.gradient_accumulation_steps % generate_every != 0):
                old_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                    model=self.model,
                    input_ids=prompt_completion_ids,
                    attention_mask=attention_mask,
                    compute_entropy=False,
                    smiles=smiles,
                    logits_to_keep=logits_to_keep,
                    batch_size=batch_size,
                )
            else:
                old_per_token_logps = None

            if self.beta != 0.0:
                wrapper = self.accelerator.unwrap_model(self.model)
                inner = getattr(wrapper, "model", None)
                if inner is None or not hasattr(inner, "disable_adapter"):
                    raise RuntimeError(
                        "beta != 0.0 requires the inner PEFT model to expose `.disable_adapter()` to compute reference "
                        "log-probs without allocating a second multimodal model."
                    )
                with inner.disable_adapter():
                    ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                        model=self.model,
                        input_ids=prompt_completion_ids,
                        attention_mask=attention_mask,
                        compute_entropy=False,
                        smiles=smiles,
                        logits_to_keep=logits_to_keep,
                        batch_size=batch_size,
                    )
            else:
                ref_per_token_logps = None

        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        completions = completions_text

        rewards_per_func = self._calculate_rewards(inputs, original_prompts, completions, completion_ids_list)
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = rewards - mean_grouped_rewards

        if self.scale_rewards in ["group", "none"]:
            std_rewards = rewards.view(-1, self.num_generations).std(dim=1)
            std_rewards = std_rewards.repeat_interleave(self.num_generations, dim=0)
        elif self.scale_rewards == "batch":
            std_rewards = rewards.std().expand_as(rewards)
        else:
            raise ValueError("scale_rewards must be one of 'batch', 'group', 'none'.")

        is_std_zero = torch.isclose(std_rewards, torch.zeros_like(std_rewards))
        if self.scale_rewards != "none":
            advantages = advantages / (std_rewards + 1e-4)
        if torch.all(advantages == 0):
            advantages = advantages + 1e-6

        # Keep only local slice
        process_slice = slice(
            self.accelerator.process_index * len(prompts_text),
            (self.accelerator.process_index + 1) * len(prompts_text),
        )
        all_process_advantages = advantages.clone()
        advantages = advantages[process_slice]

        # Logging
        if mode == "train":
            self.state.num_input_tokens_seen += self.accelerator.gather(attention_mask.sum()).sum().item()
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]
        self._metrics[mode]["completions/mean_length"].append(agg_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_length"].append(agg_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_length"].append(agg_completion_lengths.float().max().item())
        for i, name in enumerate(self.reward_func_names):
            self._metrics[mode][f"rewards/{name}/mean"].append(torch.nanmean(rewards_per_func[:, i]).item())
            self._metrics[mode][f"rewards/{name}/std"].append(nanstd(rewards_per_func[:, i]).item())
        self._metrics[mode]["reward"].append(mean_grouped_rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_rewards.mean().item())
        self._metrics[mode]["frac_reward_zero_std"].append(is_std_zero.float().mean().item())

        self._logs["prompt"].extend(gather_object(prompts_text))
        self._logs["completion"].extend(gather_object(completions_text))
        for i, name in enumerate(self.reward_func_names):
            self._logs["rewards"][name].extend(rewards_per_func[:, i].tolist())
        self._logs["advantages"].extend(all_process_advantages.tolist())

        output = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "smiles": smiles,
            "advantages": advantages,
            "multimodal_inputs": {"smiles": smiles},
        }
        if old_per_token_logps is not None:
            output["old_per_token_logps"] = old_per_token_logps
        if ref_per_token_logps is not None:
            output["ref_per_token_logps"] = ref_per_token_logps
        return output

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("GRPOTrainer does not support return_outputs in this fork.")

        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        multimodal_inputs = inputs["multimodal_inputs"]

        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)

        logits_to_keep = completion_ids.size(1)
        per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, logits_to_keep, **multimodal_inputs)

        advantages = inputs["advantages"]
        old_per_token_logps = inputs.get("old_per_token_logps", per_token_logps.detach())

        # Shape safety check
        if per_token_logps.shape != completion_mask.shape:
            print(f"DEBUG compute_loss Rank={self.accelerator.process_index}")
            print(f"  per_token_logps shape: {per_token_logps.shape}")
            print(f"  completion_mask shape: {completion_mask.shape}")
            print(f"  completion_ids shape: {completion_ids.shape}")
            print(f"  prompt_ids shape: {prompt_ids.shape}")
            print(f"  logits_to_keep: {logits_to_keep}")
            raise RuntimeError(
                f"Shape mismatch in compute_loss: per_token_logps={per_token_logps.shape}, "
                f"completion_mask={completion_mask.shape}. multimodal_inputs keys={list(multimodal_inputs.keys())}"
            )

        eps = getattr(self.args, "epsilon", 0.2)
        eps_high = getattr(self.args, "epsilon_high", eps)
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - eps, 1 + eps_high)
        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

        # KL
        if self.beta > 0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            per_token_loss = per_token_loss + self.beta * per_token_kl
            mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
            mode = "train" if model.training else "eval"
            self._metrics[mode]["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

        # Clipping is active when the clipped objective is the minimum.
        is_clipped = (per_token_loss2 < per_token_loss1).float()
        clip_ratio = (is_clipped * completion_mask).sum() / completion_mask.sum()
        mode = "train" if model.training else "eval"
        self._metrics[mode]["clip_ratio"].append(self.accelerator.gather_for_metrics(clip_ratio).mean().item())
        return loss

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {}
        for mode, mode_metrics in self._metrics.items():
            for key, val in mode_metrics.items():
                if len(val) > 0:
                    metrics[f"{mode}/{key}"] = sum(val) / len(val)
        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:
            super().log(logs)
        for mode_metrics in self._metrics.values():
            mode_metrics.clear()
