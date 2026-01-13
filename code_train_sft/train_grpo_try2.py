import os
import sys
import argparse
import logging
import re
import math
import json
from datetime import datetime
from typing import List, Optional

import torch
from peft import LoraConfig, TaskType, get_peft_model, PeftModel

from config import ModelConfig
from dataloader import load_grpo_data
from model_stage3 import Qwen3MoleculeLLM
from trainer_try2.grpo_trainer import QwenMoleculeGRPOTrainer
from trainer_try2.grpo_config import GRPOConfig


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

    # IMPORTANT: we froze everything above, which also freezes LoRA params loaded from checkpoint.
    # GRPO needs the policy parameters (LoRA adapters) to require grad; otherwise loss will be detached.
    lora_param_count = 0
    for name, p in model.model.named_parameters():
        if "lora_" in name or "modules_to_save" in name:
            p.requires_grad = True
            lora_param_count += p.numel()
    if lora_param_count == 0:
        logger.warning("No LoRA parameters were marked trainable; GRPO may fail (detached loss).")

    # Multimodal heads trainable
    for p in model.projector.parameters():
        p.requires_grad = True
    for p in model.bio_updater.parameters():
        p.requires_grad = True
    if getattr(model, "is_both_latent", False):
        if hasattr(model, "bio_thinker"):
            for p in model.bio_thinker.parameters():
                p.requires_grad = True
        if hasattr(model, "task_thinker"):
            for p in model.task_thinker.parameters():
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
        if hasattr(model, "bio_thinker"):
            model.bio_thinker.load_state_dict(checkpoint.get("bio_thinker", {}), strict=False)
        if hasattr(model, "task_thinker"):
            model.task_thinker.load_state_dict(checkpoint.get("task_thinker", {}), strict=False)
        logger.info("Loaded projector (+ bio_updater if present).")

    return model


def format_reward_answer_tag(
    prompts: List[str],
    completions: List[str],
    completion_ids=None,
    corrupt_task_latents: Optional[List[bool]] = None,
    **kwargs,
):
    """
    Minimal "try1" reward:
    - reward 1.0 if model outputs a non-empty `<answer> ... </answer>` span
    - else 0.0
    """
    if corrupt_task_latents is None:
        corrupt_flags = None
    elif isinstance(corrupt_task_latents, torch.Tensor):
        corrupt_flags = [bool(x) for x in corrupt_task_latents.detach().cpu().tolist()]
    else:
        corrupt_flags = [bool(x) for x in corrupt_task_latents]

    rewards = []
    for i, c in enumerate(completions):
        if corrupt_flags is not None and i < len(corrupt_flags) and corrupt_flags[i]:
            rewards.append(0.0)
            continue
        c = c or ""
        lo = c.lower()
        start = lo.find("<answer>")
        end = lo.find("</answer>")
        if start == -1 or end == -1 or end <= start:
            rewards.append(0.0)
            continue
        try:
            inner = c[start + len("<answer>") : end].strip()
            rewards.append(1.0 if len(inner) > 0 else 0.0)
        except Exception:
            rewards.append(0.0)
    return rewards


_ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", flags=re.IGNORECASE | re.DOTALL)

_PROMPT_EXPECT_RE = re.compile(
    r"formatted as\s*<answer>\s*(.*?)\s*</answer>", flags=re.IGNORECASE | re.DOTALL
)

_RDKit_LOGS_DISABLED = False


def _get_rdkit_chem():
    global _RDKit_LOGS_DISABLED
    try:
        from rdkit import Chem  # type: ignore
        from rdkit import RDLogger  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("RDKit is required for SMILES rewards.") from e

    if not _RDKit_LOGS_DISABLED:
        RDLogger.DisableLog("rdApp.error")
        RDLogger.DisableLog("rdApp.warning")
        _RDKit_LOGS_DISABLED = True
    return Chem


def _is_smiles_task(prompt: str) -> bool:
    # `extract_fields()` rewrites many tasks to include:
    # "Your final answer must be formatted as <answer> SMILES </answer>"
    return bool(re.search(r"<answer>\s*smiles\b", prompt, flags=re.IGNORECASE))


def _infer_expected_answer_type(prompt: str) -> str:
    """
    Infer the expected answer type from the prompt formatting instruction inserted by `extract_fields()`.

    Returns: one of {"smiles", "number", "yesno", "unknown"}.
    """
    m = _PROMPT_EXPECT_RE.search(prompt or "")
    if not m:
        return "unknown"
    spec = (m.group(1) or "").strip().lower()
    if "smiles" in spec:
        return "smiles"
    if "yes" in spec and "no" in spec:
        return "yesno"
    if "number" in spec:
        return "number"
    return "unknown"


def _extract_answer_text(completion: str) -> Optional[str]:
    m = _ANSWER_RE.search(completion or "")
    if not m:
        return None
    return m.group(1).strip()


def reward_answer_type_validity(
    prompts: List[str],
    completions: List[str],
    completion_ids=None,
    corrupt_task_latents: Optional[List[bool]] = None,
    **kwargs,
):
    """
    Reward 0.5 if the `<answer>...</answer>` content matches the *type* the prompt asks for:
    - SMILES: RDKit parses
    - Number: parses as float
    - Yes/No: matches yes/no (case-insensitive)
    Otherwise reward 0.0.
    """
    if corrupt_task_latents is None:
        corrupt_flags = None
    elif isinstance(corrupt_task_latents, torch.Tensor):
        corrupt_flags = [bool(x) for x in corrupt_task_latents.detach().cpu().tolist()]
    else:
        corrupt_flags = [bool(x) for x in corrupt_task_latents]

    rewards: list[float] = []
    for i, (p, c) in enumerate(zip(prompts, completions)):
        if corrupt_flags is not None and i < len(corrupt_flags) and corrupt_flags[i]:
            rewards.append(0.0)
            continue
        expected = _infer_expected_answer_type(p)
        answer = _extract_answer_text(c or "")
        if not answer:
            rewards.append(0.0)
            continue

        cleaned = answer.strip().strip("\"'`")
        if expected == "smiles":
            Chem = _get_rdkit_chem()

            cleaned = re.sub(r"^\s*smiles\s*[:=]\s*", "", cleaned, flags=re.IGNORECASE).strip()
            candidates = [cleaned]
            if re.search(r"\s", cleaned):
                candidates.append(cleaned.split()[0])

            ok = False
            for cand in candidates:
                try:
                    mol = Chem.MolFromSmiles(cand)  # type: ignore[attr-defined]
                except Exception:
                    mol = None
                if mol is not None:
                    ok = True
                    break
            rewards.append(0.5 if ok else 0.0)
        elif expected == "number":
            cleaned = re.sub(r"^\s*(count|number)\s*[:=]\s*", "", cleaned, flags=re.IGNORECASE).strip()
            cleaned = cleaned.replace(",", "")
            try:
                float(cleaned)
                rewards.append(0.5)
            except Exception:
                rewards.append(0.0)
        elif expected == "yesno":
            lo = cleaned.lower()
            lo = re.sub(r"^\s*(answer|output)\s*[:=]\s*", "", lo).strip()
            if lo in {"yes", "no"}:
                rewards.append(0.5)
            else:
                rewards.append(0.0)
        else:
            rewards.append(0.0)

    return rewards


def reward_answer_correctness(
    prompts: List[str],
    completions: List[str],
    completion_ids=None,
    label: Optional[List[str]] = None,
    labels: Optional[List[str]] = None,
    **kwargs,
):
    """
    Reward 1.0 if the extracted `<answer>...</answer>` matches the ground-truth label.

    Requires `load_grpo_data()` to keep the `label` column.
    """
    gt_list = labels if labels is not None else label
    if gt_list is None:
        # If labels are not present in the dataset, this reward is disabled.
        return [0.0 for _ in completions]

    rewards: list[float] = []
    for p, c, gt in zip(prompts, completions, gt_list):
        expected = _infer_expected_answer_type(p)
        pred = _extract_answer_text(c or "")
        gold = _extract_answer_text(gt or "") or (gt or "").strip()
        if pred is None:
            rewards.append(0.0)
            continue

        pred_clean = pred.strip().strip("\"'`")
        gold_clean = (gold or "").strip().strip("\"'`")

        if expected == "smiles":
            Chem = _get_rdkit_chem()

            def canon(s: str) -> Optional[str]:
                try:
                    mol = Chem.MolFromSmiles(s)  # type: ignore[attr-defined]
                except Exception:
                    mol = None
                if mol is None:
                    return None
                try:
                    return Chem.MolToSmiles(mol, canonical=True)  # type: ignore[attr-defined]
                except Exception:
                    return None

            pred_can = canon(pred_clean)
            gold_can = canon(gold_clean)
            rewards.append(1.0 if (pred_can is not None and gold_can is not None and pred_can == gold_can) else 0.0)
        elif expected == "number":
            def parse_num(s: str) -> Optional[float]:
                try:
                    return float(s)
                except Exception:
                    return None

            pn = parse_num(pred_clean)
            gn = parse_num(gold_clean)
            if pn is None or gn is None:
                rewards.append(0.0)
            else:
                rewards.append(1.0 if math.isclose(pn, gn, rel_tol=1e-3, abs_tol=1e-3) else 0.0)
        elif expected == "yesno":
            def norm_yesno(s: str) -> Optional[str]:
                s = s.strip().lower()
                if s in {"yes"}:
                    return "yes"
                if s in {"no"}:
                    return "no"
                return None

            py = norm_yesno(pred_clean)
            gy = norm_yesno(gold_clean)
            rewards.append(1.0 if (py is not None and gy is not None and py == gy) else 0.0)
        else:
            rewards.append(1.0 if pred_clean.strip().lower() == gold_clean.strip().lower() and gold_clean != "" else 0.0)

    return rewards


_BENCH_REWARD_UTILS_ON_PATH = False
_MOLOPT_EVALUATER_CACHE: dict[str, object] = {}


def _ensure_bench_reward_utils_importable() -> None:
    global _BENCH_REWARD_UTILS_ON_PATH
    if _BENCH_REWARD_UTILS_ON_PATH:
        return
    current_dir = os.path.dirname(os.path.abspath(__file__))
    reward_utils_dir = os.path.join(current_dir, "reward_utils")
    if reward_utils_dir not in sys.path:
        sys.path.append(reward_utils_dir)
    _BENCH_REWARD_UTILS_ON_PATH = True


def _clean_group_name(group: Optional[str]) -> Optional[str]:
    if group is None:
        return None
    # Some benchmark meta strings include a trailing '.' (e.g., "carboxyl.")
    return str(group).strip().rstrip(".").strip()


def _extract_smiles_candidate(answer_text: str) -> str:
    s = (answer_text or "").strip().strip("\"'`")
    s = re.sub(r"^\s*smiles\s*[:=]\s*", "", s, flags=re.IGNORECASE).strip()
    if re.search(r"\s", s):
        s = s.split()[0]
    return s.strip().rstrip(".").strip()


def reward_answer_correctness_bench(
    prompts: List[str],
    completions: List[str],
    completion_ids=None,
    label: Optional[List[str]] = None,
    labels: Optional[List[str]] = None,
    corrupt_task_latents: Optional[List[bool]] = None,
    task: Optional[List[str]] = None,
    subtask: Optional[List[str]] = None,
    meta: Optional[List[str]] = None,
    **kwargs,
):
    """
    Benchmark-aware correctness reward for ChemCoTBench-style data.

    - For numeric and yes/no answers: keep the current strict matching logic.
    - For SMILES answers:
        - mol_edit: check functional-group edit validity (no fixed target SMILES)
        - mol_opt: check property improvement using benchmark oracle
        - mol_und/reaction: exact match (as before)

    Returns:
      4.0 if correct, else 0.0
    """
    gt_list = labels if labels is not None else label

    if corrupt_task_latents is None:
        corrupt_flags = None
    elif isinstance(corrupt_task_latents, torch.Tensor):
        corrupt_flags = [bool(x) for x in corrupt_task_latents.detach().cpu().tolist()]
    else:
        corrupt_flags = [bool(x) for x in corrupt_task_latents]

    rewards: list[float] = []
    for i, (p, c) in enumerate(zip(prompts, completions, strict=True)):
        if corrupt_flags is not None and i < len(corrupt_flags) and corrupt_flags[i]:
            rewards.append(0.0)
            continue
        expected = _infer_expected_answer_type(p)
        pred = _extract_answer_text(c or "")
        if pred is None:
            rewards.append(0.0)
            continue

        pred_clean = _extract_smiles_candidate(pred) if expected == "smiles" else pred.strip().strip("\"'`")

        # Pull per-sample benchmark routing info when available
        t = task[i] if task is not None and i < len(task) else None
        st = subtask[i] if subtask is not None and i < len(subtask) else None
        m = meta[i] if meta is not None and i < len(meta) else None

        if expected == "smiles":
            # 1) Editing benchmark (functional-group constraints)
            if t == "mol_edit":
                ok = False
                try:
                    meta_dict = json.loads(m) if isinstance(m, str) and m else {}
                except Exception:
                    meta_dict = {}

                src = meta_dict.get("molecule")
                if isinstance(src, str) and src.strip():
                    _ensure_bench_reward_utils_importable()
                    try:
                        from ChemCoTBench.eval_moledit import (  # type: ignore
                            check_edit_add_valid,
                            check_edit_del_valid,
                            check_edit_sub_valid,
                        )
                    except Exception:
                        check_edit_add_valid = check_edit_del_valid = check_edit_sub_valid = None  # type: ignore

                    try:
                        if st == "add" and check_edit_add_valid is not None:
                            group = _clean_group_name(meta_dict.get("added_group"))
                            if group:
                                ok = bool(check_edit_add_valid(src=src, tgt=pred_clean, group=group))
                        elif st == "delete" and check_edit_del_valid is not None:
                            group = _clean_group_name(meta_dict.get("removed_group"))
                            if group:
                                ok = bool(check_edit_del_valid(src=src, tgt=pred_clean, group=group))
                        elif st == "sub" and check_edit_sub_valid is not None:
                            add_group = _clean_group_name(meta_dict.get("added_group"))
                            remove_group = _clean_group_name(meta_dict.get("removed_group"))
                            if add_group and remove_group:
                                ok = bool(
                                    check_edit_sub_valid(
                                        src=src, tgt=pred_clean, remove_group=remove_group, add_group=add_group
                                    )
                                )
                    except Exception:
                        ok = False

                rewards.append(4.0 if ok else 0.0)
                continue

            # 2) Molecule optimization benchmark (oracle-based improvement)
            if t == "mol_opt":
                ok = False
                try:
                    meta_dict = json.loads(m) if isinstance(m, str) and m else {}
                except Exception:
                    meta_dict = {}

                src = meta_dict.get("molecule")
                prop_dict = {
                    "logp": "logp",
                    "solubility": "solubility",
                    "qed": "qed",
                    "drd": "drd2",
                    "jnk": "jnk3",
                    "gsk": "gsk3b",
                }
                prop = prop_dict.get(str(st or "").strip().lower())
                if isinstance(src, str) and src.strip() and prop:
                    _ensure_bench_reward_utils_importable()
                    try:
                        from ChemCoTBench.core.eval_metric import mol_opt_evaluater  # type: ignore

                        evaluater = _MOLOPT_EVALUATER_CACHE.get(prop)
                        if evaluater is None:
                            evaluater = mol_opt_evaluater(prop=prop)
                            _MOLOPT_EVALUATER_CACHE[prop] = evaluater

                        oracle = getattr(evaluater, "property_oracle", None)
                        if oracle is not None:
                            src_score = oracle(src)
                            tgt_score = oracle(pred_clean)
                            if src_score is not None and tgt_score is not None:
                                ok = bool((tgt_score - src_score) > 0)
                    except Exception:
                        ok = False

                rewards.append(4.0 if ok else 0.0)
                continue

            # 3) Default SMILES: exact match (canonicalized)
            if gt_list is None:
                rewards.append(0.0)
                continue
            gt = gt_list[i] if i < len(gt_list) else ""
            gold = _extract_answer_text(gt or "") or (gt or "").strip()
            gold_clean = _extract_smiles_candidate(gold)

            Chem = _get_rdkit_chem()

            def canon(s: str) -> Optional[str]:
                try:
                    mol = Chem.MolFromSmiles(s)  # type: ignore[attr-defined]
                except Exception:
                    mol = None
                if mol is None:
                    return None
                try:
                    return Chem.MolToSmiles(mol, canonical=True)  # type: ignore[attr-defined]
                except Exception:
                    return None

            pred_can = canon(pred_clean)
            gold_can = canon(gold_clean)
            rewards.append(4.0 if (pred_can is not None and gold_can is not None and pred_can == gold_can) else 0.0)
            continue

        # Non-SMILES: keep original strict matching logic
        if gt_list is None:
            rewards.append(0.0)
            continue
        gt = gt_list[i] if i < len(gt_list) else ""
        gold = _extract_answer_text(gt or "") or (gt or "").strip()

        pred_clean = pred_clean.strip()
        gold_clean = (gold or "").strip().strip("\"'`")

        if expected == "number":
            def parse_num(s: str) -> Optional[float]:
                try:
                    return float(s)
                except Exception:
                    return None

            pn = parse_num(pred_clean)
            gn = parse_num(gold_clean)
            if pn is None or gn is None:
                rewards.append(0.0)
            else:
                rewards.append(4.0 if math.isclose(pn, gn, rel_tol=1e-3, abs_tol=1e-3) else 0.0)
        elif expected == "yesno":
            def norm_yesno(s: str) -> Optional[str]:
                s = s.strip().lower()
                if s in {"yes"}:
                    return "yes"
                if s in {"no"}:
                    return "no"
                return None

            py = norm_yesno(pred_clean)
            gy = norm_yesno(gold_clean)
            rewards.append(4.0 if (py is not None and gy is not None and py == gy) else 0.0)
        else:
            rewards.append(4.0 if pred_clean.strip().lower() == gold_clean.strip().lower() and gold_clean != "" else 0.0)

    return rewards


def reward_stage4_corrupt_or_correct(
    prompts: List[str],
    completions: List[str],
    completion_ids=None,
    label: Optional[List[str]] = None,
    labels: Optional[List[str]] = None,
    corrupt_task_latents: Optional[List[bool]] = None,
    **kwargs,
):
    """
    Stage 4 reward:
    - If corrupted: reward = -1 if correct else 0
    - If not corrupted: reward = correctness (1 if correct else 0)
    """
    gt_list = labels if labels is not None else label
    if gt_list is None:
        return [0.0 for _ in completions]

    if corrupt_task_latents is None:
        corrupt_task_latents = [False for _ in completions]

    correctness = reward_answer_correctness_bench(
        prompts=prompts,
        completions=completions,
        completion_ids=completion_ids,
        labels=gt_list,
        task=kwargs.get("task"),
        subtask=kwargs.get("subtask"),
        meta=kwargs.get("meta"),
    )
    out: list[float] = []
    for is_corrupt, corr in zip(corrupt_task_latents, correctness, strict=True):
        is_correct = bool(corr >= 0.5)
        if is_corrupt:
            out.append(-1.0 if is_correct else 0.0)
        else:
            out.append(1.0 if is_correct else 0.0)
    return out


def reward_stage4_scaled_correctness(
    prompts: List[str],
    completions: List[str],
    completion_ids=None,
    label: Optional[List[str]] = None,
    labels: Optional[List[str]] = None,
    corrupt_task_latents: Optional[List[bool]] = None,
    cot_len: Optional[List[int]] = None,
    **kwargs,
):
    """
    Stage 4 reward (latent-scaled):
    - If corrupted: reward = 0
    - If not corrupted:
        - correct: +4 / N
        - wrong:   -4 / N
      where N = CoT string length (`cot_len`, clamped to >= 1).
    """
    gt_list = labels if labels is not None else label
    if gt_list is None:
        return [0.0 for _ in completions]

    if corrupt_task_latents is None:
        corrupt_task_latents = [False for _ in completions]
    correctness = reward_answer_correctness_bench(
        prompts=prompts,
        completions=completions,
        completion_ids=completion_ids,
        labels=gt_list,
        task=kwargs.get("task"),
        subtask=kwargs.get("subtask"),
        meta=kwargs.get("meta"),
    )
    out: list[float] = []
    for is_corrupt, corr, n in zip(corrupt_task_latents, correctness, cot_len, strict=True):
        if is_corrupt:
            out.append(0.0)
            continue
        n_cot = int(n)
        is_correct = bool(corr >= 0.5)
        out.append((4.0 / n_cot) if is_correct else (-4.0 / n_cot))
    return out


def reward_stage4_double_scaled_correctness(
    prompts: List[str],
    completions: List[str],
    completion_ids=None,
    label: Optional[List[str]] = None,
    labels: Optional[List[str]] = None,
    corrupt_task_latents: Optional[List[bool]] = None,
    task_latent_count: Optional[List[int]] = None,
    cot_len: Optional[List[int]] = None,
    **kwargs,
):
    """
    Stage 4 reward (double-scaled):
    - If corrupted: reward = 0
    - If not corrupted:
        - correct: +100 * (T / C)
        - wrong:   -100 / (T * C)
      where:
        - T = number of task latent tokens (`task_latent_count`, clamped to >= 0 for the correct case, and >= 1 for
              the wrong case to avoid division by zero)
        - C = CoT string length (`cot_len`, clamped to >= 1)
    """
    gt_list = labels if labels is not None else label
    if gt_list is None:
        return [0.0 for _ in completions]

    if corrupt_task_latents is None:
        corrupt_task_latents = [False for _ in completions]
    correctness = reward_answer_correctness_bench(
        prompts=prompts,
        completions=completions,
        completion_ids=completion_ids,
        labels=gt_list,
        task=kwargs.get("task"),
        subtask=kwargs.get("subtask"),
        meta=kwargs.get("meta"),
        corrupt_task_latents=corrupt_task_latents,
    )

    out: list[float] = []
    for is_corrupt, corr, t, c in zip(corrupt_task_latents, correctness, task_latent_count, cot_len, strict=True):
        if is_corrupt:
            out.append(0.0)
            continue

        t_cnt = int(t) 
        c_len = int(c) 

        is_correct = bool(corr >= 0.5)
        if is_correct:
            out.append(100.0 * (float(t_cnt) / float(c_len)))
        else:
            denom = float(max(t_cnt, 1)) * float(c_len)
            out.append(-100.0 / denom)
    return out


def main():
    parser = argparse.ArgumentParser(description="GRPO try1 training for Bio-LatentCOT (smiles-aware, optional vLLM).")
    parser.add_argument(
        "--use_reward_answer_tag",
        type=lambda x: (str(x).lower() == "true"),
        default=False,
        help="Include `format_reward_answer_tag` in reward functions.",
    )
    parser.add_argument(
        "--use_reward_answer_type_validity",
        type=lambda x: (str(x).lower() == "true"),
        default=False,
        help="Include `reward_answer_type_validity` in reward functions.",
    )
    parser.add_argument(
        "--use_reward_answer_correctness_bench",
        type=lambda x: (str(x).lower() == "true"),
        default=False,
        help="Include `reward_answer_correctness_bench` in reward functions.",
    )
    parser.add_argument(
        "--use_reward_answer_correctness",
        type=lambda x: (str(x).lower() == "true"),
        default=False,
        help="Include legacy `reward_answer_correctness` in reward functions (non-benchmark routing).",
    )
    parser.add_argument(
        "--use_reward_stage4_corrupt_or_correct",
        type=lambda x: (str(x).lower() == "true"),
        default=False,
        help="Include `reward_stage4_corrupt_or_correct` in reward functions (uses `corrupt_prob`).",
    )
    parser.add_argument(
        "--use_reward_stage4_scaled_correctness",
        type=lambda x: (str(x).lower() == "true"),
        default=False,
        help="Include `reward_stage4_scaled_correctness` in reward functions (uses `cot_len`).",
    )
    parser.add_argument(
        "--use_reward_stage4_double_scaled_correctness",
        type=lambda x: (str(x).lower() == "true"),
        default=False,
        help="Include `reward_stage4_double_scaled_correctness` in reward functions (uses `task_latent_count` + `cot_len`).",
    )
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

    # Stage 4: task-latent corruption
    parser.add_argument(
        "--corrupt_prob",
        type=float,
        default=0.0,
        help="Probability to corrupt task latent embeddings per prompt group (stage=4/5 only).",
    )
    parser.add_argument(
        "--corrupt_latent_noise_std",
        type=float,
        default=0.0,
        help="Std of Gaussian noise to replace task latent embeddings (0 -> zeros) (stage=4/5 only).",
    )
    parser.add_argument(
        "--is_both_latent",
        type=lambda x: (str(x).lower() == "true"),
        default=True,
        help="Enable task-latent generation via is_both_latent (stage=4/5 requires true).",
    )
    parser.add_argument(
        "--bio_thinker_dropout",
        type=float,
        default=0.0,
        help="Dropout probability inside bio_thinker (TransformerEncoderLayer).",
    )
    parser.add_argument(
        "--task_thinker_dropout",
        type=float,
        default=0.0,
        help="Dropout probability inside task_thinker (MLP).",
    )

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
    parser.add_argument(
        "--vllm_ckpt",
        type=str,
        default=ModelConfig.DEFAULT_QWEN_PATH,
        help="vLLM base model checkpoint path/name (defaults to ModelConfig.DEFAULT_QWEN_PATH).",
    )
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
    use_both_latent = bool(args.is_both_latent)
    enable_corruption = float(args.corrupt_prob) > 0.0
    if enable_corruption and (not use_both_latent):
        raise ValueError("corrupt_prob>0 requires --is_both_latent true (task latents must exist to corrupt).")
    model = Qwen3MoleculeLLM(
        qwen_model_name=ModelConfig.DEFAULT_QWEN_PATH,
        mol_config=mol_config,
        is_both_latent=use_both_latent,
        is_coconut=False,
        bio_thinker_dropout=float(args.bio_thinker_dropout),
        task_thinker_dropout=float(args.task_thinker_dropout),
    )

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
        ddp_find_unused_parameters=False,
        # Some submodules (e.g. frozen molecule encoder) can contain inference-mode buffers; broadcasting them under
        # DDP may error with "Inplace update to inference tensor". LLM training does not rely on buffer sync.
        ddp_broadcast_buffers=False,
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

    reward_funcs = []
    if bool(args.use_reward_answer_tag):
        reward_funcs.append(format_reward_answer_tag)
    if bool(args.use_reward_answer_type_validity):
        reward_funcs.append(reward_answer_type_validity)
    if bool(args.use_reward_answer_correctness):
        reward_funcs.append(reward_answer_correctness)
    if bool(args.use_reward_answer_correctness_bench):
        reward_funcs.append(reward_answer_correctness_bench)
    if bool(args.use_reward_stage4_corrupt_or_correct):
        reward_funcs.append(reward_stage4_corrupt_or_correct)
    if bool(args.use_reward_stage4_scaled_correctness):
        reward_funcs.append(reward_stage4_scaled_correctness)
    if bool(args.use_reward_stage4_double_scaled_correctness):
        reward_funcs.append(reward_stage4_double_scaled_correctness)

    if not reward_funcs:
        raise ValueError("No reward functions selected. Set at least one `--use_reward_* true` flag.")

    corrupt_prob = float(args.corrupt_prob) if enable_corruption else 0.0
    corrupt_latent_noise_std = float(args.corrupt_latent_noise_std) if enable_corruption else 0.0
    training_stage = 4 if enable_corruption else 3

    trainer = QwenMoleculeGRPOTrainer(
        model=model,
        args=grpo_args,
        reward_funcs=reward_funcs,
        train_dataset=train_dataset,
        processing_class=model.tokenizer,
        training_stage=int(training_stage),
        corrupt_prob=corrupt_prob,
        corrupt_latent_noise_std=corrupt_latent_noise_std,
    )

    trainer.train()

    # Save final LoRA + multimodal heads (compatible with stage checkpoints)
    final_dir = grpo_args.output_dir
    lora_dir = os.path.join(final_dir, "lora_weights")
    os.makedirs(lora_dir, exist_ok=True)
    model.model.save_pretrained(lora_dir)
    mm_path = os.path.join(final_dir, "mm_projector.pt")
    to_save = {"projector": model.projector.state_dict(), "bio_updater": model.bio_updater.state_dict()}
    if hasattr(model, "bio_thinker"):
        to_save["bio_thinker"] = model.bio_thinker.state_dict()
    if hasattr(model, "task_thinker"):
        to_save["task_thinker"] = model.task_thinker.state_dict()
    torch.save(to_save, mm_path)
    model.tokenizer.save_pretrained(final_dir)
    logger.info(f"Saved LoRA to {lora_dir} and mm weights to {mm_path}")


if __name__ == "__main__":
    main()
