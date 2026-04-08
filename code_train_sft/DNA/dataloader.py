import json
import os
from datasets import load_dataset
from transformers import AutoTokenizer
from config import ModelConfig

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

tokenizer = AutoTokenizer.from_pretrained(ModelConfig.DEFAULT_QWEN_PATH)
tokenizer.pad_token = tokenizer.eos_token

COCONUT_TOKENS = {
    "latent": "<latent>",
    "start_latent": "<start_latent>",
    "end_latent": "<end_latent>",
    "mol_start": "<mol_start>",
    "mol_end": "<mol_end>"
}
tokenizer.add_tokens(list(COCONUT_TOKENS.values()))

LATENT_ID = tokenizer.convert_tokens_to_ids(COCONUT_TOKENS["latent"])
START_LATENT_ID = tokenizer.convert_tokens_to_ids(COCONUT_TOKENS["start_latent"])
END_LATENT_ID = tokenizer.convert_tokens_to_ids(COCONUT_TOKENS["end_latent"])

MAX_LEN = ModelConfig.MAX_TEXT_LEN

prompt_pattern = "You are a chemical assistent. Given a pair of reference and variant DNA sequences of a gene, your task is to decide the biological effect of the variant, specifically what disease it contributes to.\n\nGenome context:\n{0}.\n\nInput: Reference Gene DNA sequence, Variant Gene DNA sequence. Output: Disease Name string.\n\nYour final answer must be formatted as <answer> Your Answer </answer>."

def preprocess_example(example, is_eval: bool = False):
    raw_query = example['question'].replace("\/", "/")
    label_value = example['answer']
    cot = example['reasoning'].replace("\/", "/")
    reference = example['reference_sequence']
    variant = example['variant_sequence']

    context = raw_query[:raw_query.find("\n\nGiven this context")]
    query = prompt_pattern.format(context)

    return {
        "query": query,
        "input_smiles": [reference, variant],
        "label": f"<answer> {label_value} </answer>" if not is_eval else None,
        "cot": cot if not is_eval else None,
        "cot_len": len(cot) if not is_eval else None,
        "task": "disease_pathway_prediction",
        "subtask": "disease_pathway_prediction",
        "meta": {}
    }

def preprocess_example_text(example):
    raise NotImplementedError("Text model has not been implemented")

def tokenize_example(example, include_cot=True, max_len=ModelConfig.MAX_TEXT_LEN, is_eval: bool = False):
    prompt = example.get("query", "")
    cot = example.get("cot", "")
    label = example.get("label")

    if is_eval or (label is None):
        prompt_enc = tokenizer(f"{prompt}\n\n", truncation=True, padding=False, max_length=max_len, add_special_tokens=False)
        input_ids = prompt_enc["input_ids"][:max_len]
        attention_mask = prompt_enc["attention_mask"][:max_len]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": None,
            "sci_input": example.get("input_smiles") or "",
        }

    if include_cot and cot:
        response = f"{cot}\n\n{label}"
    else:
        response = label

    prompt_enc = tokenizer(
        f"{prompt}\n\n",
        truncation=True,
        padding=False,
        max_length=max_len,
        add_special_tokens=False
    )
    
    response_enc = tokenizer(
        f"{response}{tokenizer.eos_token}",
        truncation=True,
        padding=False,
        max_length=max_len,
        add_special_tokens=False
    )

    prompt_ids = prompt_enc["input_ids"]
    response_ids = response_enc["input_ids"]

    input_ids = (prompt_ids + response_ids)[:max_len]
    attention_mask = (prompt_enc["attention_mask"] + response_enc["attention_mask"])[:max_len]

    labels = input_ids.copy()

    actual_prompt_len = min(len(prompt_ids), max_len)

    labels[:actual_prompt_len] = [-100] * actual_prompt_len

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "sci_input": example.get("input_smiles") or "",
    }

def tokenize_example_coconut(
    example,
    scheduled_stage=0,
    c_thought=2,
    max_len=ModelConfig.MAX_TEXT_LEN,
    is_eval: bool = False,
):
    raise NotImplementedError("Coconut model has not been implemented")

# TODO: add support for train/val split
# TODO: original LatentChem has no val split, but both peft and trl supports val split
# TODO: add a config that determines whether to enable val split
# TODO: add registry support for splits
def load_data(
    path, 
    split='train',
    include_cot=True, 
    max_len=ModelConfig.MAX_TEXT_LEN,
    is_coconut=False,
    scheduled_stage=0,
    c_thought=2,
    include_tasks=None,
    exclude_tasks=['rcr'],
    eval_mode: bool = False,
    pure_text: bool = False
):
    dataset = load_dataset(path, split=split)

    if pure_text:
        dataset = dataset.map(preprocess_example_text, batched=False)
        return dataset
    
    dataset = dataset.map(
        preprocess_example,
        batched=False,
        fn_kwargs={"is_eval": eval_mode},
    )

    if is_coconut:
        dataset = dataset.map(
            tokenize_example_coconut,
            batched=False,
            fn_kwargs={
                "scheduled_stage": scheduled_stage,
                "c_thought": c_thought,
                "max_len": max_len,
                "is_eval": eval_mode
            }
        )
    else:
        dataset = dataset.map(
            tokenize_example,
            batched=False,
            fn_kwargs={"include_cot": include_cot, "max_len": max_len, "is_eval": eval_mode}
        )

    return dataset

def load_test_data(test_data_path, include_tasks, max_len=None, pure_text=False):
    return load_data(
        test_data_path,
        split='test',
        include_cot=False,
        max_len=max_len,
        include_tasks=include_tasks,
        pure_text=pure_text,
        eval_mode=True,
        is_coconut=False,
    )

def load_grpo_data(path, split="train"):
    dataset = load_dataset(path, split=split)
    dataset = dataset.map(preprocess_example, batched=False, fn_kwargs={"is_eval": False})
    return dataset