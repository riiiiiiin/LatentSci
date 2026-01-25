import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import ModelConfig


def load_qwen3_text_model(
    model_name_or_path: str | None = None,
    *,
    torch_dtype: torch.dtype = torch.bfloat16,
    device_map=None,
):
    """
    Load a plain text-only Qwen3 CausalLM + tokenizer (no multimodal modules).
    """
    ckpt = model_name_or_path or ModelConfig.DEFAULT_QWEN_PATH
    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        ckpt,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )
    if model.config.pad_token_id is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer

