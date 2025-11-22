import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Tuple, Optional

DEFAULT_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def get_dtype():
    # Use float32 for stable scoring across datasets (avoids rare NaNs in half precision)
    return torch.float32


def load_tinyllama(model_id: str = DEFAULT_MODEL_ID, trust_remote_code: bool = False):
    dtype = get_dtype()
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        device_map="auto",
        trust_remote_code=trust_remote_code,
    )
    model.eval()
    return tokenizer, model


def mean_pool_hidden_states(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    hidden_states: [batch, seq, hidden]
    attention_mask: [batch, seq] with 1 for tokens to keep, 0 for padding
    Returns [batch, hidden]
    """
    mask = attention_mask.unsqueeze(-1)  # [b, s, 1]
    masked = hidden_states * mask
    lengths = mask.sum(dim=1).clamp(min=1)
    return masked.sum(dim=1) / lengths


def get_sentence_embedding(model, tokenizer, text: str, max_length: Optional[int] = None) -> torch.Tensor:
    enc = tokenizer(
        text,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
    )
    enc = {k: v.to(model.device) for k, v in enc.items()}
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)
        last_hidden = out.hidden_states[-1]  # [1, seq, hidden]
    emb = mean_pool_hidden_states(last_hidden, enc["attention_mask"])  # [1, hidden]
    return emb.squeeze(0).detach().cpu()
