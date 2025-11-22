import math
import time
from typing import Dict, Any

import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase


def evaluate_lm(model, tokenizer: PreTrainedTokenizerBase, loader: DataLoader, device: torch.device) -> Dict[str, Any]:
    model.eval()
    total_tokens = 0
    total_nll = 0.0
    start = time.time()

    with torch.no_grad():
        for batch in loader:
            input_ids, labels = batch
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            # Shift labels by one token for causal LM? When passing labels=input_ids, HF internally shifts.
            out = model(input_ids=input_ids, labels=labels)
            # out.loss is mean over all tokens in batch
            batch_tokens = labels.numel()
            total_tokens += batch_tokens
            total_nll += out.loss.item() * batch_tokens

    mean_nll = total_nll / max(1, total_tokens)
    ppl = math.exp(mean_nll)
    runtime_s = time.time() - start
    return {
        "mean_nll": mean_nll,
        "perplexity": ppl,
        "total_tokens": int(total_tokens),
        "runtime_s": runtime_s,
    }
