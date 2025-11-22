import math
from typing import Dict, Any
import torch
from torch.nn import CrossEntropyLoss


def sentence_nll_and_ppl(model, tokenizer, text: str, stride: int = 512, max_length: int = 2048) -> Dict[str, Any]:
    """
    Compute per-sentence negative log-likelihood (avg per token) and perplexity using
    a strided/sliding window to respect context length. Adapted from HF docs.
    Returns dict with keys: nll, ppl, token_count
    """
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc["input_ids"].to(model.device)
    total_loss = 0.0
    total_tokens = 0

    # Flatten tokens
    input_ids = input_ids[0]
    seq_len = input_ids.size(0)

    # Sliding over the sequence
    prev_end_loc = 0
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # how many new tokens to predict
        input_ids_slice = input_ids[begin_loc:end_loc]
        # Labels aligned for next-token prediction
        labels = input_ids_slice[1:].clone()

        with torch.no_grad():
            out = model(input_ids_slice.unsqueeze(0))
            logits = out.logits  # [1, L, V]
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels.unsqueeze(0).contiguous()
            # Mask to keep only last trg_len positions
            mask = torch.zeros_like(shift_labels)
            mask[:, -trg_len:] = 1
            shift_labels_masked = shift_labels.masked_fill(mask == 0, -100)

            # Compute CE on CPU in float32 to avoid CUDA device-side asserts
            ce_logits = shift_logits.view(-1, shift_logits.size(-1)).detach().to(torch.float32).cpu()
            ce_targets = shift_labels_masked.view(-1).detach().to(torch.long).cpu()
            loss_per_tok = torch.nn.functional.cross_entropy(
                ce_logits,
                ce_targets,
                reduction="none",
                ignore_index=-100,
            )
            # Sum only valid tokens and count them
            valid = (shift_labels_masked.view(-1) != -100)
            # valid mask computed on GPU; bring to CPU to index loss_per_tok
            valid_cpu = valid.detach().cpu().view(-1)
            sum_loss = loss_per_tok[valid_cpu].sum().item()
            n_tok = int(valid_cpu.sum().item())
            total_loss += sum_loss
            total_tokens += n_tok
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    token_count = max(total_tokens, 1)
    avg_nll = total_loss / token_count
    ppl = math.exp(avg_nll)
    return {"nll": avg_nll, "ppl": ppl, "token_count": token_count}
