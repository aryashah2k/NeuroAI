import math
import time
from typing import Dict, Any, List, Tuple

import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase


PROMPT_TEMPLATE = (
    "Review: {text}\n"
    "Question: Is the sentiment of the review positive or negative?\n"
    "Answer:"
)
LABEL_STRINGS = ["negative", "positive"]


def _score_label_sequence(model, tokenizer, prompt_ids: torch.Tensor, label_str: str, device: torch.device) -> float:
    """
    Compute the total log-prob of generating the full label string tokens given the prompt.
    Uses teacher forcing over the label token ids.
    Returns sum log prob (natural log) over the label tokens.
    """
    label_ids = tokenizer.encode(" " + label_str, add_special_tokens=False)  # leading space to align tokenizer
    input_ids = torch.tensor(prompt_ids + label_ids[:-1], dtype=torch.long, device=device).unsqueeze(0)
    target_ids = torch.tensor(label_ids, dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad():
        out = model(input_ids=input_ids)
        logits = out.logits  # [1, seq, vocab]
        # Gather probabilities of next tokens corresponding to target_ids
        last_logits = logits[:, -len(label_ids):, :]
        log_probs = torch.log_softmax(last_logits, dim=-1)
        tgt = torch.tensor(label_ids, device=device).unsqueeze(0).unsqueeze(-1)
        token_logprobs = torch.gather(log_probs, -1, tgt).squeeze(-1)  # [1, L]
        total_logprob = token_logprobs.sum().item()
    return total_logprob


def evaluate_sst2_generative(
    model,
    tokenizer: PreTrainedTokenizerBase,
    loader: DataLoader,
    device: torch.device,
    label_names: List[str] = LABEL_STRINGS,
) -> Dict[str, Any]:
    """
    Forced-choice generative evaluation: score "negative" vs "positive" label completions.
    Accuracy computed by argmax over total label log-prob.
    """
    start = time.time()
    correct = 0
    total = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch.get("labels", None)
        if labels is None:
            # if labels not provided (e.g., on test), skip accuracy computation
            break
        labels = labels.to(device)

        # For each example in batch, build prompt text from original tokens
        for i in range(input_ids.size(0)):
            ids = input_ids[i].tolist()
            text = tokenizer.decode(ids, skip_special_tokens=True)
            prompt = PROMPT_TEMPLATE.format(text=text)
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            # score labels
            scores = [
                _score_label_sequence(model, tokenizer, prompt_ids, label_names[0], device),
                _score_label_sequence(model, tokenizer, prompt_ids, label_names[1], device),
            ]
            pred = 1 if scores[1] > scores[0] else 0
            if pred == labels[i].item():
                correct += 1
            total += 1

    runtime_s = time.time() - start
    acc = correct / total if total > 0 else float('nan')
    return {
        "accuracy": acc,
        "num_examples": total,
        "runtime_s": runtime_s,
        "mode": "generative_forced_choice",
    }
