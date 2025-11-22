import math
from typing import List, Tuple, Dict, Any

import torch
from torch.nn import functional as F


def _pair(sg: str, pl: str) -> Tuple[str, str]:
    return (sg, pl)


def _build_sva_minimal_pairs(max_examples: int = 100) -> List[Dict[str, Any]]:
    """
    Build a small, self-contained set of subject–verb agreement minimal pairs.
    Each item is a dict with:
      - prefix: the text up to (but not including) the verb token(s)
      - correct: the grammatically correct verb form given the subject number
      - incorrect: the incorrect verb form
    """
    singular_subjects = [
        "the dog", "the cat", "the child", "the key", "the car", "the idea",
        "the man", "the woman", "the hero", "the book",
    ]
    plural_subjects = [
        "the dogs", "the cats", "the children", "the keys", "the cars", "the ideas",
        "the men", "the women", "the heroes", "the books",
    ]

    verb_pairs = [
        _pair("is", "are"),
        _pair("was", "were"),
        _pair("has", "have"),
    ]

    suffixes = [
        " on the table.",
        " in the house.",
        " very loud.",
        " quite old.",
        " nearby.",
        " ready to go.",
        " under the bed.",
        " outside.",
        " important.",
        " available.",
    ]

    items: List[Dict[str, Any]] = []
    # Generate both singular and plural sets
    for subj in singular_subjects:
        for (sg, pl) in verb_pairs:
            for suff in suffixes:
                items.append({
                    "prefix": subj + " ",
                    "correct": sg,
                    "incorrect": pl,
                    "suffix": suff,
                    "number": "sg",
                })
    for subj in plural_subjects:
        for (sg, pl) in verb_pairs:
            for suff in suffixes:
                items.append({
                    "prefix": subj + " ",
                    "correct": pl,
                    "incorrect": sg,
                    "suffix": suff,
                    "number": "pl",
                })

    # limit
    return items[:max_examples]


def _encode_ids(tokenizer, text: str, add_bos: bool = True):
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=add_bos)["input_ids"]
    return ids


def _concat_ids(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.dim() == 1:
        a = a.unsqueeze(0)
    if b.dim() == 1:
        b = b.unsqueeze(0)
    return torch.cat([a, b], dim=1)


@torch.no_grad()
def _continuation_logprob(model, tokenizer, device, prefix: str, continuation: str) -> float:
    """
    Compute log P(continuation | prefix) by summing token-level logprobs for continuation tokens.
    """
    # Build joint sequence and select the continuation slice
    pref_ids = _encode_ids(tokenizer, prefix, add_bos=True).to(device)
    cont_ids = _encode_ids(tokenizer, continuation, add_bos=False).to(device)
    # Some tokenizers may introduce leading spaces; ensure at least one token
    if cont_ids.numel() == 0:
        return float("-inf")
    joint = _concat_ids(pref_ids, cont_ids)

    outputs = model(input_ids=joint)
    logits = outputs.logits  # [1, T, V]
    logprobs = F.log_softmax(logits, dim=-1)

    # continuation positions correspond to the last len(cont) tokens
    Lc = cont_ids.size(1)
    # indices in joint where each cont token is predicted
    # token at position t is predicted from logits at position t-1
    start = joint.size(1) - Lc
    # sum logprobs of the continuation tokens
    lp = 0.0
    for i in range(Lc):
        vocab_ix = cont_ids[0, i].item()
        # predicted at position start + i - 1
        pred_pos = start + i - 1
        if pred_pos < 0:
            return float("-inf")
        lp += logprobs[0, pred_pos, vocab_ix].item()
    return float(lp)


@torch.no_grad()
def evaluate_sva_probe(model, tokenizer, device: torch.device, max_examples: int = 100) -> Dict[str, Any]:
    """
    Evaluate subject–verb agreement with forced-choice between correct vs. incorrect verb forms.
    Returns accuracy and average logprob margin.
    """
    model.eval()
    items = _build_sva_minimal_pairs(max_examples=max_examples)

    correct = 0
    margins: List[float] = []

    for it in items:
        prefix = it["prefix"]
        # include a small right-context suffix to avoid degenerate continuations
        suffix = it.get("suffix", "")
        cont_correct = it["correct"]
        cont_incorrect = it["incorrect"]

        # Score only on the verb form; keep suffix outside the scored span to avoid bias
        lp_cor = _continuation_logprob(model, tokenizer, device, prefix, cont_correct)
        lp_inc = _continuation_logprob(model, tokenizer, device, prefix, cont_incorrect)

        pred_correct = lp_cor > lp_inc
        correct += int(pred_correct)
        margins.append(lp_cor - lp_inc)

        # Optionally, do a second pass with a bit of post-verb context to check stability
        # Not used for scoring, but could be logged in the future.

    total = len(items)
    acc = correct / total if total > 0 else 0.0
    avg_margin = float(sum(margins) / len(margins)) if margins else 0.0

    return {
        "task_name": "subject_verb_agreement",
        "num_items": total,
        "accuracy": acc,
        "avg_logprob_margin": avg_margin,
    }
