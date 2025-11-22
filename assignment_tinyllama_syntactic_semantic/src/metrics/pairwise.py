from typing import Tuple
from .perplexity import sentence_nll_and_ppl


def prefer_a_over_b(model, tokenizer, a: str, b: str) -> Tuple[int, float, float]:
    """
    Returns (decision, nll_a, nll_b) where decision is 1 if a has lower NLL than b, else 0.
    """
    a_metrics = sentence_nll_and_ppl(model, tokenizer, a)
    b_metrics = sentence_nll_and_ppl(model, tokenizer, b)
    nll_a, nll_b = a_metrics["nll"], b_metrics["nll"]
    decision = 1 if nll_a < nll_b else 0
    return decision, nll_a, nll_b
