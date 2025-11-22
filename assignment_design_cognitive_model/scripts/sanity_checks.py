import torch
from transformers import AutoTokenizer

from src.models import load_llama, get_device
from src.ablate import AblationSpec, apply_ablation_hooks


def logits_change_check():
    device = get_device()
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    model, tokenizer, _, _ = load_llama(model_id, device=str(device))

    text = "The quick brown fox jumps over the lazy dog."
    enc = tokenizer(text, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        base = model(**enc).logits

    # Zero a single head in layer 0 if available
    spec = AblationSpec(
        ablation_type="heads",
        severity=0.0,
        selection_policy="random",
        seeds=[123],
        selection_indices={},
    )
    # Force a deterministic single-head selection
    spec.severity = 1.0 / model.config.num_attention_heads

    with apply_ablation_hooks(model, spec):
        with torch.no_grad():
            abl = model(**enc).logits

    changed = not torch.allclose(base, abl)
    print(f"Logits changed after single-head zeroing: {changed}")
    return changed


def embedding_mask_check():
    device = get_device()
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    model, tokenizer, _, _ = load_llama(model_id, device=str(device))

    text = "Hello world"
    enc = tokenizer(text, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        base = model(**enc).logits[:, -1, :]

    spec = AblationSpec(
        ablation_type="emb_mask",
        severity=1.0,
        selection_policy="consecutive",
        seeds=[0],
        selection_indices={},
    )

    with apply_ablation_hooks(model, spec):
        with torch.no_grad():
            abl = model(**enc).logits[:, -1, :]

    # Check entropy increase as proxy for uniformity
    base_probs = torch.softmax(base, dim=-1)
    abl_probs = torch.softmax(abl, dim=-1)
    base_ent = -(base_probs * torch.log(base_probs + 1e-8)).sum(dim=-1).mean().item()
    abl_ent = -(abl_probs * torch.log(abl_probs + 1e-8)).sum(dim=-1).mean().item()
    print(f"Entropy baseline: {base_ent:.3f}, after 100% mask: {abl_ent:.3f}")
    return abl_ent >= base_ent


if __name__ == "__main__":
    ok1 = logits_change_check()
    ok2 = embedding_mask_check()
    print(f"sanity logits change: {ok1}, emb uniformity increase: {ok2}")
