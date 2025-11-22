import os
import sys
import argparse
from typing import Dict, Any, List, Tuple

# Ensure project root is on sys.path so `src` package can be imported when running as a script
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt

# Reuse project utilities without modifying existing code
from src.models import load_llama, get_device
from src.data import load_wikitext, load_sst2


def _ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)


def _prep_inputs_from_wikitext(tokenizer, device: torch.device, max_length: int, num_examples: int) -> List[torch.Tensor]:
    loader, _ = load_wikitext(tokenizer, split="validation", max_length=max_length, batch_size=1, num_workers=0)
    inputs: List[torch.Tensor] = []
    for batch in loader:
        # load_wikitext returns a TensorDataset of (input_ids, labels)
        if isinstance(batch, (list, tuple)):
            ids = batch[0].to(device)
        elif isinstance(batch, dict) and "input_ids" in batch:
            ids = batch["input_ids"].to(device)
        else:
            # Fallback: assume tensor
            ids = batch.to(device)
        inputs.append(ids)
        if len(inputs) >= num_examples:
            break
    return inputs


def _build_sst2_prompts(tokenizer, device: torch.device, max_length: int, num_examples: int) -> List[torch.Tensor]:
    # Reuse SST-2 loader to get texts, but we will format a simple prompt
    loader, total_items, dataset, _ = load_sst2(tokenizer, split="validation", max_length=max_length, batch_size=1, num_workers=0)
    # dataset has original texts under 'sentence' typically; loader yields tokenized already
    inputs: List[torch.Tensor] = []
    for batch in loader:
        ids = batch["input_ids"].to(device)
        inputs.append(ids)
        if len(inputs) >= num_examples:
            break
    return inputs


def register_hooks(model) -> Tuple[List[Any], Dict[str, List[torch.Tensor]]]:
    """
    Register forward hooks to capture layer-wise activities at:
      - layer output (block output)
      - attention output projection (o_proj)
      - MLP down projection (down_proj)
    Returns (handles, buffers) where buffers collects per-layer tensors per example.
    """
    buffers: Dict[str, List[torch.Tensor]] = {
        "block": [],  # list of [seq, hidden] per layer (last example seen)
        "attn_o": [],
        "mlp_down": [],
    }
    handles = []

    # Prepare container per layer index
    num_layers = model.config.num_hidden_layers
    buffers["block"] = [None] * num_layers
    buffers["attn_o"] = [None] * num_layers
    buffers["mlp_down"] = [None] * num_layers

    def make_block_hook(i: int):
        def hook(module, inputs, output):
            # output: [bsz, seq, hidden]
            if isinstance(output, tuple):
                out = output[0]
            else:
                out = output
            buffers["block"][i] = out.detach().float().squeeze(0).cpu()
            return output
        return hook

    def make_attn_o_hook(i: int):
        def hook(module, inputs, output):
            out = output
            buffers["attn_o"][i] = out.detach().float().squeeze(0).cpu()
            return output
        return hook

    def make_mlp_down_hook(i: int):
        def hook(module, inputs, output):
            out = output
            buffers["mlp_down"][i] = out.detach().float().squeeze(0).cpu()
            return output
        return hook

    for i in range(num_layers):
        layer = model.model.layers[i]
        handles.append(layer.register_forward_hook(make_block_hook(i)))
        handles.append(layer.self_attn.o_proj.register_forward_hook(make_attn_o_hook(i)))
        handles.append(layer.mlp.down_proj.register_forward_hook(make_mlp_down_hook(i)))

    return handles, buffers


def tensor_to_heatmap(t: torch.Tensor) -> np.ndarray:
    """
    Convert [seq, hidden] tensor to a 2D heatmap by taking L2 norm across hidden dims per token position.
    Returns [seq] vector; we stack across layers later to [layers, seq].
    """
    if t is None:
        return None
    # t: [seq, hidden]
    arr = t.numpy()
    norms = np.linalg.norm(arr, axis=-1)  # [seq]
    return norms


def save_layer_token_heatmap(mats: List[np.ndarray], title: str, out_path: str, tokens: List[str] = None, tick_stride: int = 4):
    """
    mats: list of [seq] vectors per layer; stack to [layers, seq]
    """
    valid = [m for m in mats if m is not None]
    if not valid:
        return
    max_len = max(m.shape[0] for m in valid)
    # pad to same length with NaNs
    padded = [np.pad(m, (0, max_len - m.shape[0]), constant_values=np.nan) for m in mats]
    H = np.stack(padded, axis=0)  # [layers, seq]

    plt.figure(figsize=(10, 6))
    im = plt.imshow(H, aspect='auto', interpolation='nearest', cmap='magma')
    plt.colorbar(im, fraction=0.046, pad=0.04, label='L2 norm over hidden')
    plt.xlabel('Token')
    plt.ylabel('Layer index')
    plt.title(title)
    # x-axis tokens
    if tokens is not None and len(tokens) > 0:
        max_len = H.shape[1]
        xlocs = np.arange(0, max_len, tick_stride)
        xlabels = [tokens[i] if i < len(tokens) else '' for i in xlocs]
        plt.xticks(xlocs, xlabels, rotation=90, fontsize=6)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _export_tsv(H: np.ndarray, tokens: List[str], out_path: str):
    """
    Save a TSV with rows = layers, cols = tokens, values = activity (NaN allowed).
    """
    max_len = H.shape[1]
    header = ['layer'] + [tokens[i] if i < len(tokens) else f'tok_{i}' for i in range(max_len)]
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\t'.join(header) + '\n')
        for li in range(H.shape[0]):
            row = [str(li)] + ["" if np.isnan(v) else f"{v:.6f}" for v in H[li]]
            f.write('\t'.join(row) + '\n')


def _topk_summary(H: np.ndarray, tokens: List[str], k: int = 10) -> List[List[Tuple[int, str, float]]]:
    """
    For each layer, return top-k (token_index, token_str, value) by activity.
    """
    res = []
    for li in range(H.shape[0]):
        vals = H[li]
        idxs = np.argsort(np.nan_to_num(vals, nan=-np.inf))[::-1][:k]
        triples = []
        for j in idxs:
            tok = tokens[j] if j < len(tokens) else f'tok_{j}'
            v = vals[j]
            triples.append((int(j), tok, float(v)))
        res.append(triples)
    return res


def process_example(model, tokenizer, device, input_ids: torch.Tensor, outdir: str, ex_idx: int):
    handles, buffers = register_hooks(model)
    try:
        with torch.no_grad():
            _ = model(input_ids=input_ids)
        # Derive token strings for axis labels
        ids_list = input_ids.squeeze(0).tolist()
        tokens = tokenizer.convert_ids_to_tokens(ids_list)
        # Build heatmaps per tap point
        for tap in ["block", "attn_o", "mlp_down"]:
            mats = []
            for layer_data in buffers[tap]:
                if layer_data is None:
                    mats.append(None)
                else:
                    mats.append(tensor_to_heatmap(layer_data))
            # Save figure
            out_path = os.path.join(outdir, f"ex{ex_idx:02d}_{tap}.png")
            title = f"Example {ex_idx} - {tap} token-wise L2 across hidden"
            save_layer_token_heatmap(mats, title, out_path, tokens=tokens, tick_stride=4)
            # Save raw matrix as .npy for downstream analysis
            # Stack with padding similar to plotting
            valid = [m for m in mats if m is not None]
            if valid:
                max_len = max(m.shape[0] for m in valid)
                padded = [np.pad(m, (0, max_len - (0 if m is None else m.shape[0])), constant_values=np.nan) if m is not None else np.full((max_len,), np.nan) for m in mats]
                H = np.stack(padded, axis=0)
                np.save(os.path.join(outdir, f"ex{ex_idx:02d}_{tap}.npy"), H)
                # TSV with token labels
                _export_tsv(H, tokens, os.path.join(outdir, f"ex{ex_idx:02d}_{tap}.tsv"))
                # Markdown summary of top-k tokens for key layers
                topk = _topk_summary(H, tokens, k=10)
                key_layers = [0, len(topk)//2, len(topk)-1] if len(topk) > 2 else list(range(len(topk)))
                md_path = os.path.join(outdir, f"ex{ex_idx:02d}_{tap}_summary.md")
                with open(md_path, 'w', encoding='utf-8') as f:
                    f.write(f"# Example {ex_idx} - {tap} Top Tokens by Layer\n\n")
                    f.write("Tokens (first 128):\n\n")
                    preview = tokens[:128]
                    f.write(' '.join(preview) + "\n\n")
                    for li in key_layers:
                        f.write(f"## Layer {li}\n\n")
                        f.write("| rank | token | index | value |\n|---|---|---:|---:|\n")
                        for rank, (j, tok, v) in enumerate(topk[li], start=1):
                            safe_tok = tok.replace('|','\\|')
                            f.write(f"| {rank} | {safe_tok} | {j} | {v:.4f} |\n")
    finally:
        for h in handles:
            h.remove()


def _read_tsv(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        lines = [ln.rstrip('\n') for ln in f]
    if not lines:
        return [], None
    header = lines[0].split('\t')
    tokens = header[1:]
    rows = []
    for ln in lines[1:]:
        parts = ln.split('\t')
        # parts[0] is layer index, remaining are values or empty
        vals = [np.nan if p == '' else float(p) for p in parts[1:]]
        rows.append(vals)
    H = np.array(rows, dtype=float) if rows else None
    return tokens, H


def _mk_table(rows: List[List[str]], headers: List[str]) -> str:
    head = '| ' + ' | '.join(headers) + ' |\n'
    sep = '|' + '|'.join([' --- ' for _ in headers]) + '|\n'
    body = ''.join(['| ' + ' | '.join(r) + ' |\n' for r in rows])
    return head + sep + body


def interpret_outputs(outdir: str, topk: int = 5) -> str:
    """
    Scan outdir for ex**_{tap}.tsv files, aggregate top-k tokens for early/mid/late layers
    into a consolidated markdown report. Returns the markdown string and also writes it to
    outdir/interpretation.md
    """
    taps = ['block', 'attn_o', 'mlp_down']
    # Find tsv files
    files = [fn for fn in os.listdir(outdir) if fn.endswith('.tsv') and any(t in fn for t in taps)]
    if not files:
        return "# Interpretation Report\n\nNo TSV feature maps found to analyze."

    by_example: Dict[str, Dict[str, str]] = {}
    # Structure: by_example[ex_id][tap] = markdown table
    for fn in sorted(files):
        path = os.path.join(outdir, fn)
        tokens, H = _read_tsv(path)
        if H is None or H.size == 0:
            continue
        # Identify layers of interest: early, mid, late
        L = H.shape[0]
        layers = [0, L // 2, L - 1] if L >= 3 else list(range(L))
        rows = []
        for li in layers:
            vals = H[li]
            idxs = np.argsort(np.nan_to_num(vals, nan=-np.inf))[::-1][:topk]
            for rank, j in enumerate(idxs, start=1):
                tok = tokens[j] if j < len(tokens) else f'tok_{j}'
                rows.append([
                    f"{li}",
                    f"{rank}",
                    tok.replace('|','\\|'),
                    f"{j}",
                    f"{vals[j]:.4f}",
                ])
        table = _mk_table(rows, headers=["layer", "rank", "token", "token_idx", "activity"])
        # Example id and tap
        # fn format: exXX_tap.tsv
        base = os.path.splitext(fn)[0]
        # split name into ex and tap
        parts = base.split('_')
        ex_id = parts[0]
        tap = '_'.join(parts[1:])  # in case of underscores
        by_example.setdefault(ex_id, {})[tap] = table

    # Build consolidated markdown
    md = ["# Interpretation Report\n"]
    md.append("This report summarizes top-k token activities for early/mid/late layers across taps.\n")
    for ex_id in sorted(by_example.keys()):
        md.append(f"\n## {ex_id}\n")
        taps_present = by_example[ex_id]
        for tap in ['block', 'attn_o', 'mlp_down']:
            tbl = taps_present.get(tap)
            if not tbl:
                continue
            md.append(f"\n### {tap}\n\n")
            md.append(tbl)
    out_path = os.path.join(outdir, 'interpretation.md')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md))
    return '\n'.join(md)

def main():
    ap = argparse.ArgumentParser(description="Generate layer-wise feature maps for five examples.")
    ap.add_argument("--task", type=str, default="wikitext_lm", choices=["wikitext_lm", "sst2_cls"], help="Source of text/examples")
    ap.add_argument("--model_id", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--num_examples", type=int, default=5)
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--outdir", type=str, default=os.path.join("outputs", "feature_maps"))
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--interpret", action="store_true", help="After generating, scan outdir TSVs and write interpretation.md")
    ap.add_argument("--interpret_only", action="store_true", help="Only read existing TSVs in outdir and write interpretation.md; skip generation")
    args = ap.parse_args()

    _ensure_outdir(args.outdir)

    if not args.interpret_only:
        device = get_device(args.device)
        model, tokenizer, _, _ = load_llama(model_id=args.model_id, device=str(device))
        model.eval()

        if args.task == "wikitext_lm":
            examples = _prep_inputs_from_wikitext(tokenizer, device, args.max_length, args.num_examples)
        else:
            examples = _build_sst2_prompts(tokenizer, device, args.max_length, args.num_examples)

        for i, ids in enumerate(examples, start=1):
            process_example(model, tokenizer, device, ids, args.outdir, i)
            print(f"Saved feature maps for example {i} to {args.outdir}")

    if args.interpret or args.interpret_only:
        md = interpret_outputs(args.outdir, topk=5)
        print(f"Wrote interpretation to {os.path.join(args.outdir, 'interpretation.md')}")


if __name__ == "__main__":
    main()
