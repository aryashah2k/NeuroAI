import argparse
import json
import os
import time
from datetime import datetime
from typing import Dict, Any, List

import torch

from .models import load_llama, get_device, DeviceInfo
from .data import load_wikitext, load_sst2
from .eval_lm import evaluate_lm
from .eval_cls import evaluate_sst2_generative
from .eval_syntax import evaluate_sva_probe
from .ablate import AblationSpec, apply_ablation_hooks


def run_one(config: Dict[str, Any], seed: int) -> Dict[str, Any]:
    torch.manual_seed(seed)
    device = get_device(config.get("device"))

    model_id = config["model_id"]
    model, tokenizer, hf_config, revision = load_llama(
        model_id=model_id,
        device=str(device),
        dtype=config.get("dtype"),
        trust_remote_code=config.get("trust_remote_code", False),
    )

    task = config["task"]
    split = config.get("split", "validation")
    max_length = int(config.get("max_length", 1024))
    batch_size = int(config.get("batch_size", 8))

    ablation_type = config.get("ablation_type", "none")
    severity = float(config.get("severity", 0.0))
    selection_policy = config.get("selection_policy", "random")
    component = config.get("component")  # for layers: attn/mlp/block

    spec = AblationSpec(
        ablation_type=("emb_mask" if ablation_type == "emb_mask" else ablation_type),
        severity=severity,
        selection_policy=selection_policy,
        seeds=[seed],
        selection_indices={},
        component=component,
    )

    # Data loaders and evaluation
    start = time.time()
    if task == "wikitext_lm":
        loader, total_len = load_wikitext(tokenizer, name=config.get("wikitext_name", "wikitext-2-raw-v1"), split=split,
                                          max_length=max_length, batch_size=batch_size, num_workers=int(config.get("num_workers", 0)))
        with apply_ablation_hooks(model, spec):
            metrics = evaluate_lm(model, tokenizer, loader, device)
        num_samples = metrics.pop("total_tokens", total_len)
    elif task == "sst2_cls":
        loader, total_items, _, label_names = load_sst2(tokenizer, split=split,
                                                       max_length=max_length, batch_size=batch_size,
                                                       num_workers=int(config.get("num_workers", 0)))
        with apply_ablation_hooks(model, spec):
            metrics = evaluate_sst2_generative(model, tokenizer, loader, device, label_names=label_names)
        num_samples = total_items
    elif task == "syntax_probe":
        max_examples = int(config.get("max_examples", 100))
        with apply_ablation_hooks(model, spec):
            metrics = evaluate_sva_probe(model, tokenizer, device, max_examples=max_examples)
        num_samples = metrics.get("num_items", max_examples)
    else:
        raise ValueError(f"Unknown task: {task}")

    runtime_s = time.time() - start

    device_info = DeviceInfo.collect().to_dict()
    result = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "model_id": model_id,
        "model_revision": revision,
        "model_config": hf_config.to_dict(),
        "task": task,
        "split": split,
        "ablation_type": ablation_type,
        "severity": severity,
        "selection_policy": selection_policy,
        "selection_indices": spec.selection_indices,
        "component": component,
        "seed": seed,
        "metrics": metrics,
        "num_samples": int(num_samples),
        "runtime_s": runtime_s,
        "device_info": device_info,
        "notes": config.get("notes", ""),
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config JSON")
    parser.add_argument("--outdir", type=str, default="results", help="Directory to write JSON results")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    seeds: List[int] = cfg.get("seeds", [42])
    severities_cfg = cfg.get("severity", 0.0)
    if isinstance(severities_cfg, list):
        severities: List[float] = [float(s) for s in severities_cfg]
    else:
        severities = [float(severities_cfg)]

    os.makedirs(args.outdir, exist_ok=True)

    for seed in seeds:
        for sev in severities:
            cfg_mut = dict(cfg)
            cfg_mut["severity"] = float(sev)
            res = run_one(cfg_mut, seed)
            # Filename: results/{timestamp}_{task}_{ablation}_{severity}_{seed}.json
            ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
            ablt = cfg_mut.get("ablation_type", "none")
            fname = f"{ts}_{cfg_mut['task']}_{ablt}_{sev}_{seed}.json"
            path = os.path.join(args.outdir, fname)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(res, f, indent=2)
            print(f"Wrote {path}")


if __name__ == "__main__":
    main()
