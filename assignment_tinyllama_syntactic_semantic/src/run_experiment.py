import argparse
import time
import os
import json
from typing import Dict, Any, List

import torch
from tqdm import tqdm

from src.models.tinyllama_loader import load_tinyllama, get_sentence_embedding
from src.metrics.perplexity import sentence_nll_and_ppl
from src.metrics.pairwise import prefer_a_over_b
from src.metrics.embeddings import cosine_similarity
from src.data.blimp_loader import iter_blimp
from src.data.paws_loader import iter_paws
from src.data.cola_loader import iter_cola
from src.utils.jsonl import write_jsonl, ensure_dir


def now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def base_record_header(model_id: str, dataset_name: str, split: str, config: Dict[str, Any]):
    return {
        "experiment_id": f"{dataset_name}-{int(time.time())}",
        "model_id": model_id,
        "dataset_name": dataset_name,
        "split": split,
        "created_at": now_iso(),
        "config": config,
    }


def run_blimp(args):
    tokenizer, model = load_tinyllama(args.model_id)
    out_path = os.path.join(args.output_dir, "blimp.jsonl")
    ensure_dir(args.output_dir)
    records: List[Dict[str, Any]] = []

    iterator = iter_blimp(subset=args.subset, split="train")
    for i, ex in enumerate(tqdm(iterator, desc="BLiMP")):
        if args.limit and i >= args.limit:
            break
        good = ex["sentence_good"]
        bad = ex["sentence_bad"]
        m_good = sentence_nll_and_ppl(model, tokenizer, good)
        m_bad = sentence_nll_and_ppl(model, tokenizer, bad)
        nll_good = float(m_good["nll"]) if m_good["nll"] == m_good["nll"] else float('nan')
        nll_bad = float(m_bad["nll"]) if m_bad["nll"] == m_bad["nll"] else float('nan')
        # Decision: prefer lower NLL; add small epsilon to reduce tie bias
        eps = 1e-8
        if (not (nll_good == nll_good)) or (not (nll_bad == nll_bad)):
            decision = -1  # unknown due to NaN
            correct = 0
        else:
            decision = 1 if (nll_good + eps) < nll_bad else 0
            correct = 1 if decision == 1 else 0
        rec = {
            **base_record_header(args.model_id, "blimp", "train", vars(args)),
            "example_id": ex["id"],
            "subset": ex["subset"],
            "field": ex.get("field", ex["subset"]),
            "input_a": good,
            "input_b": bad,
            "nll_a": nll_good,
            "nll_b": nll_bad,
            "ppl_a": float(m_good["ppl"]) if m_good["ppl"] == m_good["ppl"] else None,
            "ppl_b": float(m_bad["ppl"]) if m_bad["ppl"] == m_bad["ppl"] else None,
            "decision": ("a" if decision == 1 else ("b" if decision == 0 else None)),
            "correct": correct,
        }
        records.append(rec)

    write_jsonl(out_path, records)
    # Aggregate accuracy per subset
    agg = {}
    for r in records:
        k = r["subset"]
        agg.setdefault(k, {"total": 0, "correct": 0})
        agg[k]["total"] += 1
        agg[k]["correct"] += r["correct"]
    macro = sum((v["correct"]/max(v["total"],1)) for v in agg.values())/max(len(agg),1)
    overall = sum(v["correct"] for v in agg.values())/max(sum(v["total"] for v in agg.values()),1)

    summary = {
        "dataset": "blimp",
        "macro_accuracy": macro,
        "overall_accuracy": overall,
        "by_subset": {k: {"accuracy": v["correct"]/max(v["total"],1), **v} for k, v in agg.items()},
    }
    with open(os.path.join(args.output_dir, "aggregates.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def run_paws(args):
    tokenizer, model = load_tinyllama(args.model_id)
    out_path = os.path.join(args.output_dir, "paws.jsonl")
    ensure_dir(args.output_dir)
    records: List[Dict[str, Any]] = []

    iterator = iter_paws(csv_path=args.paws_path, limit=args.limit)
    for i, ex in enumerate(tqdm(iterator, desc="PAWS")):
        s1, s2, label = ex["sentence1"], ex["sentence2"], int(ex["label"])
        # Loss / ppl for each sentence
        m1 = sentence_nll_and_ppl(model, tokenizer, s1)
        m2 = sentence_nll_and_ppl(model, tokenizer, s2)
        decision, nll1, nll2 = prefer_a_over_b(model, tokenizer, s1, s2)
        # Embeddings
        emb1 = get_sentence_embedding(model, tokenizer, s1)
        emb2 = get_sentence_embedding(model, tokenizer, s2)
        cos = cosine_similarity(torch.tensor(emb1), torch.tensor(emb2))

        rec = {
            **base_record_header(args.model_id, "paws", "mixed", vars(args)),
            "example_id": ex["id"],
            "input_a": s1,
            "input_b": s2,
            "label": label,
            "nll_a": m1["nll"],
            "nll_b": m2["nll"],
            "ppl_a": m1["ppl"],
            "ppl_b": m2["ppl"],
            "decision": "a" if decision == 1 else "b",
            "embedding_cosine": cos,
        }
        records.append(rec)

    write_jsonl(out_path, records)

    # Aggregates by label
    from statistics import mean
    by_label = {0: [], 1: []}
    for r in records:
        by_label[int(r["label"])].append(r)
    agg = {
        "dataset": "paws",
        "count": len(records),
        "avg_cosine_paraphrase": mean([x["embedding_cosine"] for x in by_label.get(1, [])]) if by_label.get(1) else None,
        "avg_cosine_nonparaphrase": mean([x["embedding_cosine"] for x in by_label.get(0, [])]) if by_label.get(0) else None,
        "pairwise_pref_accuracy": sum(1 for x in records if (x["decision"] == "a") == (x["label"]==1)) / max(len(records),1),
    }
    with open(os.path.join(args.output_dir, "aggregates.json"), "a", encoding="utf-8") as f:
        f.write("\n")
        f.write(json.dumps(agg))


def run_cola(args):
    tokenizer, model = load_tinyllama(args.model_id)
    out_path = os.path.join(args.output_dir, "cola.jsonl")
    ensure_dir(args.output_dir)
    records: List[Dict[str, Any]] = []

    iterator = iter_cola(split=args.split)
    for i, ex in enumerate(tqdm(iterator, desc="CoLA")):
        if args.limit and i >= args.limit:
            break
        s = ex["sentence"]
        m = sentence_nll_and_ppl(model, tokenizer, s)
        rec = {
            **base_record_header(args.model_id, "cola", args.split, vars(args)),
            "example_id": ex["id"],
            "input": s,
            "label": int(ex["label"]),
            "nll": m["nll"],
            "ppl": m["ppl"],
        }
        records.append(rec)

    write_jsonl(out_path, records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["blimp", "paws", "cola"], required=True)
    parser.add_argument("--subset", type=str, default=None, help="BLiMP subset name (optional)")
    parser.add_argument("--split", type=str, default="validation", help="CoLA split")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--model_id", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--paws_path", type=str, default=None, help="Path to PAWS TSV/CSV file")

    args = parser.parse_args()

    if args.dataset == "blimp":
        run_blimp(args)
    elif args.dataset == "paws":
        run_paws(args)
    elif args.dataset == "cola":
        run_cola(args)


if __name__ == "__main__":
    main()
