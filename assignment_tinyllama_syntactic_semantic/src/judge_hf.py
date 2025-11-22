import argparse
import os
import json
import random
from typing import List, Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from tqdm import tqdm


def make_prompt(example: Dict[str, Any]) -> str:
    parts = []
    parts.append(
        "You are an evaluator deciding whether a model relied more on SYNTAX or SEMANTICS."
    )
    parts.append("Consider the provided text(s) and scores.")
    parts.append("Return ONLY a single JSON object. No prose before or after.")
    parts.append(
        "Schema: {\"rating\": one of [\"syntax\", \"semantics\", \"unclear\"], \"rationale\": short string}."
    )
    parts.append("Do not include any extra keys. Do not repeat 'rating:' or 'rationale:' outside JSON.")
    parts.append("\nExample JSON:\n{\"rating\": \"syntax\", \"rationale\": \"Short reason\"}\n")

    # Inputs block
    if "input" in example:
        parts.append(f"Input: {example['input']}")
    else:
        parts.append(f"Input A: {example.get('input_a','')}")
        parts.append(f"Input B: {example.get('input_b','')}")

    # Scores block
    score_lines = []
    for k in ["nll", "ppl", "nll_a", "ppl_a", "nll_b", "ppl_b", "embedding_cosine", "decision", "label"]:
        if k in example and example[k] is not None:
            score_lines.append(f"{k}: {example[k]}")
    if score_lines:
        parts.append("Scores:\n" + "\n".join(score_lines))

    parts.append("Now output the JSON object only, nothing else.")
    return "\n".join(parts)


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def judge_examples(
    pipe,
    examples: List[Dict[str, Any]],
    max_new_tokens: int = 256,
    pad_token_id: int = None,
    eos_token_id: int = None,
) -> List[Dict[str, Any]]:
    results = []
    prompts = [make_prompt(ex) for ex in examples]

    # Some instruction-tuned models prefer chat formatting; we'll use simple text prompts for broad compatibility
    outputs = pipe(
        prompts,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
    )
    for ex, out in zip(examples, outputs):
        # pipeline returns dict with 'generated_text' or list with dicts
        if isinstance(out, list):
            text = out[0].get("generated_text", "")
        else:
            text = out.get("generated_text", "")
        # Try to extract JSON: find first '{' ... last '}'
        parsed = {"rating": "unclear", "rationale": text.strip()[:500]}
        try:
            start = text.find('{')
            end = text.rfind('}')
            if start != -1 and end != -1 and end > start:
                blob = text[start:end+1]
                candidate = json.loads(blob)
                if isinstance(candidate, dict) and 'rating' in candidate:
                    parsed = candidate
        except Exception:
            pass
        results.append({
            "example_id": ex.get("example_id"),
            "dataset_name": ex.get("dataset_name"),
            "model_id": ex.get("model_id"),
            "judge_model": pipe.model.name_or_path,
            "judge_output": parsed,
        })
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--sample", type=int, default=100)
    parser.add_argument("--judge_model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    args = parser.parse_args()

    data = read_jsonl(args.input_jsonl)
    random.shuffle(data)
    sample = data[: args.sample]
    print(f"Loaded {len(sample)} examples from {args.input_jsonl}")
    print(f"Judge model: {args.judge_model}")
    print(f"Batch size: {args.batch_size}, max_new_tokens: {args.max_new_tokens}")

    # Load HF model for judging
    # Prefer float16 on CUDA for broader stability; use float32 on CPU
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device_map = "auto" if torch.cuda.is_available() else None

    tok = AutoTokenizer.from_pretrained(args.judge_model, use_fast=True)
    # Ensure a valid pad_token_id to avoid scatter/gather index issues during batching
    if tok.pad_token_id is None:
        if tok.eos_token_id is not None:
            tok.pad_token = tok.eos_token
        else:
            # As a last resort, add a pad token token and resize embeddings
            tok.add_special_tokens({"pad_token": "<|pad|>"})
    # Prefer left padding for causal LM batched generation
    tok.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(args.judge_model, device_map=device_map, dtype=dtype)
    # If we added a pad token, resize embeddings
    if getattr(tok, "pad_token_id", None) is not None and model.get_input_embeddings().num_embeddings != len(tok):
        model.resize_token_embeddings(len(tok))
    # Align model pad/eos ids
    if tok.pad_token_id is not None:
        model.config.pad_token_id = tok.pad_token_id
        if hasattr(model, "generation_config"):
            model.generation_config.pad_token_id = tok.pad_token_id
    pipe = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tok,
        device_map=device_map,
        return_full_text=False,
    )
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Model loaded on {device_str} with dtype {dtype}")

    # Run in small batches
    os.makedirs(os.path.dirname(args.output_jsonl) or ".", exist_ok=True)
    with open(args.output_jsonl, "w", encoding="utf-8") as f:
        for i in tqdm(range(0, len(sample), args.batch_size), desc="Judging", unit="batch"):
            batch = sample[i:i+args.batch_size]
            results = judge_examples(
                pipe,
                batch,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )
            for r in results:
                f.write(json.dumps(r) + "\n")
            f.flush()
    print(f"Judging complete. Wrote: {args.output_jsonl}")


if __name__ == "__main__":
    main()
