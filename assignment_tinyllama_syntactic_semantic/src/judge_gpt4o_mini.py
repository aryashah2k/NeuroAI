import argparse
import os
import json
import random
from typing import List, Dict, Any

from openai import OpenAI


def make_prompt(example: Dict[str, Any]) -> str:
    parts = [
        "You are judging whether the model relied more on syntax or semantics for this example.",
        "Consider the inputs, the model's NLLs/PPLs and embedding similarity.",
    ]
    if "input" in example:
        parts.append(f"Input: {example['input']}")
    else:
        parts.append(f"Input A: {example.get('input_a','')}")
        parts.append(f"Input B: {example.get('input_b','')}")
    for k in ["nll", "ppl", "nll_a", "ppl_a", "nll_b", "ppl_b", "embedding_cosine", "decision", "label"]:
        if k in example and example[k] is not None:
            parts.append(f"{k}: {example[k]}")
    parts.append(
        "Answer JSON with keys: {\"rating\": \"syntax|semantics|unclear\", \"rationale\": string}. Keep it brief."
    )
    return "\n".join(parts)


def judge_examples(client: OpenAI, examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    results = []
    for ex in examples:
        prompt = make_prompt(ex)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        try:
            content = resp.choices[0].message.content
            parsed = json.loads(content)
        except Exception:
            parsed = {"rating": "unclear", "rationale": content}
        results.append({
            "example_id": ex.get("example_id"),
            "dataset_name": ex.get("dataset_name"),
            "model_id": ex.get("model_id"),
            "judge_model": "gpt-4o-mini",
            "judge_output": parsed,
        })
    return results


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--sample", type=int, default=100)
    args = parser.parse_args()

    data = read_jsonl(args.input_jsonl)
    random.shuffle(data)
    sample = data[: args.sample]

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set in environment")
    client = OpenAI(api_key=api_key)

    results = judge_examples(client, sample)
    os.makedirs(os.path.dirname(args.output_jsonl) or ".", exist_ok=True)
    with open(args.output_jsonl, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
