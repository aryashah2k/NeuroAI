from typing import Dict, Any, Iterable
import os
import json


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def write_jsonl(path: str, records: Iterable[Dict[str, Any]]):
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def append_jsonl(path: str, records: Iterable[Dict[str, Any]]):
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
