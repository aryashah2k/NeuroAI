from typing import Iterable, Dict, Any, Optional
from datasets import load_dataset


def iter_cola(split: str = "validation", only_labeled: bool = True) -> Iterable[Dict[str, Any]]:
    ds = load_dataset("glue", "cola")
    dset = ds[split]
    for i, ex in enumerate(dset):
        if only_labeled and ex.get("label", -1) == -1:
            continue
        yield {
            "id": f"cola:{split}:{i}",
            "sentence": ex["sentence"],
            "label": int(ex["label"]),  # 1 acceptable, 0 unacceptable
        }
