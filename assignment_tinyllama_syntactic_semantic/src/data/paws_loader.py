from typing import Iterable, Dict, Any, Optional
import csv
import os

"""
PAWS loader. Two options:
1) If a local CSV path is provided, read pairs with columns: sentence1, sentence2, label (1 paraphrase, 0 non-paraphrase)
2) Else, attempt common default paths under data/raw/paws/
"""


def iter_paws(csv_path: Optional[str] = None, limit: Optional[int] = None) -> Iterable[Dict[str, Any]]:
    candidates = []
    if csv_path and os.path.exists(csv_path):
        candidates.append(csv_path)
    else:
        # Prefer the repo-local CSV placed under src/data/
        preferred = "src/data/paws_labeled_final_test.csv"
        if os.path.exists(preferred):
            candidates.append(preferred)
        # Fallback to common PAWS locations if the preferred file is missing
        for name in [
            "data/raw/paws/final/paws_wiki_labeled_final.tsv",
            "data/raw/paws/paws_wiki/train.tsv",
            "data/raw/paws/paws_wiki/dev.tsv",
            "data/raw/paws/paws_wiki/test.tsv",
        ]:
            if os.path.exists(name):
                candidates.append(name)
    if not candidates:
        raise FileNotFoundError(
            "PAWS CSV/TSV not found. Provide --paws_path to a local file from the official repo."
        )

    count = 0
    for path in candidates:
        with open(path, "r", encoding="utf-8") as f:
            # Try to detect delimiter
            dialect = csv.Sniffer().sniff(f.read(2048))
            f.seek(0)
            reader = csv.DictReader(f, dialect=dialect)
            # Validate header fields
            fieldnames = [fn.strip() for fn in (reader.fieldnames or [])]
            expected_any = [
                {"sentence1", "sentence2", "label"},
                {"sentence_1", "sentence_2", "label"},
                {"text_a", "text_b", "label"},
                {"sentence1", "sentence2", "gold_label"},
            ]
            header_ok = any(s.issubset(set(fieldnames)) for s in expected_any)
            if not header_ok:
                raise ValueError(
                    f"Unexpected PAWS header in {path}. Found: {fieldnames}. "
                    "Expected columns to include one of the sets: "
                    "{sentence1,sentence2,label} or aliases."
                )
            # Normalize column names
            for row in reader:
                s1 = row.get("sentence1") or row.get("sentence_1") or row.get("text_a")
                s2 = row.get("sentence2") or row.get("sentence_2") or row.get("text_b")
                label = row.get("label") or row.get("gold_label") or row.get("paraphrase")
                if s1 is None or s2 is None or label is None:
                    continue
                try:
                    y = int(label)
                except Exception:
                    # Some files use '0'/'1' or 'not_paraphrase'/'paraphrase'
                    y = 1 if str(label).strip().lower() in {"1", "paraphrase", "true"} else 0
                yield {
                    "id": f"paws:{count}",
                    "sentence1": s1.strip(),
                    "sentence2": s2.strip(),
                    "label": y,
                }
                count += 1
                if limit is not None and count >= limit:
                    return
