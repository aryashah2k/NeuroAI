from typing import Dict, Any, List, Tuple, Optional

import math
from datasets import load_dataset
from transformers import PreTrainedTokenizerBase
import torch
from torch.utils.data import DataLoader


def load_wikitext(tokenizer: PreTrainedTokenizerBase,
                  name: str = "wikitext-2-raw-v1",
                  split: str = "validation",
                  max_length: int = 1024,
                  batch_size: int = 8,
                  num_workers: int = 0,
                  ) -> Tuple[DataLoader, int]:
    ds = load_dataset("wikitext", name)[split]

    def tokenize(example: Dict[str, Any]) -> Dict[str, Any]:
        return tokenizer(example["text"])  # do not set truncation here

    tokenized = ds.map(tokenize, batched=True, remove_columns=["text"])

    # Concatenate all tokens and chunk into max_length sequences
    all_ids: List[int] = []
    for rec in tokenized:
        all_ids.extend(rec["input_ids"])

    # Drop the remainder to keep consistent chunking
    total_len = (len(all_ids) // max_length) * max_length
    all_ids = all_ids[:total_len]

    input_ids = torch.tensor(all_ids, dtype=torch.long)
    input_ids = input_ids.view(-1, max_length)

    # Labels are next-token prediction: shift by one within each row
    labels = input_ids.clone()

    dataset = torch.utils.data.TensorDataset(input_ids, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return loader, total_len


def load_sst2(tokenizer: PreTrainedTokenizerBase,
              split: str = "validation",
              max_length: int = 256,
              batch_size: int = 16,
              num_workers: int = 0,
              ) -> Tuple[DataLoader, int, List[int], List[str]]:
    ds = load_dataset("glue", "sst2")[split]

    def tokenize(example: Dict[str, Any]) -> Dict[str, Any]:
        enc = tokenizer(
            example["sentence"],
            truncation=True,
            padding=False,
            max_length=max_length,
            add_special_tokens=True,
        )
        return enc

    tokenized = ds.map(tokenize, batched=True, remove_columns=["sentence"])

    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]
    labels = tokenized["label"] if "label" in tokenized.column_names else [None] * len(input_ids)

    # to tensors
    input_ids = [torch.tensor(x, dtype=torch.long) for x in input_ids]
    attention_mask = [torch.tensor(x, dtype=torch.long) for x in attention_mask]
    labels_t = torch.tensor(labels, dtype=torch.long) if labels[0] is not None else None

    # collate function for variable-length sequences
    def collate(batch):
        ids = [b[0] for b in batch]
        masks = [b[1] for b in batch]
        lbls = [b[2] for b in batch] if batch[0][2] is not None else None
        padded = tokenizer.pad({"input_ids": ids, "attention_mask": masks}, return_tensors="pt")
        if lbls is not None:
            padded["labels"] = torch.tensor(lbls, dtype=torch.long)
        return padded

    # build list dataset
    list_ds = []
    for i in range(len(input_ids)):
        list_ds.append((input_ids[i], attention_mask[i], labels[i] if labels_t is not None else None))

    loader = DataLoader(list_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate)
    return loader, len(list_ds), ds["sentence"], ds.features["label"].names if "label" in ds.features else ["negative", "positive"]
