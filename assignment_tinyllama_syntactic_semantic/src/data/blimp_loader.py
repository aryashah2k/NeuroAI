from typing import Iterable, Dict, Any, Optional
from datasets import load_dataset, get_dataset_config_names


def iter_blimp(subset: Optional[str] = None, split: str = "train") -> Iterable[Dict[str, Any]]:
    """
    Yields BLiMP minimal pairs with keys: sentence_good, sentence_bad, field (phenomenon), and unique id.
    subset: specific BLiMP sub-dataset name, or None for all.
    """
    if subset is not None:
        # Load the specific BLiMP config
        ds = load_dataset("nyu-mll/blimp", subset)
        dset = ds[split]
        for i, ex in enumerate(dset):
            yield {
                "id": f"{subset}:{i}",
                "subset": subset,
                "sentence_good": ex["sentence_good"],
                "sentence_bad": ex["sentence_bad"],
                "field": ex.get("field", subset),
            }
    else:
        # Iterate all available BLiMP configs
        configs = get_dataset_config_names("nyu-mll/blimp")
        for name in configs:
            ds = load_dataset("nyu-mll/blimp", name)
            dset = ds[split]
            for i, ex in enumerate(dset):
                yield {
                    "id": f"{name}:{i}",
                    "subset": name,
                    "sentence_good": ex["sentence_good"],
                    "sentence_bad": ex["sentence_bad"],
                    "field": ex.get("field", name),
                }
