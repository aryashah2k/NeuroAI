import json
import os
from glob import glob
from typing import List, Dict, Any

import pandas as pd


def load_results(results_dir: str) -> List[Dict[str, Any]]:
    rows = []
    for path in glob(os.path.join(results_dir, "*.json")):
        try:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
                obj["__path"] = path
                rows.append(obj)
        except Exception as e:
            print(f"Skip {path}: {e}")
    return rows


def _flatten_device_info(device_info: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not isinstance(device_info, dict):
        return out
    for k, v in device_info.items():
        out[f"device_{k}"] = v
    return out


def _normalize_row(r: Dict[str, Any]) -> Dict[str, Any]:
    base = {
        "timestamp": r.get("timestamp"),
        "json_path": r.get("__path"),
        "model_id": r.get("model_id"),
        "model_revision": r.get("model_revision"),
        "task": r.get("task"),
        "split": r.get("split"),
        "ablation_type": r.get("ablation_type"),
        "component": r.get("component"),
        "selection_policy": r.get("selection_policy"),
        "severity": r.get("severity"),
        "seed": r.get("seed"),
        "num_samples": r.get("num_samples"),
        "runtime_s": r.get("runtime_s"),
        "notes": r.get("notes"),
    }
    # selection indices may be nested; store JSON string for fidelity
    sel = r.get("selection_indices")
    if sel is not None:
        try:
            base["selection_indices_json"] = json.dumps(sel, separators=(",", ":"))
        except Exception:
            base["selection_indices_json"] = str(sel)

    # device info flattened
    base.update(_flatten_device_info(r.get("device_info", {})))

    # metrics passthrough
    metrics = r.get("metrics", {}) or {}
    for k, v in metrics.items():
        base[k] = v

    return base


def summarize(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    recs = [_normalize_row(r) for r in rows]
    df = pd.DataFrame(recs)
    return df


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=str, default="results", help="Directory containing JSON results")
    ap.add_argument("--out", type=str, default="results_summary.csv", help="Combined CSV path")
    ap.add_argument("--outdir_csv", type=str, default=None, help="Directory to write per-ablation CSVs (defaults to --results)")
    args = ap.parse_args()

    rows = load_results(args.results)
    if not rows:
        print(f"No JSON results found in {args.results}")
        return

    df = summarize(rows)
    # Sort for readability
    sort_cols = ["task", "ablation_type", "component", "severity", "seed", "timestamp"]
    existing_sort_cols = [c for c in sort_cols if c in df.columns]
    if existing_sort_cols:
        df.sort_values(existing_sort_cols, inplace=True, na_position='last')

    # Write combined CSV
    out_combined = args.out
    df.to_csv(out_combined, index=False)
    print(f"Wrote combined summary to {out_combined} with {len(df)} rows")

    # Write per-ablation CSVs
    outdir = args.outdir_csv or args.results
    os.makedirs(outdir, exist_ok=True)
    if "ablation_type" not in df.columns:
        # Still write a single CSV per task as a fallback
        for task, dfg in df.groupby(df.get("task", "unknown")):
            out_path = os.path.join(outdir, f"summary_task_{task}.csv")
            dfg.to_csv(out_path, index=False)
            print(f"Wrote {len(dfg)} rows to {out_path}")
        return

    for ablt, dfg in df.groupby("ablation_type", dropna=False):
        name = str(ablt) if pd.notna(ablt) else "none"
        out_path = os.path.join(outdir, f"summary_{name}.csv")
        dfg.to_csv(out_path, index=False)
        print(f"Wrote {len(dfg)} rows to {out_path}")


if __name__ == "__main__":
    main()
