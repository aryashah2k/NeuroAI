# LLaMA Ablation Study Framework

This repository provides a reproducible framework to run ablation studies on LLaMA-family causal language models using PyTorch and Hugging Face. It supports three ablation mechanisms, multiple evaluation tasks, and exports JSON artifacts per run for later analysis.

## Features

- Attention head ablation via forward hooks on attention output projections
- Layer or submodule zeroing (attention-only, MLP-only, or full block)
- Input embedding masking by zeroing a fraction of hidden dims
- Tasks:
  - WikiText-2 next-token prediction (perplexity)
  - GLUE SST-2 sentiment classification (generative forced-choice accuracy)
  - NEW: Subject–Verb Agreement syntactic probe (forced-choice accuracy, avg logprob margin)
- JSON outputs per run with full metadata
- Aggregator script to combine results into a CSV

## Environment

Python 3.10+ recommended.

Install dependencies:

```bash
pip install -r requirements.txt
```

Note: If you do not have access to Meta LLaMA weights, the configs use the open checkpoint `TinyLlama/TinyLlama-1.1B-Chat-v1.0` by default.

## Usage

Run experiments via the CLI runner. Examples:

- Baseline WikiText-2 perplexity:

```bash
python -m src.run_experiment --config configs/exp_baseline_wikitext.json --outdir results
```

- Head ablation sweep on WikiText-2 (0–100%):

```bash
python -m src.run_experiment --config configs/exp_heads_wikitext_sweep.json --outdir results
```

- Layer ablation (attention-only) sweep on SST-2:

```bash
python -m src.run_experiment --config configs/exp_layers_sst2_sweep_attn.json --outdir results
```

- Embedding masking sweep on WikiText-2:

```bash
python -m src.run_experiment --config configs/exp_embmask_wikitext_sweep.json --outdir results
```

- NEW: Subject–Verb Agreement syntax probe baseline:

```bash
python -m src.run_experiment --config configs/exp_baseline_syntax.json --outdir results
```

- NEW: Syntax probe under attention-only / MLP-only / full block layer zeroing sweeps:

```bash
python -m src.run_experiment --config configs/exp_layers_syntax_sweep_attn.json --outdir results
python -m src.run_experiment --config configs/exp_layers_syntax_sweep_mlp.json --outdir results
python -m src.run_experiment --config configs/exp_layers_syntax_sweep_block.json --outdir results
```

Aggregate results into a CSV:

```bash
python scripts/aggregate_results.py --results results --out results_summary.csv
```

## Config Schema

Each config JSON may include:

```json
{
  "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
  "task": "wikitext_lm" | "sst2_cls" | "syntax_probe",
  "split": "validation" | "test",
  "batch_size": 4,
  "max_length": 512,
  "max_examples": 200,                      // only for syntax_probe
  "ablation_type": "none" | "heads" | "layers" | "emb_mask",
  "component": "attn" | "mlp" | "block",  // only for layers
  "selection_policy": "random" | "consecutive",
  "severity": 0.25 or [0.0, 0.25, 0.5, 0.75, 1.0],
  "seeds": [1, 2, 3],
  "notes": "string"
}
```

## Outputs

Each run writes a JSON to `results/` with keys:

- `model_id`, `model_revision`, `model_config`
- `task`, `split`
- `ablation_type`, `component`, `severity`, `selection_policy`, `selection_indices`
- `seed`
- `metrics` (e.g., `perplexity` for LM, `accuracy` for SST-2)
- `num_samples`, `runtime_s`, `device_info`, `timestamp`

## Sanity Checks

- Baseline runs should produce reasonable perplexity/accuracy for the chosen checkpoint.
- Single-head zeroing should change logits for a fixed input.
- 100% embedding mask should degrade metrics severely.

## Notes

- The SST-2 evaluator uses a generative forced-choice approach to remain compatible with causal-only checkpoints.
- Hooks are removed after each run to keep the model graph unmodified across runs.
 - The syntax probe implements a lightweight subject–verb agreement test using forced-choice between correct vs. incorrect verb forms. Metrics: `accuracy`, `avg_logprob_margin`.
