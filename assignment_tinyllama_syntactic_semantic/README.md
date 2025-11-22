# TinyLlama Syntax vs Semantics Experiment

This project evaluates TinyLlama on syntax vs semantics sensitivity using BLiMP, PAWS, and CoLA with three metrics: perplexity, pairwise probability accuracy, and embedding similarity. Results are saved as JSONL for downstream LLM-as-a-Judge using GPT-4o-mini.

## Quickstart

1) Create a virtual environment and install requirements:

```bash
pip install -r requirements.txt
```

2) Run a smoke test on a tiny subset (CPU/GPU will be auto-detected):

```bash
python -m src.run_experiment --dataset blimp --subset wh_questions_subject_gap --limit 50 --output_dir results
```

3) Run PAWS (requires local CSVs or HF dataset if available):

```bash
python -m src.run_experiment --dataset paws --limit 200 --output_dir results
```

4) Optional CoLA:

```bash
python -m src.run_experiment --dataset cola --limit 500 --output_dir results
```

5) Judge a sample via GPT-4o-mini (set `OPENAI_API_KEY`):

```bash
python -m src.judge_gpt4o_mini --input_jsonl results/blimp.jsonl --output_jsonl results/judge/blimp_judged.jsonl --sample 100
```

## File Structure

- `src/data/`: dataset loaders
- `src/models/`: model and tokenizer loader
- `src/metrics/`: perplexity, pairwise accuracy, embeddings
- `src/run_experiment.py`: orchestration CLI
- `src/judge_gpt4o_mini.py`: LLM-as-a-Judge
- `results/`: outputs written here

## Notes
- Uses strided/sliding-window perplexity as in HF docs.
- Avoids chat templates for scoring plain sentences.
- Embeddings obtained via mean pooling over last hidden states.
