# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project does

**PhysicsProvoker** — a QLoRA fine-tuning pipeline that trains small LLMs to generate research-depth idea provocations. Given a topic or paper, the model responds with a non-obvious reframing intended to open doors a researcher hadn't considered.

Two corpora are supported:
- **Wikipedia** (multi-domain, 9 domains, ~365 articles) — original v1–v3 pipeline
- **arXiv physics** (quant-ph / hep-th / gr-qc / cond-mat, ~1090 abstracts) — v4 physics-specialist pipeline

## Environment setup

```
pip install -r requirements.txt
```

Training dependencies are commented out in `requirements.txt` (to avoid conflicts on data-only machines). Install separately when running `finetune.py`:
```
pip install torch transformers>=4.45 peft>=0.13 trl>=0.11 bitsandbytes>=0.43 accelerate>=0.34 datasets>=2.21
```

Copy `.env.example` to `.env` and fill in credentials. Auth precedence in all scripts: `ANTHROPIC_API_KEY` → `ANTHROPIC_AUTH_TOKEN` → `~/.claude/.credentials.json` (Claude Code OAuth).

**Windows encoding**: `generate_pairs.py` reconfigures stdout to UTF-8 at import time (physics rationales contain Greek/math chars that crash cp1252). No `PYTHONUTF8=1` needed for that script, but `run_eval.py` still requires it:
```
$env:PYTHONUTF8=1; python eval/run_eval.py
```

## Pipeline commands

### Full Wikipedia pipeline (steps run in order)
```
python run_pipeline.py              # index + generate
python run_pipeline.py index        # build Wikipedia embedding index only
python run_pipeline.py generate     # generate pairs only (resumes if interrupted)
```

### arXiv physics pipeline
```
python data_prep/build_arxiv_index.py               # fetch papers + build embeddings
python data_prep/generate_pairs.py --corpus arxiv   # generate physics pairs
```

Pair generation is resumable — on 429 rate limits the script stops and prints how many pairs are saved. Re-run to continue from where it left off.

### Local training (Qwen3-0.6B, 4 GB GPU)
```
python finetune.py
```
All hyperparams are overridable via `.env` (`EPOCHS`, `LR`, `LORA_R`, `LORA_ALPHA`, `BATCH_SIZE`, `GRAD_ACCUM`, `MAX_SEQ_LEN`). Output: `output/qwen3-idea-lora/`.

### Colab training (Qwen3-1.7B or 8B)
Open the matching notebook and upload the `.jsonl` file in cell 3:
- `colab_qwen3_1_7b_qlora.ipynb` — T4, uses Wikipedia pairs
- `colab_qwen3_8b_physics_qlora.ipynb` — A100 recommended, uses arXiv pairs

### Inference
```
python inference.py "What's a non-obvious way to think about quantum decoherence?"
python inference.py "On black holes:" --compare     # base vs tuned side-by-side
python inference.py --interactive
```

### Evaluation
```
$env:PYTHONUTF8=1; python eval/run_eval.py
$env:PYTHONUTF8=1; python eval/run_eval.py --judge-model claude-haiku-4-5 --tag physics
```
Outputs `eval/results.jsonl` and `eval/summary.txt`. The judge is Claude with positions randomized to debias.

### Paper figures and PDF
```
cd paper
python figures.py        # regenerates figures/ directory
```
The paper is `paper/main.tex` (TMLR format). Compile on Overleaf using `paper/PhysicsProvoker_overleaf.zip`, which contains `main.tex`, `main.bib`, `tmlr.sty`, `tmlr.bst`, `math_commands.tex`, and all figures.

## Architecture

### Data flow
```
build_arxiv_index.py  →  data/arxiv_corpus.jsonl + arxiv_embeddings.npy
                                    ↓
generate_pairs.py --corpus arxiv  →  data/arxiv_training_pairs.jsonl
                                    ↓
finetune.py  (or Colab notebook)  →  output/qwen3-idea-lora/
                                    ↓
inference.py / eval/run_eval.py
```

### Pair generation mechanics (`data_prep/generate_pairs.py`)
1. For each X, decide cross-domain (50%) or same-domain (50%).
2. Sample K=5 candidate Y's from the cosine-distance "sweet spot" (60th–85th percentile) — close enough to be coherent, far enough to be surprising.
3. Send X + candidates + a random INPUT_TEMPLATE + OUTPUT_STYLE to Claude (JSON-schema-constrained output).
4. Claude returns `chosen_index`, `Z` (the provocation), `novelty_score`, `rationale`.
5. Discard if `novelty_score < 7`. Append accepted pairs as JSONL (resumable).

The key training objective is the **hidden-bridge**: Y's vocabulary must not appear in Z. The prompt enforces a "leak check" — Claude strips Y's terminology and keeps only the underlying structure/mechanism.

### Two system prompts
- `SYSTEM_PROMPT_TEMPLATE` — general multi-domain (Wikipedia corpus); expects broad cross-domain analogies
- `SYSTEM_PROMPT_PHYSICS` — physics-specialist (arXiv corpus); demands named physical mechanisms, scaling arguments, symmetry principles — no popularisation-level analogies

### Training configs
| Setting | Local (0.6B) | Colab 1.7B | Colab 8B |
|---|---|---|---|
| Quantization | none (fp16) | 4-bit NF4 + fp16 compute | 4-bit NF4 + **bf16** compute |
| LoRA r / α | 16 / 32 | 16 / 32 | 32 / 64 |
| Batch / grad accum | 1 / 8 | 2 / 8 | 2 / 8 |

Qwen3-8B requires `bnb_4bit_compute_dtype=torch.bfloat16` and `bf16=True, fp16=False` in `SFTConfig` — fp16 AMP triggers a `NotImplementedError` on bf16 weights.

### Qwen3 inference quirk
Qwen3 models have a built-in "thinking" mode. Without `apply_chat_template` + `enable_thinking=False`, the model generates only `<think>` tokens which are stripped by `skip_special_tokens=True`, producing empty output. Always use the two-step pattern:
```python
text = tokenizer.apply_chat_template(
    [{"role": "user", "content": prompt}],
    add_generation_prompt=True, tokenize=False, enable_thinking=False
)
inputs = tokenizer(text, return_tensors="pt").to(model.device)
prompt_len = inputs['input_ids'].shape[1]
```

### Eval protocol
`eval/run_eval.py` generates from both base and LoRA-tuned models, then sends pairs to a Claude judge with positions randomized (A/B swap, 50%). Judge scores novelty / groundedness / openness (1–5) and picks a winner. Results are appended to `eval/results.jsonl` (safe to re-run with `--label`).

## Key file locations

| Purpose | Path |
|---|---|
| Wikipedia corpus | `data/corpus.jsonl` + `data/embeddings.npy` |
| arXiv corpus | `data/arxiv_corpus.jsonl` + `data/arxiv_embeddings.npy` |
| Wikipedia training pairs | `data/training_pairs.jsonl` |
| arXiv training pairs | `data/arxiv_training_pairs.jsonl` |
| Trained adapter | `output/qwen3-idea-lora/` |
| Eval prompts | `eval/eval_prompts.json` |
| Paper source | `paper/main.tex` + `paper/main.bib` |
| Overleaf zip | `paper/PhysicsProvoker_overleaf.zip` |
