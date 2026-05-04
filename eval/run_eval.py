"""
Evaluate the LoRA-tuned model against the base model on held-out idea-provocation
prompts. For each prompt:
  1. Generate from base and from base+LoRA.
  2. Send (prompt, A=base, B=tuned) to a Claude judge with positions randomized.
  3. Judge returns: which is more idea-provoking (A/B/tie), and three 1-5 scores
     for each side: novelty, groundedness, openness.

Outputs:
    eval/results.jsonl   -- one record per prompt with both responses + judgement
    eval/summary.txt     -- aggregate metrics (win rate, mean score deltas)

Usage:
    PYTHONUTF8=1 python eval/run_eval.py
    PYTHONUTF8=1 python eval/run_eval.py --judge-model claude-haiku-4-5 --runs-per-prompt 1
    PYTHONUTF8=1 python eval/run_eval.py --tag everyday   # filter prompts by tag
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import torch
from dotenv import load_dotenv
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

import anthropic

ROOT = Path(__file__).resolve().parents[1]
ADAPTER_DIR = ROOT / "output" / "qwen3-idea-lora"
EVAL_DIR = ROOT / "eval"
PROMPTS_PATH = EVAL_DIR / "eval_prompts.json"
RESULTS_PATH = EVAL_DIR / "results.jsonl"
SUMMARY_PATH = EVAL_DIR / "summary.txt"

load_dotenv(ROOT / ".env")
BASE_MODEL = os.getenv("BASE_MODEL", "Qwen/Qwen3-0.6B")
GEN_MAX_NEW = int(os.getenv("EVAL_MAX_NEW", "350"))
GEN_TEMP = float(os.getenv("EVAL_TEMPERATURE", "0.85"))
GEN_TOP_P = float(os.getenv("EVAL_TOP_P", "0.92"))


JUDGE_SYSTEM = """You are evaluating two LLM responses for IDEA PROVOCATION.

For each (prompt, response A, response B) you receive, decide which response is more idea-provoking. Idea-provoking means: it offers a non-obvious framing, makes the reader pause and reconsider, or opens a productive direction the reader hadn't seen.

Score both responses on three 1-5 dimensions:
  novelty       -- does it offer a non-obvious angle or just restate common knowledge?
  groundedness  -- is it defensible / does it make sense / is it not gibberish?
  openness      -- does it open a door (more questions, new framings) or close it (summary, definition)?

Then pick a winner: "A", "B", or "tie".

Bias warning: do NOT favor longer responses, more confident tone, or use of jargon. Confident-sounding nonsense should score low on groundedness. A short, sharp insight can be more idea-provoking than a long polished essay.

Return strict JSON:
{
  "scores_A": {"novelty": int, "groundedness": int, "openness": int},
  "scores_B": {"novelty": int, "groundedness": int, "openness": int},
  "winner": "A" | "B" | "tie",
  "reason": "one short sentence"
}"""


JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "scores_A": {
            "type": "object",
            "properties": {
                "novelty": {"type": "integer"},
                "groundedness": {"type": "integer"},
                "openness": {"type": "integer"},
            },
            "required": ["novelty", "groundedness", "openness"],
            "additionalProperties": False,
        },
        "scores_B": {
            "type": "object",
            "properties": {
                "novelty": {"type": "integer"},
                "groundedness": {"type": "integer"},
                "openness": {"type": "integer"},
            },
            "required": ["novelty", "groundedness", "openness"],
            "additionalProperties": False,
        },
        "winner": {"type": "string"},
        "reason": {"type": "string"},
    },
    "required": ["scores_A", "scores_B", "winner", "reason"],
    "additionalProperties": False,
}


# ---------- auth (mirrors generate_pairs.py) ----------

def get_anthropic_client() -> anthropic.Anthropic:
    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if api_key and api_key != "sk-ant-...":
        return anthropic.Anthropic(api_key=api_key)
    auth_token = os.getenv("ANTHROPIC_AUTH_TOKEN", "").strip()
    if not auth_token:
        cred_path = Path.home() / ".claude" / ".credentials.json"
        if cred_path.exists():
            try:
                creds = json.loads(cred_path.read_text(encoding="utf-8"))
                auth_token = creds.get("claudeAiOauth", {}).get("accessToken", "")
            except Exception as e:
                print(f"Could not read {cred_path}: {e}", file=sys.stderr)
    if auth_token:
        print("Using Claude subscription OAuth token.")
        os.environ.pop("ANTHROPIC_API_KEY", None)
        return anthropic.Anthropic(
            auth_token=auth_token,
            default_headers={"anthropic-beta": "oauth-2025-04-20"},
        )
    sys.exit("No Claude credentials found.")


# ---------- model loading ----------

def load_model(load_adapter: bool):
    use_cuda = torch.cuda.is_available()
    dtype = torch.float16 if use_cuda else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=dtype,
        device_map="auto" if use_cuda else None,
        trust_remote_code=True,
    )
    if load_adapter:
        if not ADAPTER_DIR.exists():
            sys.exit(f"No adapter at {ADAPTER_DIR}. Train first.")
        model = PeftModel.from_pretrained(model, str(ADAPTER_DIR))
    model.eval()
    return model, tokenizer


@torch.no_grad()
def generate(model, tokenizer, prompt: str, seed: int) -> str:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=GEN_MAX_NEW,
        do_sample=True,
        temperature=GEN_TEMP,
        top_p=GEN_TOP_P,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.pad_token_id,
    )
    text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return text.strip()


# ---------- judge ----------

def judge(client, judge_model: str, prompt: str, response_A: str, response_B: str) -> dict | None:
    user_msg = (
        f"PROMPT:\n{prompt}\n\n"
        f"RESPONSE A:\n{response_A}\n\n"
        f"RESPONSE B:\n{response_B}\n\n"
        f"Score both, pick a winner."
    )
    try:
        resp = client.messages.create(
            model=judge_model,
            max_tokens=800,
            system=[
                {
                    "type": "text",
                    "text": JUDGE_SYSTEM,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": user_msg}],
            output_config={
                "format": {"type": "json_schema", "schema": JUDGE_SCHEMA}
            },
        )
    except anthropic.RateLimitError:
        print("Judge rate-limited. Stopping.", file=sys.stderr)
        return None
    except anthropic.APIError as e:
        print(f"Judge API error: {e}", file=sys.stderr)
        return None
    text_block = next((b for b in resp.content if b.type == "text"), None)
    if not text_block:
        return None
    try:
        return json.loads(text_block.text)
    except json.JSONDecodeError as e:
        print(f"Judge parse error: {e}", file=sys.stderr)
        return None


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge-model", default="claude-haiku-4-5")
    ap.add_argument("--tag", default=None, help="filter prompts by tag")
    ap.add_argument("--limit", type=int, default=None, help="cap number of prompts")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--label", default="run", help="label saved into each result row")
    args = ap.parse_args()

    EVAL_DIR.mkdir(exist_ok=True)
    prompts = json.loads(PROMPTS_PATH.read_text(encoding="utf-8"))
    if args.tag:
        prompts = [p for p in prompts if p.get("tag") == args.tag]
    if args.limit:
        prompts = prompts[: args.limit]
    print(f"Evaluating on {len(prompts)} prompts.")

    print("Loading base model...")
    base_model, tok = load_model(load_adapter=False)
    base_outputs: list[str] = []
    for i, p in enumerate(prompts, 1):
        print(f"  [base {i}/{len(prompts)}] {p['prompt'][:60]!r}")
        base_outputs.append(generate(base_model, tok, p["prompt"], seed=args.seed + i))
    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Loading tuned model...")
    tuned_model, tok = load_model(load_adapter=True)
    tuned_outputs: list[str] = []
    for i, p in enumerate(prompts, 1):
        print(f"  [tuned {i}/{len(prompts)}] {p['prompt'][:60]!r}")
        tuned_outputs.append(generate(tuned_model, tok, p["prompt"], seed=args.seed + i))
    del tuned_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\nJudging...")
    client = get_anthropic_client()
    rng = random.Random(args.seed)

    rows: list[dict] = []
    out_f = RESULTS_PATH.open("a", encoding="utf-8")
    for i, (p, base_out, tuned_out) in enumerate(zip(prompts, base_outputs, tuned_outputs), 1):
        # Randomize position to debias the judge
        tuned_is_A = rng.random() < 0.5
        if tuned_is_A:
            A, B = tuned_out, base_out
            mapping = {"A": "tuned", "B": "base"}
        else:
            A, B = base_out, tuned_out
            mapping = {"A": "base", "B": "tuned"}

        print(f"  [judge {i}/{len(prompts)}] tuned={'A' if tuned_is_A else 'B'} -- {p['prompt'][:60]!r}")
        verdict = judge(client, args.judge_model, p["prompt"], A, B)
        if verdict is None:
            print("    (skipped)")
            continue

        winner_role = mapping.get(verdict["winner"], "tie")
        scores_tuned = verdict["scores_A"] if tuned_is_A else verdict["scores_B"]
        scores_base = verdict["scores_B"] if tuned_is_A else verdict["scores_A"]

        record = {
            "label": args.label,
            "prompt": p["prompt"],
            "tag": p.get("tag"),
            "base_output": base_out,
            "tuned_output": tuned_out,
            "tuned_was_A": tuned_is_A,
            "winner_role": winner_role,
            "scores_tuned": scores_tuned,
            "scores_base": scores_base,
            "judge_reason": verdict.get("reason"),
        }
        rows.append(record)
        out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
        out_f.flush()
    out_f.close()

    if not rows:
        print("No judgements produced.")
        return

    # ---------- aggregate ----------
    n = len(rows)
    tuned_wins = sum(1 for r in rows if r["winner_role"] == "tuned")
    base_wins = sum(1 for r in rows if r["winner_role"] == "base")
    ties = sum(1 for r in rows if r["winner_role"] == "tie")

    def avg(records, key1, key2):
        return sum(r[key1][key2] for r in records) / len(records)

    summary_lines = [
        f"=== eval ({args.label}) ===",
        f"prompts judged: {n}",
        f"tuned wins: {tuned_wins} ({100*tuned_wins/n:.1f}%)",
        f"base wins:  {base_wins} ({100*base_wins/n:.1f}%)",
        f"ties:       {ties} ({100*ties/n:.1f}%)",
        "",
        f"avg novelty       tuned={avg(rows,'scores_tuned','novelty'):.2f}  base={avg(rows,'scores_base','novelty'):.2f}",
        f"avg groundedness  tuned={avg(rows,'scores_tuned','groundedness'):.2f}  base={avg(rows,'scores_base','groundedness'):.2f}",
        f"avg openness      tuned={avg(rows,'scores_tuned','openness'):.2f}  base={avg(rows,'scores_base','openness'):.2f}",
    ]
    summary = "\n".join(summary_lines)
    print("\n" + summary)
    SUMMARY_PATH.write_text(summary + "\n", encoding="utf-8")
    print(f"\nWrote {RESULTS_PATH} and {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
