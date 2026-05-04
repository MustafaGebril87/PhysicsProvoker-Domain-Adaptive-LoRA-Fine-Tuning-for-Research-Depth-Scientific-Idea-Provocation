"""
Try the trained LoRA adapter.

Usage:
    python inference.py "On Photosynthesis:"
    python inference.py "What's a non-obvious way to think about Money?" --compare
    python inference.py --interactive

The --compare flag prints base-model output alongside the LoRA-tuned output, so
you can see what the fine-tune actually changed.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from dotenv import load_dotenv
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
ADAPTER_DIR = ROOT / "output" / "qwen3-idea-lora"

load_dotenv(ROOT / ".env")
BASE_MODEL = os.getenv("BASE_MODEL", "Qwen/Qwen3-0.6B")
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "350"))


def load(load_adapter: bool):
    use_cuda = torch.cuda.is_available()
    dtype = torch.float16 if use_cuda else torch.float32

    print(f"Loading base {BASE_MODEL} ({'cuda' if use_cuda else 'cpu'}, {dtype})...")
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
            sys.exit(f"No adapter at {ADAPTER_DIR}. Run finetune.py first.")
        print(f"Loading LoRA adapter from {ADAPTER_DIR}...")
        model = PeftModel.from_pretrained(model, str(ADAPTER_DIR))

    model.eval()
    return model, tokenizer


@torch.no_grad()
def generate(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.85,
        top_p=0.92,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.pad_token_id,
    )
    text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return text.strip()


def run_one(prompt: str, compare: bool):
    if compare:
        print("\n--- BASE MODEL (no fine-tune) ---")
        base_model, tok = load(load_adapter=False)
        print(f"\nPROMPT: {prompt}\n")
        print(generate(base_model, tok, prompt, MAX_NEW_TOKENS))
        del base_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        print("\n--- TUNED MODEL (LoRA adapter) ---")
        tuned_model, tok = load(load_adapter=True)
        print(f"\nPROMPT: {prompt}\n")
        print(generate(tuned_model, tok, prompt, MAX_NEW_TOKENS))
    else:
        model, tok = load(load_adapter=True)
        print(f"\nPROMPT: {prompt}\n")
        print(generate(model, tok, prompt, MAX_NEW_TOKENS))


def run_interactive():
    model, tok = load(load_adapter=True)
    print("\nReady. Empty line to exit.")
    while True:
        try:
            prompt = input("\nprompt> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not prompt:
            break
        print()
        print(generate(model, tok, prompt, MAX_NEW_TOKENS))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt", nargs="?", default=None)
    ap.add_argument("--compare", action="store_true", help="Show base vs tuned side-by-side")
    ap.add_argument("--interactive", action="store_true", help="REPL")
    args = ap.parse_args()

    if args.interactive:
        run_interactive()
    elif args.prompt:
        run_one(args.prompt, args.compare)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
