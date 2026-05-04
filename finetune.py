o"""
Fine-tune Qwen3-0.6B with LoRA on (input, output) pairs from
data/training_pairs.jsonl. GPU-friendly without bitsandbytes (works on
older compute capabilities like Maxwell that can't run 4-bit quantization).

Memory plan for a 4GB GPU:
  base in fp16 on cuda          ~1.2 GB
  LoRA adapters (fp32)          ~0.05 GB
  activations (grad-checkpointed) ~0.5 GB
  optimizer state (LoRA only)   ~0.2 GB
  total                         ~2 GB, comfortably under 4 GB.

Run after data is generated:
    python finetune.py

Output adapters: ./output/qwen3-idea-lora/

Override defaults via .env or shell env vars:
    BASE_MODEL    (default Qwen/Qwen3-0.6B)
    EPOCHS        (default 3)
    LR            (default 2e-4)
    LORA_R        (default 16)
    LORA_ALPHA    (default 32)
    MAX_SEQ_LEN   (default 768)
    BATCH_SIZE    (default 1)
    GRAD_ACCUM    (default 8)
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from dotenv import load_dotenv
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

ROOT = Path(__file__).resolve().parent
PAIRS_PATH = ROOT / "data" / "training_pairs.jsonl"
OUTPUT_DIR = ROOT / "output" / "qwen3-idea-lora"

load_dotenv(ROOT / ".env")

BASE_MODEL = os.getenv("BASE_MODEL", "Qwen/Qwen3-0.6B")
EPOCHS = int(os.getenv("EPOCHS", "3"))
LR = float(os.getenv("LR", "2e-4"))
LORA_R = int(os.getenv("LORA_R", "16"))
LORA_ALPHA = int(os.getenv("LORA_ALPHA", "32"))
MAX_SEQ_LEN = int(os.getenv("MAX_SEQ_LEN", "768"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "1"))
GRAD_ACCUM = int(os.getenv("GRAD_ACCUM", "8"))


def load_pairs() -> Dataset:
    if not PAIRS_PATH.exists():
        raise SystemExit(f"Missing {PAIRS_PATH}. Run data_prep/generate_pairs.py first.")
    rows = [json.loads(line) for line in PAIRS_PATH.open(encoding="utf-8")]
    if not rows:
        raise SystemExit(f"{PAIRS_PATH} is empty.")
    print(f"Loaded {len(rows)} training pairs.")
    return Dataset.from_list(rows)


def format_example(example: dict) -> dict:
    text = f"{example['input']}\n\n{example['output']}"
    return {"text": text}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        gpu_name = torch.cuda.get_device_name(0)
        vram_mb = torch.cuda.get_device_properties(0).total_memory // 1024**2
        print(f"CUDA: {gpu_name}, {vram_mb} MB VRAM")
    else:
        print("Running on CPU.")

    # fp16 saves memory; on older GPUs (Maxwell/Pascal) it works without
    # Tensor Core acceleration -- slower per step than Ampere+, still fits.
    model_dtype = torch.float16 if use_cuda else torch.float32

    print(f"Loading {BASE_MODEL} in {model_dtype}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=model_dtype,
        device_map="auto" if use_cuda else None,
        trust_remote_code=True,
    )
    model.gradient_checkpointing_enable()
    model.config.use_cache = False  # required when gradient_checkpointing is on
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    lora_cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    ds = load_pairs().map(format_example)

    sft_cfg = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=1,
        bf16=False,
        fp16=False,
        optim="adamw_torch",
        report_to="none",
        max_length=MAX_SEQ_LEN,
        dataset_text_field="text",
        packing=False,
        dataloader_num_workers=0,  # Windows-friendly
        gradient_checkpointing=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=ds,
        processing_class=tokenizer,
    )
    print("Starting training...")
    trainer.train()

    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\nSaved LoRA adapters to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
