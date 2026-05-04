"""
End-to-end orchestrator. Runs the data-prep phases in order; training is launched
manually via `python finetune.py` once you've reviewed the generated Z's.

    python run_pipeline.py            # build index then generate pairs
    python run_pipeline.py index      # only build the Wikipedia embedding index
    python run_pipeline.py generate   # only run pair generation (Claude)
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(cmd: list[str]) -> None:
    print(f"\n>>> {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        sys.exit(f"Step failed with exit code {result.returncode}: {' '.join(cmd)}")


def main():
    args = sys.argv[1:]
    phase = args[0] if args else "all"

    if phase in ("index", "all"):
        run([sys.executable, str(ROOT / "data_prep" / "build_index.py")])
    if phase in ("generate", "all"):
        run([sys.executable, str(ROOT / "data_prep" / "generate_pairs.py")])

    print("\nData prep complete.")
    print(f"Inspect: {ROOT / 'data' / 'training_pairs.jsonl'}")
    print("When the Z quality looks good, run: python finetune.py")


if __name__ == "__main__":
    main()
