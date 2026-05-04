"""
Generate (input, output) training pairs that teach an LLM to provoke new ideas.

For each X drawn from the multi-domain corpus:
  1. Decide cross-domain (50%) or same-domain (50%) for Y.
  2. Sample K candidate Y's from the embedding "sweet spot" within that subset
     (60th-85th percentile distance band).
  3. Send (X, candidates) + a randomly chosen INPUT TEMPLATE + OUTPUT STYLE to
     Claude in one call. Claude returns JSON with:
       - chosen_index (int)            -- which Y was the most fertile leap
       - Z (string)                    -- the provocation, with Y subtly woven in but never named
       - novelty_score (int 1-10)      -- self-rated "would this surprise a thoughtful reader"
       - rationale (string)            -- one-line gloss for human review (not training data)
  4. Drop pairs with novelty_score < 7 (don't count toward target).

Output: data/training_pairs.jsonl with one JSON record per pair.
Resumable; halts on first 429 so the candidate pool isn't burned through.
"""

from __future__ import annotations

import io
import json
import os
import random
import sys
import time
from pathlib import Path

# Windows cp1252 stdout crashes on Greek/math chars in physics rationales.
# Reconfigure to UTF-8 at import time so no env var is needed.
if hasattr(sys.stdout, "buffer") and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
from dotenv import load_dotenv

import argparse

import anthropic

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

# Paths resolved at runtime based on --corpus flag
CORPUS_PATH = DATA_DIR / "corpus.jsonl"
EMBED_PATH = DATA_DIR / "embeddings.npy"
PAIRS_PATH = DATA_DIR / "training_pairs.jsonl"

ARXIV_CORPUS_PATH = DATA_DIR / "arxiv_corpus.jsonl"
ARXIV_EMBED_PATH = DATA_DIR / "arxiv_embeddings.npy"
ARXIV_PAIRS_PATH = DATA_DIR / "arxiv_training_pairs.jsonl"

load_dotenv(ROOT / ".env")

NUM_PAIRS = int(os.getenv("NUM_PAIRS", "200"))
K_CANDIDATES = int(os.getenv("K_CANDIDATES", "5"))
SWEETSPOT_LO = float(os.getenv("SWEETSPOT_LO", "0.60"))
SWEETSPOT_HI = float(os.getenv("SWEETSPOT_HI", "0.85"))
CROSS_DOMAIN_PROB = float(os.getenv("CROSS_DOMAIN_PROB", "0.5"))
NOVELTY_MIN = int(os.getenv("NOVELTY_MIN", "7"))
GEN_MODEL = os.getenv("GEN_MODEL", "claude-haiku-4-5")
SEED = int(os.getenv("SEED", "42"))


# ---------- prompt templates ----------

INPUT_TEMPLATES = [
    "On {title}:",
    "What's a non-obvious way to think about {title}?",
    "I'm stuck on {title}. Help me see it differently.",
    "{title} -- give me an unusual angle.",
    "Take {title} seriously for a moment. What are people missing?",
]

OUTPUT_STYLES = {
    "essay": (
        "A 130-220 word reframing that reads as a single coherent insight about X. "
        "Lead with the angle, develop it, end with what it opens up."
    ),
    "claim": (
        "An 80-160 word counterintuitive claim about X. Begin with the surface view, "
        "name the inversion, give just enough grounding to make it credible. "
        "Punchy, not preachy."
    ),
    "question": (
        "A 60-130 word piece that POSES a question about X most people don't think to ask. "
        "Set up the question with one or two sentences of context, then ask it sharply. "
        "End on the question itself, not a resolution."
    ),
    "angles": (
        "Three short angles on X (each 25-50 words), as a numbered list. "
        "Each angle should be a self-contained framing -- different lens, different stance. "
        "No introduction, no conclusion, just the three."
    ),
}


SYSTEM_PROMPT_TEMPLATE = """You are helping curate a training dataset for an LLM whose job is to PROVOKE NEW IDEAS in its user. The user comes with a topic, problem, or half-formed thought; the model should respond with something that opens a door the user hadn't seen.

You will be given:
- A primary topic X (a Wikipedia summary, possibly from any domain).
- A set of candidate inspiration topics from across domains. Each has an [index] and a domain label.
- An INPUT TEMPLATE the user will be shown.
- An OUTPUT STYLE that constrains the form of your response Z.

Your task is two steps in one response.

STEP 1 -- Pick the candidate whose connection to X is the most genuinely fertile. Avoid:
  - paraphrases or direct prerequisites of X
  - candidates whose connection is famous and well-worn
  - candidates that are merely "related" without offering a new lens
The best Y is one whose underlying STRUCTURE, MECHANISM, or PATTERN can be carried over to illuminate X in a way the reader probably hasn't encountered. Save your choice as `chosen_index`.

STEP 2 -- Write Z, a response to the INPUT TEMPLATE. Z must:
  - Read as a STANDALONE response. The reader sees only the input template and Z. They must not be able to guess what Y was.
  - Carry Y's STRUCTURE or LOGIC, but NEVER its vocabulary. Do not borrow Y's distinctive terms, named theorems, named entities, named figures, named eras, or signature phrases. If Y is "Brutalist architecture", do not say "exposed structure" or "structural honesty". If Y is "Eternalism", do not say "block universe" or "all times equally real". If Y is "Coevolution", do not say "coevolving" or "coevolution". Strip the labels; keep the shape.
  - Stay grounded -- no mysticism, no unsubstantiated metaphysics. The provocation should be defensible.
  - Match the OUTPUT STYLE constraint exactly (length range and form).
  - Be ABOUT X, with Y as flavor only. If you can't tell whose article this is, you've gone too far toward Y.
  - Open a door. The goal is "huh, I never thought of that" -- not "yes that's a fine essay."

  Before finalizing Z, do a leak check: scan your draft for any words you would only have written if you'd been thinking of Y. Replace those with neutral phrasings that describe the underlying pattern without naming it.

STEP 3 -- Self-rate `novelty_score` (1-10): "Would a thoughtful reader who already knows about X find this genuinely surprising and idea-provoking?"
  10 = "I want to send this to a friend right now."
  7  = "Solid lateral leap. Made me pause."
  5  = "Coherent but predictable."
  1  = "Bland restatement."
Be honest. If your Z is a polished restatement rather than a provocation, score it low.

STEP 4 -- One-line `rationale`: name the bridge in plain English (e.g. "X-as-flow", "X through the lens of selection pressure"). Used for review, not training.

Return strict JSON with exactly: `chosen_index` (int), `Z` (string), `novelty_score` (int), `rationale` (string)."""

SYSTEM_PROMPT_PHYSICS = """You are helping curate a training dataset for a physics research assistant whose job is to PROVOKE NEW RESEARCH DIRECTIONS. A working physicist comes with a result, open problem, or half-formed idea; the model should respond with something that opens a door they hadn't considered — at research depth, not popularisation depth.

You will be given:
- A primary paper X (arxiv abstract from physics — quant-ph, hep-th, gr-qc, or cond-mat).
- A set of candidate inspiration papers from across those subfields. Each has an [index] and a subfield label.
- An INPUT TEMPLATE the physicist will be shown.
- An OUTPUT STYLE that constrains the form of your response Z.

Your task is two steps in one response.

STEP 1 -- Pick the candidate whose connection to X is the most genuinely fertile for a working physicist. Prefer candidates where:
  - The underlying mathematical structure, scaling argument, symmetry principle, or physical mechanism in Y can be transplanted to X's setting.
  - The connection is NON-OBVIOUS — i.e., not already in the review literature on X.
  - The transplanted idea would suggest a concrete new calculation, experiment, or reframing.
Avoid:
  - Candidates that are textbook prerequisites of X.
  - Connections that are already famous in the community (e.g., AdS/CFT linking hep-th and cond-mat is known — don't re-propose it as if novel).
Save your choice as `chosen_index`.

STEP 2 -- Write Z, a response to the INPUT TEMPLATE. Z must:
  - Read as a STANDALONE response addressed to the physicist. The reader sees only the input template and Z — they must not be able to identify which specific paper Y was.
  - Carry Y's PHYSICS (mechanism, scaling, symmetry argument, technique) but NOT Y's vocabulary. Do not use Y's paper-specific terminology, named equations unique to Y, or signal phrases that would identify Y to a reader in Y's subfield.
  - Be technically grounded. Name the physical mechanism you're transplanting. If you claim a scaling argument transfers, say what scales. If you claim a symmetry applies, name the symmetry. Vague cross-domain analogies are not sufficient.
  - Open a concrete door: the provocation should suggest at least one thing the physicist could actually try — a calculation, a measurement, a reframing of an existing result.
  - Match the OUTPUT STYLE constraint exactly (length range and form).
  - Stay honest about speculation: distinguish "this mechanism might apply if…" from "this is known to work".

  Leak check: scan your draft for terminology a reader would only recognise as coming from Y's subfield. Replace with the underlying physical concept described neutrally.

STEP 3 -- Self-rate `novelty_score` (1-10) from the perspective of a working physicist in X's subfield:
  10 = "This would make me open a new notebook."
  7  = "Genuinely non-obvious. Worth a conversation with a collaborator."
  5  = "Coherent but I've seen this angle before."
  1  = "This is already in the intro of X's paper."
Be honest. Popularisation-level analogies score ≤ 4 regardless of prose quality.

STEP 4 -- One-line `rationale`: name the bridge in plain physics language (e.g. "renormalisation group flow applied to X's phase structure", "measurement-back-action formalism transplanted from quant-ph to X's gravitational setting"). Used for review, not training.

Return strict JSON with exactly: `chosen_index` (int), `Z` (string), `novelty_score` (int), `rationale` (string)."""


OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "chosen_index": {"type": "integer"},
        "Z": {"type": "string"},
        "novelty_score": {"type": "integer"},
        "rationale": {"type": "string"},
    },
    "required": ["chosen_index", "Z", "novelty_score", "rationale"],
    "additionalProperties": False,
}


# ---------- corpus loading & sampling ----------

def load_corpus(corpus_path: Path, embed_path: Path) -> tuple[list[dict], np.ndarray]:
    if not corpus_path.exists() or not embed_path.exists():
        sys.exit(
            f"Missing index. Run the appropriate build script first to create:\n"
            f"  {corpus_path}\n  {embed_path}"
        )
    corpus = [json.loads(line) for line in corpus_path.open(encoding="utf-8")]
    embeddings = np.load(embed_path)
    if len(corpus) != embeddings.shape[0]:
        sys.exit("Corpus and embeddings length mismatch -- rebuild the index.")
    return corpus, embeddings


def pick_sweet_spot_candidates(
    x_idx: int,
    embeddings: np.ndarray,
    domain_mask: np.ndarray,
    k: int,
    lo_pct: float,
    hi_pct: float,
    rng: random.Random,
) -> list[int]:
    """Sample k indices from the [lo_pct, hi_pct] distance band, restricted to domain_mask."""
    sims = embeddings @ embeddings[x_idx]
    dists = 1.0 - sims
    eligible = domain_mask.copy()
    eligible[x_idx] = False
    if eligible.sum() < k:
        return []
    eligible_dists = dists[eligible]
    eligible_idxs = np.where(eligible)[0]

    lo, hi = np.quantile(eligible_dists, [lo_pct, hi_pct])
    band_mask = (eligible_dists >= lo) & (eligible_dists <= hi)
    band = eligible_idxs[band_mask]
    if len(band) < k:
        band = eligible_idxs[eligible_dists <= hi]
    if len(band) < k:
        band = eligible_idxs
    return rng.sample(list(band), min(k, len(band)))


# ---------- prompt building ----------

def build_user_message(
    x: dict, candidates: list[dict], input_template: str, output_style_key: str
) -> str:
    cand_text = "\n\n".join(
        f"[{i}] (domain: {c['domain']}) {c['title']}\n{c['summary']}"
        for i, c in enumerate(candidates)
    )
    style_desc = OUTPUT_STYLES[output_style_key]
    return (
        f"PRIMARY TOPIC X (domain: {x['domain']}):\n{x['title']}\n{x['summary']}\n\n"
        f"CANDIDATE INSPIRATIONS:\n{cand_text}\n\n"
        f"INPUT TEMPLATE the user will see:\n  {input_template.format(title=x['title'])}\n\n"
        f"OUTPUT STYLE for Z:\n  {output_style_key}: {style_desc}\n\n"
        f"Pick the most fertile candidate, write Z in the requested style, "
        f"score novelty honestly, and give a one-line rationale."
    )


# ---------- auth ----------

def get_client() -> anthropic.Anthropic:
    """Auth precedence: ANTHROPIC_API_KEY > ANTHROPIC_AUTH_TOKEN > ~/.claude/.credentials.json
    (Claude Code OAuth, uses subscription quota)."""
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
        print("Using Claude subscription OAuth token (consumes subscription quota).")
        os.environ.pop("ANTHROPIC_API_KEY", None)
        return anthropic.Anthropic(
            auth_token=auth_token,
            default_headers={"anthropic-beta": "oauth-2025-04-20"},
        )

    sys.exit(
        "No credentials. Either set ANTHROPIC_API_KEY in .env, "
        "or sign in to Claude Code (creates ~/.claude/.credentials.json)."
    )


# ---------- resume tracking ----------

def load_existing(pairs_path: Path) -> tuple[int, set[tuple[str, str]]]:
    """Return (total pair count, set of (x_title, y_title) pairs already written)."""
    if not pairs_path.exists():
        return 0, set()
    count = 0
    seen_xy: set[tuple[str, str]] = set()
    for line in pairs_path.open(encoding="utf-8"):
        try:
            obj = json.loads(line)
            count += 1
            seen_xy.add((obj.get("source_x", ""), obj.get("source_y_hidden", "")))
        except json.JSONDecodeError:
            continue
    return count, seen_xy


# ---------- main ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus",
        choices=["wiki", "arxiv"],
        default="wiki",
        help="Which corpus to use: wiki (default) or arxiv (physics papers).",
    )
    args = parser.parse_args()

    if args.corpus == "arxiv":
        corpus_path, embed_path, pairs_path = ARXIV_CORPUS_PATH, ARXIV_EMBED_PATH, ARXIV_PAIRS_PATH
        system_prompt = SYSTEM_PROMPT_PHYSICS
        print("Mode: arxiv physics corpus")
    else:
        corpus_path, embed_path, pairs_path = CORPUS_PATH, EMBED_PATH, PAIRS_PATH
        system_prompt = SYSTEM_PROMPT_TEMPLATE
        print("Mode: Wikipedia corpus")

    client = get_client()
    rng = random.Random(SEED)
    corpus, embeddings = load_corpus(corpus_path, embed_path)
    print(f"Loaded {len(corpus)} articles across {len({a['domain'] for a in corpus})} domains.")

    domains = np.array([a["domain"] for a in corpus])

    existing_count, seen_xy = load_existing(pairs_path)
    if existing_count:
        print(f"Resuming -- {existing_count} pairs already in {pairs_path.name}")

    target = NUM_PAIRS - existing_count
    if target <= 0:
        print(f"Already have {existing_count} pairs (target {NUM_PAIRS}). Done.")
        return

    # Build a list of X candidates, weighted toward titles that appear least often
    title_counts = {a["title"]: 0 for a in corpus}
    for x_title, _ in seen_xy:
        if x_title in title_counts:
            title_counts[x_title] += 1
    # Order by least-used X first; within each tier, shuffle
    tiers: dict[int, list[int]] = {}
    for i, a in enumerate(corpus):
        n = title_counts[a["title"]]
        tiers.setdefault(n, []).append(i)
    candidate_x_indices: list[int] = []
    for n in sorted(tiers.keys()):
        bucket = tiers[n]
        rng.shuffle(bucket)
        candidate_x_indices.extend(bucket)

    print(
        f"Target: {target} new pairs (model={GEN_MODEL}, novelty>={NOVELTY_MIN}, "
        f"cross_p={CROSS_DOMAIN_PROB})"
    )

    written = 0
    rejected_novelty = 0
    other_skipped = 0
    out_f = pairs_path.open("a", encoding="utf-8")

    style_keys = list(OUTPUT_STYLES.keys())

    for x_idx in candidate_x_indices:
        if written >= target:
            break
        x = corpus[x_idx]
        x_domain = x["domain"]

        # Pick candidate domain mask
        cross = rng.random() < CROSS_DOMAIN_PROB
        if cross:
            mask = domains != x_domain
            split = "cross"
        else:
            mask = domains == x_domain
            split = "same"

        cand_idxs = pick_sweet_spot_candidates(
            x_idx, embeddings, mask, K_CANDIDATES, SWEETSPOT_LO, SWEETSPOT_HI, rng
        )
        if not cand_idxs:
            other_skipped += 1
            continue
        candidates = [corpus[i] for i in cand_idxs]

        input_template = rng.choice(INPUT_TEMPLATES)
        output_style = rng.choice(style_keys)

        user_msg = build_user_message(x, candidates, input_template, output_style)

        def _call():
            return client.messages.create(
                model=GEN_MODEL,
                max_tokens=2000,
                system=[
                    {
                        "type": "text",
                        "text": system_prompt,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[{"role": "user", "content": user_msg}],
                output_config={
                    "format": {"type": "json_schema", "schema": OUTPUT_SCHEMA}
                },
            )

        try:
            try:
                resp = _call()
            except anthropic.APIConnectionError:
                # one-shot retry for transient network blips
                time.sleep(8)
                resp = _call()
        except anthropic.RateLimitError:
            print(
                f"  X={x['title']!r} rate-limited (429). Stopping early. "
                f"Wait for quota reset, then rerun -- will resume from "
                f"{load_existing(pairs_path)[0]} pairs."
            )
            break
        except anthropic.BadRequestError as e:
            print(f"  X={x['title']!r} 400 BadRequest -- bailing out (likely schema bug). {e}")
            break
        except anthropic.APIConnectionError as e:
            print(
                f"  X={x['title']!r} connection error: {e}. "
                f"Bailing to avoid burning the candidate pool. Rerun once the network is back."
            )
            break
        except anthropic.APIError as e:
            print(f"  X={x['title']!r} API error: {e}")
            other_skipped += 1
            continue

        text_block = next((b for b in resp.content if b.type == "text"), None)
        if text_block is None:
            other_skipped += 1
            continue
        try:
            parsed = json.loads(text_block.text)
            chosen_idx = int(parsed["chosen_index"])
            z_text = str(parsed["Z"]).strip()
            novelty = max(1, min(10, int(parsed["novelty_score"])))
            rationale = str(parsed.get("rationale", "")).strip()
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"  X={x['title']!r} parse error: {e}")
            other_skipped += 1
            continue

        if not (0 <= chosen_idx < len(candidates)):
            other_skipped += 1
            continue
        if len(z_text) < 60:
            other_skipped += 1
            continue
        chosen_y = candidates[chosen_idx]
        if (x["title"], chosen_y["title"]) in seen_xy:
            # we picked an (X, Y) that's already in the dataset -- skip and try another X
            other_skipped += 1
            continue
        if novelty < NOVELTY_MIN:
            rejected_novelty += 1
            print(
                f"  X={x['title']!r} novelty={novelty} (<{NOVELTY_MIN}) -- rejected. "
                f"rationale: {rationale}"
            )
            continue

        seen_xy.add((x["title"], chosen_y["title"]))
        record = {
            "input": input_template.format(title=x["title"]),
            "output": z_text,
            "source_x": x["title"],
            "x_domain": x_domain,
            "source_y_hidden": chosen_y["title"],
            "y_domain": chosen_y["domain"],
            "split": split,
            "input_template": input_template,
            "output_style": output_style,
            "novelty_score": novelty,
            "rationale": rationale,
        }
        out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
        out_f.flush()
        written += 1

        usage = resp.usage
        print(
            f"[{written}/{target}] {split:5} X={x['title']!r} "
            f"<- Y_hidden={chosen_y['title']!r} ({chosen_y['domain']}) "
            f"style={output_style} novelty={novelty} "
            f"in={usage.input_tokens} out={usage.output_tokens} "
            f"cache_r={usage.cache_read_input_tokens}"
        )

    out_f.close()
    total = load_existing(pairs_path)[0]
    print(
        f"\nDone. {pairs_path.name} now has {total} pairs. "
        f"This run: written={written}, novelty-rejected={rejected_novelty}, "
        f"other-skipped={other_skipped}."
    )


if __name__ == "__main__":
    main()
