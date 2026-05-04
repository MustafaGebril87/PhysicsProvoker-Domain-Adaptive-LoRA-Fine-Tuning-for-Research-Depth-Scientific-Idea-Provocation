"""
Fetch recent physics papers from arxiv and build a sentence-embedding index.
Targets 4 subfields: quant-ph, hep-th, gr-qc, cond-mat.
Output mirrors build_index.py format so generate_pairs.py works with --corpus arxiv.

Output:
    data/arxiv_corpus.jsonl   -- {title, summary, domain, arxiv_id} per paper
    data/arxiv_embeddings.npy -- (N, D) float32 array, aligned with corpus order
"""

from __future__ import annotations

import json
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import requests
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

CORPUS_PATH = DATA_DIR / "arxiv_corpus.jsonl"
EMBED_PATH = DATA_DIR / "arxiv_embeddings.npy"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
ARXIV_API = "http://export.arxiv.org/api/query"
BATCH_SIZE = 100        # papers per API call; arxiv allows up to 2000 but be polite
PAPERS_PER_CAT = 300    # target per subfield → ~1200 total
SLEEP_BETWEEN = 3.0     # seconds between API calls (arxiv rate-limit policy)
MIN_ABSTRACT = 200      # skip papers with very short abstracts

ATOM_NS = "http://www.w3.org/2005/Atom"

# subfield → arxiv category string
CATEGORIES: list[str] = ["quant-ph", "hep-th", "gr-qc", "cond-mat"]


def fetch_batch(
    category: str, start: int, max_results: int, session: requests.Session
) -> list[dict]:
    params = {
        "search_query": f"cat:{category}",
        "start": start,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    try:
        r = session.get(ARXIV_API, params=params, timeout=30)
        r.raise_for_status()
    except requests.RequestException as e:
        print(f"  network error: {e}")
        return []

    try:
        root = ET.fromstring(r.text)
    except ET.ParseError as e:
        print(f"  XML parse error: {e}")
        return []

    papers: list[dict] = []
    for entry in root.findall(f"{{{ATOM_NS}}}entry"):
        id_el = entry.find(f"{{{ATOM_NS}}}id")
        title_el = entry.find(f"{{{ATOM_NS}}}title")
        summary_el = entry.find(f"{{{ATOM_NS}}}summary")
        if id_el is None or title_el is None or summary_el is None:
            continue

        arxiv_id = (id_el.text or "").strip().split("/abs/")[-1]
        title = " ".join((title_el.text or "").split())
        abstract = " ".join((summary_el.text or "").split())

        papers.append(
            {"arxiv_id": arxiv_id, "title": title, "summary": abstract, "domain": category}
        )

    return papers


def fetch_category(category: str, target: int) -> list[dict]:
    session = requests.Session()
    session.headers.update(
        {"User-Agent": "logical-finetune/1.0 (physics provocation research)"}
    )

    all_papers: list[dict] = []
    seen_ids: set[str] = set()
    start = 0

    while len(all_papers) < target:
        to_fetch = min(BATCH_SIZE, target - len(all_papers))
        print(f"  [{category}] fetching {to_fetch} starting at offset {start} ...")
        batch = fetch_batch(category, start, to_fetch, session)

        if not batch:
            print(f"  [{category}] empty batch — stopping")
            break

        added = 0
        for paper in batch:
            if paper["arxiv_id"] in seen_ids:
                continue
            if len(paper["summary"]) < MIN_ABSTRACT:
                continue
            seen_ids.add(paper["arxiv_id"])
            all_papers.append(paper)
            added += 1

        print(f"  [{category}] +{added} (total so far: {len(all_papers)})")
        start += BATCH_SIZE

        if len(batch) < to_fetch:
            print(f"  [{category}] API returned fewer than requested — likely exhausted")
            break

        time.sleep(SLEEP_BETWEEN)

    return all_papers


def main() -> None:
    if CORPUS_PATH.exists() and EMBED_PATH.exists():
        print(f"Index already exists. Delete to rebuild:\n  {CORPUS_PATH}\n  {EMBED_PATH}")
        return

    all_papers: list[dict] = []
    seen_global: set[str] = set()

    for category in CATEGORIES:
        print(f"\n=== {category} (target: {PAPERS_PER_CAT}) ===")
        papers = fetch_category(category, PAPERS_PER_CAT)
        for p in papers:
            if p["arxiv_id"] not in seen_global:
                seen_global.add(p["arxiv_id"])
                all_papers.append(p)
        print(f"  -> {len(papers)} unique papers for {category}")

    print(f"\nTotal unique papers: {len(all_papers)}")
    by_domain: dict[str, int] = {}
    for p in all_papers:
        by_domain[p["domain"]] = by_domain.get(p["domain"], 0) + 1
    for d, n in sorted(by_domain.items()):
        print(f"  {d}: {n}")

    if len(all_papers) < 200:
        sys.exit("Too few papers — aborting before embedding.")

    with open(CORPUS_PATH, "w", encoding="utf-8") as f:
        for p in all_papers:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"\nWrote {CORPUS_PATH}")

    print(f"\nLoading embedding model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)
    texts = [p["summary"] for p in all_papers]
    print(f"Embedding {len(texts)} abstracts ...")
    embeddings = model.encode(
        texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True
    ).astype(np.float32)
    np.save(EMBED_PATH, embeddings)
    print(f"Wrote {EMBED_PATH}  shape={embeddings.shape}")


if __name__ == "__main__":
    main()
