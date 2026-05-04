"""
Fetch a multi-domain Wikipedia corpus and build a sentence-embedding index.
Each article carries its domain label so generate_pairs.py can sample
cross-domain or same-domain Y candidates.

Output:
    data/corpus.jsonl        -- {title, summary, domain} per article
    data/embeddings.npy      -- (N, D) float32 array, aligned with corpus order
"""

from __future__ import annotations

import json
import sys
import time
import urllib.parse
from pathlib import Path

import numpy as np
import requests
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

CORPUS_PATH = DATA_DIR / "corpus.jsonl"
EMBED_PATH = DATA_DIR / "embeddings.npy"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
USER_AGENT = "logical-finetune-pilot/0.2 (https://example.local/contact; idea-provocation)"
SUMMARY_API = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"

DOMAINS: dict[str, list[str]] = {
    "philosophy": [
        "Free will", "Determinism", "Phenomenology (philosophy)", "Existentialism",
        "Stoicism", "Epistemology", "Skepticism", "Empiricism", "Rationalism",
        "Pragmatism", "Logical positivism", "Philosophy of mind", "Qualia",
        "Mind-body dualism", "Materialism", "Functionalism (philosophy of mind)",
        "Personal identity", "Ship of Theseus", "Trolley problem",
        "Categorical imperative", "Utilitarianism", "Virtue ethics",
        "Deontological ethics", "Consequentialism", "Moral relativism",
        "Nihilism", "Solipsism", "Hard problem of consciousness",
        "Philosophical zombie", "Simulation hypothesis", "Ontology",
        "Metaphysics", "Modal logic", "Possible world", "Counterfactual conditional",
        "Truth", "Knowledge", "Justified true belief", "Gettier problem",
        "Inductive reasoning", "Falsifiability", "Occam's razor",
        "Reductionism", "Emergence", "Supervenience", "Causality",
        "Philosophy of time", "Eternalism (philosophy of time)",
        "Allegory of the cave", "Tabula rasa",
    ],
    "complex_systems": [
        "Emergence", "Self-organization", "Cellular automaton",
        "Conway's Game of Life", "Chaos theory", "Attractor",
        "Bifurcation theory", "Phase transition", "Critical phenomena",
        "Power law", "Scale invariance", "Fractal", "Network science",
        "Small-world network", "Scale-free network", "Feedback",
        "Positive feedback", "Negative feedback", "Homeostasis",
        "Autopoiesis", "Self-organized criticality", "Tipping points (sociology)",
        "Cascading failure", "Percolation theory", "Random graph",
        "Stochastic process", "Markov chain", "Ergodicity",
        "Non-equilibrium thermodynamics", "Dissipative system",
        "Synergetics (Haken)", "Cybernetics", "Systems thinking",
        "Antifragility", "Black swan theory", "Edge of chaos",
        "Stigmergy", "Swarm intelligence", "Ant colony optimization",
        "Boids", "Complex adaptive system", "Pattern formation",
        "Reaction-diffusion system", "Synchronization",
        "Lyapunov exponent", "Catastrophe theory", "Hysteresis",
    ],
    "biology": [
        "Evolution", "Natural selection", "Genetic drift", "Speciation",
        "Adaptation", "Convergent evolution", "Mimicry", "Coevolution",
        "Symbiosis", "Mutualism", "Parasitism", "Predation",
        "Ecological niche", "Food web", "Trophic level", "Keystone species",
        "Ecosystem", "Biodiversity", "Genome", "DNA", "RNA", "Protein",
        "Enzyme", "Metabolism", "Cell (biology)", "Cell membrane",
        "Mitochondrion", "Cytoskeleton", "Apoptosis", "Stem cell",
        "Embryonic development", "Morphogenesis", "Homeobox",
        "Gene regulatory network", "Epigenetics", "Horizontal gene transfer",
        "Genetic code", "Phylogenetic tree", "Cambrian explosion",
        "Extinction event", "Punctuated equilibrium", "Sexual selection",
        "Altruism (biology)", "Kin selection", "Inclusive fitness",
        "Hawk-dove game", "Red Queen hypothesis", "Endosymbiotic theory",
        "Quorum sensing", "Slime mold",
    ],
    "mathematics": [
        "Set theory", "Cantor's diagonal argument", "Continuum hypothesis",
        "Gödel's incompleteness theorems", "Russell's paradox",
        "Axiom of choice", "Topology", "Manifold", "Group (mathematics)",
        "Symmetry group", "Lie group", "Galois theory", "Number theory",
        "Prime number", "Riemann hypothesis", "Modular arithmetic",
        "Fourier analysis", "Wavelet", "Dynamical system",
        "Hilbert space", "Functional analysis", "Linear algebra",
        "Eigenvalues and eigenvectors", "Tensor", "Differential geometry",
        "Algebraic geometry", "Category theory", "Functor",
        "Yoneda lemma", "Topos", "Combinatorics", "Graph theory",
        "Ramsey theory", "Probability theory", "Measure theory",
        "Information theory", "Kolmogorov complexity",
        "Recursion", "Fixed point (mathematics)",
        "Brouwer fixed-point theorem", "Knot theory", "Game theory",
        "Nash equilibrium", "Mathematical optimization",
        "Convex optimization", "Stochastic differential equation",
        "Cellular automaton", "Catastrophe theory",
    ],
    "design_architecture": [
        "Bauhaus", "Modernism", "Postmodernism",
        "Functionalism (architecture)", "Form follows function",
        "Brutalist architecture", "Minimalism", "Le Corbusier",
        "Christopher Alexander", "A Pattern Language",
        "Vernacular architecture", "Software design pattern",
        "Affordance", "Skeuomorph", "Information architecture",
        "Wayfinding", "User experience design", "User interface design",
        "Industrial design", "Universal design", "Inclusive design",
        "Biophilic design", "Adaptive reuse", "Sustainable architecture",
        "Passive house", "Tectonics (architecture)", "Genius loci",
        "Sacred geometry", "Golden ratio", "Symmetry",
        "Typography", "Grid (graphic design)", "White space (visual arts)",
        "Gestalt psychology", "Visual hierarchy", "Color theory",
        "Negative space", "Wabi-sabi", "Hostile architecture",
        "Defensible space theory", "Walkability", "New Urbanism",
        "Garden city movement", "Deconstructivism",
        "International Style (architecture)", "Metabolism (architecture)",
        "Phenomenology (architecture)", "Architectural theory",
    ],
    "computer_science": [
        "Turing machine", "Halting problem", "Computability theory",
        "Lambda calculus", "Algorithm", "Big O notation",
        "Time complexity", "Space complexity", "NP-completeness",
        "P versus NP problem", "Cryptography", "Public-key cryptography",
        "Cryptographic hash function", "Data structure", "Linked list",
        "Hash table", "Binary tree", "Red-black tree", "B-tree",
        "Graph (abstract data type)", "Depth-first search",
        "Breadth-first search", "Dynamic programming",
        "Greedy algorithm", "Divide-and-conquer algorithm",
        "Recursion (computer science)", "Functional programming",
        "Object-oriented programming", "Type system",
        "Polymorphism (computer science)", "Compiler",
        "Interpreter (computing)", "Virtual machine", "Operating system",
        "Concurrency (computer science)", "Distributed computing",
        "Consensus (computer science)", "CAP theorem", "Database",
        "Relational model", "Database normalization", "ACID",
        "Garbage collection (computer science)", "Memory management",
        "Cache (computing)", "Locality of reference",
        "Machine learning", "Neural network", "Reinforcement learning",
        "Programming language",
    ],
    "economics": [
        "Marginal utility", "Supply and demand", "Comparative advantage",
        "Opportunity cost", "Externality", "Public good",
        "Tragedy of the commons", "Information asymmetry", "Moral hazard",
        "Adverse selection", "Market failure", "Pareto efficiency",
        "Game theory", "Behavioral economics", "Loss aversion",
        "Sunk cost", "Rational expectations", "Efficient-market hypothesis",
        "Black-Scholes model", "Random walk hypothesis", "Phillips curve",
        "Quantity theory of money", "Velocity of money", "Kondratiev wave",
        "Creative destruction", "Liquidity trap", "Time preference",
        "Bounded rationality", "Veblen good", "Giffen good",
        "Network effect", "Economies of scale", "Positional good",
        "Principal-agent problem", "The Market for Lemons", "Coase theorem",
        "Veil of ignorance", "Endowment effect", "Prospect theory",
        "Hyperbolic discounting", "Public choice", "Auction theory",
        "Two-sided market", "Diffusion of innovations", "Inflation",
        "Mechanism design",
    ],
    "cognitive_science": [
        "Working memory", "Attention", "Cognitive load", "Cognitive bias",
        "Heuristic", "Cognitive dissonance", "Mental model",
        "Schema (psychology)", "Embodied cognition", "Predictive coding",
        "Bayesian brain", "Free energy principle", "Mirror neuron",
        "Theory of mind", "Default mode network", "Global workspace theory",
        "Multiple drafts model", "Integrated information theory",
        "Cognitive map", "Memory consolidation", "Episodic memory",
        "Semantic memory", "Procedural memory", "Forgetting curve",
        "Spaced repetition", "Flow (psychology)", "Dual process theory",
        "Confirmation bias", "Availability heuristic", "Anchoring effect",
        "Framing effect", "Cognitive science", "Connectionism",
        "Symbol grounding problem", "Chinese room", "Neuroplasticity",
        "Mental rotation", "Categorical perception", "Linguistic relativity",
        "Attention restoration theory", "Working hypothesis",
        "Stroop effect", "Priming (psychology)", "Implicit memory",
        "Metacognition", "Cognitive psychology",
    ],
    "music_theory": [
        "Tonality", "Atonality", "Chord progression", "Counterpoint",
        "Polyphony", "Harmonic series (music)", "Just intonation",
        "Equal temperament", "Pythagorean tuning", "Mode (music)",
        "Pentatonic scale", "Chromatic scale", "Microtonal music",
        "Cadence (music)", "Modulation (music)", "Key (music)",
        "Circle of fifths", "Voice leading", "Tritone substitution",
        "Suspension (music)", "Rhythm", "Polyrhythm", "Syncopation",
        "Metre (music)", "Ostinato", "Leitmotif", "Sonata form",
        "Fugue", "Twelve-tone technique", "Serialism",
        "Minimal music", "Drone (music)", "Tone row", "Pitch class",
        "Set theory (music)", "Spectral music",
        "Musique concrète", "Aleatoric music", "Phrase (music)",
        "Time signature", "Texture (music)", "Timbre",
        "Harmonic rhythm", "Consonance and dissonance",
        "Functional harmony", "Ornament (music)",
    ],
}


def fetch_summary(title: str, session: requests.Session) -> dict | None:
    encoded = urllib.parse.quote(title.replace(" ", "_"), safe="")
    url = SUMMARY_API.format(title=encoded)
    try:
        r = session.get(url, timeout=15)
    except requests.RequestException as e:
        print(f"  network error for {title!r}: {e}")
        return None
    if r.status_code == 404:
        return None
    if r.status_code != 200:
        print(f"  HTTP {r.status_code} for {title!r}")
        return None
    try:
        data = r.json()
    except ValueError:
        return None
    if data.get("type") == "disambiguation":
        return None
    extract = (data.get("extract") or "").strip()
    if not extract:
        return None
    return {"title": data.get("title", title), "summary": extract}


def fetch_articles(domain: str, topics: list[str], min_chars: int = 250) -> list[dict]:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})

    results: list[dict] = []
    seen: set[str] = set()
    for i, topic in enumerate(topics, 1):
        result = fetch_summary(topic, session)
        if result is None:
            print(f"  [{domain} {i}/{len(topics)}] miss: {topic}")
        elif result["title"] in seen:
            print(f"  [{domain} {i}/{len(topics)}] dup: {topic}")
        elif len(result["summary"]) < min_chars:
            print(f"  [{domain} {i}/{len(topics)}] short ({len(result['summary'])}): {result['title']}")
        else:
            result["domain"] = domain
            results.append(result)
            seen.add(result["title"])
            print(f"  [{domain} {i}/{len(topics)}] ok: {result['title']}")
        time.sleep(0.05)
    return results


def main():
    if CORPUS_PATH.exists() and EMBED_PATH.exists():
        print(f"Index already exists. Delete to rebuild: {CORPUS_PATH}, {EMBED_PATH}")
        return

    all_articles: list[dict] = []
    seen_global: set[str] = set()

    for domain, topics in DOMAINS.items():
        print(f"\n=== Fetching domain: {domain} ({len(topics)} topics) ===")
        domain_articles = fetch_articles(domain, topics)
        for art in domain_articles:
            if art["title"] in seen_global:
                continue  # avoid double-counting if a topic appears in two domains
            seen_global.add(art["title"])
            all_articles.append(art)
        print(f"  -> {len(domain_articles)} articles for {domain}")

    print(f"\nTotal unique articles: {len(all_articles)}")
    by_domain: dict[str, int] = {}
    for art in all_articles:
        by_domain[art["domain"]] = by_domain.get(art["domain"], 0) + 1
    for d, n in by_domain.items():
        print(f"  {d}: {n}")

    if len(all_articles) < 100:
        sys.exit("Too few articles -- aborting before embedding.")

    with open(CORPUS_PATH, "w", encoding="utf-8") as f:
        for art in all_articles:
            f.write(json.dumps(art, ensure_ascii=False) + "\n")
    print(f"\nWrote {CORPUS_PATH}")

    print(f"\nLoading embedding model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)
    texts = [a["summary"] for a in all_articles]
    print(f"Embedding {len(texts)} summaries...")
    embeddings = model.encode(
        texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True
    ).astype(np.float32)
    np.save(EMBED_PATH, embeddings)
    print(f"Wrote {EMBED_PATH}  shape={embeddings.shape}")


if __name__ == "__main__":
    main()
