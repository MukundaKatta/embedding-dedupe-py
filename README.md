# embedding-dedupe

[![PyPI](https://img.shields.io/pypi/v/embedding-dedupe.svg)](https://pypi.org/project/embedding-dedupe/)
[![Python](https://img.shields.io/pypi/pyversions/embedding-dedupe.svg)](https://pypi.org/project/embedding-dedupe/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Deduplicate near-identical embedding records by cosine similarity.** Pure Python, zero runtime dependencies.

Python port of [@mukundakatta/embedding-dedupe](https://github.com/MukundaKatta/embedding-dedupe). Same algorithm, ergonomic Python API.

## Install

```bash
pip install embedding-dedupe
# Optional: faster cosine via numpy
pip install "embedding-dedupe[numpy]"
```

## Usage

```python
from embedding_dedupe import dedupe

records = [
    {"id": "a", "embedding": [0.10, 0.20, 0.30], "text": "Cats are great."},
    {"id": "b", "embedding": [0.10, 0.20, 0.31], "text": "Cats are great pets."},   # near-dup of a
    {"id": "c", "embedding": [0.90, 0.10, 0.05], "text": "Stock prices fell."},
]

# Default: keep the lowest-id record from each cluster (deterministic).
dedupe(records, threshold=0.95)
# -> [
#      {"id": "a", "embedding": [0.10, 0.20, 0.30], "text": "Cats are great."},
#      {"id": "c", "embedding": [0.90, 0.10, 0.05], "text": "Stock prices fell."},
#    ]

# Or: keep the record with the longest text (ties -> lowest id).
dedupe(records, threshold=0.95, keep="longest")
# -> [
#      {"id": "b", "embedding": [0.10, 0.20, 0.31], "text": "Cats are great pets."},
#      {"id": "c", "embedding": [0.90, 0.10, 0.05], "text": "Stock prices fell."},
#    ]
```

## API

```python
dedupe(
    records,                  # list[dict]
    threshold=0.95,           # cosine sim above which two records are duplicates
    key="id",                 # record id field name
    vector="embedding",       # embedding field name
    keep="first",             # "first" (lowest-id) or "longest" (longest .text)
) -> list[dict]
```

`cosine(a, b)` is also exported for ad-hoc use.

## Algorithm

Greedy single-link clustering: scan records in input order, place each into the first existing cluster whose anchor vector is within `threshold`, else start a new cluster. From each cluster, return one survivor according to `keep`. Cluster output order matches input order (= order in which clusters were created).

`O(n * k)` where `k` is the cluster count -- fine up to ~100k records. For larger sets, plug in an ANN index upstream and dedupe within candidates.

## API differences from the JS sibling

* JS: `dedupeEmbeddings(items, { threshold })` returning `{ kept, duplicates }`. Python: `dedupe(records, threshold=0.95, key=, vector=, keep=)` returning a flat `list[dict]` of survivors.
* The default threshold is `0.95` (was `0.98` in JS) -- tuned for typical OpenAI/Anthropic embedding noise.
* New `keep="longest"` strategy mirrors a common request -- prefer the most informative chunk in each cluster.

See the JS sibling for the original heuristics and design notes.
