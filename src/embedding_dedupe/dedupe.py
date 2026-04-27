"""Deduplicate near-identical embedding records.

Two records are duplicates when ``cosine(a.vector, b.vector) >= threshold``.
Each record sits in exactly one cluster; from each cluster we keep one survivor
according to the ``keep`` strategy:

* ``"first"`` (default) -- keep the lowest-id (lexicographic) record in each cluster.
* ``"longest"`` -- keep the record whose ``text`` field is longest (ties broken by
  lowest id). Falls back to ``"first"`` if no records have a ``text`` field.

The clustering is greedy single-link: scan records in input order, place each into
the first existing cluster whose centroid (the first record's vector) is within the
threshold, else start a new cluster. ``O(n * k)`` where ``k`` is the cluster count.
"""

from __future__ import annotations

import math
from typing import Literal, Sequence

KeepStrategy = Literal["first", "longest"]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Pure-Python cosine similarity. Returns 0.0 when either side has zero norm."""
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    dot = aa = bb = 0.0
    for i in range(n):
        x = float(a[i])
        y = float(b[i])
        dot += x * y
        aa += x * x
        bb += y * y
    if aa == 0.0 or bb == 0.0:
        return 0.0
    return dot / (math.sqrt(aa) * math.sqrt(bb))


def _pick_survivor(cluster: list[dict], keep: KeepStrategy, key: str) -> dict:
    if keep == "longest":
        # Prefer record with the longest ``text``; tie-break on lowest id.
        with_text = [r for r in cluster if isinstance(r.get("text"), str)]
        if with_text:
            return min(
                with_text,
                key=lambda r: (-len(r["text"]), str(r.get(key, ""))),
            )
        # Fall through to ``first`` semantics.
    return min(cluster, key=lambda r: str(r.get(key, "")))


def dedupe(
    records: Sequence[dict],
    threshold: float = 0.95,
    key: str = "id",
    vector: str = "embedding",
    keep: KeepStrategy = "first",
) -> list[dict]:
    """Return a deduped list of records, one survivor per cosine-similarity cluster.

    Args:
        records: Iterable of dicts. Each must have ``key`` and ``vector`` fields.
        threshold: Cosine similarity above which two records are considered
            duplicates. Default ``0.95``.
        key: Field name to use as the record id. Default ``"id"``.
        vector: Field name to read the embedding from. Default ``"embedding"``.
        keep: Survivor strategy -- ``"first"`` (lowest-id) or ``"longest"``
            (longest ``text`` field).

    Returns:
        A new list with one record per cluster, in the same order as the input.
    """
    if not isinstance(records, (list, tuple)):
        raise TypeError("records must be a list or tuple")
    if not 0.0 <= threshold <= 1.0:
        raise TypeError("threshold must be in [0, 1]")
    if keep not in ("first", "longest"):
        raise TypeError("keep must be 'first' or 'longest'")

    if not records:
        return []

    # Greedy single-link clustering: each cluster is anchored by its first record.
    clusters: list[list[dict]] = []
    centroids: list[Sequence[float]] = []
    for r in records:
        if not isinstance(r, dict):
            raise TypeError("each record must be a dict")
        vec = r.get(vector)
        if not isinstance(vec, (list, tuple)) or not vec:
            raise TypeError(f"record {r.get(key)!r} missing or empty {vector!r} field")
        placed = False
        for i, c in enumerate(centroids):
            if cosine(vec, c) >= threshold:
                clusters[i].append(r)
                placed = True
                break
        if not placed:
            clusters.append([r])
            centroids.append(vec)

    survivors = [_pick_survivor(c, keep, key) for c in clusters]
    # Preserve cluster order = order in which clusters were created (= input order).
    return survivors
