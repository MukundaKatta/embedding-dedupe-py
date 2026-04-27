"""Tests for ``embedding_dedupe.dedupe`` and ``cosine``."""

from __future__ import annotations

import pytest

from embedding_dedupe import cosine, dedupe


def test_happy_path_collapses_near_duplicates():
    """Two near-identical embeddings collapse; a third unrelated one survives."""
    records = [
        {"id": "a", "embedding": [0.1, 0.2, 0.3]},
        {"id": "b", "embedding": [0.1, 0.2, 0.31]},   # near-dup
        {"id": "c", "embedding": [0.9, 0.1, 0.05]},
    ]
    out = dedupe(records, threshold=0.95)
    ids = [r["id"] for r in out]
    assert ids == ["a", "c"]


def test_keep_first_picks_lowest_id():
    """Even when records arrive in non-id order, lowest-id wins per cluster."""
    records = [
        {"id": "z", "embedding": [0.1, 0.2, 0.3]},
        {"id": "b", "embedding": [0.1, 0.2, 0.3]},
        {"id": "m", "embedding": [0.1, 0.2, 0.3]},
    ]
    out = dedupe(records, threshold=0.99, keep="first")
    assert len(out) == 1
    assert out[0]["id"] == "b"


def test_keep_longest_picks_longest_text():
    records = [
        {"id": "a", "embedding": [0.1, 0.2, 0.3], "text": "short"},
        {"id": "b", "embedding": [0.1, 0.2, 0.3], "text": "this is a much longer text body"},
    ]
    out = dedupe(records, threshold=0.99, keep="longest")
    assert len(out) == 1
    assert out[0]["id"] == "b"


def test_keep_longest_falls_back_when_no_text():
    """If no record in a cluster has a text field, fall back to lowest-id."""
    records = [
        {"id": "z", "embedding": [0.1, 0.2, 0.3]},
        {"id": "a", "embedding": [0.1, 0.2, 0.3]},
    ]
    out = dedupe(records, threshold=0.99, keep="longest")
    assert out[0]["id"] == "a"


def test_threshold_tuning_higher_keeps_more():
    """Raising the threshold separates clusters that lower thresholds collapse."""
    records = [
        {"id": "a", "embedding": [1.0, 0.0, 0.0]},
        {"id": "b", "embedding": [0.95, 0.31, 0.0]},  # cosine ~ 0.95 vs a
    ]
    assert len(dedupe(records, threshold=0.99)) == 2
    assert len(dedupe(records, threshold=0.90)) == 1


def test_empty_input_returns_empty_list():
    assert dedupe([]) == []


def test_custom_key_and_vector_fields():
    records = [
        {"sku": "x1", "vec": [1.0, 0.0]},
        {"sku": "x2", "vec": [1.0, 0.001]},   # near-dup
    ]
    out = dedupe(records, threshold=0.99, key="sku", vector="vec")
    assert len(out) == 1
    assert out[0]["sku"] == "x1"


def test_cosine_identical_vectors_returns_one():
    assert cosine([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]) == pytest.approx(1.0)


def test_cosine_orthogonal_vectors_returns_zero():
    assert cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_zero_vector_returns_zero():
    assert cosine([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_dedupe_rejects_non_list_records():
    with pytest.raises(TypeError):
        dedupe({"id": "a", "embedding": [0.1]})  # type: ignore[arg-type]


def test_dedupe_rejects_invalid_threshold():
    with pytest.raises(TypeError):
        dedupe([], threshold=1.5)


def test_dedupe_rejects_invalid_keep_strategy():
    with pytest.raises(TypeError):
        dedupe([], keep="random")  # type: ignore[arg-type]


def test_dedupe_rejects_record_missing_vector():
    with pytest.raises(TypeError):
        dedupe([{"id": "a"}])  # missing 'embedding'


def test_output_order_matches_input_cluster_order():
    """Cluster output order = order in which clusters were first opened."""
    records = [
        {"id": "c", "embedding": [0.9, 0.1, 0.05]},  # cluster 1
        {"id": "a", "embedding": [0.1, 0.2, 0.3]},   # cluster 2
        {"id": "b", "embedding": [0.1, 0.2, 0.31]},  # joins cluster 2
    ]
    out = dedupe(records, threshold=0.95)
    # Cluster 1 (anchor 'c') opened first, then cluster 2 (anchor 'a' -> survivor 'a').
    assert [r["id"] for r in out] == ["c", "a"]
