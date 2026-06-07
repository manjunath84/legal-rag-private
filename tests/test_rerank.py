# tests/test_rerank.py
"""Unit tests for the cross-encoder reranker.

The encoder model (~85 MB) is stubbed out so tests run offline and fast.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from raglab.chunk import Chunk
from raglab.rerank import rerank


def _make_hits(*texts: str) -> list[tuple[Chunk, float]]:
    return [
        (Chunk(id=str(i), text=t, source="x.txt", page=None), 0.5)
        for i, t in enumerate(texts)
    ]


def _stub_encoder(scores: list[float]):
    enc = MagicMock()
    enc.predict.return_value = np.array(scores)
    return enc


def test_rerank_orders_by_cross_encoder_score():
    hits = _make_hits("NDA governing law: Delaware", "MSA governing law: New York", "Payment terms")
    with patch("raglab.rerank._get_encoder", return_value=_stub_encoder([2.5, 1.0, -1.0])):
        result = rerank("governing law NDA", hits, top_k=2)
    assert len(result) == 2
    assert result[0][0].text == "NDA governing law: Delaware"
    assert result[1][0].text == "MSA governing law: New York"
    assert result[0][1] == pytest.approx(2.5)


def test_rerank_returns_top_k():
    hits = _make_hits("a", "b", "c", "d", "e")
    scores = [3.0, 1.0, 2.0, 0.5, -1.0]
    with patch("raglab.rerank._get_encoder", return_value=_stub_encoder(scores)):
        result = rerank("query", hits, top_k=3)
    assert len(result) == 3
    assert result[0][0].text == "a"  # score 3.0
    assert result[1][0].text == "c"  # score 2.0
    assert result[2][0].text == "b"  # score 1.0


def test_rerank_empty_hits_returns_empty():
    result = rerank("query", [], top_k=5)
    assert result == []


def test_rerank_fewer_hits_than_top_k():
    hits = _make_hits("only one")
    with patch("raglab.rerank._get_encoder", return_value=_stub_encoder([1.5])):
        result = rerank("query", hits, top_k=5)
    assert len(result) == 1
