from raglab.chunk import Chunk
from raglab.hybrid import _tokenize, reciprocal_rank_fusion


def _c(cid):
    return Chunk(id=cid, text=cid, source="x", page=None)


def test_tokenize_lowercases_and_splits():
    assert _tokenize("Section 3: TERM (3 years).") == ["section", "3", "term", "3", "years"]


def test_rrf_rewards_agreement_across_lists():
    a, b, d = _c("a"), _c("b"), _c("d")
    # 'a' is #2 in both lists; 'b' is #1 in one only. Appearing in both should win.
    dense = [b, a]
    sparse = [d, a]
    fused = reciprocal_rank_fusion([dense, sparse], top=3, rrf_k=60)
    assert fused[0][0].id == "a"
    # scores are descending
    assert fused[0][1] >= fused[1][1] >= fused[2][1]


def test_rrf_respects_top_and_dedupes():
    a, b = _c("a"), _c("b")
    fused = reciprocal_rank_fusion([[a, b], [a, b]], top=1, rrf_k=60)
    assert len(fused) == 1
    assert fused[0][0].id == "a"


def test_rrf_single_list_preserves_order():
    a, b, d = _c("a"), _c("b"), _c("d")
    fused = reciprocal_rank_fusion([[a, b, d]], top=3, rrf_k=60)
    assert [c.id for c, _ in fused] == ["a", "b", "d"]
