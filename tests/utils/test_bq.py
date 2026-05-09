"""Unit tests for src/utils/bq.py — no database required."""

import random

import pytest

from src.utils.bq import float_to_bq, hamming_distance, hamming_similarity, rank_by_hamming


def _vec(seed: int, n: int = 512) -> list[float]:
    rng = random.Random(seed)
    return [rng.random() for _ in range(n)]


def _opposite(v: list[float]) -> list[float]:
    return [1.0 - x for x in v]


def _shifted(v: list[float], shift: int) -> list[float]:
    w = list(v)
    for i in range(shift):
        w[i] = 1.0 - w[i]
    return w


class TestFloatToBQ:
    def test_output_length_equals_input_length(self):
        v = _vec(1)
        assert len(float_to_bq(v)) == len(v)

    def test_output_only_contains_0_and_1(self):
        assert set(float_to_bq(_vec(2))) <= {"0", "1"}

    def test_empty_input_returns_empty_string(self):
        assert float_to_bq([]) == ""

    def test_deterministic(self):
        v = _vec(3)
        assert float_to_bq(v) == float_to_bq(v)

    def test_different_vectors_different_bq(self):
        a = _vec(4)
        b = _opposite(a)
        assert float_to_bq(a) != float_to_bq(b)


class TestHammingDistance:
    def test_identical_strings_distance_zero(self):
        bq = float_to_bq(_vec(10))
        assert hamming_distance(bq, bq) == 0

    def test_single_flip(self):
        a = "0" * 512
        b = "1" + "0" * 511
        assert hamming_distance(a, b) == 1

    def test_all_different(self):
        a = "0" * 8
        b = "1" * 8
        assert hamming_distance(a, b) == 8


class TestHammingSimilarity:
    def test_identical_is_1(self):
        bq = float_to_bq(_vec(20))
        assert hamming_similarity(bq, bq) == 1.0

    def test_empty_strings_return_0(self):
        assert hamming_similarity("", "") == 0.0

    def test_similar_vectors_high_score(self):
        v = _vec(21)
        a = float_to_bq(v)
        b = float_to_bq(_shifted(v, shift=10))
        assert hamming_similarity(a, b) > 0.95

    def test_opposite_vectors_low_score(self):
        v = _vec(22)
        a = float_to_bq(v)
        b = float_to_bq(_opposite(v))
        assert hamming_similarity(a, b) < 0.15

    def test_score_in_range_0_to_1(self):
        for seed in range(5):
            v = _vec(seed)
            s = hamming_similarity(float_to_bq(v), float_to_bq(_vec(seed + 100)))
            assert 0.0 <= s <= 1.0


class TestRankByHamming:
    def test_best_match_is_first(self):
        v = _vec(30)
        bq_exact = float_to_bq(v)
        bq_close = float_to_bq(_shifted(v, shift=5))
        bq_far = float_to_bq(_opposite(v))

        results = rank_by_hamming(
            query_bq=bq_exact,
            candidates=[(bq_close, "close"), (bq_far, "far"), (bq_exact, "exact")],
            top_k=3,
        )
        assert results[0][0] == "exact"

    def test_sorted_descending(self):
        v = _vec(31)
        bq_query = float_to_bq(v)
        candidates = [
            (float_to_bq(_shifted(v, shift=i * 20)), f"rank-{i}")
            for i in range(5)
        ]
        results = rank_by_hamming(bq_query, candidates, top_k=5)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_respected(self):
        v = _vec(32)
        bq_query = float_to_bq(v)
        candidates = [(float_to_bq(_shifted(v, shift=i)), f"item-{i}") for i in range(10)]
        results = rank_by_hamming(bq_query, candidates, top_k=3)
        assert len(results) <= 3

    def test_min_similarity_filters_low_matches(self):
        v = _vec(33)
        bq_query = float_to_bq(v)
        bq_opposite = float_to_bq(_opposite(v))
        results = rank_by_hamming(
            bq_query,
            candidates=[(bq_opposite, "bad")],
            top_k=5,
            min_similarity=0.9,
        )
        assert results == []

    def test_empty_candidates_returns_empty(self):
        assert rank_by_hamming(float_to_bq(_vec(34)), [], top_k=5) == []


# ---------------------------------------------------------------------------
# Cosine distance (key-frame interleaving helper)
# ---------------------------------------------------------------------------

class TestCosineDistance:
    """Tests for src/crud/perception._cosine_distance."""

    # Import here to keep test file independent of DB fixtures
    from src.crud.perception import _cosine_distance as _cd

    def test_identical_vectors_distance_is_zero(self):
        from src.crud.perception import _cosine_distance
        v = [1.0] * 512
        assert _cosine_distance(v, v) < 1e-9

    def test_orthogonal_vectors_distance_is_1(self):
        from src.crud.perception import _cosine_distance
        a = [1.0] + [0.0] * 511
        b = [0.0, 1.0] + [0.0] * 510
        assert abs(_cosine_distance(a, b) - 1.0) < 1e-9

    def test_opposite_vectors_distance_is_2(self):
        from src.crud.perception import _cosine_distance
        v = [1.0] * 512
        opp = [-1.0] * 512
        assert abs(_cosine_distance(v, opp) - 2.0) < 1e-9

    def test_similar_vectors_small_distance(self):
        from src.crud.perception import _cosine_distance
        rng = random.Random(99)
        v = [rng.random() for _ in range(512)]
        # Tiny perturbation → distance should be well below 0.05 threshold
        noise = [x + rng.gauss(0, 0.001) for x in v]
        assert _cosine_distance(v, noise) < 0.05

    def test_zero_vector_returns_1(self):
        from src.crud.perception import _cosine_distance
        zeros = [0.0] * 512
        v = _vec(100)
        assert _cosine_distance(zeros, v) == 1.0
