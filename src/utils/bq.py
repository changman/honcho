"""Binary Quantization (BQ) utilities for multimodal fingerprint storage and search.

Converts 512-dimensional float vectors to 512-bit strings via median thresholding,
enabling fast Hamming-distance search without storing full float vectors.
"""

import statistics


def float_to_bq(vector: list[float]) -> str:
    """Convert a float vector to a binary string via median thresholding.

    Bits above the median become '1', at-or-below become '0'.
    This maximally discriminates the vector's information content.

    Args:
        vector: Float vector (any dimension, typically 512d for CLIP/SigLIP).

    Returns:
        Binary string of the same length, e.g. "10110...".
    """
    if not vector:
        return ""
    threshold = statistics.median(vector)
    return "".join("1" if v > threshold else "0" for v in vector)


def hamming_distance(a: str, b: str) -> int:
    """Bit-level Hamming distance between two binary strings.

    Strings of unequal length are compared up to the shorter length.
    """
    return sum(x != y for x, y in zip(a, b))


def hamming_similarity(a: str, b: str) -> float:
    """Normalized similarity score: 1.0 = identical, 0.0 = fully opposite.

    Returns 0.0 if either string is empty.
    """
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    return 1.0 - hamming_distance(a, b) / n


def rank_by_hamming(
    query_bq: str,
    candidates: list[tuple[str, object]],
    top_k: int = 10,
    min_similarity: float = 0.0,
) -> list[tuple[object, float]]:
    """Rank candidate (bq_string, payload) pairs by Hamming similarity to query.

    Args:
        query_bq: Binary query string.
        candidates: List of (fingerprint_bq, arbitrary_payload) tuples.
        top_k: Maximum results to return.
        min_similarity: Exclude results below this threshold (0.0–1.0).

    Returns:
        List of (payload, similarity) sorted descending by similarity.
    """
    scored = [
        (payload, hamming_similarity(query_bq, bq))
        for bq, payload in candidates
        if bq
    ]
    scored = [(p, s) for p, s in scored if s >= min_similarity]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]
