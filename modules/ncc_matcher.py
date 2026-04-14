"""
ncc_matcher.py – Normalized Cross-Correlation (NCC) computation.

Computes NCC between feature vectors and across frame sequences.
Primary similarity metric for the ClipMatch system.
"""

import numpy as np


def compute_ncc(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Compute Normalized Cross-Correlation between two feature vectors.

    NCC = sum((a - mean_a) * (b - mean_b)) / (n * std_a * std_b)

    Handles edge cases:
        - Zero-variance vectors: returns 1.0 if both are identical, 0.0 otherwise
        - Different lengths: raises ValueError

    Args:
        vec_a: First feature vector (1D numpy array).
        vec_b: Second feature vector (1D numpy array).

    Returns:
        NCC value in range [-1.0, 1.0]. Higher means more similar.
    """
    if len(vec_a) != len(vec_b):
        raise ValueError(
            f"Vector length mismatch: {len(vec_a)} vs {len(vec_b)}"
        )

    vec_a = vec_a.astype(np.float64)
    vec_b = vec_b.astype(np.float64)

    mean_a = np.mean(vec_a)
    mean_b = np.mean(vec_b)
    std_a = np.std(vec_a)
    std_b = np.std(vec_b)

    # Handle zero-variance (flat/constant) vectors
    if std_a < 1e-10 or std_b < 1e-10:
        # Both constant and equal → perfect match
        if std_a < 1e-10 and std_b < 1e-10:
            if np.allclose(vec_a, vec_b):
                return 1.0
            else:
                return 0.0
        # One constant, one not → no correlation
        return 0.0

    n = len(vec_a)
    ncc = np.sum((vec_a - mean_a) * (vec_b - mean_b)) / (n * std_a * std_b)

    # Clamp to [-1, 1] to handle floating-point precision issues
    return float(np.clip(ncc, -1.0, 1.0))


def compute_sequence_ncc(
    query_features: np.ndarray, window_features: np.ndarray
) -> float:
    """
    Compute average NCC across all frame pairs in a sequence window.

    This is the core sequence-level matching metric. Instead of matching
    single frames, we compare the entire query feature sequence against
    a window of the same length from the dataset video.

    Args:
        query_features: Query feature matrix (num_query_frames × feature_dim).
        window_features: Dataset window feature matrix (same shape as query).

    Returns:
        Average NCC score across all frame pairs. Range [-1.0, 1.0].
    """
    if query_features.shape != window_features.shape:
        raise ValueError(
            f"Shape mismatch: query {query_features.shape} vs "
            f"window {window_features.shape}"
        )

    num_frames = query_features.shape[0]
    if num_frames == 0:
        return 0.0

    total_ncc = 0.0
    valid_frames = 0

    for i in range(num_frames):
        try:
            ncc_val = compute_ncc(query_features[i], window_features[i])
            total_ncc += ncc_val
            valid_frames += 1
        except Exception:
            # Skip problematic frames
            continue

    if valid_frames == 0:
        return 0.0

    return total_ncc / valid_frames
