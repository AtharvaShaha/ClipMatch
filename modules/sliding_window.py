"""
sliding_window.py – Sequence-level sliding window matching.

Slides the query feature sequence across the dataset video's feature
sequence, computing average NCC at every position. Always scans the
entire dataset video before selecting the best match — this avoids
incorrect matches caused by static scenes.
"""

import numpy as np
from .ncc_matcher import compute_sequence_ncc


def sliding_window_match(
    query_features: np.ndarray,
    dataset_features: np.ndarray,
    step: int = 1,
    verbose: bool = True,
    video_name: str = "",
) -> dict:
    """
    Perform sliding window sequence matching.

    Slides a window of query_length across the dataset features,
    computing average NCC at each position. Scans the ENTIRE dataset
    video before returning the best match.

    This ensures:
        - Static scene ambiguity is resolved by picking highest NCC
        - Temporally consistent matches are preferred
        - No early stopping at suboptimal matches

    Args:
        query_features: Query feature matrix (Q_frames × 16).
        dataset_features: Dataset video feature matrix (D_frames × 16).
        step: Step size for sliding window (default 1 for exhaustive).
        verbose: Print progress logs.
        video_name: Name for logging purposes.

    Returns:
        Dict with keys:
            - best_start: Starting frame index of best match window
            - best_end: Ending frame index of best match window
            - best_score: Highest average NCC score found
            - scores: List of all (position, score) tuples
            - scanned: True (confirms full scan completed)
    """
    query_len = query_features.shape[0]
    dataset_len = dataset_features.shape[0]

    if query_len == 0 or dataset_len == 0:
        return {
            "best_start": -1,
            "best_end": -1,
            "best_score": 0.0,
            "scores": [],
            "scanned": True,
        }

    # If query is longer than dataset, no match possible
    if query_len > dataset_len:
        if verbose:
            print(f"  [SKIP] {video_name}: Query ({query_len} frames) longer "
                  f"than dataset ({dataset_len} frames)")
        return {
            "best_start": -1,
            "best_end": -1,
            "best_score": 0.0,
            "scores": [],
            "scanned": True,
        }

    num_windows = (dataset_len - query_len) // step + 1
    best_start = 0
    best_score = -2.0  # NCC minimum is -1
    all_scores = []

    log_interval = max(1, num_windows // 10)  # Log every ~10%

    for i in range(0, dataset_len - query_len + 1, step):
        window = dataset_features[i : i + query_len]
        score = compute_sequence_ncc(query_features, window)
        all_scores.append((i, score))

        if score > best_score:
            best_score = score
            best_start = i

        # Progress logging
        if verbose and (len(all_scores) % log_interval == 0):
            progress = len(all_scores) / num_windows * 100
            print(f"  [SCAN] {video_name}: {progress:.0f}% scanned | "
                  f"Current best NCC: {best_score:.4f} at frame {best_start}")

    best_end = best_start + query_len - 1

    if verbose:
        print(f"  [DONE] {video_name}: Best NCC = {best_score:.4f} "
              f"at frames [{best_start}, {best_end}] "
              f"({num_windows} windows scanned)")

    return {
        "best_start": best_start,
        "best_end": best_end,
        "best_score": float(best_score),
        "scores": all_scores,
        "scanned": True,
    }
