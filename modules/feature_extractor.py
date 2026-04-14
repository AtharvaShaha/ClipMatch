"""
feature_extractor.py – Block-mean intensity feature extraction.

Divides each 256x256 grayscale frame into a 4x4 grid of 64x64 blocks
and computes the mean intensity of each block, producing a 16-value
feature vector per frame. Supports caching via .npy files.
"""

import os
import numpy as np
import cv2


def compute_block_features(frame: np.ndarray, grid_size: int = 4) -> np.ndarray:
    """
    Compute block-mean intensity feature vector for a single frame.

    Divides the frame into a grid_size x grid_size grid and computes
    the mean pixel intensity of each block.

    Args:
        frame: 2D grayscale numpy array (expected 256x256).
        grid_size: Number of blocks per axis (default 4 → 16 features).

    Returns:
        1D numpy array of length grid_size^2 containing mean intensities.
    """
    h, w = frame.shape[:2]
    block_h = h // grid_size
    block_w = w // grid_size

    features = []
    for row in range(grid_size):
        for col in range(grid_size):
            y_start = row * block_h
            y_end = (row + 1) * block_h
            x_start = col * block_w
            x_end = (col + 1) * block_w

            block = frame[y_start:y_end, x_start:x_end]
            features.append(np.mean(block))

    return np.array(features, dtype=np.float64)


def extract_video_features(
    frame_paths: list,
    cache_dir: str = None,
    video_name: str = None,
    force: bool = False,
) -> np.ndarray:
    """
    Extract block-mean features for all frames of a video.

    Args:
        frame_paths: List of paths to extracted grayscale frame images.
        cache_dir: Directory to cache feature arrays as .npy files.
        video_name: Identifier for the video (used for cache filename).
        force: Force re-computation even if cache exists.

    Returns:
        2D numpy array of shape (num_frames, 16) containing feature vectors.
    """
    # Check cache
    if cache_dir and video_name and not force:
        cache_path = os.path.join(cache_dir, f"{video_name}_features.npy")
        if os.path.exists(cache_path):
            features = np.load(cache_path)
            print(f"  [CACHE] Loaded cached features for {video_name}: {features.shape}")
            return features

    # Compute features for each frame
    all_features = []
    for i, frame_path in enumerate(frame_paths):
        try:
            frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
            if frame is None:
                print(f"  [WARN] Could not read frame: {frame_path}")
                continue

            feat = compute_block_features(frame)
            all_features.append(feat)
        except Exception as e:
            print(f"  [ERROR] Frame {i}: {e}")
            continue

    if len(all_features) == 0:
        return np.array([]).reshape(0, 16)

    features = np.array(all_features)

    # Save to cache
    if cache_dir and video_name:
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"{video_name}_features.npy")
        np.save(cache_path, features)
        print(f"  [CACHE] Saved features for {video_name}: {features.shape}")

    return features
