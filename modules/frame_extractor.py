"""
frame_extractor.py – Video frame extraction and caching.

Extracts frames from video files at a configurable FPS rate,
converts to grayscale, resizes to 256x256, and caches to disk.
"""

import os
import cv2
import numpy as np


def extract_frames(
    video_path: str,
    output_dir: str,
    target_fps: int = 16,
    frame_size: tuple = (256, 256),
    force: bool = False,
) -> list:
    """
    Extract frames from a video file at the specified FPS.

    Frames are converted to grayscale and resized to frame_size.
    Extracted frames are cached on disk; re-extraction is skipped
    if cached frames already exist (unless force=True).

    Args:
        video_path: Path to the video file.
        output_dir: Directory to save extracted frames.
        target_fps: Target extraction FPS (default 16).
        frame_size: Output frame dimensions (default 256x256).
        force: Force re-extraction even if cache exists.

    Returns:
        Sorted list of absolute paths to extracted frame images.

    Raises:
        FileNotFoundError: If video_path does not exist.
        RuntimeError: If video cannot be opened.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    # Create output directory
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    frame_dir = os.path.join(output_dir, video_name)
    os.makedirs(frame_dir, exist_ok=True)

    # Check cache – if frames already exist, return them
    if not force:
        existing_frames = _get_cached_frames(frame_dir)
        if len(existing_frames) > 0:
            print(f"  [CACHE] Using {len(existing_frames)} cached frames for {video_name}")
            return existing_frames

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        video_fps = 24.0  # fallback

    # Calculate frame sampling interval
    frame_interval = max(1, int(round(video_fps / target_fps)))

    frame_count = 0
    saved_count = 0
    frame_paths = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Sample at target FPS
        if frame_count % frame_interval == 0:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Resize to target size
            resized = cv2.resize(gray, frame_size, interpolation=cv2.INTER_AREA)

            # Save frame
            frame_filename = f"frame_{saved_count:06d}.png"
            frame_path = os.path.join(frame_dir, frame_filename)
            cv2.imwrite(frame_path, resized)
            frame_paths.append(frame_path)
            saved_count += 1

        frame_count += 1

    cap.release()

    print(f"  [EXTRACT] {video_name}: {saved_count} frames extracted "
          f"(video FPS={video_fps:.1f}, target FPS={target_fps}, interval={frame_interval})")

    return sorted(frame_paths)


def _get_cached_frames(frame_dir: str) -> list:
    """
    Retrieve sorted list of cached frame paths from a directory.

    Args:
        frame_dir: Directory containing cached frame images.

    Returns:
        Sorted list of absolute paths to cached frames.
    """
    if not os.path.exists(frame_dir):
        return []

    frames = [
        os.path.join(frame_dir, f)
        for f in os.listdir(frame_dir)
        if f.endswith(".png") and f.startswith("frame_")
    ]
    return sorted(frames)


def get_video_info(video_path: str) -> dict:
    """
    Get basic metadata about a video file.

    Args:
        video_path: Path to the video file.

    Returns:
        Dict with keys: fps, frame_count, duration_sec, width, height.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"fps": 0, "frame_count": 0, "duration_sec": 0, "width": 0, "height": 0}

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0

    cap.release()

    return {
        "fps": fps,
        "frame_count": frame_count,
        "duration_sec": duration,
        "width": width,
        "height": height,
    }
