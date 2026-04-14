"""
dataset_downloader.py – Synthetic video dataset generation.

Generates a dataset of ~100 synthetic 1-minute videos and ~20 query clips
(15-second sub-segments extracted from dataset videos) using OpenCV.

Each video contains unique combinations of:
    - Moving geometric shapes (circles, rectangles, triangles)
    - Color gradients and scrolling patterns
    - Random intensity variations
    - Scene transitions
    - Noise bursts

Query clips are exact sub-segments of dataset videos with known
ground-truth for validation.
"""

import os
import json
import random
import math
import cv2
import numpy as np


# ──────────────────────────────────────────────────────────
#  Configuration
# ──────────────────────────────────────────────────────────

VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480
VIDEO_FPS = 24
DATASET_DURATION_SEC = 60      # ~1 minute per dataset video
QUERY_DURATION_SEC = 15        # ~15 seconds per query clip
NUM_DATASET_VIDEOS = 100
NUM_QUERY_CLIPS = 20
CODEC = "mp4v"


# ──────────────────────────────────────────────────────────
#  Scene Generators
# ──────────────────────────────────────────────────────────

def _gen_moving_circle(frame, t, seed):
    """Draw a moving circle with unique trajectory based on seed."""
    rng = random.Random(seed)
    radius = rng.randint(20, 60)
    color = (rng.randint(50, 255), rng.randint(50, 255), rng.randint(50, 255))
    speed_x = rng.uniform(0.5, 3.0)
    speed_y = rng.uniform(0.3, 2.0)
    phase_x = rng.uniform(0, 2 * math.pi)
    phase_y = rng.uniform(0, 2 * math.pi)

    cx = int(VIDEO_WIDTH / 2 + (VIDEO_WIDTH / 3) * math.sin(speed_x * t + phase_x))
    cy = int(VIDEO_HEIGHT / 2 + (VIDEO_HEIGHT / 3) * math.cos(speed_y * t + phase_y))

    cv2.circle(frame, (cx, cy), radius, color, -1, cv2.LINE_AA)
    return frame


def _gen_moving_rectangle(frame, t, seed):
    """Draw a moving rectangle with rotation effect."""
    rng = random.Random(seed + 1000)
    w = rng.randint(40, 120)
    h = rng.randint(30, 90)
    color = (rng.randint(50, 255), rng.randint(50, 255), rng.randint(50, 255))
    speed = rng.uniform(0.4, 2.5)
    phase = rng.uniform(0, 2 * math.pi)

    cx = int(VIDEO_WIDTH / 2 + (VIDEO_WIDTH / 4) * math.cos(speed * t + phase))
    cy = int(VIDEO_HEIGHT / 2 + (VIDEO_HEIGHT / 4) * math.sin(speed * t * 0.7 + phase))

    x1, y1 = cx - w // 2, cy - h // 2
    x2, y2 = cx + w // 2, cy + h // 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1, cv2.LINE_AA)
    return frame


def _gen_gradient_sweep(frame, t, seed):
    """Generate a sweeping color gradient background."""
    rng = random.Random(seed + 2000)
    freq = rng.uniform(0.01, 0.05)
    channel = rng.randint(0, 2)

    for y in range(VIDEO_HEIGHT):
        intensity = int(127 + 127 * math.sin(freq * y + t * 0.5))
        frame[y, :, channel] = np.clip(
            frame[y, :, channel].astype(np.int16) + intensity, 0, 255
        ).astype(np.uint8)
    return frame


def _gen_scrolling_bars(frame, t, seed):
    """Generate horizontal scrolling bar pattern."""
    rng = random.Random(seed + 3000)
    bar_width = rng.randint(10, 40)
    speed = rng.uniform(1.0, 5.0)
    color = (rng.randint(30, 200), rng.randint(30, 200), rng.randint(30, 200))

    offset = int(t * speed * bar_width) % (bar_width * 2)
    for y in range(0, VIDEO_HEIGHT, bar_width * 2):
        y_start = (y + offset) % VIDEO_HEIGHT
        y_end = min(y_start + bar_width, VIDEO_HEIGHT)
        frame[y_start:y_end, :] = np.clip(
            frame[y_start:y_end].astype(np.int16) + np.array(color), 0, 255
        ).astype(np.uint8)
    return frame


def _gen_pulsing_blob(frame, t, seed):
    """Generate a pulsing elliptical blob."""
    rng = random.Random(seed + 4000)
    cx = rng.randint(100, VIDEO_WIDTH - 100)
    cy = rng.randint(100, VIDEO_HEIGHT - 100)
    color = (rng.randint(80, 255), rng.randint(80, 255), rng.randint(80, 255))
    base_r = rng.randint(30, 80)
    pulse_speed = rng.uniform(1.0, 4.0)

    rx = int(base_r + base_r * 0.5 * math.sin(pulse_speed * t))
    ry = int(base_r + base_r * 0.3 * math.cos(pulse_speed * t * 0.8))
    cv2.ellipse(frame, (cx, cy), (max(rx, 5), max(ry, 5)), 0, 0, 360, color, -1, cv2.LINE_AA)
    return frame


def _gen_random_noise_region(frame, t, seed):
    """Add a region of random noise that moves over time."""
    rng = random.Random(seed + 5000)
    region_w = rng.randint(80, 200)
    region_h = rng.randint(60, 150)
    speed = rng.uniform(0.3, 1.5)

    cx = int((VIDEO_WIDTH / 2) + (VIDEO_WIDTH / 4) * math.sin(speed * t))
    cy = int((VIDEO_HEIGHT / 2) + (VIDEO_HEIGHT / 4) * math.cos(speed * t * 0.6))

    x1 = max(0, cx - region_w // 2)
    y1 = max(0, cy - region_h // 2)
    x2 = min(VIDEO_WIDTH, cx + region_w // 2)
    y2 = min(VIDEO_HEIGHT, cy + region_h // 2)

    noise = np.random.RandomState(seed + int(t * 100)).randint(
        0, 100, (y2 - y1, x2 - x1, 3), dtype=np.uint8
    )
    frame[y1:y2, x1:x2] = np.clip(
        frame[y1:y2, x1:x2].astype(np.int16) + noise.astype(np.int16), 0, 255
    ).astype(np.uint8)
    return frame


# List of all scene generator functions
SCENE_GENERATORS = [
    _gen_moving_circle,
    _gen_moving_rectangle,
    _gen_gradient_sweep,
    _gen_scrolling_bars,
    _gen_pulsing_blob,
    _gen_random_noise_region,
]


# ──────────────────────────────────────────────────────────
#  Video Generation
# ──────────────────────────────────────────────────────────

def generate_dataset_video(
    output_path: str,
    video_index: int,
    duration_sec: int = DATASET_DURATION_SEC,
    fps: int = VIDEO_FPS,
) -> bool:
    """
    Generate a single synthetic dataset video.

    Each video uses a unique combination of 2-4 scene generators
    with unique seeds, ensuring visual uniqueness across the dataset.

    Args:
        output_path: Path to save the output video.
        video_index: Unique index for seed generation.
        duration_sec: Video duration in seconds.
        fps: Frames per second.

    Returns:
        True if generation succeeded.
    """
    try:
        fourcc = cv2.VideoWriter_fourcc(*CODEC)
        writer = cv2.VideoWriter(output_path, fourcc, fps, (VIDEO_WIDTH, VIDEO_HEIGHT))

        if not writer.isOpened():
            print(f"  [ERROR] Cannot create video writer for {output_path}")
            return False

        rng = random.Random(video_index * 7919)  # Prime seed for uniqueness
        num_generators = rng.randint(2, 4)
        selected_gens = rng.sample(SCENE_GENERATORS, min(num_generators, len(SCENE_GENERATORS)))

        # Background base color
        bg_b = rng.randint(5, 40)
        bg_g = rng.randint(5, 40)
        bg_r = rng.randint(5, 40)

        total_frames = duration_sec * fps

        for f in range(total_frames):
            t = f / fps  # Time in seconds

            # Create base frame with dark background
            frame = np.full((VIDEO_HEIGHT, VIDEO_WIDTH, 3), (bg_b, bg_g, bg_r), dtype=np.uint8)

            # Apply selected scene generators
            for gen in selected_gens:
                frame = gen(frame, t, video_index)

            # Add frame counter watermark (unique per video)
            text = f"V{video_index:03d} F{f:05d}"
            cv2.putText(
                frame, text, (10, VIDEO_HEIGHT - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA
            )

            writer.write(frame)

        writer.release()
        return True

    except Exception as e:
        print(f"  [ERROR] Failed to generate {output_path}: {e}")
        return False


def generate_query_clip(
    source_video_path: str,
    output_path: str,
    start_sec: float,
    duration_sec: int = QUERY_DURATION_SEC,
) -> bool:
    """
    Extract a query clip from a dataset video.

    Opens the source video, seeks to start_sec, and writes
    duration_sec seconds of frames to the output path.

    Args:
        source_video_path: Path to the source dataset video.
        output_path: Path to save the query clip.
        start_sec: Start time in seconds for extraction.
        duration_sec: Duration of the query clip in seconds.

    Returns:
        True if extraction succeeded.
    """
    try:
        cap = cv2.VideoCapture(source_video_path)
        if not cap.isOpened():
            print(f"  [ERROR] Cannot open source: {source_video_path}")
            return False

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = VIDEO_FPS

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        start_frame = int(start_sec * fps)
        num_frames = int(duration_sec * fps)

        if start_frame + num_frames > total_frames:
            start_frame = max(0, total_frames - num_frames)

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        fourcc = cv2.VideoWriter_fourcc(*CODEC)
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        if not writer.isOpened():
            cap.release()
            return False

        written = 0
        while written < num_frames:
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(frame)
            written += 1

        writer.release()
        cap.release()
        return written > 0

    except Exception as e:
        print(f"  [ERROR] Failed to extract query clip: {e}")
        return False


# ──────────────────────────────────────────────────────────
#  Full Dataset Preparation
# ──────────────────────────────────────────────────────────

def prepare_dataset(
    base_dir: str,
    num_dataset_videos: int = NUM_DATASET_VIDEOS,
    num_query_clips: int = NUM_QUERY_CLIPS,
    force: bool = False,
) -> dict:
    """
    Prepare the full synthetic dataset.

    Creates:
        - {base_dir}/Dataset/  with num_dataset_videos synthetic videos
        - {base_dir}/query/    with num_query_clips extracted from dataset
        - {base_dir}/ground_truth.json  mapping query → source video + timestamps

    Skips generation if files already exist (resume support).
    Set force=True to regenerate everything.

    Args:
        base_dir: Root project directory.
        num_dataset_videos: Number of dataset videos to generate.
        num_query_clips: Number of query clips to extract.
        force: Force regeneration of all videos.

    Returns:
        Dict with keys: dataset_dir, query_dir, ground_truth, stats.
    """
    dataset_dir = os.path.join(base_dir, "Dataset")
    query_dir = os.path.join(base_dir, "query")
    gt_path = os.path.join(base_dir, "ground_truth.json")

    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(query_dir, exist_ok=True)

    ground_truth = {}
    stats = {"dataset_generated": 0, "dataset_skipped": 0, "query_generated": 0, "query_skipped": 0}

    # ── Step 1: Generate dataset videos ──
    print(f"\n{'='*60}")
    print(f"  GENERATING DATASET VIDEOS ({num_dataset_videos})")
    print(f"{'='*60}")

    for i in range(num_dataset_videos):
        filename = f"video_{i:03d}.mp4"
        filepath = os.path.join(dataset_dir, filename)

        if os.path.exists(filepath) and not force:
            stats["dataset_skipped"] += 1
            if (i + 1) % 20 == 0:
                print(f"  [SKIP] {i + 1}/{num_dataset_videos} dataset videos (cached)")
            continue

        success = generate_dataset_video(filepath, i)
        if success:
            stats["dataset_generated"] += 1
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [GEN] {i + 1}/{num_dataset_videos} dataset videos created")

    print(f"  Dataset: {stats['dataset_generated']} generated, "
          f"{stats['dataset_skipped']} cached")

    # ── Step 2: Generate query clips ──
    print(f"\n{'='*60}")
    print(f"  GENERATING QUERY CLIPS ({num_query_clips})")
    print(f"{'='*60}")

    rng = random.Random(42)  # Reproducible query selection
    dataset_files = sorted([
        f for f in os.listdir(dataset_dir)
        if f.endswith(".mp4")
    ])

    if len(dataset_files) == 0:
        print("  [ERROR] No dataset videos found!")
        return {"dataset_dir": dataset_dir, "query_dir": query_dir,
                "ground_truth": {}, "stats": stats}

    for q in range(num_query_clips):
        query_filename = f"query_{q:03d}.mp4"
        query_path = os.path.join(query_dir, query_filename)

        if os.path.exists(query_path) and not force:
            stats["query_skipped"] += 1
            continue

        # Pick a random source video and start time
        source_file = rng.choice(dataset_files)
        source_path = os.path.join(dataset_dir, source_file)

        # Random start within the first 45 seconds (leaving room for 15s clip)
        max_start = max(0, DATASET_DURATION_SEC - QUERY_DURATION_SEC)
        start_sec = rng.uniform(0, max_start)

        success = generate_query_clip(source_path, query_path, start_sec)
        if success:
            stats["query_generated"] += 1
            ground_truth[query_filename] = {
                "source_video": source_file,
                "start_sec": round(start_sec, 2),
                "end_sec": round(start_sec + QUERY_DURATION_SEC, 2),
            }
            print(f"  [GEN] {query_filename} <- {source_file} "
                  f"[{start_sec:.1f}s - {start_sec + QUERY_DURATION_SEC:.1f}s]")
        else:
            print(f"  [FAIL] Could not generate {query_filename}")

    # ── Step 3: Save ground truth ──
    # Load existing ground truth if not regenerating
    if os.path.exists(gt_path) and not force:
        try:
            with open(gt_path, "r") as f:
                existing_gt = json.load(f)
            existing_gt.update(ground_truth)
            ground_truth = existing_gt
        except Exception:
            pass

    with open(gt_path, "w") as f:
        json.dump(ground_truth, f, indent=2)

    print(f"\n  Queries: {stats['query_generated']} generated, "
          f"{stats['query_skipped']} cached")
    print(f"  Ground truth saved to: {gt_path}")
    print(f"{'='*60}\n")

    return {
        "dataset_dir": dataset_dir,
        "query_dir": query_dir,
        "ground_truth": ground_truth,
        "stats": stats,
    }


# ──────────────────────────────────────────────────────────
#  CLI Entry Point
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(f"ClipMatch Dataset Generator")
    print(f"Base directory: {base}")

    start = time.time()
    result = prepare_dataset(base)
    elapsed = time.time() - start

    print(f"\nDone in {elapsed:.1f}s")
    print(f"Dataset dir: {result['dataset_dir']}")
    print(f"Query dir: {result['query_dir']}")
    print(f"Ground truth entries: {len(result['ground_truth'])}")
