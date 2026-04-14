"""
main_pipeline.py – ClipMatch orchestration pipeline.

Coordinates the full video matching workflow:
    1. Ensure dataset exists
    2. Extract frames from query video
    3. Compute features for query
    4. For each dataset video:
       a. Extract frames
       b. Compute features
       c. Run sliding window match
    5. Select global best match across all dataset videos
    6. Convert to timestamps
    7. Return structured result

Supports progress callbacks for Streamlit integration.
"""

import os
import sys
import time
import glob

# Add parent directory to path for module imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.frame_extractor import extract_frames
from modules.feature_extractor import extract_video_features
from modules.sliding_window import sliding_window_match
from modules.timestamp_utils import get_match_timestamps
from modules.dataset_downloader import prepare_dataset


def run_pipeline(
    query_path: str,
    dataset_dir: str = None,
    base_dir: str = None,
    target_fps: int = 16,
    threshold: float = 0.7,
    step: int = 2,
    progress_callback=None,
    auto_prepare: bool = True,
) -> dict:
    """
    Run the full ClipMatch detection pipeline.

    Args:
        query_path: Path to the query video clip.
        dataset_dir: Path to the dataset directory. If None, uses base_dir/Dataset.
        base_dir: Root project directory. Defaults to script's parent dir.
        target_fps: FPS for frame extraction (default 16).
        threshold: Minimum NCC score to consider a match (default 0.7).
        step: Sliding window step size (default 2 for speed; 1 for precision).
        progress_callback: Optional function(status_str, progress_float) for UI updates.
        auto_prepare: Automatically generate dataset if missing.

    Returns:
        Dict with keys:
            - match_found: bool
            - video: matched video filename (or None)
            - start_time: start timestamp string
            - end_time: end timestamp string
            - confidence: NCC similarity score
            - execution_time: total time in seconds
            - details: additional match details
    """
    start_time = time.time()

    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))

    if dataset_dir is None:
        dataset_dir = os.path.join(base_dir, "Dataset")

    temp_frames_dir = os.path.join(base_dir, "temp_frames")
    temp_query_dir = os.path.join(base_dir, "temp_query_frames")
    cache_dir = os.path.join(base_dir, "cache")

    # Create directories
    for d in [temp_frames_dir, temp_query_dir, cache_dir]:
        os.makedirs(d, exist_ok=True)

    def _update(status, progress=0.0):
        print(f"  [{progress*100:.0f}%] {status}")
        if progress_callback:
            progress_callback(status, progress)

    # ── Step 0: Validate query ──
    if not os.path.exists(query_path):
        return _error_result(f"Query video not found: {query_path}", start_time)

    # ── Step 1: Auto-prepare dataset if needed ──
    if auto_prepare and not os.path.exists(dataset_dir):
        _update("Preparing dataset (first run)...", 0.0)
        prepare_dataset(base_dir)

    # Get dataset video files
    dataset_videos = sorted(glob.glob(os.path.join(dataset_dir, "*.mp4")))
    if not dataset_videos:
        return _error_result("No dataset videos found!", start_time)

    _update(f"Found {len(dataset_videos)} dataset videos", 0.05)

    # ── Step 2: Extract query frames ──
    _update("Extracting query frames...", 0.10)
    try:
        query_frames = extract_frames(query_path, temp_query_dir, target_fps)
    except Exception as e:
        return _error_result(f"Query frame extraction failed: {e}", start_time)

    if len(query_frames) == 0:
        return _error_result("No frames extracted from query video", start_time)

    _update(f"Query: {len(query_frames)} frames extracted", 0.15)

    # ── Step 3: Compute query features ──
    _update("Computing query features...", 0.20)
    query_name = os.path.splitext(os.path.basename(query_path))[0]
    query_features = extract_video_features(
        query_frames, cache_dir, f"query_{query_name}"
    )

    if query_features.shape[0] == 0:
        return _error_result("Could not compute query features", start_time)

    _update(f"Query features: {query_features.shape}", 0.25)

    # ── Step 4: Match against all dataset videos ──
    _update("Starting dataset matching...", 0.30)

    global_best = {
        "video": None,
        "score": -2.0,
        "start_frame": 0,
        "end_frame": 0,
        "query_length": query_features.shape[0],
    }

    for idx, video_path in enumerate(dataset_videos):
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        progress = 0.30 + 0.60 * (idx / len(dataset_videos))

        _update(f"Matching: {video_name} ({idx+1}/{len(dataset_videos)})", progress)

        # Extract dataset video frames
        try:
            ds_frames = extract_frames(video_path, temp_frames_dir, target_fps)
        except Exception as e:
            print(f"  [ERROR] Frame extraction failed for {video_name}: {e}")
            continue

        if len(ds_frames) == 0:
            continue

        # Compute dataset video features
        ds_features = extract_video_features(ds_frames, cache_dir, video_name)

        if ds_features.shape[0] == 0:
            continue

        # Sliding window match (scans entire video)
        result = sliding_window_match(
            query_features, ds_features,
            step=step, verbose=True, video_name=video_name
        )

        # Update global best
        if result["best_score"] > global_best["score"]:
            global_best["video"] = os.path.basename(video_path)
            global_best["score"] = result["best_score"]
            global_best["start_frame"] = result["best_start"]
            global_best["end_frame"] = result["best_end"]
            print(f"  [NEW BEST] {video_name}: NCC = {result['best_score']:.4f}")

    # ── Step 5: Compile result ──
    _update("Compiling results...", 0.95)
    elapsed = time.time() - start_time

    if global_best["video"] is None or global_best["score"] < threshold:
        _update("No match found above threshold", 1.0)
        return {
            "match_found": False,
            "video": global_best.get("video"),
            "start_time": "N/A",
            "end_time": "N/A",
            "confidence": round(global_best["score"], 4) if global_best["score"] > -2 else 0.0,
            "execution_time": round(elapsed, 2),
            "details": {
                "threshold": threshold,
                "videos_scanned": len(dataset_videos),
                "query_frames": len(query_frames),
                "best_raw_score": round(global_best["score"], 4) if global_best["score"] > -2 else 0.0,
            },
        }

    # Convert frame indices to timestamps
    start_ts, end_ts = get_match_timestamps(
        global_best["start_frame"],
        global_best["query_length"],
        target_fps,
    )

    _update("Match found!", 1.0)

    # Print final result
    print(f"\n{'='*60}")
    print(f"  MATCH FOUND!")
    print(f"  Video:      {global_best['video']}")
    print(f"  Start Time: {start_ts}")
    print(f"  End Time:   {end_ts}")
    print(f"  Confidence: {global_best['score']:.4f}")
    print(f"  Time:       {elapsed:.1f}s")
    print(f"{'='*60}\n")

    return {
        "match_found": True,
        "video": global_best["video"],
        "start_time": start_ts,
        "end_time": end_ts,
        "confidence": round(global_best["score"], 4),
        "execution_time": round(elapsed, 2),
        "details": {
            "threshold": threshold,
            "videos_scanned": len(dataset_videos),
            "query_frames": len(query_frames),
            "best_start_frame": global_best["start_frame"],
            "best_end_frame": global_best["end_frame"],
        },
    }


def _error_result(message: str, start_time: float) -> dict:
    """Create a standardized error result."""
    print(f"  [ERROR] {message}")
    return {
        "match_found": False,
        "video": None,
        "start_time": "N/A",
        "end_time": "N/A",
        "confidence": 0.0,
        "execution_time": round(time.time() - start_time, 2),
        "details": {"error": message},
    }


# ──────────────────────────────────────────────────────────
#  CLI Entry Point
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))
    query_dir = os.path.join(base, "query")

    # Find first query clip
    queries = sorted(glob.glob(os.path.join(query_dir, "*.mp4")))
    if not queries:
        print("No query clips found. Run dataset_downloader.py first.")
        sys.exit(1)

    query = queries[0]
    print(f"Running pipeline with query: {query}")
    result = run_pipeline(query, base_dir=base)
    print(f"\nResult: {result}")
