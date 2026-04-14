"""
timestamp_utils.py – Frame index to timestamp conversion utilities.

Converts frame indices to human-readable timestamps (HH:MM:SS)
based on the extraction FPS rate.
"""


def frame_to_timestamp(frame_index: int, fps: float) -> str:
    """
    Convert a frame index to a formatted timestamp string.

    Args:
        frame_index: Zero-based frame index.
        fps: Frames per second used during extraction.

    Returns:
        Timestamp string in HH:MM:SS format.
    """
    if fps <= 0:
        raise ValueError(f"FPS must be positive, got {fps}")
    if frame_index < 0:
        raise ValueError(f"Frame index must be non-negative, got {frame_index}")

    total_seconds = frame_index / fps
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)

    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def get_match_timestamps(
    best_start_frame: int, query_length_frames: int, fps: float
) -> tuple:
    """
    Compute start and end timestamps for a matched window.

    Args:
        best_start_frame: Starting frame index of the best match window.
        query_length_frames: Number of frames in the query clip.
        fps: Frames per second used during extraction.

    Returns:
        Tuple of (start_timestamp, end_timestamp) strings in HH:MM:SS format.
    """
    end_frame = best_start_frame + query_length_frames - 1
    start_time = frame_to_timestamp(best_start_frame, fps)
    end_time = frame_to_timestamp(end_frame, fps)
    return start_time, end_time


def seconds_to_timestamp(seconds: float) -> str:
    """
    Convert raw seconds to a formatted timestamp string.

    Args:
        seconds: Time in seconds.

    Returns:
        Timestamp string in HH:MM:SS format.
    """
    if seconds < 0:
        seconds = 0
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"
