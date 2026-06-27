"""
ClipMatch Shared Utilities
Common functions used across matcher modules to reduce code duplication
"""


def format_timestamp_range(start, end):
    """
    Format a timestamp range for display.
    
    Args:
        start: Start time in seconds (or None)
        end: End time in seconds (or None)
        
    Returns:
        Formatted string like "01:23 - 02:45" or "Unknown"
    """
    if start is None or end is None:
        return "Unknown"
    
    def fmt(seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"
    
    return f"{fmt(start)} - {fmt(end)}"


def get_confidence_label(confidence):
    """
    Get human-readable confidence label from a score.
    
    Args:
        confidence: Confidence score (0-100)
        
    Returns:
        Label string like "Very Strong Match"
    """
    if confidence >= 75:
        return "Very Strong Match"
    elif confidence >= 50:
        return "Strong Match"
    elif confidence >= 25:
        return "Possible Match"
    elif confidence >= 10:
        return "Weak Match"
    return "Very Weak Match"


def estimate_timestamp_range(matched_timestamps, all_distances, clip_duration,
                             num_clip_frames, sample_rate=2.0):
    """
    Estimate the timestamp range in the reference video where the clip appears.
    
    Uses a sliding-window approach to find the densest cluster of consecutive
    matched timestamps, instead of just taking min/max which is fragile when
    different augmentations match to different parts of the reference.
    
    Args:
        matched_timestamps: List of reference timestamps per query frame (None if no match)
        all_distances: List of hash distances per query frame
        clip_duration: Duration of query clip in seconds
        num_clip_frames: Number of query frames analyzed
        sample_rate: Frames per second used during extraction
        
    Returns:
        (start_timestamp, end_timestamp) tuple, or (None, None)
    """
    valid_ts = [(i, t) for i, t in enumerate(matched_timestamps) if t is not None]
    
    if not valid_ts:
        # Fallback: use the frame with the lowest distance
        if all_distances:
            best_idx = all_distances.index(min(all_distances))
            if best_idx < len(matched_timestamps) and matched_timestamps[best_idx] is not None:
                ts = matched_timestamps[best_idx]
                dur = clip_duration if clip_duration > 0 else num_clip_frames / sample_rate
                return round(ts, 2), round(ts + dur, 2)
        return None, None
    
    if len(valid_ts) == 1:
        ts = valid_ts[0][1]
        dur = clip_duration if clip_duration > 0 else num_clip_frames / sample_rate
        return round(ts, 2), round(ts + dur, 2)
    
    # Sort timestamps by value
    sorted_ts = sorted([t for _, t in valid_ts])
    
    # Use frame ordering: pair each query frame index with its matched ref timestamp.
    # For a correct match, ref timestamps should increase roughly linearly.
    # Find the longest subsequence of timestamps that are roughly monotonically increasing.
    
    # Simple approach: find the median timestamp, then collect all timestamps
    # within a reasonable window around it.
    
    # Expected duration of the clip in the reference
    expected_duration = clip_duration if clip_duration > 0 else num_clip_frames / sample_rate
    
    # Try sliding window: find the window of expected_duration that contains
    # the most matched timestamps
    best_window_count = 0
    best_window_start = sorted_ts[0]
    
    for ts in sorted_ts:
        window_end = ts + expected_duration * 1.5  # Allow 50% tolerance
        count = sum(1 for t in sorted_ts if ts <= t <= window_end)
        if count > best_window_count:
            best_window_count = count
            best_window_start = ts
    
    # Collect timestamps in the best window
    window_end = best_window_start + expected_duration * 1.5
    window_ts = [t for t in sorted_ts if best_window_start <= t <= window_end]
    
    if window_ts:
        start = min(window_ts)
        end = start + expected_duration
        return round(start, 2), round(end, 2)
    
    # Final fallback: use weighted approach based on distance
    # Weight timestamps by inverse distance (closer match = higher weight)
    weights = []
    timestamps_with_weight = []
    for i, t in valid_ts:
        if i < len(all_distances):
            w = max(0.01, 1.0 / (1.0 + all_distances[i]))
            weights.append(w)
            timestamps_with_weight.append(t)
    
    if timestamps_with_weight and weights:
        total_w = sum(weights)
        weighted_center = sum(t * w for t, w in zip(timestamps_with_weight, weights)) / total_w
        start = weighted_center - expected_duration / 2
        end = weighted_center + expected_duration / 2
        return round(max(0, start), 2), round(end, 2)
    
    return None, None
