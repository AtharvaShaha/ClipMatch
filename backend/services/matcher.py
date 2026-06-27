"""
ClipMatch Matcher
Core matching engine for comparing query clips against reference videos
"""

import logging
logger = logging.getLogger('clipmatch.matcher')

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import json
import sys
sys.path.append('..')
from config import MatchConfig, FeatureConfig
from models.database import get_session, ReferenceVideo, VideoFrame, MatchResult
from services.video_processor import VideoProcessor
from services.feature_extractor import FeatureExtractor, hex_to_int, fast_hamming
from services.utils import format_timestamp_range, get_confidence_label, estimate_timestamp_range


# ── Vectorised Hamming with 16-bit LUT (fully vectorised, no Python loops) ──

# Pre-compute popcount for all 16-bit values (65536 entries)
_POPCOUNT_LUT_16 = np.array([bin(i).count('1') for i in range(65536)], dtype=np.int32)


def _build_int_arrays(ref_hashes_raw):
    """
    Convert list of (id, ts, phash, dhash, whash, brightness, r, g, b) tuples
    into NumPy uint64 arrays for vectorised Hamming distance.
    """
    n = len(ref_hashes_raw)
    ph = np.zeros(n, dtype=np.uint64)
    dh = np.zeros(n, dtype=np.uint64)
    wh = np.zeros(n, dtype=np.uint64)
    ts = np.zeros(n, dtype=np.float64)
    br = np.zeros(n, dtype=np.float64)
    cr = np.zeros(n, dtype=np.float64)
    cg = np.zeros(n, dtype=np.float64)
    cb = np.zeros(n, dtype=np.float64)

    for i, (fid, fts, fp, fd, fw, fb, fr, fg, fbb) in enumerate(ref_hashes_raw):
        ph[i] = np.uint64(hex_to_int(fp) if fp else 0)
        dh[i] = np.uint64(hex_to_int(fd) if fd else 0)
        wh[i] = np.uint64(hex_to_int(fw) if fw else 0)
        ts[i] = fts or 0
        br[i] = fb or 0.5
        cr[i] = fr or 128
        cg[i] = fg or 128
        cb[i] = fbb or 128

    return ph, dh, wh, ts, br, cr, cg, cb


def _vectorised_hamming(query_int: int, ref_arr: np.ndarray) -> np.ndarray:
    """
    Fully vectorised Hamming distance using 16-bit lookup table.
    Splits each 64-bit XOR into four 16-bit chunks, looks up popcount for each,
    and sums. No Python loops — ~50× faster than the previous implementation.
    """
    xor = np.bitwise_xor(ref_arr.astype(np.uint64), np.uint64(query_int))
    mask = np.uint64(0xFFFF)
    return (
        _POPCOUNT_LUT_16[(xor & mask).astype(np.intp)] +
        _POPCOUNT_LUT_16[((xor >> np.uint64(16)) & mask).astype(np.intp)] +
        _POPCOUNT_LUT_16[((xor >> np.uint64(32)) & mask).astype(np.intp)] +
        _POPCOUNT_LUT_16[((xor >> np.uint64(48)) & mask).astype(np.intp)]
    )


def _compute_augmented_features(feature_extractor, clip_frames,
                                 video_processor=None, clip_path=None):
    """
    Compute original + augmented features for robust matching against
    cropped, flipped, and mirrored query videos.
    
    Augmentations:
    1. Original (10% query crop) — default query preprocessing
    2. Reference-crop (20% crop) — matches reference index exactly
    3. Horizontal flip of original — catches mirrored/flipped videos
    4. No-crop (just resize) — catches already-cropped/cut queries
    
    Returns list of feature sets (each is a list of dicts from batch_extract).
    """
    # 1. Original features (from the already-preprocessed clip_frames)
    original_features = feature_extractor.batch_extract(clip_frames)
    feature_sets = [original_features]
    
    # 2. Reference-crop (20% crop — matches how references were indexed)
    #    Re-extract from video file with refcrop mode
    if video_processor and clip_path:
        try:
            refcrop_frames = video_processor.extract_all_frames_to_list(
                clip_path, max_frames=len(clip_frames), crop_mode='refcrop'
            )
            if refcrop_frames and len(refcrop_frames) > 0:
                refcrop_features = feature_extractor.batch_extract(refcrop_frames)
                feature_sets.append(refcrop_features)
        except Exception:
            pass  # Skip if re-extraction fails
    
    # 3. Horizontal flip — catches mirrored/flipped videos
    flipped_frames = []
    for fn, ts, img in clip_frames:
        if len(img.shape) == 3:
            flipped = img[:, ::-1, :].copy()
        else:
            flipped = img[:, ::-1].copy()
        flipped_frames.append((fn, ts, flipped))
    flipped_features = feature_extractor.batch_extract(flipped_frames)
    feature_sets.append(flipped_features)
    
    # 4. No-crop (just resize) — catches queries that are already cropped/cut
    if video_processor and clip_path:
        try:
            nocrop_frames = video_processor.extract_all_frames_to_list(
                clip_path, max_frames=len(clip_frames), crop_mode='nocrop'
            )
            if nocrop_frames and len(nocrop_frames) > 0:
                nocrop_features = feature_extractor.batch_extract(nocrop_frames)
                feature_sets.append(nocrop_features)
        except Exception:
            pass  # Skip if re-extraction fails
    
    logger.debug("Computed %d augmented feature sets (%d frames each)", len(feature_sets), len(clip_frames))
    return feature_sets


class ClipMatcher:
    """
    Matches query video clips against indexed reference videos.
    
    Uses a sliding window approach with perceptual hash comparison
    to find potential matches and estimate timestamps.
    """
    
    def __init__(self):
        """Initialize the matcher with processor and extractor."""
        self.video_processor = VideoProcessor()
        self.feature_extractor = FeatureExtractor()
        self.window_size = MatchConfig.WINDOW_SIZE
        self.window_stride = MatchConfig.WINDOW_STRIDE
        self.hash_threshold = MatchConfig.HASH_THRESHOLD
        self.min_consecutive = MatchConfig.MIN_CONSECUTIVE_MATCHES
    
    def match_clip(self, clip_path: str, 
                   save_result: bool = True) -> Dict:
        """
        Match a query clip against all indexed reference videos.
        
        Args:
            clip_path: Path to the query video clip
            save_result: Whether to save the result to database
            
        Returns:
            Dictionary containing match results
        """
        import time
        start_time = time.time()
        
        # Validate and get clip info
        is_valid, message = self.video_processor.validate_query_clip(clip_path)
        if not is_valid:
            return {
                'success': False,
                'error': message,
                'matches': []
            }
        
        clip_info = self.video_processor.get_video_info(clip_path)
        
        # Stage 1: Extract frames from the clip (query_mode for relaxed crop)
        t_extract = time.time()
        clip_frames = self.video_processor.extract_all_frames_to_list(clip_path, query_mode=True)
        
        if not clip_frames:
            return {
                'success': False,
                'error': 'Could not extract frames from clip',
                'matches': []
            }
        frame_extraction_time = time.time() - t_extract
        
        # Stage 2: Extract features from clip frames (original + augmented crop/flip)
        t_features = time.time()
        augmented_feature_sets = _compute_augmented_features(
            self.feature_extractor, clip_frames,
            video_processor=self.video_processor, clip_path=clip_path
        )
        clip_features = augmented_feature_sets[0]  # Original features for metadata
        feature_extraction_time = time.time() - t_features
        
        # Stage 3: Match against references
        t_match = time.time()
        session = get_session()
        try:
            references = session.query(ReferenceVideo).filter(
                ReferenceVideo.status == 'indexed'
            ).all()
            
            if not references:
                return {
                    'success': False,
                    'error': 'No reference videos have been indexed',
                    'matches': []
                }
            
            # Match against each reference video
            all_matches = []
            clip_duration = clip_info.get('duration', 0)
            
            for ref_video in references:
                match_result = self._match_against_reference(
                    clip_features, ref_video, session, clip_duration,
                    augmented_feature_sets=augmented_feature_sets
                )
                if match_result and match_result['confidence'] >= MatchConfig.MIN_CONFIDENCE_THRESHOLD:
                    all_matches.append(match_result)
                    
                    # OPTIMIZATION: If we find a near-perfect match, stop searching
                    # A 95%+ confidence match is extremely reliable and likely the correct source
                    if match_result['confidence'] >= 95:
                        logger.info("Found near-perfect match (%.1f%%) in %s - stopping search", match_result['confidence'], ref_video.filename)
                        break
            
            # Sort by confidence (highest first)
            all_matches.sort(key=lambda x: x['confidence'], reverse=True)
            
            matching_time = time.time() - t_match
            processing_time = time.time() - start_time
            
            # Log timing breakdown
            logger.info("[Timing] Frame extraction: %.2fs | Feature extraction: %.2fs | Matching: %.2fs (%d refs) | Total: %.2fs",
                       frame_extraction_time, feature_extraction_time, matching_time, len(references), processing_time)
            
            # Prepare result
            best_match = all_matches[0] if all_matches else None
            
            result = {
                'success': True,
                'query': {
                    'filename': clip_info['filename'],
                    'duration': clip_info['duration'],
                    'frames_analyzed': len(clip_frames)
                },
                'best_match': best_match,
                'all_matches': all_matches,
                'processing_time': processing_time,
                'pipeline_stages': {
                    'frame_extraction_sec': round(frame_extraction_time, 3),
                    'feature_extraction_sec': round(feature_extraction_time, 3),
                    'matching_sec': round(matching_time, 3),
                    'total_sec': round(processing_time, 3),
                    'references_searched': len(references)
                }
            }
            
            # Save result to database
            if save_result and best_match:
                match_record = MatchResult(
                    query_filename=clip_info['filename'],
                    query_duration=clip_info['duration'],
                    matched_video_id=best_match['video_id'],
                    confidence_score=best_match['confidence'],
                    start_timestamp=best_match['start_timestamp'],
                    end_timestamp=best_match['end_timestamp'],
                    match_details=json.dumps(all_matches),
                    processing_time=processing_time
                )
                session.add(match_record)
                session.commit()
                result['match_id'] = match_record.id
            
            return result
            
        finally:
            session.close()
    
    def _match_against_reference(self, clip_features: List[Dict], 
                                  reference: ReferenceVideo,
                                  session, clip_duration: float = 0,
                                  augmented_feature_sets: List[List[Dict]] = None) -> Optional[Dict]:
        """
        Match clip features against a single reference video.
        Uses multiple matching strategies + augmented features for robust detection.
        
        Args:
            clip_features: List of feature dictionaries from the clip (original)
            reference: Reference video database object
            session: Database session
            clip_duration: Duration of the query clip in seconds
            augmented_feature_sets: List of feature sets [original, flipped, cropped]
            
        Returns:
            Match result dictionary or None if no match
        """
        # Get all frames for this reference video
        ref_frames = session.query(VideoFrame).filter(
            VideoFrame.video_id == reference.id
        ).order_by(VideoFrame.timestamp).all()
        
        if not ref_frames:
            return None
        
        # Build hash lookup for reference (include color info for fallback matching)
        ref_hashes_raw = [(f.id, f.timestamp, f.phash, f.dhash, f.whash, f.brightness, f.avg_color_r, f.avg_color_g, f.avg_color_b) for f in ref_frames]
        
        # Pre-compute integer arrays for vectorised matching
        ref_ph, ref_dh, ref_wh, ref_ts, ref_br, ref_cr, ref_cg, ref_cb = _build_int_arrays(ref_hashes_raw)
        n_ref = len(ref_hashes_raw)
        
        # Use augmented features if available, otherwise just original
        all_feature_sets = augmented_feature_sets or [clip_features]
        
        # EARLY REJECTION: Check first 3 frames across ALL augmentations
        strict_early_threshold = 26
        early_good_matches = 0
        
        for i in range(min(3, len(clip_features))):
            best_distance = 999
            for feat_set in all_feature_sets:
                if i >= len(feat_set):
                    continue
                ph_int = hex_to_int(feat_set[i].get('phash', '') or '')
                dh_int = hex_to_int(feat_set[i].get('dhash', '') or '')
                wh_int = hex_to_int(feat_set[i].get('whash', '') or '')
                
                ph_dist = _vectorised_hamming(ph_int, ref_ph)
                dh_dist = _vectorised_hamming(dh_int, ref_dh)
                wh_dist = _vectorised_hamming(wh_int, ref_wh)
                
                avg_dist = (ph_dist.astype(np.float64) + dh_dist + wh_dist) / 3.0
                aug_best = float(np.min(avg_dist))
                if aug_best < best_distance:
                    best_distance = aug_best
            
            if best_distance <= strict_early_threshold:
                early_good_matches += 1
        
        if early_good_matches < 1:
            logger.debug("Early rejection for %s: only %d/3 frames matched", reference.filename, early_good_matches)
            return None
        
        # Full matching with vectorised search (adaptive step based on ref size)
        search_step = 4 if n_ref > 500 else (3 if n_ref > 200 else 2)
        ref_indices = np.arange(0, n_ref, search_step)
        ref_ph_sub = ref_ph[ref_indices]
        ref_dh_sub = ref_dh[ref_indices]
        ref_wh_sub = ref_wh[ref_indices]
        ref_ts_sub = ref_ts[ref_indices]
        ref_br_sub = ref_br[ref_indices]
        ref_cr_sub = ref_cr[ref_indices]
        ref_cg_sub = ref_cg[ref_indices]
        ref_cb_sub = ref_cb[ref_indices]
        
        lenient_threshold = 32
        
        hash_matches = []
        color_matches = []
        all_distances = []
        matched_timestamps = []
        
        for i in range(len(clip_features)):
            clip_brightness = clip_features[i].get('brightness', 0.5)
            clip_r = clip_features[i].get('avg_color_r', 128)
            clip_g = clip_features[i].get('avg_color_g', 128)
            clip_b = clip_features[i].get('avg_color_b', 128)
            
            # Try ALL augmented versions of this frame, keep best hash distance
            best_hash_distance = 999.0
            best_hash_idx = 0
            
            for feat_set in all_feature_sets:
                if i >= len(feat_set):
                    continue
                aug_feat = feat_set[i]
                ph_int = hex_to_int(aug_feat.get('phash', '') or '')
                dh_int = hex_to_int(aug_feat.get('dhash', '') or '')
                wh_int = hex_to_int(aug_feat.get('whash', '') or '')
                
                ph_dist = _vectorised_hamming(ph_int, ref_ph_sub)
                dh_dist = _vectorised_hamming(dh_int, ref_dh_sub)
                wh_dist = _vectorised_hamming(wh_int, ref_wh_sub)
                aug_hash_dist = (ph_dist.astype(np.float64) + dh_dist + wh_dist) / 3.0
                
                aug_best_idx = int(np.argmin(aug_hash_dist))
                aug_best_dist = float(aug_hash_dist[aug_best_idx])
                
                if aug_best_dist < best_hash_distance:
                    best_hash_distance = aug_best_dist
                    best_hash_idx = aug_best_idx
            
            # Color distance (from original features only — flip/crop doesn't change colors much)
            color_diff = (np.abs(clip_r - ref_cr_sub) + 
                         np.abs(clip_g - ref_cg_sub) + 
                         np.abs(clip_b - ref_cb_sub)) / 3.0
            brightness_diff = np.abs(clip_brightness - ref_br_sub) * 255
            color_distance = (color_diff + brightness_diff) / 2.0
            
            best_color_distance = float(color_distance[best_hash_idx])
            best_timestamp = float(ref_ts_sub[best_hash_idx])
            
            all_distances.append(best_hash_distance)
            hash_matches.append(1 if best_hash_distance <= lenient_threshold else 0)
            color_matches.append(1 if best_color_distance < 80 else 0)
            matched_timestamps.append(
                best_timestamp if (best_hash_distance <= lenient_threshold or best_color_distance < 80) else None
            )
        
        # Calculate confidence metrics
        hash_match_ratio = sum(hash_matches) / len(hash_matches)
        color_match_ratio = sum(color_matches) / len(color_matches)
        avg_distance = sum(all_distances) / len(all_distances)
        min_distance = min(all_distances) if all_distances else 999
        distance_score = max(0, 1 - (avg_distance / 64)) * 100
        
        # Combined confidence
        confidence = (
            (hash_match_ratio * 40) +
            (color_match_ratio * 20) +
            (distance_score * 0.40)
        )
        
        if color_match_ratio < 0.3:
            confidence *= 0.8
        
        confidence = min(90, confidence)
        
        # Temporal consistency boost
        temporal_boost = self._calculate_temporal_consistency(
            [1 if d <= lenient_threshold else 0 for d in all_distances], 
            matched_timestamps
        )
        
        if confidence > 50:
            confidence = min(98, confidence + (temporal_boost * 4))
        
        confidence = min(99.9, confidence)
        
        # WEIGHTED REJECTION (replaces strict AND logic)
        # Each check contributes a penalty score; reject only if total penalty is too high
        penalty = 0.0
        
        # Check 1: Hash match ratio (target >= 35%)
        if hash_match_ratio < 0.35:
            penalty += 2.0  # Hard fail
        elif hash_match_ratio < 0.50:
            penalty += 0.5  # Soft penalty
        
        # Check 2: Average distance (target <= 28 bits)
        if avg_distance > 32:
            penalty += 2.0  # Hard fail
        elif avg_distance > 24:
            penalty += 0.5
        
        # Check 3: Color match (target >= 30%)
        if color_match_ratio < 0.20:
            penalty += 2.0  # Hard fail
        elif color_match_ratio < 0.40:
            penalty += 0.5
        
        # Check 4: Confidence (target >= 45%)
        if confidence < 35:
            penalty += 2.0  # Hard fail
        elif confidence < 45:
            penalty += 0.5
        
        # Check 5: Best frame distance (target <= 24 bits)
        if min_distance > 28:
            penalty += 2.0  # Hard fail
        elif min_distance > 20:
            penalty += 0.5
        
        # Check 6: Temporal consistency (target >= 0.3)
        temporal_consistency = self._calculate_temporal_consistency(
            [1 if d <= lenient_threshold else 0 for d in all_distances], 
            matched_timestamps
        )
        if temporal_consistency < 0.20:
            penalty += 1.0
        elif temporal_consistency < 0.40:
            penalty += 0.3
        
        # REJECT if penalty exceeds threshold (2.0 = one hard fail or multiple soft)
        if penalty >= 2.0:
            logger.debug("REJECTED %s: penalty=%.1f (hash_ratio=%.0f%%, avg_dist=%.1f, color=%.0f%%, conf=%.1f%%, min_dist=%.0f, temporal=%.2f)",
                        reference.filename, penalty, hash_match_ratio*100, avg_distance,
                        color_match_ratio*100, confidence, min_distance, temporal_consistency)
            return None
        
        logger.info("ACCEPTED %s - Confidence %.1f%% (penalty=%.1f)", reference.filename, confidence, penalty)
        
        # Estimate timestamp range using robust sliding-window clustering
        start_timestamp, end_timestamp = estimate_timestamp_range(
            matched_timestamps, all_distances, clip_duration,
            len(clip_features), sample_rate=2.0
        )
        
        return {
            'video_id': reference.id,
            'video_title': reference.title or reference.filename,
            'video_filename': reference.filename,
            'confidence': round(max(1, confidence), 2),
            'confidence_label': self._get_confidence_label(confidence),
            'start_timestamp': start_timestamp,
            'end_timestamp': end_timestamp,
            'timestamp_formatted': self._format_timestamp_range(start_timestamp, end_timestamp),
            'match_ratio': round(hash_match_ratio, 3),
            'color_match_ratio': round(color_match_ratio, 3),
            'min_distance': int(min_distance),
            'avg_distance': round(avg_distance, 1),
            'avg_similarity': round((1 - min(avg_distance, 256)/256) * 100, 2),
            'clip_frames_analyzed': len(clip_features),
            'reference_frames_total': n_ref
        }
    
    def _calculate_temporal_consistency(self, match_scores: List[float], 
                                        timestamps: List[Optional[float]]) -> float:
        """
        Calculate temporal consistency bonus.
        
        Checks if matching frames appear in sequential order,
        indicating a true continuous match rather than random similarities.
        
        Args:
            match_scores: List of match scores per frame
            timestamps: List of matched timestamps (or None)
            
        Returns:
            Temporal consistency score (0.0 to 1.0)
        """
        if len(match_scores) < 3:
            return 0.5  # Not enough frames to judge
        
        # Find consecutive matching sequences
        consecutive_count = 0
        max_consecutive = 0
        prev_timestamp = None
        
        for score, ts in zip(match_scores, timestamps):
            if score > 0 and ts is not None:
                if prev_timestamp is not None:
                    # Check if timestamps are roughly sequential
                    time_diff = ts - prev_timestamp
                    if -2 <= time_diff <= 5:  # Allow small jumps
                        consecutive_count += 1
                        max_consecutive = max(max_consecutive, consecutive_count)
                    else:
                        consecutive_count = 0
                prev_timestamp = ts
            else:
                consecutive_count = 0
                prev_timestamp = None
        
        # Score based on longest consecutive sequence
        if max_consecutive >= self.min_consecutive:
            return min(1.0, max_consecutive / len(match_scores))
        
        return 0.0
    
    def _get_confidence_label(self, confidence: float) -> str:
        """Get human-readable confidence label."""
        return get_confidence_label(confidence)
    
    def _format_timestamp_range(self, start: Optional[float], 
                                end: Optional[float]) -> str:
        """Format timestamp range for display."""
        return format_timestamp_range(start, end)
    
    def get_match_history(self, limit: int = 50) -> List[Dict]:
        """
        Get recent match history.
        
        Args:
            limit: Maximum number of results to return
            
        Returns:
            List of match result dictionaries
        """
        session = get_session()
        try:
            results = session.query(MatchResult).order_by(
                MatchResult.queried_at.desc()
            ).limit(limit).all()
            
            return [r.to_dict() for r in results]
        finally:
            session.close()


# Singleton instance
clip_matcher = ClipMatcher()
