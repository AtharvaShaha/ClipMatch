"""
ClipMatch Advanced Matcher
Complete end-to-end matching pipeline with sliding window, early rejection, and verification
"""

import logging
logger = logging.getLogger('clipmatch.advanced')

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import sys
import os
sys.path.append('..')

from config import MatchConfig
from models.database import get_session, ReferenceVideo, VideoFrame
from services.video_processor import VideoProcessor
from services.preprocessor import FramePreprocessor
from services.feature_extractor import FeatureExtractor, hex_to_int, fast_hamming
from services.verification import VerificationEngine
from services.utils import format_timestamp_range, estimate_timestamp_range
from services.matcher import _build_int_arrays, _vectorised_hamming, _compute_augmented_features


class AdvancedClipMatcher:
    """
    Complete video matching system implementing all techniques:
    - Frame preprocessing (grayscale, resize, histogram equalization)
    - Keyframe selection (distinctive frames only)
    - Dual hashing (pHash + dHash)
    - Hash indexing (offline)
    - Sliding window with early rejection
    - Anchor-based sequential matching
    - NCC/SSIM verification
    - Confidence scoring
    """
    
    # Sliding window parameters
    MIN_WINDOW_MATCHES = 4  # Lowered from 7 to allow more candidates to pass to verification
    
    # Early rejection parameters
    EARLY_REJECTION_FRAMES = 2  # Lowered from 3 to be less restrictive
    
    # Verification parameters
    VERIFICATION_MIN_CONFIDENCE = 0.75 # Lowered from 0.85
    
    def __init__(self):
        """Initialize matcher with all required components."""
        self.video_processor = VideoProcessor()
        self.preprocessor = FramePreprocessor()
        self.feature_extractor = FeatureExtractor()
        self.verifier = VerificationEngine()
    
    def select_keyframes(self, frames: List[np.ndarray], num_keyframes: int = 8) -> List[int]:
        """
        Select the most distinctive keyframes from a video clip.
        Uses frame difference to identify visually distinctive frames.
        
        Args:
            frames: List of raw frame images (BGR numpy arrays)
            num_keyframes: Number of keyframes to select
            
        Returns:
            List of indices of selected keyframes
        """
        if len(frames) <= num_keyframes:
            return list(range(len(frames)))
        
        # Calculate frame differences to find distinctive frames
        frame_differences = []
        for i in range(1, len(frames)):
            # Compare consecutive frames
            diff = self.preprocessor.get_frame_difference_score(frames[i-1], frames[i])
            frame_differences.append((i, diff))
        
        # Add first frame as always important
        keyframes = [0]
        
        # Sort by difference and pick most different frames
        frame_differences.sort(key=lambda x: x[1], reverse=True)
        
        # Add top different frames
        for idx, _ in frame_differences[:num_keyframes - 1]:
            keyframes.append(idx)
        
        # Sort keyframe indices
        keyframes = sorted(list(set(keyframes)))[:num_keyframes]
        return keyframes
    
    def preprocess_and_extract_frames(self, video_path: str, use_keyframes: bool = True) -> Dict:
        """
        Extract and extract features from video, optionally using keyframes only.
        
        Args:
            video_path: Path to video file
            use_keyframes: If True, extract only keyframes; else extract all frames
            
        Returns:
            Dictionary with frames, features, and metadata
        """
        # Extract raw frames first (use query_mode for relaxed crop)
        frames_list = self.video_processor.extract_all_frames_to_list(video_path, max_frames=40, query_mode=True)
        
        if not frames_list:
            return {
                'success': False,
                'error': 'Could not extract frames',
                'features': []
            }
        
        # Extract raw frame images (third element of tuple)
        raw_frames = [f[2] for f in frames_list]
        
        # Select keyframes if enabled
        if use_keyframes and len(raw_frames) > 8:
            keyframe_indices = self.select_keyframes(raw_frames, num_keyframes=8)
            frames_list = [frames_list[i] for i in keyframe_indices]
        
        # Extract features from frames
        features = self.feature_extractor.batch_extract(frames_list)
        
        return {
            'success': True,
            'features': features,
            'frame_count': len(features)
        }
    
    def select_query_keyframes(self, clip_features: List[Dict],
                               num_keyframes: int = 8) -> Dict:
        """
        Select keyframes from query clip features.
        
        Args:
            clip_features: Features extracted from clip frames
            num_keyframes: Number of keyframes to select
            
        Returns:
            Dictionary with keyframe features
        """
        # For simplified advanced matcher, use evenly distributed frames
        if len(clip_features) <= num_keyframes:
            keyframe_indices = list(range(len(clip_features)))
        else:
            step = len(clip_features) // num_keyframes
            keyframe_indices = [i * step for i in range(num_keyframes)]
        
        keyframe_features = [clip_features[i] for i in keyframe_indices]
        
        return {
            'keyframe_indices': keyframe_indices,
            'keyframe_features': keyframe_features,
            'num_keyframes': len(keyframe_features)
        }
    
    def early_rejection_filter(self, clip_hashes: List[Dict],
                              ref_video_features: List[Dict],
                              num_check: int = 3) -> List[int]:
        """
        Quickly filter candidate positions using early rejection.
        
        Only check first N keyframes before evaluating full window.
        This eliminates 70-80% of non-matching positions very quickly.
        
        Args:
            clip_hashes: Hashes of clip's keyframes
            ref_video_features: All reference video frame features
            num_check: Number of keyframes to check for early rejection
            
        Returns:
            List of candidate window starting positions
        """
        candidates = []
        num_check = min(num_check, len(clip_hashes))
        
        # For each position in reference video
        for start_pos in range(len(ref_video_features) - len(clip_hashes) + 1):
            # Check first N keyframes (early rejection)
            skip = False
            
            for check_idx in range(num_check):
                ref_pos = start_pos + check_idx
                if ref_pos >= len(ref_video_features):
                    skip = True
                    break
                
                # Use combined hash distance
                distance = self.feature_extractor.combined_hash_distance(
                    clip_hashes[check_idx],
                    ref_video_features[ref_pos]
                )
                
                # If first frame doesn't match, skip entire window
                if distance > 0.16:  # Higher threshold for early rejection
                    skip = True
                    break
            
            if not skip:
                candidates.append(start_pos)
        
        return candidates
    
    def anchor_sequential_matching(self, clip_hashes: List[Dict],
                                   ref_video_features: List[Dict],
                                   anchor_position: int,
                                   timestamp_tolerance: int = 1) -> Dict:
        """
        Verify sequential matching from an anchor position.
        
        An anchor is the first keyframe position that matches.
        This method verifies that remaining keyframes follow in sequence.
        
        Args:
            clip_hashes: Clip's keyframe hashes
            ref_video_features: Reference video frame features
            anchor_position: Starting position in reference
            timestamp_tolerance: Frames of timing variance to allow
            
        Returns:
            Dictionary with match score and details
        """
        matches = 0
        match_positions = []
        distances = []
        
        # Verify each keyframe from anchor
        for clip_idx, clip_hash in enumerate(clip_hashes):
            ref_idx = anchor_position + clip_idx
            
            # Allow timing tolerance
            best_distance = float('inf')
            best_ref_idx = ref_idx
            
            for tolerance_offset in range(-timestamp_tolerance, timestamp_tolerance + 1):
                test_ref_idx = ref_idx + tolerance_offset
                
                if 0 <= test_ref_idx < len(ref_video_features):
                    distance = self.feature_extractor.combined_hash_distance(
                        clip_hash,
                        ref_video_features[test_ref_idx]
                    )
                    
                    if distance < best_distance:
                        best_distance = distance
                        best_ref_idx = test_ref_idx
            
            # Check if this frame matches
            threshold = 0.156  # 10 bits out of 64
            if best_distance <= threshold:
                matches += 1
                match_positions.append(best_ref_idx)
                distances.append(best_distance)
            else:
                match_positions.append(None)
                distances.append(best_distance)
        
        # Calculate score: how many out of N keyframes matched?
        match_score = matches / len(clip_hashes) if len(clip_hashes) > 0 else 0
        
        return {
            'anchor_position': anchor_position,
            'matches': matches,
            'total': len(clip_hashes),
            'match_score': match_score,
            'match_positions': match_positions,
            'distances': distances
        }
    
    def find_candidates_in_reference(self, clip_data: Dict,
                                    ref_video: ReferenceVideo,
                                    ref_features: List[Dict]) -> List[Dict]:
        """
        Find all candidate match positions for a clip in a reference video.
        
        Process:
        1. Early rejection filtering (fast, eliminates 70-80%)
        2. Anchor-based sequential matching on survivors
        3. Score candidates
        
        Args:
            clip_data: Clip's keyframe data
            ref_video: Reference video record
            ref_features: Reference video's frame features
            
        Returns:
            List of candidate matches, scored and sorted by confidence
        """
        clip_hashes = clip_data['keyframe_hashes']
        
        # Step 1: Early rejection
        candidate_positions = self.early_rejection_filter(
            clip_hashes, ref_features, 
            num_check=self.EARLY_REJECTION_FRAMES
        )
        
        # Step 2: Anchor-based sequential matching
        scored_candidates = []
        
        for anchor_pos in candidate_positions:
            match_result = self.anchor_sequential_matching(
                clip_hashes, ref_features, anchor_pos
            )
            
            # Filter by minimum matches
            if match_result['matches'] >= self.MIN_WINDOW_MATCHES:
                match_result['video_id'] = ref_video.id
                match_result['video_filename'] = ref_video.filename
                scored_candidates.append(match_result)
        
        # Sort by match score
        scored_candidates.sort(key=lambda x: x['match_score'], reverse=True)
        
        return scored_candidates
    
    def verify_candidate_with_ncc_ssim(self, clip_raw_frames: List[np.ndarray],
                                       reference_raw_frames: List[np.ndarray],
                                       candidate_start_frame: int,
                                       num_frames: Optional[int] = None) -> Dict:
        """
        Verify a candidate match using NCC and SSIM on raw frames.
        
        This is the final verification step using pixel-level analysis.
        
        Args:
            clip_raw_frames: Raw frames from clip
            reference_raw_frames: Raw frames from reference video
            candidate_start_frame: Frame index in reference where match starts
            num_frames: Number of frames to verify (default: all clip frames)
            
        Returns:
            Dictionary with NCC/SSIM verification scores
        """
        if num_frames is None:
            num_frames = len(clip_raw_frames)
        
        # Extract the candidate segment from reference
        candidate_end = candidate_start_frame + num_frames
        if candidate_end > len(reference_raw_frames):
            candidate_end = len(reference_raw_frames)
            num_frames = candidate_end - candidate_start_frame
        
        reference_segment = reference_raw_frames[candidate_start_frame:candidate_end]
        
        # Preprocess both segments for reliable comparison
        clip_preprocessed = self.preprocessor.batch_preprocess_frames(clip_raw_frames[:num_frames])
        ref_preprocessed = self.preprocessor.batch_preprocess_frames(reference_segment)
        
        # Verify sequence
        verification = self.verifier.verify_frame_sequence(
            clip_preprocessed, ref_preprocessed
        )
        
        return verification
    
    def compute_final_confidence(self, hash_match_score: float,
                               ncc_score: float,
                               ssim_score: float,
                               hash_weight: float = 0.4,
                               ncc_weight: float = 0.35,
                               ssim_weight: float = 0.25) -> float:
        """
        Combine all evidence into final confidence score (0-100).
        
        Args:
            hash_match_score: From sliding window matching (0-1)
            ncc_score: Average NCC score (0-1)
            ssim_score: Average SSIM score (0-1)
            hash_weight: Weight for hashing evidence
            ncc_weight: Weight for NCC evidence
            ssim_weight: Weight for SSIM evidence
            
        Returns:
            Final confidence score (0-100)
        """
        combined = (hash_weight * hash_match_score +
                   ncc_weight * ncc_score +
                   ssim_weight * ssim_score)
        
        # Convert to 0-100 percentage
        confidence = min(combined * 100, 100)
        
        return confidence
    
    def match_clip(self, clip_path: str, 
                   verbose: bool = False) -> Dict:
        """
        Complete end-to-end matching pipeline with NCC-based verification.
        
        Process:
        1. Hash-based filtering to find candidate positions (fast)
        2. NCC (Normalized Cross-Correlation) verification on pixel-level data
        3. SSIM (Structural Similarity) verification as secondary metric
        4. Combined scoring with NCC having highest weight (40%) for compression robustness
        
        NCC is particularly effective for:
        - Heavily compressed videos (H.264, H.265)
        - Watermarked content
        - Brightness/contrast adjusted clips
        - Low-quality/transcoded clips
        
        Args:
            clip_path: Path to query clip
            verbose: Print progress information
            
        Returns:
            Dictionary with match results including NCC verification scores
        """
        import time
        start_time = time.time()
        
        # Step 1: Validate clip
        is_valid, message = self.video_processor.validate_query_clip(clip_path)
        if not is_valid:
            return {
                'success': False,
                'error': message,
                'matches': []
            }
        
        # Step 2: Get clip info
        clip_info = self.video_processor.get_video_info(clip_path)
        clip_duration = clip_info.get('duration', 0) if clip_info else 0
        
        # Step 3: Extract and preprocess clip
        t_extract = time.time()
        if verbose:
            logger.debug("Extracting and preprocessing clip...")
        clip_data = self.preprocess_and_extract_frames(clip_path)
        
        if not clip_data['success']:
            return {
                'success': False,
                'error': clip_data['error'],
                'matches': []
            }
        extraction_time = time.time() - t_extract
        
        # Cache clip raw frames for NCC verification AND augmentation
        clip_raw_frames_cache = None
        clip_frames_for_aug = None
        try:
            clip_raw_tuples = self.video_processor.extract_all_frames_to_list(
                clip_path, max_frames=len(clip_data['features']), query_mode=True
            )
            if clip_raw_tuples:
                clip_raw_frames_cache = [t[2] for t in clip_raw_tuples]
                clip_frames_for_aug = clip_raw_tuples
        except Exception:
            clip_raw_frames_cache = None
            clip_frames_for_aug = None
        
        # Compute augmented features (original + refcrop + flip + nocrop)
        if clip_frames_for_aug:
            augmented_feature_sets = _compute_augmented_features(
                self.feature_extractor, clip_frames_for_aug,
                video_processor=self.video_processor, clip_path=clip_path
            )
        else:
            augmented_feature_sets = [clip_data['features']]
        
        # Step 4: Get all reference videos and match
        t_match = time.time()
        session = get_session()
        try:
            references = session.query(ReferenceVideo).filter(
                ReferenceVideo.status == 'indexed'
            ).all()
            
            if not references:
                return {
                    'success': False,
                    'error': 'No reference videos indexed',
                    'matches': []
                }
            
            all_matches = []
            ncc_verification_time = 0
            
            # Step 5: Search in each reference video
            for ref_idx, ref_video in enumerate(references):
                if verbose:
                    logger.debug("Searching in %s (%d/%d)...", ref_video.filename, ref_idx+1, len(references))
                
                # Get reference video features from database
                ref_frames_query = session.query(VideoFrame).filter(
                    VideoFrame.video_id == ref_video.id
                ).order_by(VideoFrame.frame_number).all()
                
                if not ref_frames_query:
                    continue
                
                # Build reference feature list
                ref_features = [
                    {
                        'phash': f.phash,
                        'dhash': f.dhash,
                        'whash': getattr(f, 'whash', f.phash),  # Fallback if whash not available
                        'brightness': getattr(f, 'brightness', 0.5),
                        'avg_color_r': getattr(f, 'avg_color_r', 128),
                        'avg_color_g': getattr(f, 'avg_color_g', 128),
                        'avg_color_b': getattr(f, 'avg_color_b', 128),
                        'timestamp': f.timestamp,
                        'frame_number': f.frame_number
                    }
                    for f in ref_frames_query
                ]
                
                # Match clip features against reference (pass augmented features + cached frames)
                match_result = self._match_against_reference(
                    clip_data['features'], ref_video, ref_features, clip_duration, clip_path,
                    clip_raw_frames_cache=clip_raw_frames_cache,
                    augmented_feature_sets=augmented_feature_sets
                )
                
                if match_result and match_result['confidence'] >= 45:
                    all_matches.append(match_result)
                    
                    # OPTIMIZATION: If we find a near-perfect match, stop searching
                    # A 95%+ confidence match is extremely reliable and likely the correct source
                    if match_result['confidence'] >= 95:
                        logger.info("Found near-perfect match (%.1f%%) in %s - stopping search", match_result['confidence'], ref_video.filename)
                        break
            
            # Sort by confidence
            all_matches.sort(key=lambda x: x['confidence'], reverse=True)
            
            matching_time = time.time() - t_match
            elapsed = time.time() - start_time
            
            # Log timing breakdown
            logger.info("[Timing] Extraction: %.2fs | Matching: %.2fs (%d refs) | Total: %.2fs",
                       extraction_time, matching_time, len(references), elapsed)
            
            return {
                'success': True,
                'matches': all_matches,
                'best_match': all_matches[0] if all_matches else None,
                'query': {
                    'filename': os.path.basename(clip_path),
                    'duration': clip_duration,
                    'frames_analyzed': clip_data['frame_count']
                },
                'processing_time': elapsed,
                'pipeline_stages': {
                    'frame_extraction_sec': round(extraction_time, 3),
                    'matching_sec': round(matching_time, 3),
                    'total_sec': round(elapsed, 3),
                    'references_searched': len(references)
                },
                'clip_frames': clip_data['frame_count'],
                'reference_videos_searched': len(references)
            }
        
        finally:
            session.close()
    
    def _extract_reference_frames(self, ref_video: ReferenceVideo,
                                  indices: List[int]) -> Optional[List[np.ndarray]]:
        """
        Extract raw frames from reference video at specified indices.
        
        Args:
            ref_video: Reference video record
            indices: Frame indices to extract
            
        Returns:
            List of raw frames or None if extraction fails
        """
        try:
            # Get filepath from database
            filepath = ref_video.filepath
            if not filepath or not os.path.exists(filepath):
                return None
            
            # Extract frames at specified indices
            frames = []
            for frame_num in sorted(set(indices)):
                frame = self.video_processor.extract_frame_at_index(filepath, frame_num)
                if frame is not None:
                    frames.append(frame)
                else:
                    frames.append(None)
            return frames
        except Exception as e:
            if verbose := getattr(self, '_verbose', False):
                logger.warning("Could not extract reference frames: %s", e)
            return None
    
    def _match_against_reference(self, clip_features: List[Dict], reference: ReferenceVideo, ref_features: List[Dict], clip_duration: float = 0, clip_path: str = '', clip_raw_frames_cache=None, augmented_feature_sets=None) -> Optional[Dict]:
        """
        Match clip features against a reference video using vectorised hashing,
        augmented features (flip/crop), and optional NCC verification.
        
        Args:
            clip_features: Features from query clip (original)
            reference: Reference video record
            ref_features: Features from reference video
            clip_duration: Duration of the query clip in seconds
            clip_path: Path to clip file
            clip_raw_frames_cache: Pre-extracted raw frames (avoids re-reading video)
            augmented_feature_sets: List of feature sets [original, flipped, cropped]
            
        Returns:
            Match result dictionary or None
        """
        if not clip_features or not ref_features:
            return None
        
        # Build reference tuples and pre-compute integer arrays
        ref_hashes_raw = [
            (f.get('frame_number', i), f.get('timestamp', 0), 
             f.get('phash', ''), f.get('dhash', ''), f.get('whash', ''),
             f.get('brightness', 0.5), f.get('avg_color_r', 128),
             f.get('avg_color_g', 128), f.get('avg_color_b', 128))
            for i, f in enumerate(ref_features)
        ]
        
        ref_ph, ref_dh, ref_wh, ref_ts, ref_br, ref_cr, ref_cg, ref_cb = _build_int_arrays(ref_hashes_raw)
        n_ref = len(ref_hashes_raw)
        
        # Use augmented features if available
        all_feature_sets = augmented_feature_sets or [clip_features]
        
        # VECTORISED EARLY REJECTION (tries all augmentations)
        strict_threshold = 26
        early_reject_matches = 0
        
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
                
                avg_distance = (ph_dist.astype(np.float64) + dh_dist + wh_dist) / 3.0
                aug_best = float(np.min(avg_distance))
                if aug_best < best_distance:
                    best_distance = aug_best
            
            if best_distance <= strict_threshold:
                early_reject_matches += 1
        
        if early_reject_matches < 1:
            logger.debug("Early rejection for %s: only %d/3 frames matched", reference.filename, early_reject_matches)
            return None
        
        # VECTORISED FULL MATCHING (adaptive step based on ref size)
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
        
        lenient_threshold = 20
        
        hash_matches = []
        color_matches = []
        all_distances = []
        matched_timestamps = []
        matched_frame_indices = []
        
        for i in range(len(clip_features)):
            clip_brightness = clip_features[i].get('brightness', 0.5)
            clip_r = clip_features[i].get('avg_color_r', 128)
            clip_g = clip_features[i].get('avg_color_g', 128)
            clip_b = clip_features[i].get('avg_color_b', 128)
            
            # Try ALL augmented versions, keep best hash distance
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
            
            # Color distance (from original features)
            color_diff = (np.abs(clip_r - ref_cr_sub) + 
                         np.abs(clip_g - ref_cg_sub) + 
                         np.abs(clip_b - ref_cb_sub)) / 3.0
            brightness_diff = np.abs(clip_brightness - ref_br_sub) * 255
            color_distance = (color_diff + brightness_diff) / 2.0
            
            best_color_distance = float(color_distance[best_hash_idx])
            best_timestamp = float(ref_ts_sub[best_hash_idx])
            best_ref_idx = int(ref_indices[best_hash_idx])
            
            all_distances.append(best_hash_distance)
            is_hash_match = best_hash_distance <= lenient_threshold
            hash_matches.append(1 if is_hash_match else 0)
            color_matches.append(1 if best_color_distance < 80 else 0)
            
            if is_hash_match and best_timestamp is not None:
                matched_timestamps.append(best_timestamp)
                matched_frame_indices.append(best_ref_idx)
            else:
                matched_timestamps.append(None)
                matched_frame_indices.append(None)
        
        # Calculate match statistics
        hash_match_ratio = sum(hash_matches) / len(hash_matches) if hash_matches else 0
        color_match_ratio = sum(color_matches) / len(color_matches) if color_matches else 0
        avg_distance = sum(all_distances) / len(all_distances) if all_distances else 999
        min_distance = min(all_distances) if all_distances else 999
        
        # NCC verification — only for borderline matches (skip if hash is already strong)
        ncc_verification = None
        ncc_avg = 0
        ssim_avg = 0
        
        # Skip NCC if hash confidence is already high (saves significant time)
        hash_score = hash_match_ratio * 100
        distance_score = max(0, (1 - avg_distance / 64) * 100)
        estimated_confidence = hash_score * 0.4 + distance_score * 0.35
        
        if estimated_confidence < 85 and len(matched_frame_indices) > 0 and any(idx is not None for idx in matched_frame_indices):
            try:
                valid_indices = [idx for idx in matched_frame_indices if idx is not None]
                
                if valid_indices:
                    ref_raw_frames = self._extract_reference_frames(reference, valid_indices)
                    
                    # Use cached clip frames instead of re-reading video
                    clip_raw_frames = clip_raw_frames_cache
                    if clip_raw_frames is None:
                        try:
                            clip_raw_tuples = self.video_processor.extract_all_frames_to_list(
                                clip_path, max_frames=len(clip_features), query_mode=True
                            ) if clip_path else None
                            clip_raw_frames = [t[2] for t in clip_raw_tuples] if clip_raw_tuples else None
                        except Exception:
                            clip_raw_frames = None
                    
                    if ref_raw_frames and clip_raw_frames and len([f for f in ref_raw_frames if f is not None]) > 0:
                        ncc_verification = self.verify_candidate_with_ncc_ssim(
                            clip_raw_frames, ref_raw_frames, 0, len(clip_raw_frames)
                        )
                        
                        if ncc_verification and ncc_verification.get('success'):
                            ncc_avg = ncc_verification.get('avg_ncc', 0)
                            ssim_avg = ncc_verification.get('avg_ssim', 0)
            except Exception:
                pass
        
        # Calculate confidence with weighted combination
        color_score = color_match_ratio * 100
        ncc_score = ncc_avg * 100 if ncc_avg > 0 else 0
        ssim_score = ssim_avg * 100 if ssim_avg > 0 else 0
        
        if ncc_verification and ncc_verification.get('success'):
            confidence = (
                ncc_score * 0.40 +
                hash_score * 0.20 +
                ssim_score * 0.15 +
                color_score * 0.15 +
                distance_score * 0.10
            )
            if min_distance < 12:
                confidence += 3
            confidence = min(98, confidence)
        else:
            confidence = (hash_score * 0.40) + (color_score * 0.25) + (distance_score * 0.35)
            if min_distance < 12:
                confidence += 5
            elif min_distance < 16:
                confidence += 3
            confidence = min(95, confidence)
        
        # WEIGHTED REJECTION (replaces strict AND logic)
        penalty = 0.0
        
        if hash_match_ratio < 0.35:
            penalty += 2.0
        elif hash_match_ratio < 0.50:
            penalty += 0.5
        
        if avg_distance > 32:
            penalty += 2.0
        elif avg_distance > 24:
            penalty += 0.5
        
        if min_distance > 28:
            penalty += 2.0
        elif min_distance > 20:
            penalty += 0.5
        
        if confidence < 35:
            penalty += 2.0
        elif confidence < 45:
            penalty += 0.5
        
        if color_match_ratio < 0.20:
            penalty += 2.0
        elif color_match_ratio < 0.40:
            penalty += 0.5
        
        if penalty >= 2.0:
            logger.debug("REJECTED %s: penalty=%.1f (hash=%.0f%%, dist=%.1f, color=%.0f%%, conf=%.1f%%)",
                        reference.filename, penalty, hash_match_ratio*100, avg_distance,
                        color_match_ratio*100, confidence)
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
            'match_ratio': hash_match_ratio,
            'matches': sum(hash_matches),
            'total_frames': len(clip_features),
            'hash_match_ratio': hash_match_ratio,
            'color_match_ratio': color_match_ratio,
            'avg_distance': avg_distance,
            'min_distance': min_distance,
            'ncc_score': round(ncc_avg, 3) if ncc_avg > 0 else None,
            'ssim_score': round(ssim_avg, 3) if ssim_avg > 0 else None,
            'start_timestamp': start_timestamp,
            'end_timestamp': end_timestamp,
            'timestamp_formatted': self._format_timestamp_range(start_timestamp, end_timestamp),
            'avg_similarity': round((1 - min(avg_distance, 256)/256) * 100, 2),
            'verification_method': 'NCC+SSIM' if ncc_verification and ncc_verification.get('success') else 'Hash-based',
            'details': {
                'hash_ratio': hash_match_ratio,
                'color_ratio': color_match_ratio,
                'avg_distance': avg_distance,
                'ncc': round(ncc_avg, 3) if ncc_avg > 0 else None,
                'ssim': round(ssim_avg, 3) if ssim_avg > 0 else None
            }
        }
    
    def _format_timestamp_range(self, start: Optional[float], 
                                end: Optional[float]) -> str:
        """Format timestamp range for display."""
        return format_timestamp_range(start, end)


# Global instance
advanced_matcher = AdvancedClipMatcher()
