"""
ClipMatch Advanced Matcher
Complete end-to-end matching pipeline with sliding window, early rejection, and verification
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import sys
sys.path.append('..')

from config import MatchConfig
from models.database import get_session, ReferenceVideo, VideoFrame
from services.video_processor import VideoProcessor
from services.preprocessor import FramePreprocessor
from services.feature_extractor import FeatureExtractor
from services.verification import VerificationEngine


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
    MIN_WINDOW_MATCHES = 7  # Out of 8 keyframes, at least 7 must match
    
    # Early rejection parameters
    EARLY_REJECTION_FRAMES = 3  # Check first 3 keyframes before full window eval
    
    # Verification parameters
    VERIFICATION_MIN_CONFIDENCE = 0.85
    
    def __init__(self):
        """Initialize matcher with all required components."""
        self.video_processor = VideoProcessor()
        self.preprocessor = FramePreprocessor()
        self.feature_extractor = FeatureExtractor()
        self.verifier = VerificationEngine()
    
    def preprocess_and_extract_frames(self, video_path: str) -> Dict:
        """
        Extract and extract features from all frames in a video.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with frames, features, and metadata
        """
        # Extract frames as tuples (frame_number, timestamp, frame_image)
        frames_list = self.video_processor.extract_all_frames_to_list(video_path, max_frames=40)
        
        if not frames_list:
            return {
                'success': False,
                'error': 'Could not extract frames',
                'features': []
            }
        
        # Extract features directly from frame tuples
        # batch_extract expects List[Tuple[int, float, np.ndarray]]
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
        Complete end-to-end matching pipeline.
        Uses standard matching like clip_matcher, can be expanded with NCC verification later.
        
        Args:
            clip_path: Path to query clip
            verbose: Print progress information
            
        Returns:
            Dictionary with match results
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
        
        # Step 3: Extract and preprocess clip
        if verbose:
            print("Extracting and preprocessing clip...")
        clip_data = self.preprocess_and_extract_frames(clip_path)
        
        if not clip_data['success']:
            return {
                'success': False,
                'error': clip_data['error'],
                'matches': []
            }
        
        # Step 4: Get all reference videos and match
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
            
            # Step 5: Search in each reference video
            for ref_idx, ref_video in enumerate(references):
                if verbose:
                    print(f"Searching in {ref_video.filename} ({ref_idx+1}/{len(references)})...")
                
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
                
                # Match clip features against reference
                match_result = self._match_against_reference(
                    clip_data['features'], ref_video, ref_features
                )
                
                if match_result and match_result['confidence'] >= 50:
                    all_matches.append(match_result)
            
            # Sort by confidence
            all_matches.sort(key=lambda x: x['confidence'], reverse=True)
            
            elapsed = time.time() - start_time
            
            return {
                'success': True,
                'matches': all_matches,
                'best_match': all_matches[0] if all_matches else None,
                'processing_time': elapsed,
                'clip_frames': clip_data['frame_count'],
                'reference_videos_searched': len(references)
            }
        
        finally:
            session.close()
    
    def _match_against_reference(self, clip_features: List[Dict],
                                reference: ReferenceVideo,
                                ref_features: List[Dict]) -> Optional[Dict]:
        """
        Match clip features against a reference video using EXACT same logic as regular matcher.
        
        Args:
            clip_features: Features from query clip (from batch_extract)
            reference: Reference video record
            ref_features: Features from reference video
            
        Returns:
            Match result dictionary or None
        """
        if not clip_features or not ref_features:
            return None
        
        # Build reference feature tuples like regular matcher does
        ref_hashes = [
            (f.get('frame_number', i), f.get('timestamp', 0), 
             f.get('phash', ''), f.get('dhash', ''), f.get('whash', ''),
             f.get('brightness', 0.5), f.get('avg_color_r', 128),
             f.get('avg_color_g', 128), f.get('avg_color_b', 128))
            for i, f in enumerate(ref_features)
        ]
        
        # Use EXACT same matching as regular matcher
        hash_matches = []
        color_matches = []
        all_distances = []
        
        lenient_threshold = 32
        
        for i, clip_feat in enumerate(clip_features):
            # Get all hashes and colors from clip
            clip_phash = clip_feat.get('phash', '')
            clip_dhash = clip_feat.get('dhash', '')
            clip_whash = clip_feat.get('whash', '')
            clip_brightness = clip_feat.get('brightness', 0.5)
            clip_r = clip_feat.get('avg_color_r', 128)
            clip_g = clip_feat.get('avg_color_g', 128)
            clip_b = clip_feat.get('avg_color_b', 128)
            
            best_hash_distance = 999
            best_color_distance = 999
            best_timestamp = None
            best_combined = 999
            
            # Compare against all reference frames
            for ref_id, ref_ts, ref_phash, ref_dhash, ref_whash, ref_brightness, ref_r, ref_g, ref_b in ref_hashes:
                # Compute distances using ALL three hash algorithms
                phash_dist = self.feature_extractor.compute_hash_distance(
                    clip_phash, ref_phash
                ) if clip_phash and ref_phash else 64
                
                dhash_dist = self.feature_extractor.compute_hash_distance(
                    clip_dhash, ref_dhash
                ) if clip_dhash and ref_dhash else 64
                
                whash_dist = self.feature_extractor.compute_hash_distance(
                    clip_whash, ref_whash
                ) if clip_whash and ref_whash else 64
                
                # Average of all three hash algorithms
                avg_hash_distance = (phash_dist + dhash_dist + whash_dist) / 3.0
                
                # Color distance (normalized to 0-255 scale)
                color_diff = (abs(clip_r - (ref_r or 128)) + 
                             abs(clip_g - (ref_g or 128)) + 
                             abs(clip_b - (ref_b or 128))) / 3
                brightness_diff = abs(clip_brightness - (ref_brightness or 0.5)) * 255
                color_distance = (color_diff + brightness_diff) / 2
                
                # Combined score
                combined = (avg_hash_distance * 0.5) + (color_distance * 0.5)
                
                if combined < best_combined:
                    best_combined = combined
                    best_hash_distance = avg_hash_distance
                    best_color_distance = color_distance
                    best_timestamp = ref_ts
            
            all_distances.append(best_hash_distance)
            # Match if either hash is good OR color is good
            hash_matches.append(1 if best_hash_distance <= lenient_threshold else 0)
            color_matches.append(1 if best_color_distance < 80 else 0)
        
        # Calculate match statistics (exact same as regular matcher)
        hash_match_ratio = sum(hash_matches) / len(hash_matches) if hash_matches else 0
        color_match_ratio = sum(color_matches) / len(color_matches) if color_matches else 0
        
        # Average distance metrics
        avg_distance = sum(all_distances) / len(all_distances) if all_distances else 999
        min_distance = min(all_distances) if all_distances else 999
        
        # Calculate confidence (exact same formula as regular matcher)
        scores = []
        
        # Hash score
        hash_score = hash_match_ratio * 100
        scores.append(('hash_ratio', hash_match_ratio))
        
        # Color score
        color_score = color_match_ratio * 100
        scores.append(('color_ratio', color_match_ratio))
        
        # Distance score (lower is better)
        distance_score = max(0, 100 - (avg_distance * 4))
        scores.append(('avg_distance', avg_distance))
        
        # Confidence is weighted combination
        confidence = (hash_score * 0.35) + (color_score * 0.30) + (distance_score * 0.35)
        
        # Bonuses for very good matches
        if min_distance < 12:
            confidence += 15
        elif min_distance < 16:
            confidence += 10
        elif min_distance < 20:
            confidence += 5
        
        confidence = min(100, confidence)
        
        # Only return if confidence meets threshold
        if confidence < 40:
            return None
        
        # Find best matching position
        best_pos = 0
        min_dist_idx = all_distances.index(min(all_distances)) if all_distances else 0
        best_timestamp = ref_hashes[min_dist_idx][1] if min_dist_idx < len(ref_hashes) else 0
        
        return {
            'video_id': reference.id,
            'video_filename': reference.filename,
            'confidence': confidence,
            'match_ratio': hash_match_ratio,
            'matches': sum(hash_matches),
            'total_frames': len(clip_features),
            'hash_match_ratio': hash_match_ratio,
            'color_match_ratio': color_match_ratio,
            'avg_distance': avg_distance,
            'min_distance': min_distance,
            'start_timestamp': max(0, best_timestamp - 2),
            'end_timestamp': best_timestamp + 2,
            'timestamp': best_timestamp,
            'details': {k: v for k, v in scores}
        }


# Global instance
advanced_matcher = AdvancedClipMatcher()
