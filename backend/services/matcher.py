"""
ClipMatch Matcher
Core matching engine for comparing query clips against reference videos
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import json
import sys
sys.path.append('..')
from config import MatchConfig, FeatureConfig
from models.database import get_session, ReferenceVideo, VideoFrame, MatchResult
from services.video_processor import VideoProcessor
from services.feature_extractor import FeatureExtractor


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
        
        # Extract frames from the clip
        clip_frames = self.video_processor.extract_all_frames_to_list(clip_path)
        
        if not clip_frames:
            return {
                'success': False,
                'error': 'Could not extract frames from clip',
                'matches': []
            }
        
        # Extract features from clip frames
        clip_features = self.feature_extractor.batch_extract(clip_frames)
        
        # Get all reference videos
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
            
            for ref_video in references:
                match_result = self._match_against_reference(
                    clip_features, ref_video, session
                )
                if match_result and match_result['confidence'] >= MatchConfig.MIN_CONFIDENCE_THRESHOLD:
                    all_matches.append(match_result)
            
            # Sort by confidence (highest first)
            all_matches.sort(key=lambda x: x['confidence'], reverse=True)
            
            processing_time = time.time() - start_time
            
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
                'processing_time': processing_time
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
                                  session) -> Optional[Dict]:
        """
        Match clip features against a single reference video.
        Uses multiple matching strategies for robust detection.
        
        Args:
            clip_features: List of feature dictionaries from the clip
            reference: Reference video database object
            session: Database session
            
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
        ref_hashes = [(f.id, f.timestamp, f.phash, f.brightness, f.avg_color_r, f.avg_color_g, f.avg_color_b) for f in ref_frames]
        
        # Strategy 1: Hash-based matching with very lenient threshold
        hash_matches = []
        color_matches = []
        all_distances = []
        matched_timestamps = []
        
        # Use a VERY lenient threshold for initial detection (128 = 50% bit difference allowed)
        lenient_threshold = 128
        
        for i, clip_feat in enumerate(clip_features):
            clip_hash = clip_feat['phash']
            clip_brightness = clip_feat.get('brightness', 0.5)
            clip_r = clip_feat.get('avg_color_r', 128)
            clip_g = clip_feat.get('avg_color_g', 128)
            clip_b = clip_feat.get('avg_color_b', 128)
            
            best_hash_distance = 999
            best_color_distance = 999
            best_timestamp = None
            best_combined = 999
            
            for ref_id, ref_ts, ref_hash, ref_brightness, ref_r, ref_g, ref_b in ref_hashes:
                # Hash distance
                hash_distance = self.feature_extractor.compute_hash_distance(
                    clip_hash, ref_hash
                )
                
                # Color distance (normalized to 0-255 scale)
                color_diff = (abs(clip_r - (ref_r or 128)) + 
                             abs(clip_g - (ref_g or 128)) + 
                             abs(clip_b - (ref_b or 128))) / 3
                brightness_diff = abs(clip_brightness - (ref_brightness or 0.5)) * 255
                color_distance = (color_diff + brightness_diff) / 2
                
                # Combined score (weighted)
                # For re-encoded/reformatted content, color is often more reliable
                combined = (hash_distance * 0.6) + (color_distance * 0.4)
                
                if combined < best_combined:
                    best_combined = combined
                    best_hash_distance = hash_distance
                    best_color_distance = color_distance
                    best_timestamp = ref_ts
            
            all_distances.append(best_hash_distance)
            hash_matches.append(1 if best_hash_distance <= lenient_threshold else 0)
            color_matches.append(1 if best_color_distance < 50 else 0)
            matched_timestamps.append(best_timestamp if best_hash_distance <= lenient_threshold else None)
        
        # Debug logging
        print(f"[DEBUG] Reference: {reference.filename}")
        print(f"[DEBUG] Clip frames: {len(clip_features)}, Ref frames: {len(ref_hashes)}")
        print(f"[DEBUG] Hash distances (min/avg/max): {min(all_distances):.0f}/{sum(all_distances)/len(all_distances):.0f}/{max(all_distances):.0f}")
        
        # Calculate confidence using multiple signals
        hash_match_ratio = sum(hash_matches) / len(hash_matches)
        color_match_ratio = sum(color_matches) / len(color_matches)
        
        # Distance-based score (inverse of average distance)
        avg_distance = sum(all_distances) / len(all_distances)
        distance_score = max(0, 1 - (avg_distance / 256)) * 100  # 256 is max distance
        
        print(f"[DEBUG] Hash match ratio: {hash_match_ratio:.2f}, Color match ratio: {color_match_ratio:.2f}")
        print(f"[DEBUG] Avg distance: {avg_distance:.0f}, Distance score: {distance_score:.1f}")
        
        # Combined confidence from multiple signals
        # Even if hash doesn't match well, color similarity suggests related content
        confidence = (
            (hash_match_ratio * 40) +      # Hash matches
            (color_match_ratio * 25) +      # Color matches  
            (distance_score * 0.35)         # Overall similarity
        )
        
        # Bonus for low minimum distance (at least some frames matched well)
        min_distance = min(all_distances)
        if min_distance < 32:
            confidence += 15  # Strong frame match bonus
        elif min_distance < 64:
            confidence += 10  # Good frame match bonus
        elif min_distance < 96:
            confidence += 5   # Weak frame match bonus
        
        # Temporal consistency boost
        temporal_boost = self._calculate_temporal_consistency(
            [1 if d <= lenient_threshold else 0 for d in all_distances], 
            matched_timestamps
        )
        confidence = min(100, confidence + (temporal_boost * 10))
        
        print(f"[DEBUG] Final confidence: {confidence:.1f}")
        
        # ALWAYS return a result if there's ANY similarity (let UI show weak matches)
        # Only filter out completely unrelated content
        if confidence < 1 and min_distance > 200:
            print(f"[DEBUG] No similarity detected")
            return None
        
        # Estimate timestamp range
        valid_timestamps = [t for t in matched_timestamps if t is not None]
        
        if valid_timestamps:
            start_timestamp = min(valid_timestamps)
            end_timestamp = max(valid_timestamps)
        else:
            # Use distance-based estimation
            start_timestamp = None
            end_timestamp = None
        
        return {
            'video_id': reference.id,
            'video_title': reference.title or reference.filename,
            'video_filename': reference.filename,
            'confidence': round(max(1, confidence), 2),  # Minimum 1% if returned
            'confidence_label': self._get_confidence_label(confidence),
            'start_timestamp': start_timestamp,
            'end_timestamp': end_timestamp,
            'timestamp_formatted': self._format_timestamp_range(start_timestamp, end_timestamp),
            'match_ratio': hash_match_ratio,
            'color_match_ratio': color_match_ratio,
            'min_distance': min_distance,
            'avg_distance': round(avg_distance, 1),
            'avg_similarity': round((1 - avg_distance/256) * 100, 2)
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
        if confidence >= MatchConfig.VERY_STRONG_MATCH:
            return "Very Strong Match"
        elif confidence >= MatchConfig.STRONG_MATCH:
            return "Strong Match"
        elif confidence >= MatchConfig.POSSIBLE_MATCH:
            return "Possible Match"
        elif confidence >= MatchConfig.WEAK_MATCH:
            return "Weak Match"
        return "Very Weak Match"
    
    def _format_timestamp_range(self, start: Optional[float], 
                                end: Optional[float]) -> str:
        """Format timestamp range for display."""
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
