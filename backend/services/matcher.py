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
            clip_duration = clip_info.get('duration', 0)
            
            for ref_video in references:
                match_result = self._match_against_reference(
                    clip_features, ref_video, session, clip_duration
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
                                  session, clip_duration: float = 0) -> Optional[Dict]:
        """
        Match clip features against a single reference video.
        Uses multiple matching strategies for robust detection.
        
        Args:
            clip_features: List of feature dictionaries from the clip
            reference: Reference video database object
            session: Database session
            clip_duration: Duration of the query clip in seconds
            
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
        ref_hashes = [(f.id, f.timestamp, f.phash, f.dhash, f.whash, f.brightness, f.avg_color_r, f.avg_color_g, f.avg_color_b) for f in ref_frames]
        
        # Strategy 1: Hash-based matching with very lenient threshold
        hash_matches = []
        color_matches = []
        all_distances = []
        matched_timestamps = []
        
        # Use lenient threshold - we're looking for partial matches in large videos
        # For 64-bit hashes: anything under 32 bits difference (~50%) is very similar
        lenient_threshold = 32
        
        for i, clip_feat in enumerate(clip_features):
            # Use multiple hash algorithms for robustness
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
            
            for ref_id, ref_ts, ref_phash, ref_dhash, ref_whash, ref_brightness, ref_r, ref_g, ref_b in ref_hashes:
                # Compute distances using multiple hash algorithms for robustness
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
                # When frames are from same video source, color match is VERY reliable
                combined = (avg_hash_distance * 0.5) + (color_distance * 0.5)
                
                if combined < best_combined:
                    best_combined = combined
                    best_hash_distance = avg_hash_distance
                    best_color_distance = color_distance
                    best_timestamp = ref_ts
            
            all_distances.append(best_hash_distance)
            # Match if either hash is good OR color is good (lenient approach)
            hash_matches.append(1 if best_hash_distance <= lenient_threshold else 0)
            color_matches.append(1 if best_color_distance < 80 else 0)  # More lenient color threshold
            # Timestamp is matched if EITHER hash OR color matches reasonably well
            matched_timestamps.append(
                best_timestamp if (best_hash_distance <= lenient_threshold or best_color_distance < 80) else None
            )
        
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
        min_distance = min(all_distances) if all_distances else 999
        
        confidence = (
            (hash_match_ratio * 35) +      # Hash matches (slightly reduced)
            (color_match_ratio * 30) +      # Color matches (increased - re-encoding affects color)
            (distance_score * 0.35)         # Overall similarity
        )
        
        # Aggressive bonus for frames with good distance match (key for finding clips in large videos)
        if min_distance < 12:
            confidence += 15  # Good frame match bonus
        elif min_distance < 16:
            confidence += 10  # Moderate frame match bonus
        elif min_distance < 20:
            confidence += 5  # Small frame match bonus
        
        # Temporal consistency boost
        temporal_boost = self._calculate_temporal_consistency(
            [1 if d <= lenient_threshold else 0 for d in all_distances], 
            matched_timestamps
        )
        confidence = min(100, confidence + (temporal_boost * 10))
        
        print(f"[DEBUG] Final confidence: {confidence:.1f}")
        
        # VERY STRICT VALIDATION - reject any video that isn't a clear match
        # All signals must confirm it's the same content
        
        rejection_reasons = []
        
        # Check 1: STRICT hash match ratio (at least 60% of frames must match well)
        # This ensures we have strong frame-level agreement
        if hash_match_ratio < 0.60:
            rejection_reasons.append(f"Hash match ratio {hash_match_ratio*100:.0f}% (FAIL - need ≥60%)")
        
        # Check 2: VERY STRICT average distance (max 16 bits = ~25% different)
        # Only nearly identical frames should match
        if avg_distance > 16:
            rejection_reasons.append(f"Avg distance {avg_distance:.1f} bits (FAIL - max 16)")
        
        # Check 3: STRICT color match (at least 60% of frames must match colors)
        if color_match_ratio < 0.60:
            rejection_reasons.append(f"Color match ratio {color_match_ratio*100:.0f}% (FAIL - need ≥60%)")
        
        # Check 4: STRICT matching frames (at least 55% of clip must match)
        matching_frames = sum(hash_matches)
        min_matching_frames = max(5, int(len(clip_features) * 0.55))  # At least 55% of frames
        if matching_frames < min_matching_frames:
            rejection_reasons.append(f"Only {matching_frames}/{len(clip_features)} frames matched (FAIL - need ≥{min_matching_frames})")
        
        # Check 5: VERY HIGH confidence (at least 50% = very strong match)
        # This is a strong signal threshold
        if confidence < 50:
            rejection_reasons.append(f"Confidence {confidence:.1f}% (FAIL - need ≥50%)")
        
        # Check 6: VERY STRICT best frame match (best frame < 12 bits = ~19% different)
        if min_distance > 12:
            rejection_reasons.append(f"Best frame distance {min_distance:.0f} bits (FAIL - need <12)")
        
        # REJECT if ANY check fails (very strict AND logic)
        if rejection_reasons:
            print(f"[DEBUG] ❌ REJECTED - Video doesn't match criteria:")
            for reason in rejection_reasons:
                print(f"        {reason}")
            return None
        
        print(f"[DEBUG] ✅ ACCEPTED - Video matches ALL criteria")
        
        # Estimate timestamp range from matched frames
        valid_timestamps = [t for t in matched_timestamps if t is not None]
        
        if valid_timestamps:
            start_timestamp = round(min(valid_timestamps), 2)
            # Use actual clip duration (passed as parameter) instead of window stride calculation
            # This fixes the issue where 10 second clip was showing as 2 second duration
            actual_duration = clip_duration if clip_duration > 0 else len(clip_features)
            end_timestamp = round(start_timestamp + actual_duration, 2)
        else:
            # Fallback: estimate from best matching frame
            if all_distances:
                best_idx = all_distances.index(min(all_distances))
                if best_idx < len(matched_timestamps) and matched_timestamps[best_idx]:
                    start_timestamp = round(matched_timestamps[best_idx], 2)
                    actual_duration = clip_duration if clip_duration > 0 else len(clip_features)
                    end_timestamp = round(start_timestamp + actual_duration, 2)
                else:
                    start_timestamp = None
                    end_timestamp = None
            else:
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
            'match_ratio': round(hash_match_ratio, 3),
            'color_match_ratio': round(color_match_ratio, 3),
            'min_distance': int(min_distance),
            'avg_distance': round(avg_distance, 1),
            'avg_similarity': round((1 - min(avg_distance, 256)/256) * 100, 2),
            'clip_frames_analyzed': len(clip_features),
            'reference_frames_total': len(ref_hashes)
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
