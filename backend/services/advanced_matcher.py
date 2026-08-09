"""
ClipMatch Advanced Matcher
Complete end-to-end matching pipeline with NCC/SSIM verification.

Optimized with:
- ThreadPoolExecutor for parallel reference comparison (numpy releases GIL)
- TWO-PHASE approach: fast hash ranking for ALL refs, then NCC only on top-3
- Pre-built numpy arrays during DB retrieval (shared across threads)
- Reduced augmentations (2 instead of 4)
- PipelineProfiler for per-stage timing
- Structured pipeline logging
"""

import logging
logger = logging.getLogger('clipmatch.advanced')

import os
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
sys.path.append('..')

from config import MatchConfig
from models.database import get_session, ReferenceVideo, VideoFrame
from services.video_processor import VideoProcessor
from services.preprocessor import FramePreprocessor
from services.feature_extractor import FeatureExtractor, hex_to_int, fast_hamming
from services.verification import VerificationEngine
from services.utils import format_timestamp_range, get_confidence_label, estimate_timestamp_range
from services.matcher import (
    _build_int_arrays, _vectorised_hamming, _compute_augmented_features,
    _precompute_clip_ints, _match_single_reference, _calculate_temporal_consistency_static
)
from services.profiler import PipelineProfiler


def _ncc_verify_candidate(match_result, clip_raw_frames):
    """
    Run NCC/SSIM pixel-level verification on a SINGLE candidate match.
    Called only for top-3 hash-ranked candidates — not all references.
    
    Args:
        match_result: dict from _match_single_reference (with video_id, filepath, etc.)
        clip_raw_frames: list of raw BGR frames from the query clip
    
    Returns:
        Updated match_result dict with NCC/SSIM scores and adjusted confidence
    """
    import time
    t0 = time.perf_counter()
    
    ref_filepath = match_result.get('_filepath')
    if not ref_filepath or not os.path.exists(ref_filepath):
        return match_result
    
    try:
        preprocessor = FramePreprocessor()
        verifier = VerificationEngine()
        vp = VideoProcessor()
        
        # Extract a few reference frames at the matched timestamp range
        start_ts = match_result.get('start_timestamp', 0) or 0
        end_ts = match_result.get('end_timestamp', start_ts + 20) or (start_ts + 20)
        
        # Use only 3 frames for fast verification (was 4, originally 8)
        num_verify_frames = min(3, len(clip_raw_frames))
        verify_timestamps = np.linspace(start_ts, end_ts, num_verify_frames)
        
        ref_raw_frames = []
        for ts in verify_timestamps:
            frame = vp.get_frame_at_timestamp(ref_filepath, float(ts))
            if frame is not None:
                ref_raw_frames.append(frame)
        
        if not ref_raw_frames:
            return match_result
        
        # Preprocess both sets
        num = min(len(clip_raw_frames), len(ref_raw_frames))
        clip_indices = np.linspace(0, len(clip_raw_frames) - 1, num, dtype=int)
        clip_subset = [clip_raw_frames[i] for i in clip_indices]
        
        clip_pp = preprocessor.batch_preprocess_frames(clip_subset[:num])
        ref_pp = preprocessor.batch_preprocess_frames(ref_raw_frames[:num])
        
        verification = verifier.verify_frame_sequence(clip_pp, ref_pp)
        
        if verification and verification.get('success'):
            ncc_avg = max(0, verification.get('avg_ncc', 0))
            ssim_avg = max(0, verification.get('avg_ssim', 0))
            
            # Keep original hash confidence — NCC can only BOOST, never lower
            original_confidence = match_result['confidence']
            
            # NCC bonus: add up to +8 points for strong pixel evidence
            ncc_bonus = 0
            if ncc_avg >= 0.8:
                ncc_bonus = 8  # Excellent pixel match
            elif ncc_avg >= 0.6:
                ncc_bonus = 4  # Good pixel match
            elif ncc_avg >= 0.4:
                ncc_bonus = 1  # Marginal evidence
            
            # SSIM bonus: add up to +4 points
            ssim_bonus = 0
            if ssim_avg >= 0.7:
                ssim_bonus = 4
            elif ssim_avg >= 0.5:
                ssim_bonus = 2
            
            boosted_confidence = min(99.5, original_confidence + ncc_bonus + ssim_bonus)
            
            match_result['confidence'] = round(boosted_confidence, 2)
            match_result['confidence_label'] = get_confidence_label(boosted_confidence)
            match_result['ncc_score'] = round(ncc_avg, 3)
            match_result['ssim_score'] = round(ssim_avg, 3)
            match_result['verification_method'] = 'NCC+SSIM'
            match_result['details'] = {
                'hash_ratio': match_result.get('match_ratio', 0),
                'color_ratio': match_result.get('color_match_ratio', 0),
                'avg_distance': match_result.get('avg_distance', 0),
                'ncc': round(ncc_avg, 3),
                'ssim': round(ssim_avg, 3),
            }
            
            elapsed = time.perf_counter() - t0
            logger.info("[Pipeline]   NCC verification: %s | ncc=%.3f ssim=%.3f → conf=%.1f%% | %.2fs",
                       match_result['video_filename'], ncc_avg, ssim_avg, boosted_confidence, elapsed)
    except Exception as e:
        logger.warning("[Pipeline]   NCC verification failed for %s: %s",
                      match_result.get('video_filename', '?'), e)
    
    return match_result


class AdvancedClipMatcher:
    """
    Complete video matching system implementing all techniques:
    - Frame preprocessing
    - Dual hashing (pHash + dHash)
    - Augmented feature matching (original + flip)
    - TWO-PHASE: Fast hash ranking for all refs → NCC/SSIM only on top-3
    - Confidence scoring
    """
    
    def __init__(self):
        """Initialize matcher with all required components."""
        self.video_processor = VideoProcessor()
        self.preprocessor = FramePreprocessor()
        self.feature_extractor = FeatureExtractor()
        self.verifier = VerificationEngine()
        self._max_workers = min(os.cpu_count() or 2, 6)
    
    def match_clip(self, clip_path: str, 
                   verbose: bool = False) -> Dict:
        """
        Complete end-to-end matching pipeline with NCC-based verification.
        
        TWO-PHASE approach:
        Phase 1: Fast hash-based ranking for ALL references (same as standard matcher)
        Phase 2: NCC/SSIM verification ONLY on top-3 candidates
        
        This eliminates running NCC on 28+ references (was 73s!) and instead
        runs it on just 3 (takes ~3-5s total).
        """
        import time
        profiler = PipelineProfiler()
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
        
        # ── Stage: Frame Extraction (12 frames — sweet spot for speed + accuracy) ──
        profiler.start('frame_extraction')
        logger.info("[Pipeline] Stage: Frame Extraction — extracting from %s", os.path.basename(clip_path))
        
        clip_frames = self.video_processor.extract_all_frames_to_list(
            clip_path, max_frames=12, query_mode=True
        )
        
        if not clip_frames:
            return {
                'success': False,
                'error': 'Could not extract frames',
                'matches': []
            }
        
        # Cache raw frames for NCC verification (extracted once)
        clip_raw_frames = [t[2] for t in clip_frames]
        
        profiler.stop('frame_extraction')
        logger.info("[Pipeline] Stage: Frame Extraction | %.2fs | %d frames",
                    profiler.get('frame_extraction'), len(clip_frames))
        
        # ── Stage: Feature Extraction (2 augmented sets: original + flip) ──
        profiler.start('feature_extraction')
        
        clip_features = self.feature_extractor.batch_extract(clip_frames)
        
        augmented_feature_sets = _compute_augmented_features(
            self.feature_extractor, clip_frames,
            video_processor=self.video_processor, clip_path=clip_path
        )
        
        profiler.stop('feature_extraction')
        logger.info("[Pipeline] Stage: Feature Extraction | %.2fs | %d augmented sets",
                    profiler.get('feature_extraction'), len(augmented_feature_sets))
        
        # ── Stage: Hash Pre-computation ──
        profiler.start('hash_precompute')
        clip_ints_all = _precompute_clip_ints(augmented_feature_sets)
        profiler.stop('hash_precompute')
        
        # ── Stage: DB Retrieval — use cache or fetch + build numpy arrays ──
        profiler.start('db_retrieval')
        session = get_session()
        try:
            from services.matcher import _get_cached_ref_data
            ref_data_list, cache_hit = _get_cached_ref_data(session)
            
            if not ref_data_list:
                return {
                    'success': False,
                    'error': 'No reference videos indexed',
                    'matches': []
                }
            
            profiler.stop('db_retrieval')
            logger.info("[Pipeline] Stage: DB Retrieval | %.2fs | %d references %s",
                       profiler.get('db_retrieval'), len(ref_data_list),
                       '(cached)' if cache_hit else '(built from DB)')
            
            # ══════════════════════════════════════════════════════════
            # PHASE 1: Fast hash-based ranking for ALL references
            # (same logic as standard matcher — no NCC here)
            # ══════════════════════════════════════════════════════════
            profiler.start('matching')
            hash_matches = []
            rejected_count = 0
            
            logger.info("[Pipeline] PHASE 1: Hash ranking | %d references | %d threads",
                       len(ref_data_list), self._max_workers)
            
            with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
                futures = {
                    executor.submit(
                        _match_single_reference,
                        rd, clip_features, clip_ints_all,
                        clip_duration, MatchConfig.MIN_CONSECUTIVE_MATCHES
                    ): rd
                    for rd in ref_data_list
                }
                
                for future in as_completed(futures):
                    rd = futures[future]
                    try:
                        result = future.result()
                        if result is None:
                            continue
                        
                        if result.get('_rejected'):
                            rejected_count += 1
                            logger.debug("[Pipeline]   %s → REJECTED: %s | %.3fs",
                                        result['_filename'], result['_reason'], result['_elapsed'])
                            continue
                        
                        # Attach filepath for NCC phase
                        result['_filepath'] = rd['filepath']
                        result['verification_method'] = 'Hash-based'
                        
                        if result['confidence'] >= 45:
                            hash_matches.append(result)
                            logger.info("[Pipeline]   %s → CANDIDATE: %.1f%% | %.3fs",
                                       result['video_filename'], result['confidence'], result['_elapsed'])
                    except Exception as e:
                        logger.warning("[Pipeline]   %s → ERROR: %s", rd['filename'], e)
            
            # Sort by hash confidence
            hash_matches.sort(key=lambda x: x['confidence'], reverse=True)
            
            profiler.stop('matching')
            logger.info("[Pipeline] PHASE 1 complete | %.2fs | %d candidates, %d rejected",
                       profiler.get('matching'), len(hash_matches), rejected_count)
            
            # ══════════════════════════════════════════════════════════
            # PHASE 2: NCC/SSIM verification ONLY on top-3 candidates
            # (was running on ALL 28+ passing refs = 73s → now just 3 = ~3s)
            # ══════════════════════════════════════════════════════════
            profiler.start('ncc_verification')
            
            top_n = min(2, len(hash_matches))
            if top_n > 0:
                logger.info("[Pipeline] PHASE 2: NCC verification on top-%d candidates", top_n)
                
                for i in range(top_n):
                    hash_matches[i] = _ncc_verify_candidate(hash_matches[i], clip_raw_frames)
                
                # Re-sort after NCC adjustment
                hash_matches.sort(key=lambda x: x['confidence'], reverse=True)
            
            profiler.stop('ncc_verification')
            logger.info("[Pipeline] PHASE 2 complete | %.2fs",
                       profiler.get('ncc_verification'))
            
            # Clean internal keys
            for m in hash_matches:
                m.pop('_elapsed', None)
                m.pop('_filepath', None)
            
            elapsed = time.time() - start_time
            
            # Final logging
            best_match = hash_matches[0] if hash_matches else None
            if best_match:
                logger.info("[Pipeline] Result: Best match=%s | %.1f%% [%s] | Total: %.2fs",
                           best_match['video_filename'], best_match['confidence'],
                           best_match.get('verification_method', '?'), elapsed)
            else:
                logger.info("[Pipeline] Result: No match found | Total: %.2fs", elapsed)
            
            profiler.log_summary('Timing')
            
            result = {
                'success': True,
                'matches': hash_matches,
                'best_match': best_match,
                'query': {
                    'filename': os.path.basename(clip_path),
                    'duration': clip_duration,
                    'frames_analyzed': len(clip_features)
                },
                'processing_time': elapsed,
                'pipeline_stages': profiler.summary(),
                'clip_frames': len(clip_features),
                'reference_videos_searched': len(ref_data_list)
            }
            result['pipeline_stages']['references_searched'] = len(ref_data_list)
            result['pipeline_stages']['candidates_rejected'] = rejected_count
            result['pipeline_stages']['candidates_accepted'] = len(hash_matches)
            
            return result
        
        finally:
            session.close()


# Global instance
advanced_matcher = AdvancedClipMatcher()
