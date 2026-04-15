"""
ClipMatch Verification Engine
NCC and SSIM verification for final precise matching
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
from scipy import signal
from skimage.metrics import structural_similarity as ssim
import sys
sys.path.append('..')


class VerificationEngine:
    """
    Performs precise pixel-level verification using NCC and SSIM.
    
    After hashing finds a candidate match area, this engine does detailed
    verification at full quality to confirm legitimacy with high confidence.
    """
    
    NCC_THRESHOLD = 0.85      # NCC score above 0.85 = strong match
    SSIM_THRESHOLD = 0.80     # SSIM score above 0.80 = strong match
    MIN_CONSECUTIVE_MATCHES = 5  # Need at least 5 consecutive high scores
    
    @staticmethod
    def normalized_cross_correlation(template: np.ndarray, 
                                     image: np.ndarray) -> float:
        """
        Compute Normalized Cross-Correlation (NCC) between template and image.
        
        NCC measures similarity between two patches, normalized by their intensities.
        Result is -1 to +1, where 1 = perfect match, 0 = no correlation.
        
        Key advantage: Robust to brightness differences!
        A darker version of the same scene still gets high NCC score.
        
        Args:
            template: Query patch (preprocessed grayscale)
            image: Reference patch (preprocessed grayscale, same size as template)
            
        Returns:
            NCC score (-1 to 1, typically 0 to 1 for similar images)
        """
        # Ensure same size
        if template.shape != image.shape:
            image = cv2.resize(image, (template.shape[1], template.shape[0]))
        
        # Convert to float for computation
        template = template.astype(np.float32)
        image = image.astype(np.float32)
        
        # Compute means
        template_mean = np.mean(template)
        image_mean = np.mean(image)
        
        # Center the signals
        template_centered = template - template_mean
        image_centered = image - image_mean
        
        # Compute correlation components
        numerator = np.sum(template_centered * image_centered)
        denominator = np.sqrt(np.sum(template_centered ** 2) * 
                            np.sum(image_centered ** 2))
        
        if denominator == 0:
            return 0.0
        
        ncc = numerator / denominator
        
        # Clamp to [-1, 1] due to floating point errors
        return float(np.clip(ncc, -1.0, 1.0))
    
    @staticmethod
    def structural_similarity(template: np.ndarray, 
                             image: np.ndarray) -> float:
        """
        Compute Structural Similarity Index (SSIM) between template and image.
        
        SSIM is more perceptually meaningful than raw pixel difference.
        Considers three components:
        1. Luminance (brightness) similarity
        2. Contrast similarity
        3. Structural similarity
        
        Result is -1 to 1, where 1 = identical images.
        Typically for images rated 0.8+ = perceptually very similar.
        
        Args:
            template: Query patch (preprocessed grayscale)
            image: Reference patch (same size)
            
        Returns:
            SSIM score (-1 to 1, typically 0 to 1)
        """
        # Ensure same size
        if template.shape != image.shape:
            image = cv2.resize(image, (template.shape[1], template.shape[0]))
        
        # Convert to uint8 if needed
        template_uint8 = (template * 255).astype(np.uint8) if template.max() <= 1 else template.astype(np.uint8)
        image_uint8 = (image * 255).astype(np.uint8) if image.max() <= 1 else image.astype(np.uint8)
        
        # Compute SSIM
        # data_range is the maximum possible value (255 for uint8)
        sim = ssim(template_uint8, image_uint8, data_range=255)
        
        return float(sim)
    
    @staticmethod
    def verify_frame_match(clip_frame: np.ndarray,
                          reference_frame: np.ndarray,
                          ncc_weight: float = 0.5,
                          ssim_weight: float = 0.5) -> Dict[str, float]:
        """
        Verify if two frames match using combined NCC and SSIM.
        
        Args:
            clip_frame: Frame from the query clip (preprocessed)
            reference_frame: Frame from the reference video (preprocessed)
            ncc_weight: Weight for NCC in combined score
            ssim_weight: Weight for SSIM in combined score
            
        Returns:
            Dictionary with 'ncc', 'ssim', and 'combined' scores
        """
        ncc_score = VerificationEngine.normalized_cross_correlation(
            clip_frame, reference_frame
        )
        
        ssim_score = VerificationEngine.structural_similarity(
            clip_frame, reference_frame
        )
        
        # Normalize both to 0-1 range
        ncc_norm = (ncc_score + 1) / 2  # Convert from [-1,1] to [0,1]
        ssim_norm = (ssim_score + 1) / 2  # Convert from [-1,1] to [0,1]
        
        # Combined score
        combined = (ncc_weight * ncc_norm + ssim_weight * ssim_norm)
        
        return {
            'ncc': ncc_norm,
            'ssim': ssim_norm,
            'combined': combined
        }
    
    @staticmethod
    def verify_frame_sequence(clip_frames: List[np.ndarray],
                              reference_frames: List[np.ndarray],
                              timestamp_tolerance: int = 1) -> Dict:
        """
        Verify that a sequence of clip frames matches a reference sequence.
        
        timestamp_tolerance allows ±1 frame timing variance due to encoding differences.
        
        Args:
            clip_frames: List of preprocessed clip frames
            reference_frames: List of preprocessed reference frames
            timestamp_tolerance: Frames allowed timing deviation
            
        Returns:
            Dictionary with verification scores and match quality
        """
        if len(clip_frames) == 0 or len(reference_frames) == 0:
            return {
                'success': False,
                'error': 'Empty frame list',
                'matches': 0,
                'total': 0,
                'match_percentage': 0,
                'avg_ncc': 0,
                'avg_ssim': 0,
                'avg_combined': 0
            }
        
        scores = []
        matched = 0
        
        # Compare each clip frame against corresponding reference frame
        # Allowing for timing tolerance
        for clip_idx, clip_frame in enumerate(clip_frames):
            best_score = None
            
            # Search in reference within tolerance window
            start_ref_idx = max(0, clip_idx - timestamp_tolerance)
            end_ref_idx = min(len(reference_frames), clip_idx + timestamp_tolerance + 1)
            
            for ref_idx in range(start_ref_idx, end_ref_idx):
                if ref_idx < len(reference_frames):
                    verification = VerificationEngine.verify_frame_match(
                        clip_frame, reference_frames[ref_idx]
                    )
                    
                    if best_score is None or verification['combined'] > best_score['combined']:
                        best_score = verification
            
            if best_score:
                scores.append(best_score)
                # Consider it a match if combined score > threshold
                if best_score['combined'] > VerificationEngine.NCC_THRESHOLD:
                    matched += 1
        
        if len(scores) == 0:
            return {
                'success': False,
                'error': 'Could not compute verification scores',
                'matches': 0,
                'total': len(clip_frames),
                'match_percentage': 0,
                'avg_ncc': 0,
                'avg_ssim': 0,
                'avg_combined': 0
            }
        
        # Calculate average scores
        avg_ncc = np.mean([s['ncc'] for s in scores])
        avg_ssim = np.mean([s['ssim'] for s in scores])
        avg_combined = np.mean([s['combined'] for s in scores])
        
        match_percentage = (matched / len(scores)) * 100
        
        return {
            'success': True,
            'matches': matched,
            'total': len(scores),
            'match_percentage': match_percentage,
            'avg_ncc': avg_ncc,
            'avg_ssim': avg_ssim,
            'avg_combined': avg_combined,
            'individual_scores': scores
        }
