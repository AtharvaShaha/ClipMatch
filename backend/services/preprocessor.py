"""
ClipMatch Preprocessor
Comprehensive frame preprocessing for robust video matching
"""

import cv2
import numpy as np
from typing import List, Tuple
import sys
sys.path.append('..')
from config import VideoConfig


class FramePreprocessor:
    """
    Preprocesses video frames to make them robust to variations
    like re-encoding, brightness changes, resolution differences, etc.
    """
    
    # Standard preprocessing size
    PREPROCESS_WIDTH = 256
    PREPROCESS_HEIGHT = 256
    
    @staticmethod
    def preprocess_frame(frame: np.ndarray) -> np.ndarray:
        """
        Comprehensive preprocessing of a single frame.
        
        Applies: resize, grayscale, subtitle crop, histogram equalization
        
        Args:
            frame: BGR image as numpy array from OpenCV
            
        Returns:
            Preprocessed grayscale frame
        """
        # Step 1: Resize to standard size
        frame = cv2.resize(frame, (FramePreprocessor.PREPROCESS_WIDTH, 
                                   FramePreprocessor.PREPROCESS_HEIGHT))
        
        # Step 2: Convert to grayscale - removes color variations
        # Colour often changes between different exports, but content structure stays same
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Step 3: Crop bottom 12% to remove subtitles
        # Subtitles are unreliable for matching (can be added/removed)
        crop_height = int(frame.shape[0] * 0.12)
        frame = frame[:-crop_height, :]
        
        # Step 4: Histogram equalization for brightness normalization
        # Makes darker and brighter versions of same content comparable
        frame = cv2.equalizeHist(frame)
        
        return frame
    
    @staticmethod
    def batch_preprocess_frames(frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Preprocess multiple frames efficiently.
        
        Args:
            frames: List of BGR frames
            
        Returns:
            List of preprocessed frames
        """
        return [FramePreprocessor.preprocess_frame(f) for f in frames]
    
    @staticmethod
    def get_frame_difference_score(frame1: np.ndarray, 
                                   frame2: np.ndarray) -> float:
        """
        Calculate how different two frames are.
        Used for keyframe selection - high difference = informative frame.
        
        Args:
            frame1: First frame (grayscale)
            frame2: Second frame (grayscale)
            
        Returns:
            Difference score (0-255 range)
        """
        # Ensure both frames are same size
        if frame1.shape != frame2.shape:
            frame2 = cv2.resize(frame2, (frame1.shape[1], frame1.shape[0]))
        
        # Absolute difference between frames
        diff = cv2.absdiff(frame1.astype(np.float32), frame2.astype(np.float32))
        
        # Return mean difference
        return np.mean(diff)
    
    @staticmethod
    def select_keyframes(frames: List[np.ndarray], 
                        num_keyframes: int = 8,
                        min_diff_threshold: float = 5.0) -> List[int]:
        """
        Select the most visually distinctive frames from a clip.
        These are frames with high information content for reliable matching.
        
        Distinctive frames are:
        - Scene changes
        - Action moments
        - Unique visual compositions
        
        Boring frames to avoid:
        - Slow pans
        - Static shots
        - Uniform colors
        
        Args:
            frames: List of preprocessed frames
            num_keyframes: Number of keyframes to select
            min_diff_threshold: Minimum difference score to consider
            
        Returns:
            List of indices of selected keyframes (sorted chronologically)
        """
        if len(frames) <= num_keyframes:
            # If we have fewer frames than requested, use all
            return list(range(len(frames)))
        
        # Calculate frame difference scores
        scores = []
        for i in range(1, len(frames)):
            diff = FramePreprocessor.get_frame_difference_score(
                frames[i-1], frames[i]
            )
            if diff >= min_diff_threshold:
                scores.append((diff, i))
        
        # If not enough high-difference frames, use all transitions
        if len(scores) < num_keyframes:
            scores = [(FramePreprocessor.get_frame_difference_score(
                frames[i-1], frames[i]), i) for i in range(1, len(frames))]
        
        # Sort by difference and take top N
        scores.sort(reverse=True)
        keyframe_indices = [idx for _, idx in scores[:num_keyframes]]
        
        # Return in chronological order
        keyframe_indices.sort()
        
        return keyframe_indices
    
    @staticmethod
    def normalize_frame_for_correlation(frame: np.ndarray) -> np.ndarray:
        """
        Normalize frame for NCC/SSIM computation.
        Removes mean and scales to unit variance.
        
        Args:
            frame: Grayscale frame
            
        Returns:
            Normalized frame
        """
        frame = frame.astype(np.float32)
        mean = np.mean(frame)
        std = np.std(frame)
        
        if std == 0:
            return frame - mean
        
        return (frame - mean) / std
