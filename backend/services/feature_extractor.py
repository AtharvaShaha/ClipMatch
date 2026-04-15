"""
ClipMatch Feature Extractor
Handles dual perceptual hashing (pHash + dHash) for robust frame matching
"""

import imagehash
from PIL import Image
import numpy as np
from typing import Dict, List, Tuple, Optional
import sys
sys.path.append('..')
from config import FeatureConfig


class FeatureExtractor:
    """
    Extracts visual features from video frames using dual perceptual hashing.
    
    Uses pHash (DCT-based, global structure) and dHash (gradient-based, edges)
    together for maximum robustness across different video encodings.
    """
    
    HASH_SIZE = 8  # 8x8 = 64-bit hashes
    HAMMING_THRESHOLD = 10  # Out of 64 bits, allow up to 10 bits different
    
    def __init__(self, hash_size: int = None):
        """
        Initialize the feature extractor.
        
        Args:
            hash_size: Size of the hash (default 8x8 = 64 bits)
        """
        self.hash_size = hash_size or self.HASH_SIZE
    
    def extract_dual_hash(self, frame: np.ndarray) -> Dict[str, str]:
        """
        Extract dual hashes (pHash + dHash) from a frame.
        
        pHash: Captures global structure using DCT - robust to brightness/compression
        dHash: Captures edges/gradients - robust to scaling
        
        Args:
            frame: Grayscale frame as numpy array (already preprocessed)
            
        Returns:
            Dict with 'phash' and 'dhash' as hex strings
        """
        # Convert grayscale numpy array to PIL Image
        # If frame is already in uint8 format, use directly
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8) if frame.max() <= 1 else frame.astype(np.uint8)
        
        # Convert to 3-channel for imagehash (library expects RGB)
        if len(frame.shape) == 2:
            frame_rgb = np.stack([frame] * 3, axis=2)
        else:
            frame_rgb = frame
        
        pil_image = Image.fromarray(frame_rgb.astype(np.uint8))
        
        features = {}
        
        try:
            # pHash: Perceptual hash using DCT (Discrete Cosine Transform)
            # Captures overall structure - resilient to re-encoding
            phash = imagehash.phash(pil_image, hash_size=self.hash_size)
            features['phash'] = str(phash)
        except Exception as e:
            print(f"pHash extraction failed: {e}")
            features['phash'] = None
        
        try:
            # dHash: Difference hash - compares adjacent pixels
            # Captures edges and gradients - good for structural matching
            dhash = imagehash.dhash(pil_image, hash_size=self.hash_size)
            features['dhash'] = str(dhash)
        except Exception as e:
            print(f"dHash extraction failed: {e}")
            features['dhash'] = None
        
        return features
    
    def extract_features(self, frame: np.ndarray) -> Dict[str, str]:
        """
        Extract all features (pHash + dHash + wHash) from a frame.
        
        This is a wrapper around extract_dual_hash that also adds wavelet hash.
        
        Args:
            frame: Grayscale frame as numpy array (already preprocessed)
            
        Returns:
            Dict with 'phash', 'dhash', and 'whash' as hex strings
        """
        # Get dual hashes
        features = self.extract_dual_hash(frame)
        
        # Add wavelet hash
        try:
            # Convert grayscale numpy array to PIL Image
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8) if frame.max() <= 1 else frame.astype(np.uint8)
            
            # Convert to 3-channel for imagehash (library expects RGB)
            if len(frame.shape) == 2:
                frame_rgb = np.stack([frame] * 3, axis=2)
            else:
                frame_rgb = frame
            
            pil_image = Image.fromarray(frame_rgb.astype(np.uint8))
            
            # wHash: Wavelet hash using Haar wavelets
            # Captures multi-scale structure
            whash = imagehash.whash(pil_image, hash_size=self.hash_size)
            features['whash'] = str(whash)
        except Exception as e:
            print(f"wHash extraction failed: {e}")
            features['whash'] = None
        
        return features
    
    def batch_extract(self, frames: List[np.ndarray]) -> List[Dict[str, str]]:
        """
        Extract dual hashes from multiple frames.
        
        Args:
            frames: List of preprocessed grayscale frames
            
        Returns:
            List of feature dictionaries, one per frame
        """
        return [self.extract_dual_hash(frame) for frame in frames]
    
    @staticmethod
    def hamming_distance(hash1_str: str, hash2_str: str) -> int:
        """
        Calculate Hamming distance between two hash strings.
        Counts number of bits that differ.
        
        Args:
            hash1_str: First hash as hex string
            hash2_str: Second hash as hex string
            
        Returns:
            Number of differing bits (0-64 for 64-bit hashes)
        """
        if hash1_str is None or hash2_str is None:
            return 999  # High distance if either hash is invalid
        
        # Convert hex strings to integers
        try:
            h1 = imagehash.ImageHash(hash1_str)
            h2 = imagehash.ImageHash(hash2_str)
            return h1 - h2  # imagehash supports subtraction = hamming distance
        except:
            return 999
    
    @staticmethod
    def combined_hash_distance(features1: Dict, 
                               features2: Dict,
                               phash_weight: float = 0.6,
                               dhash_weight: float = 0.4) -> float:
        """
        Calculate combined distance between two frames using both hashes.
        
        Weights emphasize pHash (global structure) slightly more than dHash.
        
        Args:
            features1: First frame's features {'phash': ..., 'dhash': ...}
            features2: Second frame's features
            phash_weight: Weight for pHash distance (default 0.6)
            dhash_weight: Weight for dHash distance (default 0.4)
            
        Returns:
            Normalized combined distance (0-1, where 1 is completely different)
        """
        phash_dist = FeatureExtractor.hamming_distance(
            features1.get('phash'), features2.get('phash')
        )
        
        dhash_dist = FeatureExtractor.hamming_distance(
            features1.get('dhash'), features2.get('dhash')
        )
        
        # Normalize to 0-1 range (max distance for 64-bit hash is 64)
        phash_normalized = min(phash_dist / 64.0, 1.0)
        dhash_normalized = min(dhash_dist / 64.0, 1.0)
        
        # Weighted combination
        combined = (phash_weight * phash_normalized + 
                   dhash_weight * dhash_normalized)
        
        return combined
    
    @staticmethod
    def is_hash_match(features1: Dict, features2: Dict, 
                     threshold: float = None) -> bool:
        """
        Check if two frames match based on dual hashing.
        
        Args:
            features1: First frame's features
            features2: Second frame's features
            threshold: Maximum allowed combined distance (default from HAMMING_THRESHOLD)
            
        Returns:
            True if combined distance is below threshold
        """
        if threshold is None:
            threshold = FeatureExtractor.HAMMING_THRESHOLD / 64.0  # Convert to 0-1 range
        
        distance = FeatureExtractor.combined_hash_distance(features1, features2)
        return distance <= threshold
    
    def compute_hash_distance(self, hash1: str, hash2: str) -> int:
        """
        Compute Hamming distance between two hash strings.
        
        The Hamming distance is the number of bit positions where the
        corresponding bits differ. Lower distance = more similar.
        
        Args:
            hash1: First hash as hex string
            hash2: Second hash as hex string
            
        Returns:
            Hamming distance (integer)
        """
        try:
            h1 = imagehash.hex_to_hash(hash1)
            h2 = imagehash.hex_to_hash(hash2)
            return h1 - h2  # ImageHash overloads subtraction for Hamming distance
        except Exception as e:
            print(f"Error computing hash distance: {e}")
            return 999  # Return high distance on error
    
    def compute_similarity(self, hash1: str, hash2: str, max_distance: int = None) -> float:
        """
        Compute similarity score between two hashes.
        
        Args:
            hash1: First hash as hex string
            hash2: Second hash as hex string
            max_distance: Maximum possible distance (default: hash_size^2)
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        if max_distance is None:
            max_distance = self.hash_size * self.hash_size
        
        distance = self.compute_hash_distance(hash1, hash2)
        similarity = 1.0 - (distance / max_distance)
        return max(0.0, min(1.0, similarity))
    
    def are_similar(self, hash1: str, hash2: str, threshold: int = None) -> bool:
        """
        Check if two hashes are similar within a threshold.
        
        Args:
            hash1: First hash as hex string
            hash2: Second hash as hex string
            threshold: Maximum Hamming distance for similarity
            
        Returns:
            True if hashes are similar
        """
        if threshold is None:
            threshold = FeatureConfig.HASH_SIZE  # Default threshold
        
        distance = self.compute_hash_distance(hash1, hash2)
        return distance <= threshold
    
    def batch_extract(self, frames: List[Tuple[int, float, np.ndarray]]) -> List[Dict]:
        """
        Extract features from a batch of frames.
        
        Args:
            frames: List of (frame_number, timestamp, frame_image) tuples
            
        Returns:
            List of dictionaries containing frame info and features
        """
        results = []
        
        for frame_num, timestamp, frame in frames:
            features = self.extract_features(frame)
            
            # Calculate color features for quick filtering
            avg_color = np.mean(frame, axis=(0, 1))
            brightness = (0.299 * avg_color[0] + 0.587 * avg_color[1] + 0.114 * avg_color[2]) / 255.0
            
            results.append({
                'frame_number': frame_num,
                'timestamp': timestamp,
                'phash': features.get('phash', ''),
                'dhash': features.get('dhash', ''),
                'whash': features.get('whash', ''),
                'avg_color_r': int(avg_color[0]),
                'avg_color_g': int(avg_color[1]),
                'avg_color_b': int(avg_color[2]),
                'brightness': brightness
            })
        
        return results
    
    def find_similar_frames(self, query_hash: str, 
                           reference_hashes: List[Tuple[int, str]], 
                           threshold: int = None) -> List[Tuple[int, int]]:
        """
        Find frames in a reference set that are similar to a query hash.
        
        Args:
            query_hash: Hash of the query frame
            reference_hashes: List of (frame_id, hash) tuples from reference
            threshold: Maximum Hamming distance for a match
            
        Returns:
            List of (frame_id, distance) tuples for matching frames
        """
        if threshold is None:
            threshold = FeatureConfig.HASH_SIZE
        
        matches = []
        for frame_id, ref_hash in reference_hashes:
            distance = self.compute_hash_distance(query_hash, ref_hash)
            if distance <= threshold:
                matches.append((frame_id, distance))
        
        # Sort by distance (closest matches first)
        matches.sort(key=lambda x: x[1])
        
        return matches


# Singleton instance
feature_extractor = FeatureExtractor()
