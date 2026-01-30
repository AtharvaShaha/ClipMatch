"""
ClipMatch Feature Extractor
Handles perceptual hashing and feature extraction from video frames
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
    Extracts visual features from video frames using perceptual hashing.
    
    Perceptual hashes are designed to produce similar outputs for visually
    similar images, making them ideal for content matching.
    """
    
    def __init__(self, hash_size: int = None):
        """
        Initialize the feature extractor.
        
        Args:
            hash_size: Size of the hash (default from config)
        """
        self.hash_size = hash_size or FeatureConfig.HASH_SIZE
        self.algorithms = FeatureConfig.HASH_ALGORITHMS
        self.primary_hash = FeatureConfig.PRIMARY_HASH
    
    def extract_features(self, frame: np.ndarray) -> Dict[str, str]:
        """
        Extract all configured hash features from a frame.
        
        Args:
            frame: RGB frame as numpy array
            
        Returns:
            Dictionary mapping hash algorithm names to hash hex strings
        """
        # Convert numpy array to PIL Image
        pil_image = Image.fromarray(frame.astype('uint8'))
        
        features = {}
        
        for algo in self.algorithms:
            hash_value = self._compute_hash(pil_image, algo)
            if hash_value is not None:
                features[algo] = str(hash_value)
        
        return features
    
    def extract_primary_hash(self, frame: np.ndarray) -> str:
        """
        Extract only the primary hash from a frame.
        
        Args:
            frame: RGB frame as numpy array
            
        Returns:
            Primary hash as hex string
        """
        pil_image = Image.fromarray(frame.astype('uint8'))
        hash_value = self._compute_hash(pil_image, self.primary_hash)
        return str(hash_value) if hash_value else ""
    
    def _compute_hash(self, image: Image.Image, algorithm: str) -> Optional[imagehash.ImageHash]:
        """
        Compute a specific perceptual hash.
        
        Args:
            image: PIL Image
            algorithm: Hash algorithm name ('phash', 'dhash', 'whash', 'ahash')
            
        Returns:
            ImageHash object or None if algorithm not supported
        """
        try:
            if algorithm == 'phash':
                # Perceptual hash using DCT (most robust to scaling/compression)
                return imagehash.phash(image, hash_size=self.hash_size)
            
            elif algorithm == 'dhash':
                # Difference hash (fast, good for gradient detection)
                return imagehash.dhash(image, hash_size=self.hash_size)
            
            elif algorithm == 'whash':
                # Wavelet hash (good for texture detection)
                return imagehash.whash(image, hash_size=self.hash_size)
            
            elif algorithm == 'ahash':
                # Average hash (fastest, least robust)
                return imagehash.average_hash(image, hash_size=self.hash_size)
            
            else:
                return None
                
        except Exception as e:
            print(f"Error computing {algorithm}: {e}")
            return None
    
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
