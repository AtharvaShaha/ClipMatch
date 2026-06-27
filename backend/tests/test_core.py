"""
ClipMatch Test Suite
Basic tests for video processing and matching
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
from PIL import Image
import tempfile

from services.feature_extractor import FeatureExtractor
from services.video_processor import VideoProcessor


class TestFeatureExtractor:
    """Tests for the feature extraction module"""
    
    def setup_method(self):
        self.extractor = FeatureExtractor()
    
    def test_extract_features_from_frame(self):
        """Test feature extraction from a numpy array frame"""
        # Create a test frame (random colored image)
        frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        
        features = self.extractor.extract_features(frame)
        
        assert 'phash' in features
        assert 'dhash' in features
        assert 'whash' in features
        assert len(features['phash']) > 0
    
    def test_hash_similarity_identical(self):
        """Test that identical frames produce identical hashes"""
        frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        
        features1 = self.extractor.extract_features(frame)
        features2 = self.extractor.extract_features(frame)
        
        distance = self.extractor.compute_hash_distance(
            features1['phash'], 
            features2['phash']
        )
        
        assert distance == 0
    
    def test_hash_similarity_different(self):
        """Test that different frames produce different hashes"""
        frame1 = np.zeros((240, 320, 3), dtype=np.uint8)  # Black image
        frame2 = np.ones((240, 320, 3), dtype=np.uint8) * 255  # White image
        
        features1 = self.extractor.extract_features(frame1)
        features2 = self.extractor.extract_features(frame2)
        
        distance = self.extractor.compute_hash_distance(
            features1['phash'], 
            features2['phash']
        )
        
        assert distance > 0
    
    def test_similarity_score(self):
        """Test similarity score calculation"""
        frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        features = self.extractor.extract_features(frame)
        
        # Same hash should have similarity of 1.0
        similarity = self.extractor.compute_similarity(
            features['phash'], 
            features['phash']
        )
        
        assert similarity == 1.0
    
    def test_are_similar_threshold(self):
        """Test similarity threshold checking"""
        frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        features = self.extractor.extract_features(frame)
        
        # Same hash should be similar
        is_similar = self.extractor.are_similar(
            features['phash'], 
            features['phash'],
            threshold=10
        )
        
        assert is_similar is True


class TestVideoProcessor:
    """Tests for the video processor module"""
    
    def setup_method(self):
        self.processor = VideoProcessor()
    
    def test_get_average_color(self):
        """Test average color calculation"""
        # Create a solid red frame
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        frame[:, :, 0] = 255  # Red channel
        
        r, g, b, brightness = self.processor.get_average_color(frame)
        
        assert r == 255
        assert g == 0
        assert b == 0
        assert 0 <= brightness <= 1
    
    def test_preprocess_frame_dimensions(self):
        """Test that frame preprocessing produces correct dimensions"""
        # Create a large frame
        large_frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        
        processed = self.processor._preprocess_frame(large_frame)
        
        assert processed.shape[0] == self.processor.frame_height
        assert processed.shape[1] == self.processor.frame_width
        assert processed.shape[2] == 3


class TestIntegration:
    """Integration tests for the full pipeline"""
    
    def test_feature_extraction_pipeline(self):
        """Test the full feature extraction pipeline"""
        extractor = FeatureExtractor()
        processor = VideoProcessor()
        
        # Create test frames
        frames = []
        for i in range(5):
            frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
            frames.append((i, i * 1.0, frame))  # (frame_num, timestamp, frame)
        
        # Batch extract features
        results = extractor.batch_extract(frames)
        
        assert len(results) == 5
        for result in results:
            assert 'frame_number' in result
            assert 'timestamp' in result
            assert 'phash' in result
            assert 'brightness' in result


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
