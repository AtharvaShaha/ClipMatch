"""
ClipMatch Configuration Module
Centralized configuration for all system parameters
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
REFERENCES_DIR = DATA_DIR / "references"
UPLOADS_DIR = DATA_DIR / "uploads"
FEATURES_DIR = DATA_DIR / "features"
DATABASE_PATH = DATA_DIR / "clipmatch.db"

# Ensure directories exist
for directory in [DATA_DIR, REFERENCES_DIR, UPLOADS_DIR, FEATURES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Video Processing Configuration
class VideoConfig:
    # Frame sampling rate (frames per second to extract)
    SAMPLE_RATE = 2  # 2 frames per second (optimal for hash-based matching)
    
    # Supported video formats
    SUPPORTED_FORMATS = ['.mp4', '.avi', '.mkv', '.mov', '.webm', '.flv']
    
    # Maximum file sizes (in bytes)
    MAX_REFERENCE_SIZE = 10 * 1024 * 1024 * 1024  # 10 GB
    MAX_CLIP_SIZE = 500 * 1024 * 1024  # 500 MB
    
    # Clip duration limits (in seconds)
    MIN_CLIP_DURATION = 5
    MAX_CLIP_DURATION = 60
    
    # Reference video duration limits (in seconds)
    MIN_REFERENCE_DURATION = 30 * 60  # 30 minutes
    MAX_REFERENCE_DURATION = 10 * 60 * 60  # 10 hours
    
    # Frame resize dimensions for processing
    FRAME_WIDTH = 320
    FRAME_HEIGHT = 240


# Feature Extraction Configuration
class FeatureConfig:
    # Hash size for perceptual hashing
    # Must match FeatureExtractor.HASH_SIZE (8x8 = 64 bits)
    HASH_SIZE = 8  # 8x8 = 64 bits
    
    # Hash algorithms to use
    HASH_ALGORITHMS = ['phash', 'dhash', 'whash']
    
    # Primary hash algorithm
    PRIMARY_HASH = 'phash'
    
    # Feature vector dimensions
    FEATURE_DIM = 64


# Matching Configuration
class MatchConfig:
    # Sliding window size (in frames)
    WINDOW_SIZE = 10
    
    # Window stride (in frames)
    WINDOW_STRIDE = 5
    
    # Hamming distance threshold for hash similarity (higher = more lenient)
    # For 16x16 hash, max distance is 256. 
    # 128 allows ~50% bit difference (very lenient for re-encoded/reformatted videos)
    HASH_THRESHOLD = 128
    
    # Minimum consecutive matching frames for temporal consistency
    MIN_CONSECUTIVE_MATCHES = 2
    
    # Confidence score thresholds
    WEAK_MATCH = 10
    POSSIBLE_MATCH = 25
    STRONG_MATCH = 50
    VERY_STRONG_MATCH = 75
    
    # Minimum confidence to report a match
    # RELAXED threshold: Return matches at 45%+ confidence
    # This allows videos in database to match while filtering low-confidence false positives
    MIN_CONFIDENCE_THRESHOLD = 45  # Relaxed - catches videos in database while filtering obvious mismatches


# Database Configuration
class DatabaseConfig:
    SQLALCHEMY_DATABASE_URI = f"sqlite:///{DATABASE_PATH}"
    SQLALCHEMY_TRACK_MODIFICATIONS = False


# Flask Configuration
class FlaskConfig:
    DEBUG = True
    SECRET_KEY = os.environ.get('SECRET_KEY', 'clipmatch-dev-secret-key-change-in-prod')
    MAX_CONTENT_LENGTH = 10 * 1024 * 1024 * 1024  # 10 GB max upload (for large reference videos)
    UPLOAD_FOLDER = str(UPLOADS_DIR)
    
    # CORS settings
    CORS_ORIGINS = ['http://localhost:3000', 'http://127.0.0.1:3000', 
                    'http://localhost:5500', 'http://127.0.0.1:5500']
