# 📚 ClipMatch API Reference - All Functions & Classes

## Module: `services/preprocessor.py`

### Class: `FramePreprocessor`

#### Static Methods

```python
FramePreprocessor.preprocess_frame(frame: np.ndarray) -> np.ndarray
    """
    Preprocess single frame: resize, grayscale, crop, equalize
    Input: BGR frame from OpenCV
    Output: Grayscale preprocessed frame (224x256)
    Time: 10-20ms per frame
    """

FramePreprocessor.batch_preprocess_frames(frames: List[np.ndarray]) -> List[np.ndarray]
    """
    Preprocess multiple frames efficiently
    Input: List of BGR frames
    Output: List of preprocessed grayscale frames
    Time: 10-20ms per frame
    """

FramePreprocessor.get_frame_difference_score(frame1: np.ndarray, frame2: np.ndarray) -> float
    """
    Calculate pixel difference between two frames
    Usage: For keyframe selection (high difference = informative)
    Output: Mean absolute pixel difference (0-255)
    """

FramePreprocessor.select_keyframes(frames: List[np.ndarray], 
                                   num_keyframes: int = 8,
                                   min_diff_threshold: float = 5.0) -> List[int]
    """
    Select most distinctive frames from video clip
    Returns: List of frame indices sorted chronologically
    Strategy: Frame difference scoring + top-N selection
    Time: 50-100ms for 60 frames
    """

FramePreprocessor.normalize_frame_for_correlation(frame: np.ndarray) -> np.ndarray
    """
    Normalize frame for NCC/SSIM computation
    Removes mean, scales to unit variance
    Output: Normalized frame
    """
```

---

## Module: `services/feature_extractor.py`

### Class: `FeatureExtractor`

#### Constants
```python
HASH_SIZE = 8              # 8x8 grid = 64-bit hashes
HAMMING_THRESHOLD = 10     # Max bits allowed different (out of 64)
```

#### Instance Methods

```python
__init__(hash_size: int = None)
    """Initialize feature extractor with hash size"""

extract_dual_hash(frame: np.ndarray) -> Dict[str, str]
    """
    Extract pHash and dHash from single frame
    Input: Grayscale preprocessed frame
    Output: {'phash': '...', 'dhash': '...'}
    Time: 5-10ms per frame
    """

batch_extract(frames: List[np.ndarray]) -> List[Dict[str, str]]
    """
    Extract dual hashes from multiple frames
    Time: 5-10ms per frame (parallel-friendly)
    """
```

#### Static Methods

```python
FeatureExtractor.hamming_distance(hash1_str: str, hash2_str: str) -> int
    """
    Calculate bits that differ between two hashes
    Output: 0-64 (0 = identical, 64 = opposite)
    Use: Compare individual hashes
    """

FeatureExtractor.combined_hash_distance(features1: Dict,
                                        features2: Dict,
                                        phash_weight: float = 0.6,
                                        dhash_weight: float = 0.4) -> float
    """
    Combined distance between frames using both hashes
    Output: 0.0-1.0 (0 = identical, 1 = opposite)
    Formula: 0.6 * phash_normalized + 0.4 * dhash_normalized
    Use: Frame comparison with higher robustness
    """

FeatureExtractor.is_hash_match(features1: Dict,
                              features2: Dict,
                              threshold: float = None) -> bool
    """
    Check if two frames match based on hashes
    Output: Boolean match result
    Threshold default: 10/64 = 0.156
    """
```

---

## Module: `services/verification.py`

### Class: `VerificationEngine`

#### Constants
```python
NCC_THRESHOLD = 0.85       # Strong match threshold
SSIM_THRESHOLD = 0.80      # Strong match threshold
MIN_CONSECUTIVE_MATCHES = 5
```

#### Static Methods

```python
VerificationEngine.normalized_cross_correlation(template: np.ndarray,
                                                image: np.ndarray) -> float
    """
    NCC between two frame patches
    Output: -1.0 to +1.0 (typically 0 to 1)
    Robust to: Brightness changes
    Time: 1-2ms per frame pair
    Use: Pixel-level similarity measurement
    """

VerificationEngine.structural_similarity(template: np.ndarray,
                                        image: np.ndarray) -> float
    """
    SSIM between two frame patches
    Output: -1.0 to +1.0 (typically 0 to 1)
    Robust to: Luminance, contrast variations
    Time: 2-3ms per frame pair
    Use: Perceptually meaningful similarity
    """

VerificationEngine.verify_frame_match(clip_frame: np.ndarray,
                                     reference_frame: np.ndarray,
                                     ncc_weight: float = 0.5,
                                     ssim_weight: float = 0.5) -> Dict
    """
    Verify single frame match
    Output: {'ncc': 0-1, 'ssim': 0-1, 'combined': 0-1}
    Combined: Weighted average of NCC and SSIM
    """

VerificationEngine.verify_frame_sequence(clip_frames: List[np.ndarray],
                                        reference_frames: List[np.ndarray],
                                        timestamp_tolerance: int = 1) -> Dict
    """
    Verify entire sequence of clip frames
    Output: {
        'success': bool,
        'matches': int,
        'total': int,
        'match_percentage': float,
        'avg_ncc': float,
        'avg_ssim': float,
        'avg_combined': float,
        'individual_scores': List[Dict]
    }
    Time: 100-200ms for 15-20 frames
    """
```

---

## Module: `services/advanced_matcher.py`

### Class: `AdvancedClipMatcher`

#### Instance Methods

```python
__init__()
    """Initialize with all required components"""

preprocess_and_extract_frames(video_path: str,
                             sample_rate: float = 4.0) -> Dict
    """
    Extract, preprocess, and hash all frames
    Input: Path to video file
    Output: {
        'success': bool,
        'raw_frames': List,
        'preprocessed_frames': List,
        'features': List[Dict],
        'frame_count': int
    }
    Time: 100-500ms total
    """

select_query_keyframes(clip_frames: List[np.ndarray],
                      num_keyframes: int = 8) -> Dict
    """
    Select distinctive keyframes from clip
    Output: {
        'keyframe_indices': List[int],
        'keyframe_frames': List[np.ndarray],
        'keyframe_hashes': List[Dict],
        'num_keyframes': int
    }
    """

early_rejection_filter(clip_hashes: List[Dict],
                      ref_video_features: List[Dict],
                      num_check: int = 3) -> List[int]
    """
    Quick filter using first N keyframes
    Output: List of candidate window starting positions
    Efficiency: Eliminates ~98% of positions
    """

anchor_sequential_matching(clip_hashes: List[Dict],
                          ref_video_features: List[Dict],
                          anchor_position: int,
                          timestamp_tolerance: int = 1) -> Dict
    """
    Verify sequential matching from anchor
    Output: {
        'anchor_position': int,
        'matches': int,
        'total': int,
        'match_score': float,
        'match_positions': List,
        'distances': List[float]
    }
    Time: 1-2ms per window
    """

find_candidates_in_reference(clip_data: Dict,
                            ref_video: ReferenceVideo,
                            ref_features: List[Dict]) -> List[Dict]
    """
    Find all candidates in single reference video
    Output: List of scored candidate matches
    Sorted by match_score descending
    """

verify_candidate_with_ncc_ssim(clip_raw_frames: List[np.ndarray],
                              reference_raw_frames: List[np.ndarray],
                              candidate_start_frame: int,
                              num_frames: Optional[int] = None) -> Dict
    """
    Pixel-level verification of candidate
    Output: Dict with NCC, SSIM, and combined scores
    """

compute_final_confidence(hash_match_score: float,
                        ncc_score: float,
                        ssim_score: float,
                        hash_weight: float = 0.4,
                        ncc_weight: float = 0.35,
                        ssim_weight: float = 0.25) -> float
    """
    Calculate final 0-100% confidence
    Formula: (0.4 * hash + 0.35 * ncc + 0.25 * ssim) * 100
    """

match_clip(clip_path: str,
          verbose: bool = False) -> Dict
    """
    Complete end-to-end matching pipeline
    Input: Path to query clip video
    Output: {
        'success': bool,
        'matches': List[Dict],  # Sorted by confidence
        'processing_time': float,
        'clip_keyframes': int,
        'reference_videos_searched': int
    }
    Time: 2-30 seconds depending on database size
    
    Each match in 'matches':
    {
        'video_id': int,
        'video_filename': str,
        'timestamp': float,     # In seconds
        'match_score': float,   # 0-1
        'ncc_score': float,     # 0-1
        'ssim_score': float,    # 0-1
        'confidence': float,    # 0-100
        'matches': int,         # Keyframes matched
        'total_keyframes': int
    }
    """
```

#### Module-level Instances

```python
advanced_matcher = AdvancedClipMatcher()
    """Global instance for use throughout application"""
```

---

## Module: `services/feature_extractor.py` (Updated)

### New Properties Added

```python
# Algorithm names (formerly HASH_ALGORITHMS)
algorithms = ['phash', 'dhash']       # Dual hashing

# Primary algorithm
primary_hash = 'phash'

# Hash size
hash_size = 8  # 8x8 = 64 bits
```

---

## Key Data Structures

### Frame Features Dictionary
```python
{
    'phash': '0xa1b2c3d4e5f6a7b8',  # 64-bit pHash as hex
    'dhash': '0xf8e7d6c5b4a39291'     # 64-bit dHash as hex
}
```

### Match Result Dictionary
```python
{
    'video_id': 42,
    'video_filename': 'lecture_python.mp4',
    'timestamp': 324.5,                    # In seconds
    'match_score': 0.875,                  # 7/8 keyframes
    'ncc_score': 0.92,                     # Normalized 0-1
    'ssim_score': 0.88,                    # Normalized 0-1
    'confidence': 89.2,                    # 0-100%
    'matches': 7,                          # Keyframes matched
    'total_keyframes': 8
}
```

### Processing Result Dictionary
```python
{
    'success': True,
    'raw_frames': [...],                   # List of BGR frames
    'preprocessed_frames': [...],          # List of grayscale frames
    'features': [                          # List of feature dicts
        {'phash': '...', 'dhash': '...'},
        ...
    ],
    'frame_count': 60
}
```

---

## Typical Usage Patterns

### Pattern 1: Simple Match Query
```python
from backend.services.advanced_matcher import advanced_matcher

result = advanced_matcher.match_clip('path/to/clip.mp4')
if result['success'] and result['matches']:
    best_match = result['matches'][0]
    print(f"Match found: {best_match['video_filename']}")
    print(f"Confidence: {best_match['confidence']:.1f}%")
```

### Pattern 2: Batch Processing
```python
from pathlib import Path

for clip_file in Path('clips').glob('*.mp4'):
    result = advanced_matcher.match_clip(str(clip_file))
    # Store result...
```

### Pattern 3: Frame-level Operations
```python
from backend.services.preprocessor import FramePreprocessor
from backend.services.feature_extractor import FeatureExtractor

# Get frames
frames = [...]  # From video

# Preprocess
preprocessed = FramePreprocessor.batch_preprocess_frames(frames)

# Extract features
extractor = FeatureExtractor()
features = extractor.batch_extract(preprocessed)

# Compare
for f1, f2 in zip(features[:-1], features[1:]):
    distance = FeatureExtractor.combined_hash_distance(f1, f2)
```

---

## Performance Characteristics

### Time per Operation (ms)
```
preprocess_frame:                     10-20 ms
select_keyframes (50 frames):          50-100 ms
extract_dual_hash:                     5-10 ms
hamming_distance:                      0.1 ms
combined_hash_distance:                0.5 ms
normalized_cross_correlation:          1-2 ms
structural_similarity:                 2-3 ms
verify_frame_sequence (15 frames):     100-200 ms
```

### Memory Usage
```
Per frame:
  - Raw frame (720p):           3.1 MB
  - Preprocessed frame:         0.06 MB
  - Feature dict:               0.1 KB

For 1-hour video at 4fps:
  - Raw: ~400 MB
  - Preprocessed: ~8 MB
  - Features database: ~0.4 MB
```

---

## Error Handling

### Common Exceptions

```python
FileNotFoundError
    """Video file not found"""
    Handle: Verify file path exists

ValueError
    """Cannot open video file"""
    Handle: Check file format is supported

RuntimeError
    """OpenCV/FFmpeg error"""
    Handle: Reinstall opencv-python, check FFmpeg PATH

MemoryError
    """Out of memory during processing"""
    Handle: Process in smaller chunks, increase RAM
```

---

## Configuration & Tuning

### Key Thresholds

```python
# In FeatureExtractor
HAMMING_THRESHOLD = 10        # 0-64 bits
HASH_SIZE = 8                 # 8x8 = 64 bits

# In AdvancedClipMatcher
MIN_WINDOW_MATCHES = 7        # 0-8 keyframes
EARLY_REJECTION_FRAMES = 3    # 1-8 keyframes
VERIFICATION_MIN_CONFIDENCE = 0.85

# In VerificationEngine
NCC_THRESHOLD = 0.85          # 0-1 range
SSIM_THRESHOLD = 0.80         # 0-1 range

# In preprocessing
FramePreprocessor.PREPROCESS_WIDTH = 256
FramePreprocessor.PREPROCESS_HEIGHT = 256
CROP_BOTTOM_PERCENT = 0.12    # 12% = ~31 pixels
```

---

## Integration Points

### With Database
```python
# Store features
video_record = ReferenceVideo(...)
frame_record = VideoFrame(
    video_id=video_record.id,
    phash=features['phash'],
    dhash=features['dhash'],
    ...
)
```

### With API
```python
# Flask route handler
@app.route('/api/match', methods=['POST'])
def match():
    clip_file = request.files['clip']
    result = advanced_matcher.match_clip(clip_file_path)
    return jsonify(result)
```

---

## Debugging & Logging

### Enable Verbose Output
```python
result = advanced_matcher.match_clip(
    'path/to/clip.mp4',
    verbose=True  # Prints progress information
)
```

### Check Individual Scores
```python
result = advanced_matcher.match_clip('clip.mp4')
for match in result['matches']:
    print(f"{match['video_filename']}: {match['confidence']}%")
    print(f"  Hash: {match['match_score']*100:.1f}%")
    print(f"  NCC: {match['ncc_score']:.3f}")
    print(f"  SSIM: {match['ssim_score']:.3f}")
```

---

**This completes the API reference for ClipMatch v1.0**
