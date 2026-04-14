# ClipMatch – Video Similarity & Clip Localization System

An academic MVP system that detects whether a short query video clip (~15 seconds)
exists inside a dataset of longer videos (~1 minute each), and returns the matched
video filename, start/end timestamps, and similarity confidence score.

## Quick Start

### 1. Install Dependencies

```bash
cd ClipMatch
pip install -r requirements.txt
```

### 2. Generate Dataset (First Run)

The dataset is automatically generated when you launch the Streamlit app.
Or generate it manually:

```bash
python -m modules.dataset_downloader
```

This creates:
- `Dataset/` – 100 synthetic 1-minute videos
- `query/` – 20 query clips (15-second segments from dataset videos)
- `ground_truth.json` – mapping of query clips to source videos

### 3. Run Streamlit App

```bash
streamlit run app.py
```

### 4. CLI Usage (Optional)

```bash
python main_pipeline.py
```

## Architecture

```
ClipMatch/
├── Dataset/              # 100 synthetic dataset videos
├── query/                # 20 query clips
├── temp_frames/          # Cached extracted frames
├── temp_query_frames/    # Cached query frames
├── cache/                # Feature vector caches (.npy)
├── modules/
│   ├── dataset_downloader.py  # Synthetic video generation
│   ├── frame_extractor.py     # Frame extraction & caching
│   ├── feature_extractor.py   # Block-mean feature vectors
│   ├── ncc_matcher.py         # Normalized cross-correlation
│   ├── sliding_window.py      # Sequence-level sliding window
│   └── timestamp_utils.py     # Frame-to-timestamp conversion
├── main_pipeline.py           # Orchestration pipeline
├── app.py                     # Streamlit frontend
└── requirements.txt
```

## Algorithm Overview

### Frame Processing Pipeline

1. **Extract frames** at 16 FPS from both query and dataset videos
2. **Convert to grayscale** and **resize to 256×256**
3. **Divide each frame** into a 4×4 grid of 64×64 blocks
4. **Compute block-mean intensity** → 16-value feature vector per frame
5. **Slide query sequence** across dataset video features
6. **Compute average NCC** at each window position
7. **Scan entire video** → select highest-scoring window
8. **Convert frame indices** to timestamps

### Sliding Window Matching

The system does NOT perform single-frame matching. Instead, it uses **sequence-level
sliding window matching**:

- The query feature sequence (Q frames × 16 features) slides across the dataset
  video's feature sequence (D frames × 16 features)
- At each position, the average Normalized Cross-Correlation (NCC) is computed
  across all frame pairs in the window
- The window with the highest average NCC across the **entire** dataset video is selected
- This handles static scenes correctly because temporally consistent matches
  produce higher aggregate scores than coincidental single-frame similarities

### Static Scene Ambiguity Handling

Static scenes (lecture slides, CCTV idle footage) produce similar frames at
multiple timestamps. The system avoids mismatches by:

1. **Never stopping at the first match** – scans the complete video
2. **Sequence-level comparison** – a window of frames must all be similar, not just one
3. **Temporal consistency** – real matches show consistently high NCC across
   consecutive frames, while coincidental matches only match on a few frames
4. **Global best selection** – picks the single highest-scoring window after full scan

### Timestamp Calculation

```
timestamp = frame_index / extraction_fps
start_time = best_window_start_frame / fps
end_time = (best_window_start_frame + query_length_frames - 1) / fps
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `target_fps` | 16 | Frame extraction rate |
| `threshold` | 0.7 | Minimum NCC score for valid match |
| `step` | 2 | Sliding window step size |
| `frame_size` | 256×256 | Processed frame dimensions |
| `grid_size` | 4×4 | Feature extraction grid |

## Requirements

- Python 3.10+
- OpenCV (`opencv-python`)
- NumPy
- Streamlit
- CPU-only (no GPU required)
- No deep learning models
