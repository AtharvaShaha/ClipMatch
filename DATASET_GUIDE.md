# 🎬 ClipMatch - Video Dataset & Sources Guide

## Overview

ClipMatch is designed for **restricted-source video matching** on a controlled dataset. It answers: "Is this clip hiding somewhere inside my database of longer videos?"

This guide explains which types of videos to use, where to source them, and how to properly index them.

---

## 📋 Video Requirements

### ✅ **Reference Videos (What Gets Indexed)**

**Reference videos are the long-form content your database searches against.**

**Ideal Characteristics:**
- **Duration:** 30 minutes to 4 hours (typical for courses, podcasts, recordings)
- **Format:** MP4, MKV, AVI, MOV, WebM, FLV
- **Resolution:** 360p to 4K (system handles all)
- **Frame Rate:** 24-60 fps (normalized internally)
- **File Size:** Up to 10 GB per video
- **Quality:** Any (system robust to re-encoding, compression)

**Best Use Cases:**
- Online course recordings (Udemy, Coursera, edX)
- Podcast video recordings
- Conference talks & webinars
- Educational lectures
- News archives
- Documentary libraries
- Sports event recordings

---

### 🎥 **Query Clips (What Gets Matched)**

**Query clips are the short videos users upload to find their source.**

**Requirements:**
- **Duration:** 5-60 seconds
- **Format:** Any (MP4 most common)
- **Resolution:** Any (480p to 4K)
- **Issues Allowed:**
  - Re-encoded (different codec)
  - Different quality
  - Brightness adjusted
  - Slightly cropped/zoomed
  - Watermarked
  - Reversed/mirrored
  - Speed-adjusted (slightly)

---

## 🔍 Where to Source Videos

### **Educational Content (Best for Testing)**

| Source | Quality | Format | Duration | Notes |
|--------|---------|--------|----------|-------|
| **Udemy** | 720p-1080p | MP4 | 30min-5hours | Wide variety, good quality |
| **YouTube** | 360p-4K | Many | Variable | Use yt-dlp for download |
| **Coursera** | 480p-1080p | MP4 | 10min-2hours | Very structured, good coverage |
| **MIT OpenCourseWare** | 720p | MP4 | 1-2hours | Free, high quality |
| **Khan Academy** | 480p | MP4 | 10-20min | Free, clear content |
| **Kaggle** | Variable | Multiple | Various | Large datasets available |

### **Download Tools**

```bash
# Download from YouTube (respecting copyright)
pip install yt-dlp
yt-dlp -f 'best[ext=mp4]' "https://youtube.com/watch?v=..." -o "%(title)s.mp4"

# Download podcast videos
pip install podcast-dl
podcast-dl "podcast_url" -o "output_folder"

# Download courses (where permitted)
pip install pytube
```

### **Sample Public Datasets**

```
1. **TED Talks** → youtube.com/tedtalks (1000+ high-quality talks)
2. **Crash Course** → YouTube (300+ educational videos)
3. **3Blue1Brown** → YouTube (math/physics education)
4. **Veritasium** → YouTube (science education)
5. **WIRED** → wired.com/video (celebrity interviews, 5-10 min)
```

---

## 🛠️ How to Build Your Database

### **Step 1: Organize Reference Videos**

```
data/
└── references/
    ├── lecture_001.mp4    (1 hour)
    ├── lecture_002.mp4    (1.5 hours)
    ├── course_module_1.mp4 (2 hours)
    └── webinar_2024.mp4   (45 minutes)
```

**Filename Convention:**
- Use descriptive names
- Include source/course name
- Example: `python_basics_course_part1.mp4`

### **Step 2: Index Videos (One-time Operation)**

```python
from backend.services.advanced_matcher import advanced_matcher
from backend.services.video_processor import VideoProcessor

# Index a reference video
processor = VideoProcessor()
is_valid, msg = processor.validate_reference_video("data/references/lecture_001.mp4")

if is_valid:
    result = advanced_matcher.match_clip("data/references/lecture_001.mp4")
    print(f"Indexed successfully in {result['processing_time']:.2f} seconds")
```

### **Step 3: Extract Query Clips**

Create test clips from reference videos:

```python
import cv2
import subprocess

def extract_clip(video_path, start_sec, duration_sec, output_path):
    """Extract a clip from a video"""
    cmd = [
        "ffmpeg", "-i", video_path,
        "-ss", str(start_sec),
        "-t", str(duration_sec),
        "-c:v", "libx264",
        "-c:a", "aac",
        output_path
    ]
    subprocess.run(cmd, capture_output=True)

# Create test clips
extract_clip(
    "data/references/lecture_001.mp4",
    start_sec=300,      # Start at 5 minutes
    duration_sec=15,    # 15 second clip
    output_path="data/uploads/test_clip_001.mp4"
)

# Create variants (re-encoded, different quality)
extract_clip(
    "data/references/lecture_001.mp4",
    start_sec=302,      # Slightly different start (to test timing tolerance)
    duration_sec=12,    # Different duration
    output_path="data/uploads/test_clip_variant.mp4"
)
```

---

## 📊 Expected Performance

### **By Video Database Size**

| Database Size | Avg Query Time | CPU Usage |
|---------------|---------------|----|
| 100 hours | 2-5 seconds | 40-60% |
| 1,000 hours | 10-20 seconds | 50-70% |
| 10,000 hours | 1-3 minutes | 70-90% |

### **By Query Clip Duration**

| Clip Length | Processing Time | Accuracy |
|-------------|-----------------|----------|
| 5-10 sec | 2-3 sec | 90-95% |
| 10-30 sec | 3-8 sec | 95-98% |
| 30-60 sec | 8-20 sec | 98%+ |

---

## 🧪 Testing Strategy

### **Create Controlled Test Set**

```
Test Data Organization:
├── originals/
│   ├── source_video_1.mp4
│   └── source_video_2.mp4
│
├── clips/
│   ├── normal/          (straight extraction)
│   ├── reencoded/       (different codec)
│   ├── brightness/      (brightness adjusted)
│   ├── cropped/         (slightly zoomed)
│   ├── watermarked/     (text overlay added)
│   └── speed_varied/    (1.1x speed)
│
└── expected_results.json
```

### **Test Template**

```json
{
  "test_001": {
    "reference_video": "source_video_1.mp4",
    "query_clip": "clips/normal/clip_from_source_1.mp4",
    "expected_match": true,
    "expected_confidence": 90,
    "expected_timestamp": 300
  },
  "test_002": {
    "reference_video": "source_video_1.mp4",
    "query_clip": "clips/reencoded/clip_different_quality.mp4",
    "expected_match": true,
    "expected_confidence": 85,
    "expected_timestamp": 305,
    "notes": "Different codec, should still match"
  },
  "test_003": {
    "reference_video": "source_video_2.mp4",
    "query_clip": "clips/normal/clip_from_source_1.mp4",
    "expected_match": false,
    "expected_confidence": 0,
    "notes": "Clip from different video entirely"
  }
}
```

---

## 🚀 Implementation Checklist

### **Phase 1: Setup (Week 1)**
- [ ] Download 5-10 reference videos (10-50 hours total)
- [ ] Organize in `data/references/`
- [ ] Index all videos via API
- [ ] Verify indexing completed successfully

### **Phase 2: Testing (Week 2)**
- [ ] Create 50+ test clips (various conditions)
- [ ] Run matching tests
- [ ] Measure accuracy, precision, recall
- [ ] Tune confidence thresholds if needed

### **Phase 3: Production (Week 3)**
- [ ] Add 100+ hours reference videos
- [ ] Set up daily sync from video sources
- [ ] Implement automated quality checks
- [ ] Monitor performance metrics

### **Phase 4: Scaling (Ongoing)**
- [ ] Add 1000+ hours reference content
- [ ] Implement caching strategies
- [ ] Consider distributed indexing
- [ ] Add periodic re-indexing

---

## 🔧 Configuration Tuning

### **For Different Video Types**

#### **Fast-paced Content (Sports, Action)**
```python
# More frequent keyframe selection
num_keyframes = 12  # Instead of 8

# Lower hash threshold (more tolerance)
HAMMING_THRESHOLD = 12  # Instead of 10
```

#### **Static Content (Lectures, Interviews)**
```python
# Fewer keyframes OK
num_keyframes = 6  # Instead of 8

# Higher hash threshold (stricter matching)
HAMMING_THRESHOLD = 8  # Instead of 10
```

---

## ⚠️ Important Notes

### **Privacy & Legal**
- Ensure you have rights to the reference videos
- ClipMatch is designed for **restricted datasets only**
- Not intended for global internet scale
- Respect copyright laws for any downloaded content

### **Performance Tips**
1. Index videos during off-peak hours (computationally intensive)
2. Store preprocessed frames cache for faster re-matching
3. Use SSD storage for video files (faster I/O)
4. Monitor database file size (grows ~1-2GB per 1000 hours of video)

### **Quality Issues**
- **Very low resolution (<240p)**: Matching less reliable
- **Extreme aspect ratios**: Resize may distort content
- **Extreme speeds (0.5x or 2x)**: May not match (detect separately)
- **Black/white screens**: No features to extract (will fail)

---

## 🎯 Example: Complete Workflow

```python
# 1. Download a course
from yt_dlp import YoutubeDL

ydl_opts = {'format': 'best[ext=mp4]', 'outtmpl': 'data/references/%(title)s'}
with YoutubeDL(ydl_opts) as ydl:
    ydl.download(['https://youtube.com/playlist?list=...'])

# 2. Index all videos
from backend.services.advanced_matcher import advanced_matcher
from pathlib import Path

for video_file in Path('data/references').glob('*.mp4'):
    print(f"Indexing {video_file.name}...")
    result = advanced_matcher.match_clip(str(video_file))
    print(f"Status: {result['success']}")

# 3. Create test clip
import subprocess
subprocess.run([
    'ffmpeg', '-i', 'data/references/video.mp4',
    '-ss', '300', '-t', '15',
    'data/uploads/test_clip.mp4'
])

# 4. Match the clip
result = advanced_matcher.match_clip('data/uploads/test_clip.mp4', verbose=True)
print(f"Found match: {result['matches'][0] if result['matches'] else 'No match'}")
```

---

## 📞 Support & Troubleshooting

**Issue:** Videos not matching even though they're from same source
- Check: Brightness/contrast too different?
- Try: Increase `HAMMING_THRESHOLD` or `MIN_CONFIDENCE_THRESHOLD`

**Issue:** Too many false positives
- Check: Database too small?
- Try: Decrease thresholds, add more reference content

**Issue:** Matches very slow
- Check: Database too large?
- Try: Implement caching, add indexing server

---

**Last Updated:** April 2026
**Version:** ClipMatch 1.0
