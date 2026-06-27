# ClipMatch - Restricted Source Video Matching System

```
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║   ██████╗██╗     ██╗██████╗ ███╗   ███╗ █████╗ ████████╗  ║
║  ██╔════╝██║     ██║██╔══██╗████╗ ████║██╔══██╗╚══██╔══╝  ║
║  ██║     ██║     ██║██████╔╝██╔████╔██║███████║   ██║     ║
║  ██║     ██║     ██║██╔═══╝ ██║╚██╔╝██║██╔══██║   ██║     ║
║  ╚██████╗███████╗██║██║     ██║ ╚═╝ ██║██║  ██║   ██║     ║
║   ╚═════╝╚══════╝╚═╝╚═╝     ╚═╝     ╚═╝╚═╝  ╚═╝   ╚═╝     ║
║                                                           ║
║         Restricted Source Video Matching System           ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
```

## Overview

ClipMatch is a content-based video matching system designed to determine whether a short video clip (5–60 seconds) originates from any video within a predefined set of long-form reference videos (30 minutes to 4 hours).

**This is an academic/educational prototype** that operates only on a controlled dataset and does not perform global web-scale search.

## Features

- **Visual Fingerprinting**: Uses perceptual hashing (pHash, dHash, wHash) to create non-reversible visual signatures
- **Temporal Matching**: Sliding window algorithm with temporal consistency checking
- **Confidence Scoring**: ClipMatch Confidence Score (CCS) from 0-100
- **Reference Library**: Index and manage reference videos
- **Match History**: Track previous analysis results
- **CPU-Only**: Designed to run without GPU requirements

## 🛠️ Tech Stack

<table>
<tr>
<td align="center" width="50%">

### 🔙 Backend
</td>
<td align="center" width="50%">

### 🎨 Frontend
</td>
</tr>
<tr>
<td align="center">

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-07405E?style=for-the-badge&logo=sqlite&logoColor=white)

</td>
<td align="center">

![HTML5](https://img.shields.io/badge/HTML5-E34F26?style=for-the-badge&logo=html5&logoColor=white)
![CSS3](https://img.shields.io/badge/CSS3-1572B6?style=for-the-badge&logo=css3&logoColor=white)
![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black)

</td>
</tr>
</table>

### 📦 Core Dependencies

| Category | Technology | Purpose |
|----------|------------|---------|
| **Web Framework** | Flask 3.0 | REST API & static file serving |
| **Database** | SQLAlchemy + SQLite | ORM & persistent storage |
| **Video Processing** | OpenCV | Frame extraction & preprocessing |
| **Media Handling** | FFmpeg + imageio | Video decoding & format support |
| **Feature Extraction** | ImageHash | Perceptual hashing (pHash, dHash, wHash) |
| **Image Processing** | Pillow | Image manipulation & conversion |
| **Scientific Computing** | NumPy + SciPy | Numerical operations & algorithms |
| **CORS Support** | Flask-CORS | Cross-origin resource sharing |

### 🔧 Development Tools

- **Testing**: pytest
- **Code Style**: flake8, black
- **Version Control**: Git

## Project Structure

```
Clipmatch/
├── backend/
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py          # API endpoints
│   ├── models/
│   │   ├── __init__.py
│   │   └── database.py        # SQLAlchemy models
│   ├── services/
│   │   ├── __init__.py
│   │   ├── video_processor.py # Frame extraction
│   │   ├── feature_extractor.py # Perceptual hashing
│   │   ├── matcher.py         # Matching algorithm
│   │   └── indexer.py         # Reference indexing
│   ├── app.py                 # Flask application
│   ├── config.py              # Configuration
│   └── requirements.txt       # Dependencies
├── frontend/
│   ├── css/
│   │   └── styles.css         # Dark investigative theme
│   ├── js/
│   │   └── app.js             # Frontend logic
│   └── index.html             # Main page
├── data/
│   ├── references/            # Reference videos
│   ├── uploads/               # Temporary uploads
│   ├── features/              # Extracted features
│   └── clipmatch.db           # SQLite database
└── README.md
```

## Installation

### Prerequisites

- Python 3.9+
- FFmpeg (for video processing)

### Setup

1. **Clone the repository**
   ```bash
   cd Clipmatch
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

4. **Install FFmpeg** (if not already installed)
   
   - **Windows**: Download from https://ffmpeg.org/download.html and add to PATH
   - **macOS**: `brew install ffmpeg`
   - **Linux**: `sudo apt install ffmpeg`

## Usage

### Starting the Server

```bash
cd backend
python app.py
```

The server will start at `http://localhost:5000`

### Using the Web Interface

1. Open `http://localhost:5000` in your browser
2. **Library Tab**: Add reference videos to build your detection library
3. **Analyze Tab**: Upload short clips to match against references
4. **History Tab**: View previous analysis results

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check |
| GET | `/api/references` | List indexed videos |
| POST | `/api/references` | Upload & index reference |
| DELETE | `/api/references/<id>` | Remove reference |
| POST | `/api/match` | Match a clip |
| GET | `/api/match/history` | Get match history |
| GET | `/api/stats` | System statistics |

### Example API Usage

**Index a Reference Video:**
```bash
curl -X POST -F "video=@podcast_episode.mp4" -F "title=Episode 42" \
     http://localhost:5000/api/references
```

**Match a Clip:**
```bash
curl -X POST -F "clip=@short_clip.mp4" \
     http://localhost:5000/api/match
```

**Response:**
```json
{
  "success": true,
  "best_match": {
    "video_title": "Episode 42",
    "confidence": 78.5,
    "confidence_label": "Strong Match",
    "timestamp_formatted": "12:34 - 12:45"
  },
  "processing_time": 3.24
}
```

## Confidence Score Interpretation

| Score | Label | Meaning |
|-------|-------|---------|
| 0-30 | Weak/No Match | Clip likely not from reference library |
| 30-60 | Possible Match | Some visual similarity detected |
| 60-80 | Strong Match | High probability of origin match |
| 80-100 | Very Strong Match | Near-certain identification |

## Configuration

Edit `backend/config.py` to adjust:

- `SAMPLE_RATE`: Frames per second to extract (default: 1)
- `HASH_SIZE`: Perceptual hash size (default: 16x16)
- `HASH_THRESHOLD`: Hamming distance for similarity (default: 12)
- `MIN_CONSECUTIVE_MATCHES`: Temporal consistency requirement

## Limitations

- Works only within predefined reference videos
- Performance degrades with heavy edits or overlays
- Does not guarantee identification of original creator
- Not suitable for web-scale deployment
- CPU-only (no GPU acceleration)

## Technical Details

### Feature Extraction

The system uses three perceptual hash algorithms:

1. **pHash (Perceptual Hash)**: DCT-based, most robust to scaling/compression
2. **dHash (Difference Hash)**: Gradient-based, fast computation
3. **wHash (Wavelet Hash)**: Texture detection using wavelets

### Matching Algorithm

1. Extract frames from query clip at configured sample rate
2. Compute perceptual hashes for each frame
3. For each reference video:
   - Compare query hashes against indexed hashes
   - Calculate Hamming distance for similarity
   - Track temporal consistency of matches
4. Aggregate scores with temporal consistency bonus
5. Return best match above confidence threshold

## Development

### Running Tests
```bash
cd backend
python -m pytest tests/
```

### Code Style
```bash
pip install flake8 black
black .
flake8 .
```

## License

This is an academic/educational prototype. Not intended for commercial use or copyright enforcement.

## Disclaimer

ClipMatch does not claim to determine copyright ownership or infringement. The confidence score is a technical similarity measure within a controlled dataset and has no legal standing.
