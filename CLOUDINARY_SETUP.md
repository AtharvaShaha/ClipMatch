# ClipMatch — Cloudinary Setup Guide

## Overview

ClipMatch uses **Cloudinary** as remote video storage and **SQLite** as the local metadata/index database. This guide explains how to set up, sync, and maintain the dataset.

## Architecture

```
                 CLOUDINARY
                     |
          +----------+-----------+
          |                      |
    Source Videos           Query Upload
    (source_001...)              |
          |                      v
          v                Temporary File
      SQLite DB                  |
    (metadata +                  |
     frame hashes)               |
          |                      |
          +----------+-----------+
                     v
              ClipMatch Engine
                     |
          +----------+-----------+
          v                      v
    Candidate Search       Frame Matching
                              (NCC + Hash)
                                 |
                                 v
                          Match Result
```

## 1. Configure Cloudinary

### Create `.env` file

Copy `.env.example` to `.env` and fill in your Cloudinary credentials:

```bash
cp .env.example .env
```

Edit `.env`:

```env
CLOUDINARY_CLOUD_NAME=your_cloud_name
CLOUDINARY_API_KEY=your_api_key
CLOUDINARY_API_SECRET=your_api_secret
```

> **NEVER** commit the `.env` file or share your API secret.

### Where to find credentials

1. Log in to [Cloudinary Console](https://console.cloudinary.com/)
2. Go to **Settings** > **API Keys**
3. Copy Cloud Name, API Key, and API Secret

## 2. Create the Database

### Fresh setup (first time)

```bash
python setup_cloudinary_database.py --reset
```

This will:
1. Connect to Cloudinary
2. List ALL video assets (all folders)
3. Rename them to clean names: `source_001`, `source_002`, ...
4. Create a fresh SQLite database with metadata records
5. Set all videos to `pending` status (frames not yet extracted)

### Skip renaming (if videos are already named)

```bash
python setup_cloudinary_database.py --reset --no-rename
```

## 3. Index Videos (Extract Frame Hashes)

After setup, download and index videos:

```bash
python index_cached_videos.py
```

This will:
1. Download each `pending` video from Cloudinary to `cache/`
2. Extract frames at 2fps
3. Compute pHash, dHash, wHash for each frame
4. Store hashes in SQLite
5. Update status to `indexed`

### Index a limited number (for testing)

```bash
python index_cached_videos.py --limit 10
```

## 4. Synchronize New Videos

When you or a collaborator uploads new videos to Cloudinary:

```bash
python sync_cloudinary_database.py
```

This will:
1. List all Cloudinary videos
2. Compare with existing SQLite records
3. Rename new videos to next available `source_NNN`
4. Insert new records with `pending` status
5. **Never renumber existing records**

Then index the new videos:

```bash
python index_cached_videos.py
```

## 5. Collaborator Workflow

### For a collaborator to add videos:

1. Log in to the shared Cloudinary account
2. Upload videos to the Media Library (any folder)
3. Notify the database owner

### For the database owner:

```bash
python sync_cloudinary_database.py    # Detect + rename + insert
python index_cached_videos.py         # Download + index frames
python verify_dataset.py              # Verify everything is synced
```

## 6. Verify Dataset

```bash
python verify_dataset.py
```

Example output:

```
  Cloudinary videos:      170
  SQLite records:         170

  Missing from SQLite:    0
  Missing from Cloudinary:0

  Indexing status:
    Pending (no frames):  0
    Indexed (has frames): 170

  Dataset is SYNCHRONIZED.
```

## 7. Reset the Database

To rebuild the database from scratch:

```bash
python setup_cloudinary_database.py --reset
```

This will:
- **Delete** the local SQLite database
- **Never** delete Cloudinary videos
- Rebuild from Cloudinary metadata

## 8. Local Caching

Videos are cached in `cache/` to avoid repeated downloads:

```
cache/
    source_001.mp4
    source_002.mp4
    ...
```

The cache is checked before every download. To clear the cache:

```bash
# Delete all cached videos (they can be re-downloaded)
rm -rf cache/
mkdir cache/
```

## 9. Running ClipMatch

After setup + indexing:

```bash
python backend/app.py
```

Then open: **http://localhost:5000/app**

## Commands Summary

| Command | Purpose |
|---------|---------|
| `python setup_cloudinary_database.py --reset` | Create fresh DB from Cloudinary |
| `python index_cached_videos.py` | Download + extract frames + compute hashes |
| `python sync_cloudinary_database.py` | Add new Cloudinary videos to DB |
| `python verify_dataset.py` | Check Cloudinary vs SQLite consistency |
| `python backend/app.py` | Start the web server |
