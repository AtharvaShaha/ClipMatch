#!/usr/bin/env python3
"""Verify video indexing and debug matcher"""

import sqlite3
import sys

conn = sqlite3.connect('data/clipmatch.db')
cursor = conn.cursor()

print("=" * 60)
print("CLIPMATCH DATABASE STATUS")
print("=" * 60)

# Check reference videos
print("\n1. REFERENCE VIDEOS:")
cursor.execute("""
    SELECT id, filename, title, status, frame_count, duration
    FROM reference_videos
    ORDER BY id
""")
videos = cursor.fetchall()

if videos:
    for vid_id, filename, title, status, frame_count, duration in videos:
        print(f"   ID {vid_id}: {filename}")
        print(f"      Title: {title}")
        print(f"      Status: {status}")
        print(f"      Frames: {frame_count}, Duration: {duration:.1f}s")
else:
    print("   ⚠ NO REFERENCE VIDEOS INDEXED")

# Check frames per video
print("\n2. FRAME COUNTS:")
cursor.execute("""
    SELECT video_id, COUNT(*) as frame_count, 
           MIN(timestamp) as first_ts, MAX(timestamp) as last_ts
    FROM video_frames
    GROUP BY video_id
""")
frames = cursor.fetchall()

if frames:
    for video_id, count, first_ts, last_ts in frames:
        print(f"   Video {video_id}: {count} frames ({first_ts:.1f}s - {last_ts:.1f}s)")
else:
    print("   ⚠ NO FRAMES INDEXED")

# Check for None hashes
print("\n3. HASH EXTRACTION QUALITY:")
cursor.execute("""
    SELECT 
        SUM(CASE WHEN phash IS NOT NULL AND phash != '' THEN 1 ELSE 0 END) as phash_count,
        SUM(CASE WHEN dhash IS NOT NULL AND dhash != '' THEN 1 ELSE 0 END) as dhash_count,
        SUM(CASE WHEN whash IS NOT NULL AND whash != '' THEN 1 ELSE 0 END) as whash_count,
        COUNT(*) as total_frames
    FROM video_frames
""")
result = cursor.fetchone()
if result:
    phash_count, dhash_count, whash_count, total = result
    print(f"   Total frames: {total}")
    print(f"   pHash extracted: {phash_count}/{total} ({100*phash_count//total if total else 0}%)")
    print(f"   dHash extracted: {dhash_count}/{total} ({100*dhash_count//total if total else 0}%)")
    print(f"   wHash extracted: {whash_count}/{total} ({100*whash_count//total if total else 0}%)")

# Check match history
print("\n4. MATCH HISTORY:")
cursor.execute("""
    SELECT COUNT(*), AVG(confidence_score), MAX(confidence_score)
    FROM match_results
""")
result = cursor.fetchone()
if result:
    count, avg_conf, max_conf = result
    if count > 0:
        print(f"   Total matches attempted: {count}")
        print(f"   Average confidence: {avg_conf:.1f}%")
        print(f"   Best confidence: {max_conf:.1f}%")
    else:
        print("   No match history yet")
else:
    print("   Match history table empty")

print("\n" + "=" * 60)

conn.close()
