"""
Batch index all videos from Database/ folder that are not yet indexed.
Reads each .mp4, extracts frames at 2fps, computes pHash/dHash/wHash,
and stores in the SQLite database.

Usage:
    python backend/index_database_videos.py
"""
import sys
import os
import time

# Ensure backend is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

# Force UTF-8 output
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# Suppress OpenCV warnings
os.environ['OPENCV_VIDEOIO_DEBUG'] = '0'
os.environ['OPENCV_LOG_LEVEL'] = 'OFF'

from config import VideoConfig
from models.database import get_session, ReferenceVideo, VideoFrame, init_db
from services.video_processor import VideoProcessor
from services.feature_extractor import FeatureExtractor

# Path to the Database folder (contains all 181 videos)
DATABASE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Database')

# Also check 'new db' folder
NEW_DB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'new db')


def get_video_files(directory):
    """Get all video files from a directory."""
    files = []
    if not os.path.exists(directory):
        return files
    for f in sorted(os.listdir(directory)):
        ext = os.path.splitext(f)[1].lower()
        if ext in VideoConfig.SUPPORTED_FORMATS:
            files.append(os.path.join(directory, f))
    return files


def index_single_video(filepath, session, vp, fe, video_num, total):
    """Index a single video file."""
    filename = os.path.basename(filepath)
    
    # Check if already indexed
    existing = session.query(ReferenceVideo).filter(
        ReferenceVideo.filename == filename
    ).first()
    
    if existing:
        return 'skipped', f'Already indexed (ID={existing.id})'
    
    try:
        # Get video info
        video_info = vp.get_video_info(filepath)
        
        # Check minimum duration
        if video_info['duration'] < 5:
            return 'skipped', f'Too short ({video_info["duration"]:.1f}s)'
        
        # Create database record
        video_record = ReferenceVideo(
            filename=video_info['filename'],
            filepath=video_info['filepath'],
            title=video_info['filename'],
            duration=video_info['duration'],
            frame_count=video_info['frame_count'],
            width=video_info['width'],
            height=video_info['height'],
            fps=video_info['fps'],
            file_size=video_info['file_size'],
            checksum=video_info['checksum'],
            status='indexing'
        )
        
        session.add(video_record)
        session.commit()
        video_id = video_record.id
        
        # Extract and process frames
        frames_indexed = 0
        frame_records = []
        
        for frame_num, timestamp, frame in vp.extract_frames(filepath):
            features = fe.extract_features(frame)
            avg_r, avg_g, avg_b, brightness = vp.get_average_color(frame)
            
            frame_record = VideoFrame(
                video_id=video_id,
                frame_number=frame_num,
                timestamp=timestamp,
                phash=features.get('phash', ''),
                dhash=features.get('dhash', ''),
                whash=features.get('whash', ''),
                avg_color_r=avg_r,
                avg_color_g=avg_g,
                avg_color_b=avg_b,
                brightness=brightness
            )
            
            frame_records.append(frame_record)
            frames_indexed += 1
            
            # Batch insert every 500 frames
            if len(frame_records) >= 500:
                session.bulk_save_objects(frame_records)
                session.commit()
                frame_records = []
        
        # Insert remaining frames
        if frame_records:
            session.bulk_save_objects(frame_records)
            session.commit()
        
        # Update video status
        video_record.status = 'indexed'
        video_record.frame_count = frames_indexed
        session.commit()
        
        return 'indexed', f'{frames_indexed} frames ({video_info["duration"]:.0f}s)'
        
    except Exception as e:
        session.rollback()
        try:
            bad = session.query(ReferenceVideo).filter(
                ReferenceVideo.filename == filename
            ).first()
            if bad:
                session.delete(bad)
                session.commit()
        except Exception:
            session.rollback()
        return 'failed', str(e)[:100]


def main():
    print("=" * 70)
    print("  CLIPMATCH - Batch Video Indexer")
    print("=" * 70)
    
    init_db()
    session = get_session()
    vp = VideoProcessor()
    fe = FeatureExtractor()
    
    # Collect all video files
    all_files = []
    
    if os.path.exists(DATABASE_DIR):
        db_files = get_video_files(DATABASE_DIR)
        print(f"\n  Database/ folder: {len(db_files)} video files")
        all_files.extend(db_files)
    else:
        print(f"\n  Database/ folder NOT FOUND at {DATABASE_DIR}")
    
    if os.path.exists(NEW_DB_DIR):
        new_files = get_video_files(NEW_DB_DIR)
        print(f"  new db/ folder:   {len(new_files)} video files")
        all_files.extend(new_files)
    
    if not all_files:
        print("\n  No video files found. Exiting.")
        return
    
    # Check current state
    indexed_count = session.query(ReferenceVideo).filter(
        ReferenceVideo.status == 'indexed'
    ).count()
    print(f"\n  Already indexed:  {indexed_count} videos")
    print(f"  Total to process: {len(all_files)} files")
    print("=" * 70)
    
    # Process each video
    stats = {'indexed': 0, 'skipped': 0, 'failed': 0}
    start_time = time.time()
    
    for i, filepath in enumerate(all_files, 1):
        filename = os.path.basename(filepath)
        # Truncate display name for readability
        display_name = filename[:55] + '...' if len(filename) > 58 else filename
        
        t0 = time.time()
        status, message = index_single_video(filepath, session, vp, fe, i, len(all_files))
        elapsed = time.time() - t0
        
        stats[status] = stats.get(status, 0) + 1
        
        icon = {'indexed': '[+]', 'skipped': '[=]', 'failed': '[!]'}[status]
        print(f"  {icon} [{i:3d}/{len(all_files)}] {display_name}")
        if status != 'skipped':
            print(f"      {status}: {message} ({elapsed:.1f}s)")
    
    # Final summary
    total_time = time.time() - start_time
    final_count = session.query(ReferenceVideo).filter(
        ReferenceVideo.status == 'indexed'
    ).count()
    total_frames = session.query(VideoFrame).count()
    
    print("\n" + "=" * 70)
    print(f"  RESULTS:")
    print(f"    Indexed:  {stats['indexed']}")
    print(f"    Skipped:  {stats['skipped']}")
    print(f"    Failed:   {stats['failed']}")
    print(f"    Total indexed videos: {final_count}")
    print(f"    Total frames in DB:   {total_frames}")
    print(f"    Time: {total_time:.1f}s ({total_time/60:.1f}m)")
    print("=" * 70)
    
    session.close()
    
    # Invalidate matcher cache
    try:
        from services.matcher import invalidate_ref_cache
        invalidate_ref_cache()
    except Exception:
        pass


if __name__ == '__main__':
    main()
