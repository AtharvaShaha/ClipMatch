"""
ClipMatch — Index Cached Videos

Downloads 'pending' videos from Cloudinary to the local cache,
then extracts frames and computes perceptual hashes for matching.

Usage:
    python index_cached_videos.py               # Index all pending videos
    python index_cached_videos.py --limit 10     # Index only first 10 pending
"""
import sys
import os
import time
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['OPENCV_VIDEOIO_DEBUG'] = '0'
os.environ['OPENCV_LOG_LEVEL'] = 'OFF'
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from config import CACHE_DIR


def main():
    parser = argparse.ArgumentParser(description='Index cached videos for ClipMatch')
    parser.add_argument('--limit', type=int, default=0,
                        help='Limit number of videos to process (0 = all)')
    args = parser.parse_args()

    from models.database import init_db, get_session, ReferenceVideo, VideoFrame
    from services.video_processor import VideoProcessor
    from services.feature_extractor import FeatureExtractor
    from services.cloudinary_service import get_or_download_video

    init_db()
    session = get_session()
    vp = VideoProcessor()
    fe = FeatureExtractor()

    print("=" * 64)
    print("  CLIPMATCH - Index Cached Videos")
    print("=" * 64)

    # Get pending videos
    query = session.query(ReferenceVideo).filter(
        ReferenceVideo.status == 'pending'
    ).order_by(ReferenceVideo.source_name)

    if args.limit > 0:
        query = query.limit(args.limit)

    pending = query.all()

    if not pending:
        indexed = session.query(ReferenceVideo).filter(
            ReferenceVideo.status == 'indexed'
        ).count()
        print(f"\n  No pending videos. {indexed} already indexed.")
        session.close()
        return

    print(f"\n  Pending videos: {len(pending)}")
    print(f"  Cache dir: {CACHE_DIR}\n")

    stats = {'indexed': 0, 'failed': 0, 'download_failed': 0}
    start_time = time.time()

    for i, video in enumerate(pending, 1):
        source = video.source_name or video.filename
        print(f"  [{i:3d}/{len(pending)}] {source}...", end=" ", flush=True)

        t0 = time.time()

        # Step 1: Get or download video to cache
        local_path = get_or_download_video(video.id)

        if not local_path or not os.path.exists(local_path):
            print(f"DOWNLOAD FAILED")
            video.status = 'error'
            session.commit()
            stats['download_failed'] += 1
            continue

        # Step 2: Get video info
        try:
            video_info = vp.get_video_info(local_path)
        except Exception as e:
            print(f"INFO FAILED: {e}")
            video.status = 'error'
            session.commit()
            stats['failed'] += 1
            continue

        # Update video metadata from actual file
        video.duration = video_info.get('duration', video.duration)
        video.width = video_info.get('width', video.width)
        video.height = video_info.get('height', video.height)
        video.fps = video_info.get('fps', video.fps)
        video.filepath = local_path
        video.status = 'indexing'
        session.commit()

        # Step 3: Extract frames and compute hashes
        try:
            frames_indexed = 0
            frame_records = []

            for frame_num, timestamp, frame in vp.extract_frames(local_path):
                features = fe.extract_features(frame)
                avg_r, avg_g, avg_b, brightness = vp.get_average_color(frame)

                frame_record = VideoFrame(
                    video_id=video.id,
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

            # Insert remaining
            if frame_records:
                session.bulk_save_objects(frame_records)
                session.commit()

            # Update status
            video.status = 'indexed'
            video.frame_count = frames_indexed
            session.commit()

            elapsed = time.time() - t0
            print(f"{frames_indexed} frames ({elapsed:.1f}s)")
            stats['indexed'] += 1

        except Exception as e:
            session.rollback()
            video.status = 'error'
            session.commit()
            print(f"FAILED: {str(e)[:60]}")
            stats['failed'] += 1

    # Final summary
    total_time = time.time() - start_time
    total_indexed = session.query(ReferenceVideo).filter(
        ReferenceVideo.status == 'indexed'
    ).count()
    total_frames = session.query(VideoFrame).count()
    session.close()

    # Invalidate matcher cache
    try:
        from services.matcher import invalidate_ref_cache
        invalidate_ref_cache()
    except Exception:
        pass

    print(f"\n" + "=" * 64)
    print(f"  RESULTS:")
    print(f"    Indexed:           {stats['indexed']}")
    print(f"    Download failed:   {stats['download_failed']}")
    print(f"    Processing failed: {stats['failed']}")
    print(f"    Total indexed:     {total_indexed} videos")
    print(f"    Total frames:      {total_frames}")
    print(f"    Time:              {total_time:.1f}s ({total_time/60:.1f}m)")
    print("=" * 64)


if __name__ == '__main__':
    main()
