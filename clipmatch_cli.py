"""
ClipMatch — Terminal CLI Tool

List all indexed videos and match query clips directly from the command line.

Usage:
    python clipmatch_cli.py list                           # List all indexed videos
    python clipmatch_cli.py match <video_path>             # Match with Original quality
    python clipmatch_cli.py match <video_path> --enhanced  # Match with Enhanced quality
    python clipmatch_cli.py info <source_name>             # Get details about a specific video
    python clipmatch_cli.py stats                          # Show database statistics
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

# Suppress noisy logs for clean terminal output
import logging
logging.basicConfig(level=logging.WARNING)


def print_header(title):
    """Print a styled header."""
    width = 70
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def print_separator():
    print("-" * 70)


def format_duration(seconds):
    """Format seconds into mm:ss or hh:mm:ss."""
    if not seconds or seconds <= 0:
        return "N/A"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def format_size(size_bytes):
    """Format bytes into human-readable size."""
    if not size_bytes:
        return "N/A"
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def cmd_list(args):
    """List all indexed reference videos."""
    from models.database import init_db, get_session, ReferenceVideo, VideoFrame
    from sqlalchemy import func

    init_db()
    session = get_session()

    videos = session.query(ReferenceVideo).order_by(ReferenceVideo.source_name).all()

    if not videos:
        print("\n  No videos indexed. Run setup_cloudinary_database.py first.")
        session.close()
        return

    print_header("CLIPMATCH - Indexed Reference Videos")
    print()

    # Table header
    print(f"  {'#':>4}  {'Source Name':<14}  {'Frames':>7}  {'Duration':>10}  {'Size':>10}  {'Status':<8}")
    print_separator()

    total_frames = 0
    total_size = 0

    for i, v in enumerate(videos, 1):
        frames = v.frame_count or 0
        total_frames += frames
        size = v.file_size or 0
        total_size += size
        status_icon = "[OK]" if v.status == 'indexed' else "[..]"

        print(f"  {i:>4}  {v.source_name or 'N/A':<14}  {frames:>7}  {format_duration(v.duration):>10}  {format_size(size):>10}  {status_icon:<8}")

    print_separator()
    print(f"  Total: {len(videos)} videos | {total_frames} frames | {format_size(total_size)}")
    print()

    # Stats
    indexed = sum(1 for v in videos if v.status == 'indexed')
    pending = sum(1 for v in videos if v.status == 'pending')
    cloud = sum(1 for v in videos if v.cloudinary_url)

    print(f"  Indexed:   {indexed}")
    print(f"  Pending:   {pending}")
    print(f"  On Cloud:  {cloud}")
    print("=" * 70)

    session.close()


def cmd_match(args):
    """Match a query video clip against the database."""
    clip_path = args.video_path

    if not os.path.exists(clip_path):
        print(f"\n  ERROR: File not found: {clip_path}")
        sys.exit(1)

    mode = "Enhanced (NCC+SSIM)" if args.enhanced else "Original (Hash-based)"

    print_header("CLIPMATCH - Video Matching")
    print(f"\n  Query video:  {os.path.basename(clip_path)}")
    print(f"  File size:    {format_size(os.path.getsize(clip_path))}")
    print(f"  Match mode:   {mode}")
    print()

    # Get video info
    from services.video_processor import VideoProcessor
    vp = VideoProcessor()
    try:
        info = vp.get_video_info(clip_path)
        dur = info.get('duration', 0)
        print(f"  Duration:     {format_duration(dur)}")
        print(f"  Resolution:   {info.get('width', '?')}x{info.get('height', '?')}")
        print(f"  FPS:          {info.get('fps', '?')}")
    except Exception:
        pass

    print()
    print("  Matching in progress...")
    print_separator()

    start_time = time.time()

    if args.enhanced:
        from services.advanced_matcher import advanced_matcher
        result = advanced_matcher.match_clip(clip_path)
    else:
        from services.matcher import clip_matcher
        result = clip_matcher.match_clip(clip_path)

    elapsed = time.time() - start_time

    print()

    if not result.get('success', False):
        print(f"  RESULT: NO MATCH FOUND")
        print(f"  Reason: {result.get('error', result.get('message', 'Unknown'))}")
        print(f"  Time:   {elapsed:.2f}s")
        print("=" * 70)
        return

    # Display results
    match_data = result.get('best_match') or result.get('match') or result
    confidence = match_data.get('confidence_score') or match_data.get('confidence') or 0.0
    
    # Get video title or source_name
    matched_video_dict = match_data.get('matched_video') or {}
    video_name = matched_video_dict.get('source_name') or \
                 matched_video_dict.get('title') or \
                 match_data.get('video_title') or \
                 match_data.get('video_filename') or 'Unknown'
                 
    timestamp = match_data.get('timestamp_range') or \
                match_data.get('timestamp_formatted') or \
                format_timestamp_range(match_data.get('start_timestamp'), match_data.get('end_timestamp'))

    # Confidence label
    if confidence >= 80:
        label = "VERY STRONG MATCH"
        bar = "##########"
    elif confidence >= 60:
        label = "STRONG MATCH"
        bar = "########.."
    elif confidence >= 30:
        label = "POSSIBLE MATCH"
        bar = "#####....."
    else:
        label = "WEAK/NO MATCH"
        bar = "##........"

    print(f"  +{'=' * 50}+")
    print(f"  |{'MATCH RESULT':^50}|")
    print(f"  +{'=' * 50}+")
    print(f"  |                                                  |")
    print(f"  |  Confidence:   {confidence:>6.1f}%  [{bar}]         |")
    print(f"  |  Verdict:      {label:<34}|")
    print(f"  |                                                  |")
    print(f"  |  Found in:     {video_name:<34}|")
    print(f"  |  Timestamp:    {timestamp:<34}|")
    print(f"  |  Match mode:   {mode:<34}|")
    print(f"  |  Process time: {elapsed:>5.2f}s{' ' * 29}|")
    print(f"  |                                                  |")

    # Extra details for enhanced mode
    ncc = match_data.get('ncc_score', None)
    ssim = match_data.get('ssim_score', None)
    if ncc is not None:
        print(f"  |  NCC Score:    {ncc:>6.3f}{' ' * 30}|")
    if ssim is not None:
        print(f"  |  SSIM Score:   {ssim:>6.3f}{' ' * 30}|")

    print(f"  +{'=' * 50}+")
    print()

    # Video present confirmation
    print(f"  VIDEO PRESENT IN DATABASE: YES")
    print(f"  Matched reference: {video_name}")
    if match_data.get('matched_video', {}).get('cloudinary_url'):
        print(f"  Cloud URL: {match_data['matched_video']['cloudinary_url'][:60]}...")

    print("=" * 70)


def cmd_info(args):
    """Get details about a specific source video."""
    from models.database import init_db, get_session, ReferenceVideo, VideoFrame

    init_db()
    session = get_session()

    name = args.source_name
    video = session.query(ReferenceVideo).filter(
        (ReferenceVideo.source_name == name) |
        (ReferenceVideo.filename == name) |
        (ReferenceVideo.filename == name + '.mp4')
    ).first()

    if not video:
        # Try partial match
        video = session.query(ReferenceVideo).filter(
            ReferenceVideo.source_name.like(f'%{name}%')
        ).first()

    if not video:
        print(f"\n  Video '{name}' not found in database.")
        session.close()
        return

    frames = session.query(VideoFrame).filter(VideoFrame.video_id == video.id).count()

    print_header(f"Video Details: {video.source_name}")
    print()
    print(f"  Source Name:    {video.source_name}")
    print(f"  Filename:       {video.filename}")
    print(f"  Database ID:    {video.id}")
    print(f"  Duration:       {format_duration(video.duration)}")
    print(f"  Resolution:     {video.width or '?'}x{video.height or '?'}")
    print(f"  FPS:            {video.fps or 'N/A'}")
    print(f"  File Size:      {format_size(video.file_size)}")
    print(f"  Frame Count:    {frames}")
    print(f"  Status:         {video.status}")
    print(f"  Indexed At:     {video.indexed_at}")
    print_separator()
    print(f"  Cloudinary ID:  {video.cloudinary_public_id or 'N/A'}")
    print(f"  Cloudinary URL: {video.cloudinary_url or 'N/A'}")
    print(f"  Local Path:     {video.filepath}")

    # Check if cached locally
    from config import CACHE_DIR
    cache_path = CACHE_DIR / video.filename
    if cache_path.exists():
        print(f"  Cached:         YES ({format_size(cache_path.stat().st_size)})")
    else:
        print(f"  Cached:         NO")

    print("=" * 70)
    session.close()


def cmd_stats(args):
    """Show database statistics."""
    from models.database import init_db, get_session, ReferenceVideo, VideoFrame
    from config import DATABASE_PATH, CACHE_DIR

    init_db()
    session = get_session()

    total = session.query(ReferenceVideo).count()
    indexed = session.query(ReferenceVideo).filter(ReferenceVideo.status == 'indexed').count()
    pending = session.query(ReferenceVideo).filter(ReferenceVideo.status == 'pending').count()
    errors = session.query(ReferenceVideo).filter(ReferenceVideo.status == 'error').count()
    frames = session.query(VideoFrame).count()
    cloud = session.query(ReferenceVideo).filter(ReferenceVideo.cloudinary_url != None).count()

    # Cache stats
    cached = 0
    cache_size = 0
    if CACHE_DIR.exists():
        for f in CACHE_DIR.iterdir():
            if f.suffix == '.mp4':
                cached += 1
                cache_size += f.stat().st_size

    # DB size
    db_size = DATABASE_PATH.stat().st_size if DATABASE_PATH.exists() else 0

    print_header("CLIPMATCH - Database Statistics")
    print()
    print(f"  Total Videos:     {total}")
    print(f"  Indexed:          {indexed}")
    print(f"  Pending:          {pending}")
    print(f"  Errors:           {errors}")
    print(f"  Total Frames:     {frames}")
    print_separator()
    print(f"  On Cloudinary:    {cloud}")
    print(f"  Cached Locally:   {cached}")
    print(f"  Cache Size:       {format_size(cache_size)}")
    print(f"  Database Size:    {format_size(db_size)}")
    print(f"  Database Path:    {DATABASE_PATH}")
    print("=" * 70)

    session.close()


def main():
    parser = argparse.ArgumentParser(
        description='ClipMatch CLI - Video Matching from Terminal',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python clipmatch_cli.py list                              List all videos
  python clipmatch_cli.py match query.mp4                   Match with Original quality
  python clipmatch_cli.py match query.mp4 --enhanced        Match with Enhanced quality
  python clipmatch_cli.py info source_081                   Get video details
  python clipmatch_cli.py stats                             Show statistics
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # list
    list_parser = subparsers.add_parser('list', help='List all indexed videos')

    # match
    match_parser = subparsers.add_parser('match', help='Match a query video')
    match_parser.add_argument('video_path', help='Path to query video clip')
    match_parser.add_argument('--enhanced', '-e', action='store_true',
                             help='Use Enhanced quality (NCC+SSIM verification)')

    # info
    info_parser = subparsers.add_parser('info', help='Get video details')
    info_parser.add_argument('source_name', help='Source name (e.g., source_081)')

    # stats
    stats_parser = subparsers.add_parser('stats', help='Show database statistics')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    if args.command == 'list':
        cmd_list(args)
    elif args.command == 'match':
        cmd_match(args)
    elif args.command == 'info':
        cmd_info(args)
    elif args.command == 'stats':
        cmd_stats(args)


if __name__ == '__main__':
    main()
