"""
ClipMatch — Sync Cloudinary Database

Detects NEW videos on Cloudinary that aren't in the local SQLite DB,
renames them to the next available source_NNN, and inserts records.

Existing records are NEVER renumbered or modified.

Usage:
    python sync_cloudinary_database.py
"""
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))
os.environ['PYTHONIOENCODING'] = 'utf-8'
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from config import DATABASE_PATH, CACHE_DIR

CLOUD_FOLDER = "clipmatch_references"


def main():
    import cloudinary
    import cloudinary.api
    import cloudinary.uploader
    from config import CloudinaryConfig
    from models.database import init_db, get_session, ReferenceVideo

    if not CloudinaryConfig.CLOUD_NAME or not CloudinaryConfig.API_KEY:
        print("\n  ERROR: Cloudinary credentials not found! Check .env file.")
        sys.exit(1)

    cloudinary.config(
        cloud_name=CloudinaryConfig.CLOUD_NAME,
        api_key=CloudinaryConfig.API_KEY,
        api_secret=CloudinaryConfig.API_SECRET,
        secure=True
    )

    print("=" * 64)
    print("  CLIPMATCH - Cloudinary Database Sync")
    print("=" * 64)

    # Step 1: List all Cloudinary videos
    print("\n  Fetching Cloudinary video assets...")
    all_videos = []
    next_cursor = None
    while True:
        params = dict(type='upload', resource_type='video', max_results=500)
        if next_cursor:
            params['next_cursor'] = next_cursor
        result = cloudinary.api.resources(**params)
        all_videos.extend(result.get('resources', []))
        next_cursor = result.get('next_cursor')
        if not next_cursor:
            break

    all_videos = [v for v in all_videos if v.get('resource_type') == 'video']
    all_videos.sort(key=lambda v: v['public_id'])
    print(f"  Cloudinary videos: {len(all_videos)}")

    # Step 2: Read existing DB records
    init_db()
    session = get_session()

    existing_pids = set()
    max_num = 0
    for v in session.query(ReferenceVideo).all():
        if v.cloudinary_public_id:
            existing_pids.add(v.cloudinary_public_id)
        if v.source_name and v.source_name.startswith('source_'):
            try:
                num = int(v.source_name.split('_')[1])
                max_num = max(max_num, num)
            except (ValueError, IndexError):
                pass

    existing_count = len(existing_pids)
    print(f"  SQLite records:    {existing_count}")

    # Step 3: Find new videos (not in DB)
    new_videos = [v for v in all_videos if v['public_id'] not in existing_pids]

    if not new_videos:
        print(f"\n  No new videos found. Database is up to date.")
        session.close()
        print("=" * 64)
        return

    print(f"\n  New videos found:  {len(new_videos)}")
    print(f"\n  Renaming and inserting new records...\n")

    # Step 4: Rename new videos and insert into DB
    added = 0
    for video in new_videos:
        old_public_id = video['public_id']
        max_num += 1
        new_public_id = f"{CLOUD_FOLDER}/source_{max_num:03d}"
        source_name = f"source_{max_num:03d}"

        # Rename on Cloudinary
        try:
            rename_result = cloudinary.uploader.rename(
                old_public_id,
                new_public_id,
                resource_type="video",
                overwrite=False
            )
            secure_url = rename_result.get('secure_url', video.get('secure_url', ''))
            actual_pid = new_public_id
        except Exception as e:
            if 'already exists' in str(e).lower():
                secure_url = video.get('secure_url', '')
                actual_pid = old_public_id
                source_name = old_public_id.split('/')[-1]
            else:
                print(f"  [!] FAILED: {old_public_id[:50]} -> {e}")
                max_num -= 1
                continue

        # Insert DB record
        fmt = video.get('format', 'mp4')
        filename = f"{source_name}.{fmt}"

        record = ReferenceVideo(
            source_name=source_name,
            filename=filename,
            filepath=str(CACHE_DIR / filename),
            title=source_name,
            duration=video.get('duration', 0) or 0,
            frame_count=0,
            width=video.get('width', 0) or 0,
            height=video.get('height', 0) or 0,
            fps=0,
            file_size=video.get('bytes', 0) or 0,
            cloudinary_url=secure_url,
            cloudinary_public_id=actual_pid,
            status='pending'
        )
        session.add(record)
        added += 1

        old_short = old_public_id[-50:] if len(old_public_id) > 50 else old_public_id
        mb = round(video.get('bytes', 0) / 1024 / 1024, 1)
        print(f"  [+] {source_name} <- {old_short} ({mb}MB)")

        if added % 20 == 0:
            time.sleep(0.5)

    session.commit()
    total = session.query(ReferenceVideo).count()
    session.close()

    print(f"\n" + "=" * 64)
    print(f"  Sync complete.")
    print(f"  Added:  {added}")
    print(f"  Total:  {total}")
    print("=" * 64)


if __name__ == '__main__':
    main()
