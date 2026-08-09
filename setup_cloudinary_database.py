"""
ClipMatch — Setup Cloudinary Database

Connects to Cloudinary, retrieves all video assets (all folders),
RENAMES them to clean sequential names (source_001, source_002...),
and creates a fresh SQLite database.

Usage:
    python setup_cloudinary_database.py          # Add new videos only (idempotent)
    python setup_cloudinary_database.py --reset   # Delete DB and rebuild from scratch
"""
import sys
import os
import argparse
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))
os.environ['PYTHONIOENCODING'] = 'utf-8'
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from config import DATABASE_PATH, CACHE_DIR

# Target folder on Cloudinary for all renamed videos
CLOUD_FOLDER = "clipmatch_references"


def get_cloudinary_client():
    """Configure and validate Cloudinary connection."""
    import cloudinary
    import cloudinary.api
    import cloudinary.uploader
    from config import CloudinaryConfig

    if not CloudinaryConfig.CLOUD_NAME or not CloudinaryConfig.API_KEY:
        print("\n  ERROR: Cloudinary credentials not found!")
        print("  Create a .env file with CLOUDINARY_CLOUD_NAME, CLOUDINARY_API_KEY, CLOUDINARY_API_SECRET")
        print("  See .env.example for the template.")
        sys.exit(1)

    cloudinary.config(
        cloud_name=CloudinaryConfig.CLOUD_NAME,
        api_key=CloudinaryConfig.API_KEY,
        api_secret=CloudinaryConfig.API_SECRET,
        secure=True
    )
    return cloudinary


def get_all_cloudinary_videos():
    """Retrieve ALL video assets from Cloudinary with pagination."""
    import cloudinary.api

    all_videos = []
    next_cursor = None
    page = 0

    print("\n  Connecting to Cloudinary...")

    while True:
        page += 1
        params = dict(type='upload', resource_type='video', max_results=500)
        if next_cursor:
            params['next_cursor'] = next_cursor
        try:
            result = cloudinary.api.resources(**params)
        except Exception as e:
            print(f"\n  ERROR: Cloudinary API call failed: {e}")
            sys.exit(1)

        resources = result.get('resources', [])
        all_videos.extend(resources)
        print(f"    Page {page}: {len(resources)} videos (total: {len(all_videos)})")

        next_cursor = result.get('next_cursor')
        if not next_cursor:
            break

    # Filter to video resources only
    all_videos = [v for v in all_videos if v.get('resource_type') == 'video']

    # Sort deterministically by public_id
    all_videos.sort(key=lambda v: v['public_id'])

    print(f"\n  Found {len(all_videos)} video assets on Cloudinary.")
    return all_videos


def rename_cloudinary_videos(cloud_videos):
    """
    Rename ALL Cloudinary videos to clean sequential names:
      clipmatch_references/source_001
      clipmatch_references/source_002
      ...
    
    Videos already named correctly are skipped.
    Returns updated list of video metadata.
    """
    import cloudinary.uploader

    print(f"\n  Renaming {len(cloud_videos)} videos to clean names...\n")

    renamed = []
    skipped = 0
    failed = 0

    for i, video in enumerate(cloud_videos, 1):
        old_public_id = video['public_id']
        new_public_id = f"{CLOUD_FOLDER}/source_{i:03d}"

        # Skip if already correctly named
        if old_public_id == new_public_id:
            skipped += 1
            renamed.append(video)
            print(f"  [=] {i:3d}. {new_public_id} (already correct)")
            continue

        try:
            result = cloudinary.uploader.rename(
                old_public_id,
                new_public_id,
                resource_type="video",
                overwrite=False  # Don't overwrite if target exists
            )

            # Update the video dict with new info
            video['public_id'] = new_public_id
            video['secure_url'] = result.get('secure_url', video.get('secure_url', ''))
            renamed.append(video)

            old_short = old_public_id[-55:] if len(old_public_id) > 55 else old_public_id
            print(f"  [+] {i:3d}. {old_short}")
            print(f"         -> {new_public_id}")

            # Small delay to avoid rate limiting
            if i % 20 == 0:
                time.sleep(0.5)

        except Exception as e:
            err_msg = str(e)
            if 'already exists' in err_msg.lower():
                # Target name already taken — skip
                skipped += 1
                renamed.append(video)
                print(f"  [=] {i:3d}. {new_public_id} (target exists, skipping)")
            else:
                failed += 1
                renamed.append(video)  # Keep original
                print(f"  [!] {i:3d}. FAILED to rename {old_public_id[:50]}: {err_msg[:60]}")

    print(f"\n  Rename summary: {len(renamed) - skipped - failed} renamed, {skipped} skipped, {failed} failed")
    return renamed


def reset_database():
    """Delete and recreate the SQLite database."""
    if DATABASE_PATH.exists():
        print(f"\n  WARNING: This will delete the local SQLite database.")
        print(f"  Database: {DATABASE_PATH}")
        print(f"  Cloudinary videos will NOT be deleted.\n")
        confirm = input("  Continue? [y/N] ").strip().lower()
        if confirm != 'y':
            print("  Aborted.")
            sys.exit(0)

        DATABASE_PATH.unlink()
        for suffix in ['-wal', '-shm']:
            p = DATABASE_PATH.parent / (DATABASE_PATH.name + suffix)
            if p.exists():
                p.unlink()
        print(f"  Deleted: {DATABASE_PATH}")


def create_database(cloud_videos, is_reset=False):
    """Create/update SQLite database from Cloudinary video metadata."""
    from models.database import init_db, get_session, ReferenceVideo

    engine = init_db()
    session = get_session()

    # Get existing records (by cloudinary_public_id)
    existing = {}
    try:
        for v in session.query(ReferenceVideo).all():
            if v.cloudinary_public_id:
                existing[v.cloudinary_public_id] = v
    except Exception:
        pass

    new_count = 0
    skipped_count = 0

    print(f"\n  Creating database records...\n")

    for i, video in enumerate(cloud_videos, 1):
        public_id = video['public_id']

        # Skip if already in DB
        if public_id in existing:
            skipped_count += 1
            continue

        # Source name from public_id (e.g., "clipmatch_references/source_001" -> "source_001")
        source_name = public_id.split('/')[-1]

        # Filename for local cache
        fmt = video.get('format', 'mp4')
        filename = f"{source_name}.{fmt}"

        # Secure URL
        secure_url = video.get('secure_url', '')

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
            cloudinary_public_id=public_id,
            status='pending'
        )

        session.add(record)
        new_count += 1

        mb = round(video.get('bytes', 0) / 1024 / 1024, 1)
        print(f"  {i:3d}. {source_name} -> {public_id} ({mb}MB)")

    session.commit()

    total = session.query(ReferenceVideo).count()
    session.close()

    return new_count, skipped_count, total


def main():
    parser = argparse.ArgumentParser(description='Setup ClipMatch database from Cloudinary')
    parser.add_argument('--reset', action='store_true',
                        help='Delete existing database and rebuild from scratch')
    parser.add_argument('--no-rename', action='store_true',
                        help='Skip renaming Cloudinary assets (use existing names)')
    args = parser.parse_args()

    print("=" * 64)
    print("  CLIPMATCH - Cloudinary Database Setup")
    print("=" * 64)

    # Step 1: Configure Cloudinary
    get_cloudinary_client()

    # Step 2: Get all Cloudinary videos
    cloud_videos = get_all_cloudinary_videos()

    if not cloud_videos:
        print("\n  No video assets found on Cloudinary. Nothing to do.")
        return

    # Step 3: Rename videos to clean names (unless --no-rename)
    if not args.no_rename:
        cloud_videos = rename_cloudinary_videos(cloud_videos)
    else:
        print("\n  Skipping rename (--no-rename flag set).")

    # Step 4: Reset DB if requested
    if args.reset:
        reset_database()

    # Step 5: Create/update database
    new_count, skipped_count, total = create_database(cloud_videos, is_reset=args.reset)

    # Summary
    print("\n" + "=" * 64)
    print(f"  Database created successfully.")
    print(f"")
    print(f"  New records:     {new_count}")
    print(f"  Skipped (exist): {skipped_count}")
    print(f"  Total videos:    {total}")
    print(f"  Database:        {DATABASE_PATH}")
    print(f"  Cache dir:       {CACHE_DIR}")
    print(f"")
    print(f"  NEXT STEP: Run 'python index_cached_videos.py' to download")
    print(f"  videos and extract frame hashes for matching.")
    print("=" * 64)


if __name__ == '__main__':
    main()
