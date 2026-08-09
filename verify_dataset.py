"""
ClipMatch — Verify Dataset Integrity

Compares Cloudinary assets with SQLite records and reports discrepancies.

Usage:
    python verify_dataset.py
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))
os.environ['PYTHONIOENCODING'] = 'utf-8'
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from config import DATABASE_PATH, CACHE_DIR


def main():
    import cloudinary
    import cloudinary.api
    from config import CloudinaryConfig
    from models.database import init_db, get_session, ReferenceVideo, VideoFrame

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
    print("  CLIPMATCH - Dataset Verification")
    print("=" * 64)

    # 1. List Cloudinary videos
    print("\n  Fetching Cloudinary assets...")
    all_cloud = []
    next_cursor = None
    while True:
        params = dict(type='upload', resource_type='video', max_results=500)
        if next_cursor:
            params['next_cursor'] = next_cursor
        result = cloudinary.api.resources(**params)
        all_cloud.extend(result.get('resources', []))
        next_cursor = result.get('next_cursor')
        if not next_cursor:
            break
    cloud_pids = {v['public_id'] for v in all_cloud if v.get('resource_type') == 'video'}

    # 2. Read SQLite records
    init_db()
    session = get_session()
    db_records = session.query(ReferenceVideo).all()
    db_pids = {v.cloudinary_public_id for v in db_records if v.cloudinary_public_id}

    # 3. Compare
    missing_from_db = cloud_pids - db_pids
    missing_from_cloud = db_pids - cloud_pids

    # 4. Indexing status
    total_db = len(db_records)
    pending = sum(1 for v in db_records if v.status == 'pending')
    indexed = sum(1 for v in db_records if v.status == 'indexed')
    errors = sum(1 for v in db_records if v.status == 'error')
    total_frames = session.query(VideoFrame).count()

    # 5. Cache status
    cached = 0
    if CACHE_DIR.exists():
        for v in db_records:
            cache_path = CACHE_DIR / v.filename
            if cache_path.exists():
                cached += 1

    session.close()

    # Report
    print(f"\n  Cloudinary videos:      {len(cloud_pids)}")
    print(f"  SQLite records:         {total_db}")
    print(f"")
    print(f"  Missing from SQLite:    {len(missing_from_db)}")
    print(f"  Missing from Cloudinary:{len(missing_from_cloud)}")

    if missing_from_db:
        print(f"\n  Videos on Cloudinary but NOT in SQLite:")
        for pid in sorted(missing_from_db)[:10]:
            print(f"    - {pid}")
        if len(missing_from_db) > 10:
            print(f"    ... and {len(missing_from_db) - 10} more")

    if missing_from_cloud:
        print(f"\n  Videos in SQLite but NOT on Cloudinary (deleted?):")
        for pid in sorted(missing_from_cloud)[:10]:
            print(f"    - {pid}")

    print(f"\n  Indexing status:")
    print(f"    Pending (no frames):  {pending}")
    print(f"    Indexed (has frames): {indexed}")
    print(f"    Error:                {errors}")
    print(f"    Total frames in DB:   {total_frames}")

    print(f"\n  Local cache:")
    print(f"    Cached videos:        {cached} / {total_db}")
    print(f"    Cache directory:       {CACHE_DIR}")

    # Verdict
    print(f"\n  {'=' * 40}")
    if len(missing_from_db) == 0 and len(missing_from_cloud) == 0:
        print(f"  Dataset is SYNCHRONIZED.")
    else:
        print(f"  Dataset has DISCREPANCIES.")
        if missing_from_db:
            print(f"  Run: python sync_cloudinary_database.py")
    print(f"  {'=' * 40}")

    print("=" * 64)


if __name__ == '__main__':
    main()
