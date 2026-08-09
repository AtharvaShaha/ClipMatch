"""
Batch upload all indexed reference videos to Cloudinary.
After uploading, you can delete local video files to free space.
KEEP the database file (clipmatch.db) — it's only 4.3 MB and needed for matching.
"""
import sys
import os
import time
sys.path.insert(0, '.')

from services.cloudinary_service import upload_video
from models.database import get_session, ReferenceVideo

session = get_session()

# Get all indexed videos
videos = session.query(ReferenceVideo).filter(
    ReferenceVideo.status == 'indexed'
).order_by(ReferenceVideo.filename).all()

# Calculate totals
total_size = 0
already_uploaded = 0
to_upload = []

for v in videos:
    if v.cloudinary_url:
        already_uploaded += 1
    elif os.path.exists(v.filepath):
        to_upload.append(v)
        total_size += (v.file_size or 0)
    else:
        print(f"  SKIP {v.filename} - local file not found at {v.filepath}")

print("=" * 60)
print("  CLOUDINARY BATCH UPLOAD")
print("=" * 60)
print(f"  Total indexed videos : {len(videos)}")
print(f"  Already on cloud     : {already_uploaded}")
print(f"  To upload            : {len(to_upload)}")
print(f"  Total upload size    : {total_size / 1024 / 1024:.0f} MB ({total_size / 1024 / 1024 / 1024:.2f} GB)")
print(f"  Cloudinary free tier : 25 GB")
print()

if not to_upload:
    print("Nothing to upload!")
    sys.exit(0)

# Upload each video
uploaded = 0
failed = 0
start = time.time()

for i, v in enumerate(to_upload, 1):
    size_mb = (v.file_size or 0) / 1024 / 1024
    print(f"[{i}/{len(to_upload)}] Uploading {v.filename} ({size_mb:.1f} MB)...", end=" ", flush=True)
    
    try:
        result = upload_video(v.filepath)
        
        if result['success']:
            v.cloudinary_url = result['secure_url']
            v.cloudinary_public_id = result['public_id']
            session.commit()
            uploaded += 1
            print("OK")
        else:
            failed += 1
            print(f"FAILED: {result.get('error', 'Unknown')}")
    except Exception as e:
        failed += 1
        print(f"ERROR: {e}")

elapsed = time.time() - start

print()
print("=" * 60)
print("  UPLOAD COMPLETE")
print("=" * 60)
print(f"  Uploaded  : {uploaded}")
print(f"  Failed    : {failed}")
print(f"  Time      : {elapsed/60:.1f} minutes")
print()
print("  You can now delete local video files to free disk space.")
print("  KEEP these files:")
print(f"    - D:\\Desktop\\mismatch\\data\\clipmatch.db (4.3 MB)")
print(f"    - D:\\Desktop\\mismatch\\backend\\ (all code)")

session.close()
