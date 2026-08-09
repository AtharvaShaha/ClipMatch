"""
Retry failed Cloudinary uploads — uses chunked upload for large files.
Only uploads videos that don't have a cloudinary_url yet.
"""
import sys
import os
import time
sys.path.insert(0, '.')

import cloudinary
import cloudinary.uploader
from config import CloudinaryConfig, REFERENCES_DIR
from models.database import get_session, ReferenceVideo

# Configure
cloudinary.config(
    cloud_name=CloudinaryConfig.CLOUD_NAME,
    api_key=CloudinaryConfig.API_KEY,
    api_secret=CloudinaryConfig.API_SECRET,
    secure=True
)

session = get_session()

# Find videos not yet on cloud
videos = session.query(ReferenceVideo).filter(
    ReferenceVideo.status == 'indexed',
    ReferenceVideo.cloudinary_url == None
).order_by(ReferenceVideo.filename).all()

print(f"Found {len(videos)} videos not yet on Cloudinary:")
for v in videos:
    size_mb = (v.file_size or 0) / 1024 / 1024
    exists = "OK" if os.path.exists(v.filepath) else "MISSING"
    print(f"  {v.filename} ({size_mb:.0f} MB) [{exists}]")

print()

uploaded = 0
failed = 0
start = time.time()

for i, v in enumerate(videos, 1):
    if not os.path.exists(v.filepath):
        print(f"[{i}/{len(videos)}] SKIP {v.filename} - file not found")
        continue

    size_mb = (v.file_size or 0) / 1024 / 1024
    public_id = f"{CloudinaryConfig.CLOUD_FOLDER}/{os.path.splitext(v.filename)[0]}"

    print(f"[{i}/{len(videos)}] Uploading {v.filename} ({size_mb:.0f} MB)...", end=" ", flush=True)

    try:
        # Use chunk_size for large files (>100MB)
        upload_kwargs = {
            "public_id": public_id,
            "resource_type": "video",
            "overwrite": True,
            "timeout": 600,  # 10 min timeout
        }

        # For files > 100MB, use chunked upload
        if size_mb > 100:
            upload_kwargs["chunk_size"] = 20_000_000  # 20MB chunks
            result = cloudinary.uploader.upload_large(v.filepath, **upload_kwargs)
        else:
            result = cloudinary.uploader.upload(v.filepath, **upload_kwargs)

        v.cloudinary_url = result['secure_url']
        v.cloudinary_public_id = result['public_id']
        session.commit()
        uploaded += 1
        print("OK")

    except Exception as e:
        failed += 1
        err = str(e)[:80]
        print(f"FAILED: {err}")

elapsed = time.time() - start

print()
print("=" * 60)
print(f"  Uploaded: {uploaded} | Failed: {failed} | Time: {elapsed/60:.1f} min")
print("=" * 60)

session.close()
