"""
Compress large videos (>100MB) using OpenCV (no FFmpeg needed), then upload to Cloudinary.
"""
import sys, os, time
sys.path.insert(0, '.')

import cv2
import cloudinary
import cloudinary.uploader
from config import CloudinaryConfig
from models.database import get_session, ReferenceVideo

cloudinary.config(
    cloud_name=CloudinaryConfig.CLOUD_NAME,
    api_key=CloudinaryConfig.API_KEY,
    api_secret=CloudinaryConfig.API_SECRET,
    secure=True
)

session = get_session()
videos = session.query(ReferenceVideo).filter(
    ReferenceVideo.status == 'indexed',
    ReferenceVideo.cloudinary_url == None
).order_by(ReferenceVideo.filename).all()

print(f"Found {len(videos)} videos to compress and upload\n")

TEMP_DIR = os.path.join(os.path.dirname(__file__), '_temp_compressed')
os.makedirs(TEMP_DIR, exist_ok=True)

uploaded = 0
failed = 0

for i, v in enumerate(videos, 1):
    if not os.path.exists(v.filepath):
        print(f"[{i}/{len(videos)}] SKIP {v.filename} - file not found")
        continue

    size_mb = (v.file_size or 0) / 1024 / 1024
    compressed_path = os.path.join(TEMP_DIR, os.path.splitext(v.filename)[0] + '_compressed.mp4')

    print(f"[{i}/{len(videos)}] {v.filename} ({size_mb:.0f}MB)")

    try:
        # Read video properties
        cap = cv2.VideoCapture(v.filepath)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 300

        # Scale down to 480p max width and reduce fps to 15
        target_width = min(480, width)
        scale = target_width / width
        target_height = int(height * scale)
        # Ensure even dimensions
        target_height = target_height + (target_height % 2)
        target_width = target_width + (target_width % 2)
        target_fps = min(15, fps)

        # Skip frames to match target fps
        frame_skip = max(1, int(fps / target_fps))

        print(f"  Compressing: {width}x{height}@{fps:.0f} -> {target_width}x{target_height}@{target_fps:.0f}fps...", end=" ", flush=True)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(compressed_path, fourcc, target_fps, (target_width, target_height))

        frame_count = 0
        written = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % frame_skip == 0:
                resized = cv2.resize(frame, (target_width, target_height))
                out.write(resized)
                written += 1
            frame_count += 1

        cap.release()
        out.release()

        comp_size = os.path.getsize(compressed_path) / 1024 / 1024
        print(f"OK ({comp_size:.0f}MB, {written} frames)")

        if comp_size > 100:
            print(f"  Still >100MB, trying lower resolution...", end=" ", flush=True)
            # Try 320px width
            os.remove(compressed_path)
            cap2 = cv2.VideoCapture(v.filepath)
            tw2 = 320
            s2 = tw2 / width
            th2 = int(height * s2) + (int(height * s2) % 2)
            out2 = cv2.VideoWriter(compressed_path, fourcc, 10, (tw2, th2))
            fc2 = 0
            skip2 = max(1, int(fps / 10))
            while True:
                ret, frame = cap2.read()
                if not ret:
                    break
                if fc2 % skip2 == 0:
                    out2.write(cv2.resize(frame, (tw2, th2)))
                fc2 += 1
            cap2.release()
            out2.release()
            comp_size = os.path.getsize(compressed_path) / 1024 / 1024
            print(f"OK ({comp_size:.0f}MB)")

            if comp_size > 100:
                print(f"  STILL >100MB, skipping")
                os.remove(compressed_path)
                failed += 1
                continue

        # Upload
        public_id = f"{CloudinaryConfig.CLOUD_FOLDER}/{os.path.splitext(v.filename)[0]}"
        print(f"  Uploading to Cloudinary...", end=" ", flush=True)

        result = cloudinary.uploader.upload(
            compressed_path, public_id=public_id,
            resource_type="video", overwrite=True, timeout=120
        )

        v.cloudinary_url = result['secure_url']
        v.cloudinary_public_id = result['public_id']
        session.commit()
        uploaded += 1
        print("OK")
        os.remove(compressed_path)

    except Exception as e:
        failed += 1
        print(f"FAILED: {str(e)[:80]}")
        if os.path.exists(compressed_path):
            os.remove(compressed_path)

try:
    os.rmdir(TEMP_DIR)
except Exception:
    pass

print(f"\n{'='*60}")
print(f"  Uploaded: {uploaded} | Failed: {failed}")
print(f"{'='*60}")
session.close()
