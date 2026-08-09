"""List ALL videos on Cloudinary account (all folders, with pagination)."""
import sys, os
sys.path.insert(0, 'backend')
os.environ['PYTHONIOENCODING'] = 'utf-8'
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import cloudinary, cloudinary.api
from config import CloudinaryConfig

cloudinary.config(
    cloud_name=CloudinaryConfig.CLOUD_NAME,
    api_key=CloudinaryConfig.API_KEY,
    api_secret=CloudinaryConfig.API_SECRET,
    secure=True
)

all_videos = []
next_cursor = None
page = 0
while True:
    page += 1
    params = dict(type='upload', resource_type='video', max_results=500)
    if next_cursor:
        params['next_cursor'] = next_cursor
    result = cloudinary.api.resources(**params)
    resources = result.get('resources', [])
    all_videos.extend(resources)
    print(f"  Page {page}: got {len(resources)} videos (total so far: {len(all_videos)})")
    next_cursor = result.get('next_cursor')
    if not next_cursor:
        break

print(f"\nTotal videos on Cloudinary: {len(all_videos)}")

# Group by folder
folders = {}
for v in all_videos:
    pid = v['public_id']
    parts = pid.split('/')
    folder = '/'.join(parts[:-1]) if len(parts) > 1 else '(root)'
    folders.setdefault(folder, []).append(v)

print("\nVideos by folder:")
for folder in sorted(folders.keys()):
    vids = folders[folder]
    print(f"  {folder}: {len(vids)} videos")

print("\nFirst 15 public_ids (sorted):")
for v in sorted(all_videos, key=lambda x: x['public_id'])[:15]:
    pid = v['public_id']
    fmt = v.get('format', '?')
    mb = round(v.get('bytes', 0) / 1024 / 1024, 1)
    print(f"  {pid} | {fmt} | {mb}MB")

print("\nLast 15 public_ids (sorted):")
for v in sorted(all_videos, key=lambda x: x['public_id'])[-15:]:
    pid = v['public_id']
    fmt = v.get('format', '?')
    mb = round(v.get('bytes', 0) / 1024 / 1024, 1)
    print(f"  {pid} | {fmt} | {mb}MB")
