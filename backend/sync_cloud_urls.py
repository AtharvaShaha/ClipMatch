"""
Sync Cloudinary URLs to locally-indexed videos in the database.
Does NOT re-upload anything — just matches existing Cloudinary resources
to DB records by filename.

Usage:
    python backend/sync_cloud_urls.py
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

# Force UTF-8 output
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

from models.database import init_db
from services import cloudinary_service

def main():
    print("=" * 60)
    print("  CLIPMATCH - Cloudinary URL Sync")
    print("=" * 60)
    
    init_db()
    
    print("\n  Fetching Cloudinary resources and matching to DB...")
    result = cloudinary_service.sync_cloudinary_urls()
    
    if result['success']:
        print(f"\n  Results:")
        print(f"    Newly synced:    {result['synced']}")
        print(f"    Already synced:  {result['already_synced']}")
        print(f"    Total indexed:   {result['total_indexed']}")
        print(f"    Total on cloud:  {result['total_cloud']}")
        
        if result['details']:
            print(f"\n  Newly synced videos:")
            for d in result['details']:
                name = d['filename'][:50]
                print(f"    [+] {name}")
    else:
        print(f"\n  Error: {result.get('error', 'Unknown')}")
    
    print("\n" + "=" * 60)


if __name__ == '__main__':
    main()
