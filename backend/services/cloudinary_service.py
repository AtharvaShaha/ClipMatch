"""
ClipMatch Cloudinary Service
Handles uploading reference videos to Cloudinary, downloading them for indexing,
and maintaining a local cache to avoid repeated downloads.

Videos are stored in the 'clipmatch_references' folder on Cloudinary.
After indexing, only the hash database is needed for matching — not the video files.
"""

import os
import logging
import tempfile
import requests
import cloudinary
import cloudinary.uploader
import cloudinary.api
from pathlib import Path

from config import CloudinaryConfig, REFERENCES_DIR, CACHE_DIR

logger = logging.getLogger('clipmatch.cloud')


def _configure():
    """Configure Cloudinary SDK with credentials from .env."""
    cloudinary.config(
        cloud_name=CloudinaryConfig.CLOUD_NAME,
        api_key=CloudinaryConfig.API_KEY,
        api_secret=CloudinaryConfig.API_SECRET,
        secure=True
    )


# Auto-configure on import
_configure()


def list_cloud_videos():
    """
    List ALL video assets on Cloudinary (all folders, with pagination).
    
    Returns:
        dict with 'success', 'count', 'videos' list
    """
    try:
        _configure()
        
        all_videos = []
        next_cursor = None
        
        while True:
            params = dict(type='upload', resource_type='video', max_results=500)
            if next_cursor:
                params['next_cursor'] = next_cursor
            
            result = cloudinary.api.resources(**params)
            resources = result.get('resources', [])
            
            for r in resources:
                all_videos.append({
                    'public_id': r['public_id'],
                    'format': r.get('format', 'mp4'),
                    'bytes': r.get('bytes', 0),
                    'width': r.get('width', 0),
                    'height': r.get('height', 0),
                    'duration': r.get('duration', 0),
                    'secure_url': r.get('secure_url', ''),
                    'created_at': r.get('created_at', ''),
                    'resource_type': r.get('resource_type', 'video'),
                })
            
            next_cursor = result.get('next_cursor')
            if not next_cursor:
                break
        
        logger.info("[Cloud] Listed %d video assets", len(all_videos))
        
        return {
            'success': True,
            'count': len(all_videos),
            'videos': all_videos
        }
        
    except Exception as e:
        logger.error("[Cloud] List failed: %s", e)
        return {
            'success': False,
            'error': str(e),
            'count': 0,
            'videos': []
        }


def upload_video(filepath):
    """
    Upload a video file to Cloudinary.
    
    Args:
        filepath: Local path to video file
    
    Returns:
        dict with 'success', 'public_id', 'secure_url'
    """
    try:
        _configure()
        
        filename = os.path.basename(filepath)
        name_without_ext = os.path.splitext(filename)[0]
        public_id = f"{CloudinaryConfig.CLOUD_FOLDER}/{name_without_ext}"
        
        logger.info("[Cloud] Uploading %s...", filename)
        
        result = cloudinary.uploader.upload(
            filepath,
            resource_type="video",
            public_id=public_id,
            overwrite=False,
            chunk_size=20000000  # 20MB chunks for large files
        )
        
        logger.info("[Cloud] Uploaded: %s -> %s", filename, result.get('public_id', ''))
        
        return {
            'success': True,
            'public_id': result.get('public_id', ''),
            'secure_url': result.get('secure_url', ''),
            'bytes': result.get('bytes', 0),
            'duration': result.get('duration', 0),
            'format': result.get('format', 'mp4'),
            'width': result.get('width', 0),
            'height': result.get('height', 0),
        }
        
    except Exception as e:
        logger.error("[Cloud] Upload failed for %s: %s", filepath, e)
        return {
            'success': False,
            'error': str(e)
        }


def download_video(public_id, output_dir=None):
    """
    Download a video from Cloudinary by public_id.
    
    Args:
        public_id: Cloudinary public ID
        output_dir: Directory to save to (defaults to REFERENCES_DIR)
    
    Returns:
        dict with 'success', 'filepath', 'filename'
    """
    try:
        _configure()
        
        if output_dir is None:
            output_dir = str(REFERENCES_DIR)
        
        # Build download URL
        url = cloudinary.utils.cloudinary_url(
            public_id,
            resource_type="video"
        )[0]
        
        filename = public_id.split('/')[-1] + '.mp4'
        filepath = os.path.join(output_dir, filename)
        
        logger.info("[Cloud] Downloading %s -> %s", public_id, filepath)
        
        response = requests.get(url, stream=True, timeout=120)
        response.raise_for_status()
        
        os.makedirs(output_dir, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        logger.info("[Cloud] Downloaded: %s (%d bytes)", filename, os.path.getsize(filepath))
        
        return {
            'success': True,
            'filepath': filepath,
            'filename': filename,
            'bytes': os.path.getsize(filepath)
        }
        
    except Exception as e:
        logger.error("[Cloud] Download failed for %s: %s", public_id, e)
        return {
            'success': False,
            'error': str(e)
        }


def get_or_download_video(source_name_or_id):
    """
    Get a local cached path to a video, downloading from Cloudinary if not cached.
    
    This is the PRIMARY function for accessing reference videos. It:
    1. Checks if the video exists in the local cache (cache/ directory)
    2. If not, looks up the Cloudinary URL from SQLite
    3. Downloads the video to cache/
    4. Returns the local file path
    
    Args:
        source_name_or_id: Either a source_name like 'source_001',
                           a cloudinary_public_id, or a database ID (int)
    
    Returns:
        str: Local file path to the cached video, or None if not found
    """
    from models.database import get_session, ReferenceVideo
    
    session = get_session()
    try:
        # Look up the video record
        video = None
        if isinstance(source_name_or_id, int):
            video = session.query(ReferenceVideo).get(source_name_or_id)
        elif source_name_or_id.startswith('source_'):
            video = session.query(ReferenceVideo).filter(
                ReferenceVideo.source_name == source_name_or_id
            ).first()
        else:
            video = session.query(ReferenceVideo).filter(
                ReferenceVideo.cloudinary_public_id == source_name_or_id
            ).first()
        
        if not video:
            logger.warning("[Cache] Video not found: %s", source_name_or_id)
            return None
        
        if not video.cloudinary_url:
            logger.warning("[Cache] No Cloudinary URL for: %s", video.source_name)
            return None
        
        # Check local cache
        cache_path = CACHE_DIR / video.filename
        if cache_path.exists() and cache_path.stat().st_size > 0:
            logger.debug("[Cache] HIT: %s -> %s", video.source_name, cache_path)
            return str(cache_path)
        
        # Download from Cloudinary
        logger.info("[Cache] MISS: downloading %s from Cloudinary...", video.source_name)
        
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        
        response = requests.get(video.cloudinary_url, stream=True, timeout=300)
        response.raise_for_status()
        
        with open(cache_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=65536):
                f.write(chunk)
        
        file_size = cache_path.stat().st_size
        logger.info("[Cache] Downloaded %s: %d bytes -> %s",
                    video.source_name, file_size, cache_path)
        
        return str(cache_path)
    
    except Exception as e:
        logger.error("[Cache] Failed to get video %s: %s", source_name_or_id, e)
        return None
    finally:
        session.close()


def delete_cloud_video(public_id):
    """
    Delete a video from Cloudinary.
    
    Args:
        public_id: Cloudinary public ID
    
    Returns:
        dict with 'success'
    """
    try:
        _configure()
        
        result = cloudinary.uploader.destroy(
            public_id,
            resource_type="video"
        )
        
        logger.info("[Cloud] Deleted %s: %s", public_id, result.get('result', ''))
        
        return {
            'success': result.get('result') == 'ok',
            'result': result.get('result', 'unknown')
        }
        
    except Exception as e:
        logger.error("[Cloud] Delete failed for %s: %s", public_id, e)
        return {
            'success': False,
            'error': str(e)
        }
