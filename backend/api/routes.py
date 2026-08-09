"""
ClipMatch API Routes
Flask blueprint with all API endpoints
"""

import os
from pathlib import Path
from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
import sys
sys.path.append('..')
from config import UPLOADS_DIR, REFERENCES_DIR, VideoConfig
from services.indexer import video_indexer
from services.matcher import clip_matcher
from services.advanced_matcher import advanced_matcher
from services.video_processor import video_processor
from services import cloudinary_service

api = Blueprint('api', __name__, url_prefix='/api')


# ============================================================
# Health Check
# ============================================================

@api.route('/health', methods=['GET'])
def health_check():
    """API health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'service': 'ClipMatch API',
        'version': '1.0.0'
    })


# ============================================================
# Reference Videos Management
# ============================================================

@api.route('/references', methods=['GET'])
def get_references():
    """Get list of all indexed reference videos."""
    try:
        videos = video_indexer.get_indexed_videos()
        stats = video_indexer.get_index_stats()
        
        return jsonify({
            'success': True,
            'videos': videos,
            'stats': stats
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@api.route('/references/<int:video_id>', methods=['GET'])
def get_reference_details(video_id):
    """Get detailed information about a specific reference video."""
    try:
        details = video_indexer.get_video_details(video_id)
        
        if not details:
            return jsonify({
                'success': False,
                'error': f'Video with ID {video_id} not found'
            }), 404
        
        return jsonify({
            'success': True,
            'video': details
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@api.route('/references', methods=['POST'])
def upload_reference():
    """Upload and index a new reference video."""
    try:
        if 'video' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No video file provided'
            }), 400
        
        file = request.files['video']
        
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400
        
        # Check file extension
        filename = secure_filename(file.filename)
        ext = Path(filename).suffix.lower()
        
        if ext not in VideoConfig.SUPPORTED_FORMATS:
            return jsonify({
                'success': False,
                'error': f'Unsupported format: {ext}. Supported: {VideoConfig.SUPPORTED_FORMATS}'
            }), 400
        
        # Save file to references directory
        filepath = REFERENCES_DIR / filename
        file.save(str(filepath))
        
        # Get optional title
        title = request.form.get('title', None)
        
        # Index the video
        result = video_indexer.index_video(str(filepath), title=title)
        
        if result['success']:
            return jsonify(result), 201
        else:
            # Clean up file if indexing failed
            if filepath.exists():
                filepath.unlink()
            return jsonify(result), 400
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@api.route('/references/<int:video_id>', methods=['DELETE'])
def delete_reference(video_id):
    """Remove a reference video from the index."""
    try:
        result = video_indexer.delete_video(video_id)
        
        if result['success']:
            return jsonify(result)
        else:
            return jsonify(result), 404
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@api.route('/references/<int:video_id>/reindex', methods=['POST'])
def reindex_reference(video_id):
    """Re-index a reference video."""
    try:
        result = video_indexer.reindex_video(video_id)
        
        if result['success']:
            return jsonify(result)
        else:
            return jsonify(result), 400
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@api.route('/references/index-directory', methods=['POST'])
def index_directory():
    """Index all videos in the references directory."""
    try:
        directory = request.json.get('directory', None) if request.is_json else None
        result = video_indexer.index_directory(directory)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ============================================================
# Clip Matching
# ============================================================

@api.route('/match', methods=['POST'])
def match_clip():
    """
    Match a query clip against indexed reference videos.
    
    Expects a video file upload with key 'clip' and optional 'clip_quality'.
    clip_quality options:
    - 'original' (default): Fast hash-based matching for clear/unedited clips
    - 'edited': Advanced NCC-based verification for edited/compressed/watermarked clips
    
    Returns the best match with confidence score and timestamp range.
    """
    try:
        if 'clip' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No clip file provided. Use form key "clip"'
            }), 400
        
        file = request.files['clip']
        
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400
        
        # Check file extension
        filename = secure_filename(file.filename)
        ext = Path(filename).suffix.lower()
        
        if ext not in VideoConfig.SUPPORTED_FORMATS:
            return jsonify({
                'success': False,
                'error': f'Unsupported format: {ext}'
            }), 400
        
        # Get clip quality setting (default to 'original')
        clip_quality = request.form.get('clip_quality', 'original')
        
        # Save clip temporarily
        filepath = UPLOADS_DIR / filename
        file.save(str(filepath))
        
        try:
            # Select matcher based on clip quality
            if clip_quality == 'edited':
                # Use advanced NCC-based matcher for edited/compressed/watermarked videos
                # NCC (Normalized Cross-Correlation) is robust to compression artifacts and brightness changes
                result = advanced_matcher.match_clip(str(filepath))
            else:
                # Use standard fast hash matcher (default for clear/unedited clips)
                result = clip_matcher.match_clip(str(filepath))
            
            return jsonify(result)
            
        finally:
            # Clean up uploaded file
            if filepath.exists():
                filepath.unlink()
                
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@api.route('/match/history', methods=['GET'])
def get_match_history():
    """Get recent match query history."""
    try:
        limit = request.args.get('limit', 50, type=int)
        history = clip_matcher.get_match_history(limit=limit)
        
        return jsonify({
            'success': True,
            'history': history,
            'count': len(history)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ============================================================
# Statistics
# ============================================================

@api.route('/stats', methods=['GET'])
def get_stats():
    """Get overall system statistics."""
    try:
        index_stats = video_indexer.get_index_stats()
        
        return jsonify({
            'success': True,
            'index': index_stats,
            'config': {
                'sample_rate': VideoConfig.SAMPLE_RATE,
                'min_clip_duration': VideoConfig.MIN_CLIP_DURATION,
                'max_clip_duration': VideoConfig.MAX_CLIP_DURATION,
                'supported_formats': VideoConfig.SUPPORTED_FORMATS
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ============================================================
# Validation
# ============================================================

@api.route('/validate/clip', methods=['POST'])
def validate_clip():
    """Validate a clip file before matching."""
    try:
        if 'clip' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No clip file provided'
            }), 400
        
        file = request.files['clip']
        filename = secure_filename(file.filename)
        filepath = UPLOADS_DIR / filename
        file.save(str(filepath))
        
        try:
            is_valid, message = video_processor.validate_query_clip(str(filepath))
            info = video_processor.get_video_info(str(filepath)) if is_valid else None
            
            return jsonify({
                'success': True,
                'valid': is_valid,
                'message': message,
                'info': info
            })
            
        finally:
            if filepath.exists():
                filepath.unlink()
                
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ============================================================
# Cloud Storage (Cloudinary)
# ============================================================

@api.route('/cloud/videos', methods=['GET'])
def list_cloud_videos():
    """List all videos stored on Cloudinary."""
    try:
        result = cloudinary_service.list_cloud_videos()
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@api.route('/cloud/upload', methods=['POST'])
def upload_to_cloud():
    """
    Upload a video to Cloudinary AND index it locally.
    Flow: Save locally → Index → Upload to Cloudinary → Store cloud URL in DB → Optionally delete local file.
    """
    try:
        if 'video' not in request.files:
            return jsonify({'success': False, 'error': 'No video file provided'}), 400
        
        file = request.files['video']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        filename = secure_filename(file.filename)
        ext = Path(filename).suffix.lower()
        
        if ext not in VideoConfig.SUPPORTED_FORMATS:
            return jsonify({
                'success': False,
                'error': f'Unsupported format: {ext}'
            }), 400
        
        # Save locally to references dir
        filepath = REFERENCES_DIR / filename
        file.save(str(filepath))
        
        title = request.form.get('title', None)
        
        # Step 1: Index the video locally (extracts frames → computes hashes → stores in DB)
        index_result = video_indexer.index_video(str(filepath), title=title)
        
        if not index_result['success']:
            if filepath.exists():
                filepath.unlink()
            return jsonify(index_result), 400
        
        # Step 2: Upload to Cloudinary
        cloud_result = cloudinary_service.upload_video(str(filepath))
        
        if cloud_result['success']:
            # Step 3: Store cloud URL in database
            from models.database import get_session, ReferenceVideo
            session = get_session()
            try:
                ref_video = session.query(ReferenceVideo).filter(
                    ReferenceVideo.filename == filename
                ).first()
                if ref_video:
                    ref_video.cloudinary_url = cloud_result['secure_url']
                    ref_video.cloudinary_public_id = cloud_result['public_id']
                    session.commit()
            finally:
                session.close()
        
        return jsonify({
            'success': True,
            'indexed': index_result.get('success', False),
            'cloud_uploaded': cloud_result.get('success', False),
            'cloud_url': cloud_result.get('secure_url', ''),
            'video_id': index_result.get('video_id'),
            'message': f'{filename} indexed and uploaded to cloud'
        }), 201
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@api.route('/cloud/index-from-url', methods=['POST'])
def index_from_cloud_url():
    """
    Download a video from Cloudinary URL and index it.
    Body: { "public_id": "clipmatch_references/VideoName" }
    """
    try:
        data = request.get_json()
        if not data or 'public_id' not in data:
            return jsonify({'success': False, 'error': 'public_id is required'}), 400
        
        public_id = data['public_id']
        
        # Download from Cloudinary
        dl_result = cloudinary_service.download_video(public_id)
        if not dl_result['success']:
            return jsonify(dl_result), 400
        
        filepath = dl_result['filepath']
        title = data.get('title', None)
        
        # Index the downloaded video
        index_result = video_indexer.index_video(filepath, title=title)
        
        if index_result['success']:
            # Store cloud info in DB
            from models.database import get_session, ReferenceVideo
            import cloudinary.utils
            session = get_session()
            try:
                ref_video = session.query(ReferenceVideo).filter(
                    ReferenceVideo.filename == dl_result['filename']
                ).first()
                if ref_video:
                    cloud_url = cloudinary.utils.cloudinary_url(
                        public_id, resource_type="video"
                    )[0]
                    ref_video.cloudinary_url = cloud_url
                    ref_video.cloudinary_public_id = public_id
                    session.commit()
            finally:
                session.close()
        
        return jsonify({
            'success': index_result.get('success', False),
            'message': f'Indexed {dl_result["filename"]} from cloud',
            'video_id': index_result.get('video_id'),
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@api.route('/cloud/sync', methods=['POST'])
def sync_to_cloud():
    """
    Upload all locally-indexed videos that aren't yet on Cloudinary.
    This is a batch operation — uploads each local reference video to the cloud.
    """
    try:
        from models.database import get_session, ReferenceVideo
        session = get_session()
        try:
            # Find videos without cloud URLs
            local_only = session.query(ReferenceVideo).filter(
                ReferenceVideo.status == 'indexed',
                ReferenceVideo.cloudinary_url == None
            ).all()
            
            results = []
            for ref in local_only:
                if os.path.exists(ref.filepath):
                    cloud_result = cloudinary_service.upload_video(ref.filepath)
                    if cloud_result['success']:
                        ref.cloudinary_url = cloud_result['secure_url']
                        ref.cloudinary_public_id = cloud_result['public_id']
                        session.commit()
                        results.append({
                            'filename': ref.filename,
                            'status': 'uploaded',
                            'cloud_url': cloud_result['secure_url']
                        })
                    else:
                        results.append({
                            'filename': ref.filename,
                            'status': 'failed',
                            'error': cloud_result.get('error', 'Unknown')
                        })
                else:
                    results.append({
                        'filename': ref.filename,
                        'status': 'skipped',
                        'error': 'Local file not found'
                    })
            
            uploaded = sum(1 for r in results if r['status'] == 'uploaded')
            return jsonify({
                'success': True,
                'total': len(local_only),
                'uploaded': uploaded,
                'results': results
            })
            
        finally:
            session.close()
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
