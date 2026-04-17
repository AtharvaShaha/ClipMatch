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
    clip_quality can be 'original' (fast) or 'edited' (deep analysis).
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
                # Use advanced matcher with NCC/SSIM verification
                result = advanced_matcher.match_clip(str(filepath))
            else:
                # Use standard fast matcher (default)
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
