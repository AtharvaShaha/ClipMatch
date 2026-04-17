"""
ClipMatch - Main Flask Application
Restricted Source Video Matching System
"""

from flask import Flask, send_from_directory, render_template
from flask_cors import CORS
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import FlaskConfig, DATA_DIR, REFERENCES_DIR, UPLOADS_DIR
from api.routes import api
from models.database import init_db, get_session, ReferenceVideo
from services.indexer import VideoIndexer


def create_app():
    """Create and configure the Flask application."""
    
    # Initialize Flask app with backend/static as static folder
    app = Flask(__name__, 
                static_folder='static',
                static_url_path='/static',
                template_folder='templates')
    
    # Load configuration
    app.config.from_object(FlaskConfig)
    
    # Enable CORS for all origins (development)
    CORS(app, resources={r"/api/*": {"origins": "*"}})
    
    # Initialize database
    init_db()
    
    # Auto-index reference videos if they exist and aren't indexed
    try:
        print("[Init] Checking for existing reference videos...")
        indexer = VideoIndexer()
        session = get_session()
        
        # Check if there are indexed videos
        indexed_count = session.query(ReferenceVideo).filter(
            ReferenceVideo.status == 'indexed'
        ).count()
        
        session.close()
        
        # If no indexed videos but files exist, index them
        if indexed_count == 0:
            print("[Init] No indexed videos found. Scanning references directory...")
            result = indexer.index_directory()
            if result['success']:
                print(f"[Init] ✅ Auto-indexed {len(result['indexed'])} video(s)")
                for video_info in result['indexed']:
                    print(f"     - {video_info['filename']}")
            else:
                print(f"[Init] No reference videos to index or all failed")
        else:
            print(f"[Init] ✅ Found {indexed_count} already indexed video(s)")
    except Exception as e:
        print(f"[Init] Warning: Could not auto-index videos: {str(e)}")
    
    # Register API blueprint
    app.register_blueprint(api)
    
    # Serve templates
    @app.route('/')
    @app.route('/home')
    def home():
        return render_template('home.html')
    
    @app.route('/app')
    def app_interface():
        return render_template('app.html')
    
    @app.route('/<path:path>')
    def serve_static(path):
        if os.path.exists(os.path.join(app.static_folder, path)):
            return send_from_directory(app.static_folder, path)
        return render_template('app.html')
    
    # Error handlers
    @app.errorhandler(404)
    def not_found(error):
        return {'error': 'Resource not found'}, 404
    
    @app.errorhandler(500)
    def server_error(error):
        return {'error': 'Internal server error'}, 500
    
    @app.errorhandler(413)
    def file_too_large(error):
        return {'error': 'File too large. Maximum size is 500MB'}, 413
    
    return app


# Create app instance
app = create_app()


if __name__ == '__main__':
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║   ██████╗██╗     ██╗██████╗ ███╗   ███╗ █████╗ ████████╗ ║
    ║  ██╔════╝██║     ██║██╔══██╗████╗ ████║██╔══██╗╚══██╔══╝ ║
    ║  ██║     ██║     ██║██████╔╝██╔████╔██║███████║   ██║    ║
    ║  ██║     ██║     ██║██╔═══╝ ██║╚██╔╝██║██╔══██║   ██║    ║
    ║  ╚██████╗███████╗██║██║     ██║ ╚═╝ ██║██║  ██║   ██║    ║
    ║   ╚═════╝╚══════╝╚═╝╚═╝     ╚═╝     ╚═╝╚═╝  ╚═╝   ╚═╝    ║
    ║                                                           ║
    ║         Restricted Source Video Matching System           ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    
    Starting server on http://localhost:5000
    API available at http://localhost:5000/api
    
    Directories:
    - References: {refs}
    - Uploads: {uploads}
    - Database: {db}
    """.format(
        refs=REFERENCES_DIR,
        uploads=UPLOADS_DIR,
        db=DATA_DIR / 'clipmatch.db'
    ))
    
    app.run(host='0.0.0.0', port=5000, debug=True)
