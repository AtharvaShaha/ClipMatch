"""
ClipMatch - Main Flask Application
Restricted Source Video Matching System
"""

from flask import Flask, send_from_directory
from flask_cors import CORS
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import FlaskConfig, DATA_DIR, REFERENCES_DIR, UPLOADS_DIR
from api.routes import api
from models.database import init_db


def create_app():
    """Create and configure the Flask application."""
    
    # Initialize Flask app
    app = Flask(__name__, 
                static_folder='../frontend',
                static_url_path='')
    
    # Load configuration
    app.config.from_object(FlaskConfig)
    
    # Enable CORS for all origins (development)
    CORS(app, resources={r"/api/*": {"origins": "*"}})
    
    # Initialize database
    init_db()
    
    # Register API blueprint
    app.register_blueprint(api)
    
    # Serve frontend
    @app.route('/')
    def serve_frontend():
        return send_from_directory(app.static_folder, 'index.html')
    
    @app.route('/<path:path>')
    def serve_static(path):
        if os.path.exists(os.path.join(app.static_folder, path)):
            return send_from_directory(app.static_folder, path)
        return send_from_directory(app.static_folder, 'index.html')
    
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
