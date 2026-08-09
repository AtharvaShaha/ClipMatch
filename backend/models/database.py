"""
ClipMatch Database Models
SQLAlchemy models for storing video metadata and features
"""

from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, LargeBinary, ForeignKey, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session, relationship
import sys
sys.path.append('..')
from config import DATABASE_PATH

Base = declarative_base()


class ReferenceVideo(Base):
    """Model for storing reference video metadata"""
    __tablename__ = 'reference_videos'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    source_name = Column(String(20), unique=True, nullable=True)  # source_001, source_002...
    filename = Column(String(500), nullable=False, unique=True)
    filepath = Column(String(1000), nullable=False)
    title = Column(String(500), nullable=True)
    duration = Column(Float, nullable=False)  # Duration in seconds
    frame_count = Column(Integer, nullable=False)
    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)
    fps = Column(Float, nullable=True)
    file_size = Column(Integer, nullable=True)  # Size in bytes
    checksum = Column(String(64), nullable=True)  # SHA256 hash
    cloudinary_url = Column(String(1000), nullable=True)  # Cloudinary secure URL
    cloudinary_public_id = Column(String(500), nullable=True)  # Cloudinary public ID
    indexed_at = Column(DateTime, default=datetime.utcnow)
    status = Column(String(50), default='pending')  # pending, indexing, indexed, error
    
    # Relationship to frames
    frames = relationship("VideoFrame", back_populates="video", cascade="all, delete-orphan")
    
    def to_dict(self):
        return {
            'id': self.id,
            'source_name': self.source_name,
            'filename': self.filename,
            'title': self.title or self.source_name or self.filename,
            'duration': self.duration,
            'duration_formatted': self._format_duration(self.duration),
            'frame_count': self.frame_count,
            'resolution': f"{self.width}x{self.height}" if self.width and self.height else "Unknown",
            'fps': self.fps,
            'file_size': self.file_size,
            'file_size_formatted': self._format_size(self.file_size),
            'indexed_at': self.indexed_at.isoformat() if self.indexed_at else None,
            'status': self.status,
            'cloudinary_url': self.cloudinary_url,
            'cloudinary_public_id': self.cloudinary_public_id,
            'cloud_stored': self.cloudinary_url is not None,
        }
    
    @staticmethod
    def _format_duration(seconds):
        if not seconds:
            return "Unknown"
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        if hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        return f"{minutes}m {secs}s"
    
    @staticmethod
    def _format_size(size_bytes):
        if not size_bytes:
            return "Unknown"
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} TB"


class VideoFrame(Base):
    """Model for storing extracted frame features"""
    __tablename__ = 'video_frames'
    __table_args__ = (
        Index('idx_frame_video_number', 'video_id', 'frame_number'),
        Index('idx_frame_video_timestamp', 'video_id', 'timestamp'),
        Index('idx_frame_video_phash', 'video_id', 'phash'),
    )
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    video_id = Column(Integer, ForeignKey('reference_videos.id'), nullable=False)
    frame_number = Column(Integer, nullable=False)
    timestamp = Column(Float, nullable=False)  # Timestamp in seconds
    
    # Perceptual hashes stored as hex strings
    phash = Column(String(64), nullable=False)  # Perceptual hash
    dhash = Column(String(64), nullable=True)   # Difference hash
    whash = Column(String(64), nullable=True)   # Wavelet hash
    
    # Optional: average color features for quick filtering
    avg_color_r = Column(Integer, nullable=True)
    avg_color_g = Column(Integer, nullable=True)
    avg_color_b = Column(Integer, nullable=True)
    brightness = Column(Float, nullable=True)
    
    # Relationship to video
    video = relationship("ReferenceVideo", back_populates="frames")
    
    def to_dict(self):
        return {
            'id': self.id,
            'video_id': self.video_id,
            'frame_number': self.frame_number,
            'timestamp': self.timestamp,
            'timestamp_formatted': self._format_timestamp(self.timestamp),
            'phash': self.phash,
            'dhash': self.dhash,
            'whash': self.whash
        }
    
    @staticmethod
    def _format_timestamp(seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"


class MatchResult(Base):
    """Model for storing match query results"""
    __tablename__ = 'match_results'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    query_filename = Column(String(500), nullable=False)
    query_duration = Column(Float, nullable=True)
    matched_video_id = Column(Integer, ForeignKey('reference_videos.id'), nullable=True)
    confidence_score = Column(Float, nullable=False)
    start_timestamp = Column(Float, nullable=True)
    end_timestamp = Column(Float, nullable=True)
    match_details = Column(Text, nullable=True)  # JSON string with detailed results
    queried_at = Column(DateTime, default=datetime.utcnow)
    processing_time = Column(Float, nullable=True)  # Time taken in seconds
    
    # Relationship to matched video
    matched_video = relationship("ReferenceVideo")
    
    def to_dict(self):
        return {
            'id': self.id,
            'query_filename': self.query_filename,
            'query_duration': self.query_duration,
            'matched_video_id': self.matched_video_id,
            'matched_video': self.matched_video.to_dict() if self.matched_video else None,
            'confidence_score': self.confidence_score,
            'confidence_label': self._get_confidence_label(self.confidence_score),
            'start_timestamp': self.start_timestamp,
            'end_timestamp': self.end_timestamp,
            'timestamp_range': self._format_range(self.start_timestamp, self.end_timestamp),
            'queried_at': self.queried_at.isoformat() if self.queried_at else None,
            'processing_time': self.processing_time
        }
    
    @staticmethod
    def _get_confidence_label(score):
        if score >= 80:
            return "Very Strong Match"
        elif score >= 60:
            return "Strong Match"
        elif score >= 30:
            return "Possible Match"
        return "Weak/No Match"
    
    @staticmethod
    def _format_range(start, end):
        if start is None or end is None:
            return "Unknown"
        
        def fmt(seconds):
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            if hours > 0:
                return f"{hours:02d}:{minutes:02d}:{secs:02d}"
            return f"{minutes:02d}:{secs:02d}"
        
        return f"{fmt(start)} - {fmt(end)}"


# Database initialization
_engine = None
_session_factory = None


def init_db():
    """Initialize the database and create tables with SQLite optimizations."""
    global _engine, _session_factory
    from sqlalchemy import event
    
    _engine = create_engine(f"sqlite:///{DATABASE_PATH}")
    
    # SQLite performance pragmas: WAL mode for concurrent reads during writes,
    # NORMAL sync is safe for WAL mode and significantly faster for bulk inserts.
    @event.listens_for(_engine, 'connect')
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute('PRAGMA journal_mode=WAL')
        cursor.execute('PRAGMA synchronous=NORMAL')
        cursor.execute('PRAGMA cache_size=-64000')  # 64MB cache
        cursor.execute('PRAGMA temp_store=MEMORY')   # Temp tables in RAM
        cursor.close()
    
    Base.metadata.create_all(_engine)
    _session_factory = scoped_session(sessionmaker(bind=_engine))
    return _engine


def get_session():
    """Get a database session from the scoped factory (reuses per-thread)."""
    global _engine, _session_factory
    if _engine is None:
        init_db()
    return _session_factory()


# Initialize database on import
init_db()
