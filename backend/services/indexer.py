"""
ClipMatch Indexer
Handles indexing of reference videos for later matching
"""

import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
import sys
sys.path.append('..')
from config import REFERENCES_DIR, VideoConfig
from models.database import get_session, ReferenceVideo, VideoFrame
from services.video_processor import VideoProcessor
from services.feature_extractor import FeatureExtractor


class VideoIndexer:
    """
    Indexes reference videos by extracting and storing visual features.
    
    The indexing process:
    1. Validates the video file
    2. Extracts frames at the configured sample rate
    3. Computes perceptual hashes for each frame
    4. Stores metadata and features in the database
    """
    
    def __init__(self):
        """Initialize the indexer with processor and extractor."""
        self.video_processor = VideoProcessor()
        self.feature_extractor = FeatureExtractor()
    
    def index_video(self, video_path: str, 
                    title: str = None,
                    progress_callback=None) -> Dict:
        """
        Index a single reference video.
        
        Args:
            video_path: Path to the video file
            title: Optional title for the video
            progress_callback: Optional callback function for progress updates
            
        Returns:
            Dictionary with indexing result
        """
        path = Path(video_path)
        
        # Validate video
        is_valid, message = self.video_processor.validate_reference_video(str(path))
        if not is_valid:
            return {
                'success': False,
                'error': message
            }
        
        session = get_session()
        
        try:
            # Check if already indexed
            existing = session.query(ReferenceVideo).filter(
                ReferenceVideo.filename == path.name
            ).first()
            
            if existing:
                return {
                    'success': False,
                    'error': f'Video "{path.name}" is already indexed',
                    'video_id': existing.id
                }
            
            # Get video info
            video_info = self.video_processor.get_video_info(str(path))
            
            # Create database record
            video_record = ReferenceVideo(
                filename=video_info['filename'],
                filepath=video_info['filepath'],
                title=title or video_info['filename'],
                duration=video_info['duration'],
                frame_count=video_info['frame_count'],
                width=video_info['width'],
                height=video_info['height'],
                fps=video_info['fps'],
                file_size=video_info['file_size'],
                checksum=video_info['checksum'],
                status='indexing'
            )
            
            session.add(video_record)
            session.commit()
            
            video_id = video_record.id
            
            # Extract and process frames
            frames_indexed = 0
            total_frames = int(video_info['duration'] * VideoConfig.SAMPLE_RATE)
            
            if progress_callback:
                progress_callback(0, total_frames, "Starting frame extraction...")
            
            frame_records = []
            
            for frame_num, timestamp, frame in self.video_processor.extract_frames(str(path)):
                # Extract features
                features = self.feature_extractor.extract_features(frame)
                avg_r, avg_g, avg_b, brightness = self.video_processor.get_average_color(frame)
                
                # Create frame record
                frame_record = VideoFrame(
                    video_id=video_id,
                    frame_number=frame_num,
                    timestamp=timestamp,
                    phash=features.get('phash', ''),
                    dhash=features.get('dhash', ''),
                    whash=features.get('whash', ''),
                    avg_color_r=avg_r,
                    avg_color_g=avg_g,
                    avg_color_b=avg_b,
                    brightness=brightness
                )
                
                frame_records.append(frame_record)
                frames_indexed += 1
                
                # Batch insert every 500 frames for better performance
                if len(frame_records) >= 500:
                    session.bulk_save_objects(frame_records)
                    session.commit()
                    frame_records = []
                
                if progress_callback:
                    progress_callback(frames_indexed, total_frames, 
                                    f"Processing frame {frames_indexed}...")
            
            # Insert remaining frames
            if frame_records:
                session.bulk_save_objects(frame_records)
                session.commit()
            
            # Update video status
            video_record.status = 'indexed'
            video_record.frame_count = frames_indexed
            session.commit()
            
            if progress_callback:
                progress_callback(frames_indexed, frames_indexed, "Indexing complete!")
            
            return {
                'success': True,
                'video_id': video_id,
                'filename': video_info['filename'],
                'title': title or video_info['filename'],
                'duration': video_info['duration'],
                'frames_indexed': frames_indexed,
                'message': f'Successfully indexed {frames_indexed} frames'
            }
            
        except Exception as e:
            session.rollback()
            # Mark as error if record was created
            try:
                video_record = session.query(ReferenceVideo).filter(
                    ReferenceVideo.filename == path.name
                ).first()
                if video_record:
                    video_record.status = 'error'
                    session.commit()
            except:
                pass
            
            return {
                'success': False,
                'error': f'Indexing failed: {str(e)}'
            }
            
        finally:
            session.close()
    
    def index_directory(self, directory_path: str = None,
                        progress_callback=None) -> Dict:
        """
        Index all videos in a directory.
        
        Args:
            directory_path: Path to directory (default: references directory)
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary with indexing results
        """
        directory = Path(directory_path) if directory_path else REFERENCES_DIR
        
        if not directory.exists():
            return {
                'success': False,
                'error': f'Directory not found: {directory}'
            }
        
        # Find all video files
        video_files = []
        for ext in VideoConfig.SUPPORTED_FORMATS:
            video_files.extend(directory.glob(f'*{ext}'))
            video_files.extend(directory.glob(f'*{ext.upper()}'))
        
        if not video_files:
            return {
                'success': False,
                'error': 'No video files found in directory',
                'directory': str(directory)
            }
        
        results = {
            'success': True,
            'total_videos': len(video_files),
            'indexed': [],
            'failed': [],
            'skipped': []
        }
        
        for i, video_path in enumerate(video_files):
            if progress_callback:
                progress_callback(i, len(video_files), f"Indexing: {video_path.name}")
            
            result = self.index_video(str(video_path))
            
            if result['success']:
                results['indexed'].append({
                    'filename': video_path.name,
                    'video_id': result['video_id'],
                    'frames': result['frames_indexed']
                })
            elif 'already indexed' in result.get('error', '').lower():
                results['skipped'].append({
                    'filename': video_path.name,
                    'reason': 'Already indexed'
                })
            else:
                results['failed'].append({
                    'filename': video_path.name,
                    'error': result['error']
                })
        
        return results
    
    def reindex_video(self, video_id: int) -> Dict:
        """
        Re-index a video by removing old features and re-extracting.
        
        Args:
            video_id: Database ID of the video
            
        Returns:
            Dictionary with reindexing result
        """
        session = get_session()
        
        try:
            video = session.query(ReferenceVideo).filter(
                ReferenceVideo.id == video_id
            ).first()
            
            if not video:
                return {
                    'success': False,
                    'error': f'Video with ID {video_id} not found'
                }
            
            filepath = video.filepath
            
            # Delete existing frames
            session.query(VideoFrame).filter(
                VideoFrame.video_id == video_id
            ).delete()
            
            # Delete video record
            session.delete(video)
            session.commit()
            
            # Re-index
            return self.index_video(filepath)
            
        except Exception as e:
            session.rollback()
            return {
                'success': False,
                'error': f'Reindexing failed: {str(e)}'
            }
        finally:
            session.close()
    
    def delete_video(self, video_id: int) -> Dict:
        """
        Remove a video from the index.
        
        Args:
            video_id: Database ID of the video
            
        Returns:
            Dictionary with deletion result
        """
        session = get_session()
        
        try:
            video = session.query(ReferenceVideo).filter(
                ReferenceVideo.id == video_id
            ).first()
            
            if not video:
                return {
                    'success': False,
                    'error': f'Video with ID {video_id} not found'
                }
            
            filename = video.filename
            
            # Delete frames first (cascade should handle this, but be explicit)
            session.query(VideoFrame).filter(
                VideoFrame.video_id == video_id
            ).delete()
            
            # Delete video record
            session.delete(video)
            session.commit()
            
            return {
                'success': True,
                'message': f'Removed "{filename}" from index'
            }
            
        except Exception as e:
            session.rollback()
            return {
                'success': False,
                'error': f'Deletion failed: {str(e)}'
            }
        finally:
            session.close()
    
    def get_indexed_videos(self) -> List[Dict]:
        """
        Get list of all indexed videos.
        
        Returns:
            List of video dictionaries
        """
        session = get_session()
        
        try:
            videos = session.query(ReferenceVideo).all()
            return [v.to_dict() for v in videos]
        finally:
            session.close()
    
    def get_video_details(self, video_id: int) -> Optional[Dict]:
        """
        Get detailed information about an indexed video.
        
        Args:
            video_id: Database ID of the video
            
        Returns:
            Video details dictionary or None
        """
        session = get_session()
        
        try:
            video = session.query(ReferenceVideo).filter(
                ReferenceVideo.id == video_id
            ).first()
            
            if not video:
                return None
            
            frame_count = session.query(VideoFrame).filter(
                VideoFrame.video_id == video_id
            ).count()
            
            details = video.to_dict()
            details['actual_frame_count'] = frame_count
            
            return details
            
        finally:
            session.close()
    
    def get_index_stats(self) -> Dict:
        """
        Get statistics about the indexed video library.
        
        Returns:
            Dictionary with index statistics
        """
        session = get_session()
        
        try:
            total_videos = session.query(ReferenceVideo).count()
            indexed_videos = session.query(ReferenceVideo).filter(
                ReferenceVideo.status == 'indexed'
            ).count()
            total_frames = session.query(VideoFrame).count()
            
            # Calculate total duration
            videos = session.query(ReferenceVideo).filter(
                ReferenceVideo.status == 'indexed'
            ).all()
            total_duration = sum(v.duration for v in videos)
            
            return {
                'total_videos': total_videos,
                'indexed_videos': indexed_videos,
                'pending_videos': total_videos - indexed_videos,
                'total_frames': total_frames,
                'total_duration': total_duration,
                'total_duration_formatted': self._format_duration(total_duration)
            }
            
        finally:
            session.close()
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable form."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        parts = []
        if hours > 0:
            parts.append(f"{hours}h")
        if minutes > 0:
            parts.append(f"{minutes}m")
        if secs > 0 or not parts:
            parts.append(f"{secs}s")
        
        return " ".join(parts)


# Singleton instance
video_indexer = VideoIndexer()
