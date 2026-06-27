"""
ClipMatch Video Processor
Handles video loading, frame extraction, and preprocessing
"""

import cv2
import numpy as np
from pathlib import Path
import hashlib
from typing import Generator, Tuple, Optional, Dict, Any
import sys
sys.path.append('..')
from config import VideoConfig, REFERENCES_DIR, UPLOADS_DIR


class VideoProcessor:
    """
    Processes videos for frame extraction and analysis.
    Handles both reference videos and query clips.
    """
    
    def __init__(self, sample_rate: int = None):
        """
        Initialize the video processor.
        
        Args:
            sample_rate: Frames to extract per second. Defaults to config value.
        """
        self.sample_rate = sample_rate or VideoConfig.SAMPLE_RATE
        self.frame_width = VideoConfig.FRAME_WIDTH
        self.frame_height = VideoConfig.FRAME_HEIGHT
    
    def get_video_info(self, video_path: str) -> Dict[str, Any]:
        """
        Extract metadata from a video file.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary containing video metadata
        """
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        try:
            # Extract properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 30.0  # Assume 30fps as safe default
            
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Robust fallback for containers that return 0 frame count
            if frame_count <= 0:
                # Try seeking to end to get duration via milliseconds
                cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
                duration_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                if duration_ms > 0:
                    frame_count = int((duration_ms / 1000.0) * fps)
                else:
                    # Manual count: read all frames (expensive but reliable)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    count = 0
                    while True:
                        ret = cap.grab()
                        if not ret:
                            break
                        count += 1
                    frame_count = count
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset
            
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            
            # If duration is still 0 but we have frames, try millisecond-based duration
            if duration <= 0 and frame_count > 0:
                cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
                duration_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                if duration_ms > 0:
                    duration = duration_ms / 1000.0
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            # Calculate file checksum
            checksum = self._calculate_checksum(path)
            
            return {
                'filename': path.name,
                'filepath': str(path.absolute()),
                'duration': duration,
                'frame_count': frame_count,
                'width': width,
                'height': height,
                'fps': fps,
                'file_size': path.stat().st_size,
                'checksum': checksum,
                'format': path.suffix.lower()
            }
        finally:
            cap.release()
    
    def validate_reference_video(self, video_path: str) -> Tuple[bool, str]:
        """
        Validate a video file for use as a reference.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Tuple of (is_valid, message)
        """
        try:
            info = self.get_video_info(video_path)
            
            # Check format
            if info['format'] not in VideoConfig.SUPPORTED_FORMATS:
                return False, f"Unsupported format: {info['format']}"
            
            # Check file size
            if info['file_size'] > VideoConfig.MAX_REFERENCE_SIZE:
                return False, f"File too large: {info['file_size']} bytes"
            
            # Check duration (relaxed for testing)
            # In production, enforce MIN_REFERENCE_DURATION
            if info['duration'] < 10:  # Minimum 10 seconds for testing
                return False, f"Video too short: {info['duration']:.1f} seconds"
            
            if info['duration'] > VideoConfig.MAX_REFERENCE_DURATION:
                return False, f"Video too long: {info['duration']:.1f} seconds"
            
            return True, "Video is valid for indexing"
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    def validate_query_clip(self, video_path: str) -> Tuple[bool, str]:
        """
        Validate a video file for use as a query clip.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Tuple of (is_valid, message)
        """
        try:
            info = self.get_video_info(video_path)
            
            # Check format
            if info['format'] not in VideoConfig.SUPPORTED_FORMATS:
                return False, f"Unsupported format: {info['format']}"
            
            # Check file size
            if info['file_size'] > VideoConfig.MAX_CLIP_SIZE:
                return False, f"File too large: {info['file_size']} bytes"
            
            # Check duration
            if info['duration'] < VideoConfig.MIN_CLIP_DURATION:
                return False, f"Clip too short: {info['duration']:.1f} seconds (minimum: {VideoConfig.MIN_CLIP_DURATION}s)"
            
            if info['duration'] > VideoConfig.MAX_CLIP_DURATION:
                return False, f"Clip too long: {info['duration']:.1f} seconds (maximum: {VideoConfig.MAX_CLIP_DURATION}s)"
            
            return True, "Clip is valid for matching"
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    def extract_frames(self, video_path: str, 
                       start_time: float = 0, 
                       end_time: float = None,
                       query_mode: bool = False,
                       crop_mode: str = None) -> Generator[Tuple[int, float, np.ndarray], None, None]:
        """
        Extract frames from a video at the configured sample rate.
        
        Args:
            video_path: Path to the video file
            start_time: Start time in seconds (default: 0)
            end_time: End time in seconds (default: entire video)
            query_mode: If True, use relaxed crop for query clips
            crop_mode: Override crop mode: 'query', 'reference', 'refcrop', 'nocrop'
            
        Yields:
            Tuple of (frame_number, timestamp, frame_image)
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            
            # Calculate frame interval based on sample rate
            frame_interval = int(fps / self.sample_rate) if fps > 0 else 1
            frame_interval = max(1, frame_interval)
            
            # Set start position
            if start_time > 0:
                start_frame = int(start_time * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            # Calculate end frame
            end_frame = total_frames
            if end_time is not None and end_time < duration:
                end_frame = int(end_time * fps)
            
            # Choose preprocessor based on mode
            if crop_mode == 'nocrop':
                preprocess_fn = self._preprocess_frame_nocrop
            elif crop_mode in ('refcrop', 'reference'):
                preprocess_fn = self._preprocess_frame_refcrop
            elif crop_mode == 'query' or query_mode:
                preprocess_fn = self._preprocess_frame_query
            else:
                preprocess_fn = self._preprocess_frame
            
            frame_number = 0
            extracted_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                if current_frame >= end_frame:
                    break
                
                # Only yield frames at the sample interval
                if frame_number % frame_interval == 0:
                    timestamp = current_frame / fps if fps > 0 else 0
                    
                    # Resize frame for consistent processing
                    processed_frame = preprocess_fn(frame)
                    
                    yield extracted_count, timestamp, processed_frame
                    extracted_count += 1
                
                frame_number += 1
                
        finally:
            cap.release()
    
    def extract_all_frames_to_list(self, video_path: str, max_frames: int = 40,
                                    query_mode: bool = False,
                                    crop_mode: str = None) -> list:
        """
        Extract frames and return as a list (limited to max_frames for speed).
        
        Args:
            video_path: Path to the video file
            max_frames: Maximum number of frames to extract (default: 40 for fast processing)
            query_mode: If True, use relaxed crop for query clips
            crop_mode: Override crop mode: 'query', 'reference', 'refcrop', 'nocrop'
            
        Returns:
            List of (frame_number, timestamp, frame_image) tuples
        """
        frames = list(self.extract_frames(video_path, query_mode=query_mode, crop_mode=crop_mode))
        
        # Limit to max_frames by sampling evenly
        if len(frames) > max_frames:
            step = len(frames) // max_frames
            frames = frames[::step][:max_frames]
        
        return frames
    
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess a frame for REFERENCE indexing.
        Extracts center region to avoid captions, watermarks, and letterboxing.
        Uses 20% crop for reference videos (established index).
        
        Args:
            frame: Raw frame from video
            
        Returns:
            Preprocessed frame (center region only)
        """
        h, w = frame.shape[:2]
        
        # Extract center 60% of the frame to avoid:
        # - Captions (usually at bottom)
        # - Watermarks (usually at corners)
        # - Letterboxing/pillarboxing
        crop_percent = 0.20  # Remove 20% from each edge
        
        top = int(h * crop_percent)
        bottom = int(h * (1 - crop_percent))
        left = int(w * crop_percent)
        right = int(w * (1 - crop_percent))
        
        # Crop to center region
        center_crop = frame[top:bottom, left:right]
        
        # Resize to standard dimensions
        resized = cv2.resize(center_crop, (self.frame_width, self.frame_height))
        
        # Convert BGR to RGB (OpenCV uses BGR by default)
        rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        return rgb_frame
    
    def _preprocess_frame_query(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess a frame for QUERY matching.
        Uses a smaller 10% crop to preserve more content — query clips may have
        different framing, aspect ratios, or cropping than reference videos.
        
        Args:
            frame: Raw frame from video
            
        Returns:
            Preprocessed frame
        """
        h, w = frame.shape[:2]
        
        # Smaller crop for queries — keep 80% of pixels (vs 60% for references)
        # This is intentionally asymmetric: the hash comparison is still useful
        # because perceptual hashes are robust to moderate crop differences.
        crop_percent = 0.10  # Remove 10% from each edge
        
        top = int(h * crop_percent)
        bottom = int(h * (1 - crop_percent))
        left = int(w * crop_percent)
        right = int(w * (1 - crop_percent))
        
        center_crop = frame[top:bottom, left:right]
        resized = cv2.resize(center_crop, (self.frame_width, self.frame_height))
        rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        return rgb_frame
    
    def _preprocess_frame_refcrop(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess a frame using the SAME 20% crop as reference indexing.
        Use this as an augmentation to see if the query matches the reference
        when both use identical preprocessing.
        """
        h, w = frame.shape[:2]
        crop_percent = 0.20  # Same as _preprocess_frame
        top = int(h * crop_percent)
        bottom = int(h * (1 - crop_percent))
        left = int(w * crop_percent)
        right = int(w * (1 - crop_percent))
        center_crop = frame[top:bottom, left:right]
        resized = cv2.resize(center_crop, (self.frame_width, self.frame_height))
        rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        return rgb_frame
    
    def _preprocess_frame_nocrop(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess a frame with NO center crop — just resize and convert.
        Use this for query clips that are already cropped/cut from the original,
        so we don't lose even more content by cropping again.
        """
        resized = cv2.resize(frame, (self.frame_width, self.frame_height))
        rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        return rgb_frame
    
    def get_frame_at_timestamp(self, video_path: str, timestamp: float) -> Optional[np.ndarray]:
        """
        Extract a single frame at a specific timestamp.
        
        Args:
            video_path: Path to the video file
            timestamp: Time in seconds
            
        Returns:
            Frame image or None if extraction fails
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_number = int(timestamp * fps)
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            
            if ret:
                return self._preprocess_frame(frame)
            return None
            
        finally:
            cap.release()
    
    def extract_frame_at_index(self, video_path: str, frame_index: int) -> Optional[np.ndarray]:
        """
        Extract a single frame at a specific frame index.
        
        Args:
            video_path: Path to the video file
            frame_index: Frame number (0-indexed)
            
        Returns:
            Preprocessed frame image or None if extraction fails
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            
            if ret:
                return self._preprocess_frame(frame)
            return None
            
        finally:
            cap.release()
    
    def _calculate_checksum(self, filepath: Path, chunk_size: int = 8192) -> str:
        """
        Calculate SHA256 checksum of a file.
        
        Args:
            filepath: Path to the file
            chunk_size: Size of chunks to read
            
        Returns:
            Hex string of the checksum
        """
        sha256 = hashlib.sha256()
        with open(filepath, 'rb') as f:
            # Only hash first 10MB for performance
            bytes_read = 0
            max_bytes = 10 * 1024 * 1024
            while bytes_read < max_bytes:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                sha256.update(chunk)
                bytes_read += len(chunk)
        return sha256.hexdigest()
    
    def get_average_color(self, frame: np.ndarray) -> Tuple[int, int, int, float]:
        """
        Calculate average color and brightness of a frame.
        
        Args:
            frame: RGB frame image
            
        Returns:
            Tuple of (avg_r, avg_g, avg_b, brightness)
        """
        avg_color = np.mean(frame, axis=(0, 1))
        avg_r, avg_g, avg_b = int(avg_color[0]), int(avg_color[1]), int(avg_color[2])
        
        # Calculate perceived brightness
        brightness = (0.299 * avg_r + 0.587 * avg_g + 0.114 * avg_b) / 255.0
        
        return avg_r, avg_g, avg_b, brightness


# Singleton instance for convenience
video_processor = VideoProcessor()
