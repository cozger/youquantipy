"""
Frame Router & Buffer Manager
Maintains circular buffer of recent frames and routes them to detection/tracking
"""

import numpy as np
import time
from collections import deque
from threading import Lock
import cv2
from typing import Optional, Tuple, Dict, List

class FrameRouterBuffer:
    def __init__(self, buffer_size: int = 30, detection_interval: int = 7):
        """
        Initialize Frame Router and Buffer Manager.
        
        Args:
            buffer_size: Number of frames to keep in circular buffer
            detection_interval: Submit frame to detector every N frames
        """
        self.buffer_size = buffer_size
        self.detection_interval = detection_interval
        
        # Circular buffer for frames
        self.frame_buffer = deque(maxlen=buffer_size)
        self.frame_metadata = deque(maxlen=buffer_size)
        self.buffer_lock = Lock()
        
        # Frame counting
        self.frame_count = 0
        self.last_detection_frame = 0
        
        # Resolution info
        self.full_resolution = None
        self.downscale_resolution = (640, 480)
        
    def add_frame(self, frame: np.ndarray, timestamp: float) -> Tuple[bool, bool, np.ndarray]:
        """
        Add frame to buffer and determine routing.
        
        Returns:
            Tuple of (should_detect, should_track, downscaled_frame)
        """
        with self.buffer_lock:
            # Store full resolution
            if self.full_resolution is None:
                self.full_resolution = frame.shape[:2]
            
            # Add to buffer
            self.frame_buffer.append(frame)
            self.frame_metadata.append({
                'frame_id': self.frame_count,
                'timestamp': timestamp
            })
            
            # Determine if we should run detection
            should_detect = (self.frame_count % self.detection_interval == 0)
            
            # Always track (every frame)
            should_track = True
            
            # Create downscaled frame for tracking
            if frame.shape[:2][::-1] != self.downscale_resolution:
                frame_downscaled = cv2.resize(frame, self.downscale_resolution)
            else:
                frame_downscaled = frame
            
            self.frame_count += 1
            
            return should_detect, should_track, frame_downscaled
    
    def get_frame_by_id(self, frame_id: int) -> Optional[np.ndarray]:
        """Get a specific frame from buffer by ID."""
        with self.buffer_lock:
            for i, metadata in enumerate(self.frame_metadata):
                if metadata['frame_id'] == frame_id:
                    return self.frame_buffer[i]
        return None
    
    def get_latest_frame(self) -> Optional[Tuple[np.ndarray, Dict]]:
        """Get the most recent frame and its metadata."""
        with self.buffer_lock:
            if self.frame_buffer:
                return self.frame_buffer[-1], self.frame_metadata[-1]
        return None
    
    def get_frame_for_roi(self, frame_id: int) -> Optional[np.ndarray]:
        """
        Get frame for ROI extraction.
        Falls back to latest frame if specific frame not in buffer.
        """
        frame = self.get_frame_by_id(frame_id)
        if frame is None:
            # Frame might have been evicted, use latest
            latest = self.get_latest_frame()
            if latest:
                frame = latest[0]
        return frame
    
    def clear(self):
        """Clear the buffer."""
        with self.buffer_lock:
            self.frame_buffer.clear()
            self.frame_metadata.clear()
            self.frame_count = 0