"""
ROI Manager
Extracts face regions from 4K buffer, applies padding, resizes to fixed size
"""

import numpy as np
import cv2
import time
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import queue
import threading

@dataclass
class ROIRequest:
    """Request for ROI extraction."""
    frame_id: int
    track_id: int
    bbox: List[float]  # [x1, y1, x2, y2]
    padding: float = 1.5

@dataclass
class ROIResult:
    """Result of ROI extraction."""
    roi_image: np.ndarray
    track_id: int
    frame_id: int
    original_bbox: np.ndarray
    padded_bbox: np.ndarray
    transform_matrix: np.ndarray

class ROIManager:
    def __init__(self, target_size: Tuple[int, int] = (256, 256), padding_ratio: float = 1.5):
        """
        Initialize ROI Manager.
        
        Args:
            target_size: Target size for ROIs (width, height)
            padding_ratio: Padding ratio for bounding boxes
        """
        self.target_size = target_size
        self.padding_ratio = padding_ratio
        
        # Queues for async processing
        self.request_queue = queue.Queue(maxsize=100)
        self.result_queue = queue.Queue(maxsize=100)
        
        # Frame buffer reference (will be set by camera worker)
        self.frame_buffer = None
        
        # Processing thread
        self.processing_thread = None
        self.is_running = False
        
        # Face ID to ROI mapping
        self.face_roi_mapping = {}
        self.mapping_lock = threading.Lock()
        
    def set_frame_buffer(self, frame_buffer):
        """Set reference to frame buffer."""
        self.frame_buffer = frame_buffer
        
    def start(self):
        """Start ROI processing thread."""
        if not self.is_running:
            self.is_running = True
            self.processing_thread = threading.Thread(target=self._process_loop)
            self.processing_thread.daemon = True
            self.processing_thread.start()
            print("[ROI Manager] Started")
    
    def stop(self):
        """Stop ROI processing thread."""
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=1.0)
        print("[ROI Manager] Stopped")
    
    def submit_roi_request(self, frame_id: int, track_id: int, bbox: List[float]):
        """Submit ROI extraction request."""
        try:
            request = ROIRequest(
                frame_id=frame_id,
                track_id=track_id,
                bbox=bbox,
                padding=self.padding_ratio
            )
            self.request_queue.put_nowait(request)
            return True
        except queue.Full:
            return False
    
    def submit_batch_requests(self, frame_id: int, tracks: List[Dict]):
        """Submit multiple ROI requests for a frame."""
        submitted = 0
        for track in tracks:
            if 'bbox' in track and 'track_id' in track:
                if self.submit_roi_request(frame_id, track['track_id'], track['bbox']):
                    submitted += 1
        return submitted
    
    def get_roi_results(self, timeout: float = 0.001) -> List[ROIResult]:
        """Get available ROI results."""
        results = []
        deadline = time.time() + timeout
        
        while time.time() < deadline:
            try:
                result = self.result_queue.get_nowait()
                results.append(result)
            except queue.Empty:
                break
        
        return results
    
    def _process_loop(self):
        """Main processing loop for ROI extraction."""
        while self.is_running:
            try:
                request = self.request_queue.get(timeout=0.1)
                if request and self.frame_buffer:
                    result = self._extract_roi(request)
                    if result:
                        try:
                            self.result_queue.put_nowait(result)
                        except queue.Full:
                            pass
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[ROI Manager] Error: {e}")
    
    def _extract_roi(self, request: ROIRequest) -> Optional[ROIResult]:
        """Extract ROI from frame."""
        # Get frame from buffer
        frame = self.frame_buffer.get_frame_for_roi(request.frame_id)
        if frame is None:
            return None
        
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = request.bbox
        
        # Calculate padded bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        width = x2 - x1
        height = y2 - y1
        
        # Apply padding
        padded_width = width * request.padding
        padded_height = height * request.padding
        
        # Keep aspect ratio square for face ROIs
        max_dim = max(padded_width, padded_height)
        
        # Calculate padded coordinates
        px1 = int(max(0, cx - max_dim / 2))
        py1 = int(max(0, cy - max_dim / 2))
        px2 = int(min(w, cx + max_dim / 2))
        py2 = int(min(h, cy + max_dim / 2))
        
        # Extract ROI
        roi = frame[py1:py2, px1:px2]
        
        if roi.size == 0:
            return None
        
        # Resize to target size
        roi_resized = cv2.resize(roi, self.target_size)
        
        # Calculate transform matrix for mapping back to original coordinates
        scale_x = (px2 - px1) / self.target_size[0]
        scale_y = (py2 - py1) / self.target_size[1]
        
        transform_matrix = np.array([
            [scale_x, 0, px1],
            [0, scale_y, py1],
            [0, 0, 1]
        ])
        
        # Update mapping
        with self.mapping_lock:
            self.face_roi_mapping[request.track_id] = {
                'frame_id': request.frame_id,
                'original_bbox': np.array(request.bbox),
                'padded_bbox': np.array([px1, py1, px2, py2]),
                'transform': transform_matrix
            }
        
        return ROIResult(
            roi_image=roi_resized,
            track_id=request.track_id,
            frame_id=request.frame_id,
            original_bbox=np.array(request.bbox),
            padded_bbox=np.array([px1, py1, px2, py2]),
            transform_matrix=transform_matrix
        )
    
    def transform_landmarks_to_original(self, track_id: int, landmarks: np.ndarray) -> Optional[np.ndarray]:
        """Transform landmarks from ROI coordinates back to original frame coordinates."""
        with self.mapping_lock:
            if track_id in self.face_roi_mapping:
                mapping = self.face_roi_mapping[track_id]
                transform = mapping['transform']
                
                # Add homogeneous coordinate
                landmarks_h = np.hstack([landmarks, np.ones((landmarks.shape[0], 1))])
                
                # Apply transform
                landmarks_original = (transform @ landmarks_h.T).T[:, :2]
                
                return landmarks_original
        
        return None