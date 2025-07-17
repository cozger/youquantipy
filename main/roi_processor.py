import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
import time

@dataclass
class ROIData:
    """Container for ROI data with transformation info."""
    roi_image: np.ndarray
    original_bbox: np.ndarray  # [x1, y1, x2, y2] in original frame
    padded_bbox: np.ndarray   # [x1, y1, x2, y2] with padding
    transform_matrix: np.ndarray  # 3x3 transformation matrix
    track_id: int
    frame_id: int
    quality_score: float
    
    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """Transform points from ROI coordinates to original frame coordinates."""
        if points is None or len(points) == 0:
            return points
            
        # Ensure points are in homogeneous coordinates
        if points.shape[-1] == 2:
            ones = np.ones((points.shape[0], 1))
            points_h = np.hstack([points, ones])
        else:
            points_h = points
            
        # Apply transformation
        transformed = points_h @ self.transform_matrix.T
        
        # Convert back to 2D
        return transformed[:, :2] / transformed[:, 2:3]

class ROIProcessor:
    def __init__(self,
                 target_size: Tuple[int, int] = (256, 256),
                 padding_ratio: float = 0.3,
                 min_quality_score: float = 0.5,
                 max_workers: int = 4):
        """
        ROI processor with coordinate transformation tracking.
        
        Args:
            target_size: Target size for ROI extraction
            padding_ratio: Padding ratio relative to bbox size
            min_quality_score: Minimum quality score for processing
            max_workers: Number of parallel workers
        """
        self.target_size = target_size
        self.padding_ratio = padding_ratio
        self.min_quality_score = min_quality_score
        self.max_workers = max_workers
        
        # Processing queues
        self.input_queue = queue.Queue(maxsize=100)
        self.output_queue = queue.Queue(maxsize=100)
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Processing thread
        self.is_running = False
        self.processing_thread = None
        
        # Cache for transformation matrices
        self.transform_cache = {}
        
    def start(self):
        """Start the ROI processing thread."""
        if not self.is_running:
            self.is_running = True
            self.processing_thread = threading.Thread(target=self._processing_loop)
            self.processing_thread.daemon = True
            self.processing_thread.start()
    
    def stop(self):
        """Stop the ROI processing thread."""
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=1.0)
        self.executor.shutdown(wait=False)
    
    def submit_frame_tracks(self, frame: np.ndarray, tracks: List[Dict], frame_id: int):
        """
        Submit frame and tracks for ROI extraction.
        
        Args:
            frame: Original frame
            tracks: List of tracks from tracker
            frame_id: Frame identifier
        """
        try:
            self.input_queue.put_nowait((frame, tracks, frame_id))
        except queue.Full:
            pass  # Drop frame if queue is full
    
    def get_processed_rois(self, timeout: float = 0.001) -> Optional[List[ROIData]]:
        """Get processed ROIs with transformation data."""
        try:
            return self.output_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def extract_roi(self, frame: np.ndarray, bbox: List[float], track_id: int, timestamp: float) -> Optional[Dict]:
        """Synchronous ROI extraction with frame dimensions"""
        track_data = {
            'bbox': bbox,
            'track_id': track_id
        }
        
        # Use internal extraction method
        roi_data = self._extract_roi(frame, track_data, 0)
        
        if roi_data is None:
            return None
            
        # Convert to expected format with frame dimensions
        return {
            'roi': roi_data.roi_image,
            'transform': {
                'scale': roi_data.transform_matrix[0, 0],
                'offset_x': roi_data.transform_matrix[0, 2],
                'offset_y': roi_data.transform_matrix[1, 2],
                'frame_width': frame.shape[1],  # Add frame dimensions
                'frame_height': frame.shape[0]
            },
            'quality_score': roi_data.quality_score,
            'original_bbox': roi_data.original_bbox,
            'padded_bbox': roi_data.padded_bbox
        }    

    def _processing_loop(self):
        """Main processing loop."""
        while self.is_running:
            try:
                # Get input data
                data = self.input_queue.get(timeout=0.1)
                if data is None:
                    continue
                    
                frame, tracks, frame_id = data
                
                # Process tracks in parallel
                futures = []
                for track in tracks:
                    future = self.executor.submit(
                        self._extract_roi, frame, track, frame_id
                    )
                    futures.append(future)
                
                # Collect results
                rois = []
                for future in futures:
                    roi_data = future.result()
                    if roi_data is not None:
                        rois.append(roi_data)
                
                # Output results
                if rois:
                    self.output_queue.put(rois)
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"ROI processing error: {e}")
    
    def _extract_roi(self, frame: np.ndarray, track: Dict, frame_id: int) -> Optional[ROIData]:
        """Extract and process a single ROI."""
        bbox = np.array(track['bbox'])
        track_id = track['track_id']
        
        # Calculate padded bbox
        padded_bbox = self._calculate_padded_bbox(bbox, frame.shape)
        
        # Check quality
        quality_score = self._calculate_quality_score(bbox, frame.shape)
        if quality_score < self.min_quality_score:
            return None
        
        # Extract ROI with padding
        x1, y1, x2, y2 = padded_bbox.astype(int)
        roi = frame[y1:y2, x1:x2]
        
        if roi.size == 0:
            return None
        
        # Calculate transformation matrix
        transform_matrix = self._calculate_transform_matrix(
            padded_bbox, self.target_size
        )
        
        # Resize ROI to target size
        roi_resized = cv2.resize(roi, self.target_size, interpolation=cv2.INTER_LINEAR)
        
        # Create ROI data
        roi_data = ROIData(
            roi_image=roi_resized,
            original_bbox=bbox,
            padded_bbox=padded_bbox,
            transform_matrix=transform_matrix,
            track_id=track_id,
            frame_id=frame_id,
            quality_score=quality_score
        )
        
        return roi_data
    
    def _calculate_padded_bbox(self, bbox: np.ndarray, frame_shape: Tuple) -> np.ndarray:
        """Calculate bbox with padding."""
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        
        # Calculate padding
        pad_w = w * self.padding_ratio
        pad_h = h * self.padding_ratio
        
        # Apply padding
        x1_pad = max(0, x1 - pad_w)
        y1_pad = max(0, y1 - pad_h)
        x2_pad = min(frame_shape[1], x2 + pad_w)
        y2_pad = min(frame_shape[0], y2 + pad_h)
        
        # Ensure square aspect ratio if needed
        w_pad = x2_pad - x1_pad
        h_pad = y2_pad - y1_pad
        
        if w_pad > h_pad:
            # Adjust height
            diff = (w_pad - h_pad) / 2
            y1_pad = max(0, y1_pad - diff)
            y2_pad = min(frame_shape[0], y2_pad + diff)
        elif h_pad > w_pad:
            # Adjust width
            diff = (h_pad - w_pad) / 2
            x1_pad = max(0, x1_pad - diff)
            x2_pad = min(frame_shape[1], x2_pad + diff)
        
        return np.array([x1_pad, y1_pad, x2_pad, y2_pad])
    
    def _calculate_quality_score(self, bbox: np.ndarray, frame_shape: Tuple) -> float:
        """Calculate quality score for ROI."""
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        
        # Size score (prefer larger faces)
        size_score = min(1.0, (w * h) / (frame_shape[0] * frame_shape[1] * 0.1))
        
        # Aspect ratio score (prefer square-ish faces)
        aspect_ratio = w / h if h > 0 else 0
        ar_score = 1.0 - abs(1.0 - aspect_ratio) * 0.5
        
        # Position score (prefer centered faces)
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        pos_score = 1.0 - abs(cx - frame_shape[1]/2) / frame_shape[1] * 0.5
        pos_score *= 1.0 - abs(cy - frame_shape[0]/2) / frame_shape[0] * 0.5
        
        # Boundary score (penalize faces near edges)
        margin = min(x1, y1, frame_shape[1] - x2, frame_shape[0] - y2)
        boundary_score = min(1.0, margin / 50.0)
        
        # Combined score
        quality_score = (size_score * 0.4 + 
                        ar_score * 0.2 + 
                        pos_score * 0.2 + 
                        boundary_score * 0.2)
        
        return quality_score
    
    def _calculate_transform_matrix(self, 
                                   padded_bbox: np.ndarray,
                                   target_size: Tuple[int, int]) -> np.ndarray:
        """
        Calculate transformation matrix from ROI to original coordinates.
        
        Returns:
            3x3 transformation matrix
        """
        x1, y1, x2, y2 = padded_bbox
        w = x2 - x1
        h = y2 - y1
        
        # Scale factors
        scale_x = w / target_size[0]
        scale_y = h / target_size[1]
        
        # Translation
        tx = x1
        ty = y1
        
        # Build transformation matrix
        # This transforms from ROI coordinates to original frame coordinates
        transform = np.array([
            [scale_x, 0, tx],
            [0, scale_y, ty],
            [0, 0, 1]
        ], dtype=np.float32)
        
        return transform
    
    def transform_landmarks(self, 
                           landmarks: np.ndarray,
                           roi_data: ROIData) -> np.ndarray:
        """
        Transform landmarks from ROI space to original frame space.
        
        Args:
            landmarks: Landmarks in ROI coordinates
            roi_data: ROI data with transformation info
            
        Returns:
            Landmarks in original frame coordinates
        """
        if landmarks is None:
            return None
            
        return roi_data.transform_points(landmarks)
    
    def get_roi_for_recognition(self, roi_data: ROIData) -> np.ndarray:
        """
        Get ROI prepared for face recognition.
        
        Args:
            roi_data: ROI data
            
        Returns:
            Preprocessed face image for recognition
        """
        # Extract face region more tightly for recognition
        roi = roi_data.roi_image
        
        # Convert to RGB if needed
        if len(roi.shape) == 2:
            roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)
        elif roi.shape[2] == 4:
            roi = cv2.cvtColor(roi, cv2.COLOR_BGRA2RGB)
        elif roi.shape[2] == 3:
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        
        # Resize to recognition model input size (typically 112x112 for ArcFace)
        roi_recognition = cv2.resize(roi, (112, 112), interpolation=cv2.INTER_LINEAR)
        
        # Normalize
        roi_recognition = roi_recognition.astype(np.float32)
        roi_recognition = (roi_recognition - 127.5) / 128.0
        
        return roi_recognition
    
    def visualize_roi_transforms(self, frame: np.ndarray, roi_data: ROIData) -> np.ndarray:
        """
        Visualize ROI and transformation for debugging.
        
        Args:
            frame: Original frame
            roi_data: ROI data with transformation
            
        Returns:
            Visualization image
        """
        vis = frame.copy()
        
        # Draw original bbox
        x1, y1, x2, y2 = roi_data.original_bbox.astype(int)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw padded bbox
        x1p, y1p, x2p, y2p = roi_data.padded_bbox.astype(int)
        cv2.rectangle(vis, (x1p, y1p), (x2p, y2p), (255, 0, 0), 1)
        
        # Draw quality score
        cv2.putText(vis, f"Q: {roi_data.quality_score:.2f}", 
                   (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, (255, 255, 255), 1)
        
        # Test transformation by drawing corners
        corners_roi = np.array([
            [0, 0],
            [self.target_size[0], 0],
            [self.target_size[0], self.target_size[1]],
            [0, self.target_size[1]]
        ])
        
        corners_frame = roi_data.transform_points(corners_roi)
        
        for i in range(4):
            pt = tuple(corners_frame[i].astype(int))
            cv2.circle(vis, pt, 3, (0, 0, 255), -1)
        
        return vis