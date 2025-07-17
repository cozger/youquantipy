import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from scipy.optimize import linear_sum_assignment
import time

@dataclass
class Track:
    """Represents a single face track."""
    track_id: int
    bbox: np.ndarray  # [x1, y1, x2, y2]
    kalman: cv2.KalmanFilter
    confidence: float
    age: int
    hits: int
    hit_streak: int
    time_since_update: int
    features: Optional[np.ndarray] = None
    landmarks: Optional[np.ndarray] = None
    
    # Drift correction
    reference_features: Optional[np.ndarray] = None
    drift_accumulated: np.ndarray = None
    
    # Sliding window hit tracking
    hit_history: List[int] = None  # Frame numbers where hits occurred
    
    def __post_init__(self):
        if self.drift_accumulated is None:
            self.drift_accumulated = np.zeros(2)
        if self.hit_history is None:
            self.hit_history = []

class LightweightTracker:
    def __init__(self,
                 max_age: int = 30,
                 min_hits: int = 3,
                 iou_threshold: float = 0.3,
                 max_drift: float = 50.0,
                 drift_correction_rate: float = 0.1,
                 detection_interval: int = 7,
                 min_hits_in_window: int = 3,
                 window_cycles: int = 4,
                 max_tracks: int = None):
        """
        Lightweight tracker with optical flow and drift correction.
        
        Args:
            max_age: Maximum frames to keep track alive without detection
            min_hits: Minimum hits before track is confirmed (deprecated, use min_hits_in_window)
            iou_threshold: IOU threshold for matching
            max_drift: Maximum allowed drift in pixels
            drift_correction_rate: Rate of drift correction (0-1)
            detection_interval: How often detection runs (every N frames)
            min_hits_in_window: Minimum hits required within sliding window
            window_cycles: Number of detection cycles for sliding window
            max_tracks: Maximum number of tracks to maintain (None for unlimited)
        """
        self.max_age = max_age
        self.min_hits = min_hits  # Keep for backward compatibility
        self.iou_threshold = iou_threshold
        self.max_drift = max_drift
        self.drift_correction_rate = drift_correction_rate
        self.detection_interval = detection_interval
        self.min_hits_in_window = min_hits_in_window
        self.window_size = detection_interval * window_cycles  # Adaptive window size
        self.max_tracks = max_tracks
        
        self.tracks: List[Track] = []
        self.track_id_counter = 0
        self.frame_count = 0
        
        # Optical flow parameters
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        # Previous frame for optical flow
        self.prev_gray = None
        
    def _has_enough_hits_in_window(self, track: Track) -> bool:
        """Check if track has enough hits within the sliding window."""
        if not track.hit_history:
            return False
            
        # Remove old hits outside the window
        current_frame = self.frame_count
        window_start = current_frame - self.window_size
        track.hit_history = [f for f in track.hit_history if f > window_start]
        
        # Count hits in window
        hits_in_window = len(track.hit_history)
        return hits_in_window >= self.min_hits_in_window
        
    def update(self, frame: np.ndarray, detections: List[Dict]) -> List[Dict]:
        """
        Update tracker with new detections.
        
        Args:
            frame: Current frame
            detections: List of detections from RetinaFace
            
        Returns:
            List of tracked objects with IDs
        """
        self.frame_count += 1
        
        if self.frame_count % 240 == 0 or len(detections) > 0:
            print(f"[TRACKER] update() called: frame {self.frame_count}, {len(detections)} detections")
            if len(detections) > 0:
                print(f"[TRACKER] First detection: {detections[0]}")
        
        # Frame distributor passes RGB frames, not BGR
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # Predict new locations of existing tracks
        for track in self.tracks:
            self._predict_track(track)
        
        # Apply optical flow if previous frame exists
        if self.prev_gray is not None and len(self.tracks) > 0:
            self._apply_optical_flow(self.prev_gray, gray)
        
        # Match detections to tracks
        matched, unmatched_dets, unmatched_trks = self._match_detections(detections)
        
        if self.frame_count % 240 == 0 or len(detections) > 0:
            print(f"[TRACKER] Frame {self.frame_count}: {len(detections)} detections, {len(self.tracks)} tracks")
            print(f"[TRACKER] Matching: {len(matched)} matched, {len(unmatched_dets)} unmatched dets, {len(unmatched_trks)} unmatched tracks")
        
        # Update matched tracks
        for m in matched:
            self._update_with_detection(self.tracks[m[1]], detections[m[0]], frame)
        
        # Create new tracks for unmatched detections
        for i in unmatched_dets:
            # Check if we're at max tracks limit
            if self.max_tracks is not None and len(self.tracks) >= self.max_tracks:
                print(f"[TRACKER] Cannot create new track - at max_tracks limit ({self.max_tracks})")
                continue
            
            trk = self._init_track(detections[i])
            self.tracks.append(trk)
            print(f"[TRACKER] Created new track {trk.track_id} at frame {self.frame_count}")
        
        # Handle unmatched tracks
        for i in unmatched_trks:
            self.tracks[i].time_since_update += 1
            self.tracks[i].hit_streak = 0
        
        # Remove dead tracks
        self.tracks = [t for t in self.tracks if t.time_since_update < self.max_age]
        
        # Apply drift correction
        self._apply_drift_correction()
        
        # Prepare output
        results = []
        for track in self.tracks:
            # Use sliding window check for track confirmation
            if self._has_enough_hits_in_window(track) or track.age < self.detection_interval:
                result = {
                    'track_id': track.track_id,
                    'bbox': track.bbox.tolist(),
                    'confidence': track.confidence,
                    'age': track.age,
                    'landmarks': track.landmarks.tolist() if track.landmarks is not None else None
                }
                results.append(result)
                
        if self.frame_count % 240 == 0 and len(results) > 0:
            print(f"[TRACKER] Returning {len(results)} confirmed tracks at frame {self.frame_count}")
        
        self.prev_gray = gray
        return results
    
    def _init_track(self, detection: Dict) -> Track:
        """Initialize a new track from detection."""
        bbox = np.array(detection['bbox'])
        
        # Initialize Kalman filter (constant velocity model)
        kf = cv2.KalmanFilter(7, 4)
        kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ], np.float32)
        
        kf.transitionMatrix = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ], np.float32)
        
        kf.processNoiseCov = 0.03 * np.eye(7, dtype=np.float32)
        kf.measurementNoiseCov = 0.1 * np.eye(4, dtype=np.float32)
        
        # Initialize state
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        
        kf.statePre = np.array([cx, cy, w, h, 0, 0, 0], np.float32)
        kf.statePost = kf.statePre.copy()
        
        track = Track(
            track_id=self.track_id_counter + 1,
            bbox=bbox,
            kalman=kf,
            confidence=detection['confidence'],
            age=0,
            hits=1,
            hit_streak=1,
            time_since_update=0,
            landmarks=detection.get('landmarks'),
            hit_history=[self.frame_count]  # Initialize with first hit
        )
        
        self.track_id_counter += 1
        return track
    
    def _predict_track(self, track: Track):
        """Predict new location using Kalman filter."""
        prediction = track.kalman.predict()
        
        cx, cy, w, h = prediction[:4].flatten()
        track.bbox = np.array([
            cx - w/2,
            cy - h/2,
            cx + w/2,
            cy + h/2
        ])
        
        track.age += 1
        if track.time_since_update > 0:
            track.hit_streak = 0
    
    def _apply_optical_flow(self, prev_gray: np.ndarray, curr_gray: np.ndarray):
        """Apply optical flow to refine track positions."""
        for track in self.tracks:
            if track.time_since_update > 0:  # Only for tracks without detection
                # Get feature points in bbox region
                x1, y1, x2, y2 = track.bbox.astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(prev_gray.shape[1], x2), min(prev_gray.shape[0], y2)
                
                # Extract good features to track
                roi = prev_gray[y1:y2, x1:x2]
                features = cv2.goodFeaturesToTrack(
                    roi, maxCorners=20, qualityLevel=0.3,
                    minDistance=7, blockSize=7
                )
                
                if features is not None and len(features) > 5:
                    # Convert to absolute coordinates
                    features = features.reshape(-1, 2)
                    features[:, 0] += x1
                    features[:, 1] += y1
                    
                    # Calculate optical flow
                    next_pts, status, error = cv2.calcOpticalFlowPyrLK(
                        prev_gray, curr_gray, features, None, **self.lk_params
                    )
                    
                    # Filter good points
                    good_old = features[status.flatten() == 1]
                    good_new = next_pts[status.flatten() == 1]
                    
                    if len(good_new) > 3:
                        # Calculate median displacement
                        displacement = np.median(good_new - good_old, axis=0)
                        
                        # Update bbox with optical flow
                        track.bbox[:2] += displacement
                        track.bbox[2:] += displacement
                        
                        # Accumulate drift
                        track.drift_accumulated += displacement
    
    def _match_detections(self, detections: List[Dict]) -> Tuple[List, List, List]:
        """Match detections to existing tracks using Hungarian algorithm."""
        if len(self.tracks) == 0:
            return [], list(range(len(detections))), []
        
        if len(detections) == 0:
            return [], [], list(range(len(self.tracks)))
        
        # Calculate IOU matrix
        iou_matrix = np.zeros((len(detections), len(self.tracks)))
        for d, det in enumerate(detections):
            for t, trk in enumerate(self.tracks):
                iou_matrix[d, t] = self._iou(det['bbox'], trk.bbox)
        
        # Hungarian assignment
        matched_indices = []
        if iou_matrix.size > 0:
            det_ids, trk_ids = linear_sum_assignment(-iou_matrix)
            
            for d, t in zip(det_ids, trk_ids):
                if iou_matrix[d, t] >= self.iou_threshold:
                    matched_indices.append([d, t])
        
        unmatched_detections = []
        for d in range(len(detections)):
            if d not in [m[0] for m in matched_indices]:
                unmatched_detections.append(d)
        
        unmatched_tracks = []
        for t in range(len(self.tracks)):
            if t not in [m[1] for m in matched_indices]:
                unmatched_tracks.append(t)
        
        return matched_indices, unmatched_detections, unmatched_tracks
    
    def _iou(self, bbox1: List[float], bbox2: np.ndarray) -> float:
        """Calculate Intersection over Union."""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        
        bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        
        union_area = bbox1_area + bbox2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0
    
    def cleanup_inactive_tracks(self, active_ids: List[int]):
        """Remove tracks that are not in the active list."""
        self.tracks = [t for t in self.tracks if t.track_id in active_ids]
    
    def _update_with_detection(self, track: Track, detection: Dict, frame: np.ndarray):
        """Update track with matched detection."""
        # Update Kalman filter
        measurement = np.array([
            (detection['bbox'][0] + detection['bbox'][2]) / 2,
            (detection['bbox'][1] + detection['bbox'][3]) / 2,
            detection['bbox'][2] - detection['bbox'][0],
            detection['bbox'][3] - detection['bbox'][1]
        ], dtype=np.float32)
        
        track.kalman.correct(measurement)
        
        # Update track properties
        track.bbox = np.array(detection['bbox'])
        track.confidence = detection['confidence']
        track.hits += 1
        track.hit_streak += 1
        track.time_since_update = 0
        track.landmarks = detection.get('landmarks')
        
        # Record hit in history for sliding window
        track.hit_history.append(self.frame_count)
        
        # Extract features for drift correction
        x1, y1, x2, y2 = track.bbox.astype(int)
        if 0 <= x1 < frame.shape[1] and 0 <= y1 < frame.shape[0]:
            roi = frame[y1:y2, x1:x2]
            if roi.size > 0:
                track.features = cv2.resize(roi, (64, 64)).flatten()
                
                # Set reference features on first good detection
                if track.reference_features is None and track.hit_streak > 3:
                    track.reference_features = track.features.copy()
    
    def _apply_drift_correction(self):
        """Apply drift correction to long-term tracks."""
        for track in self.tracks:
            if (track.reference_features is not None and 
                track.features is not None and
                track.hit_streak > 5):
                
                # Check if drift is excessive
                drift_magnitude = np.linalg.norm(track.drift_accumulated)
                if drift_magnitude > self.max_drift:
                    # Calculate feature similarity
                    similarity = np.dot(track.features, track.reference_features) / (
                        np.linalg.norm(track.features) * np.linalg.norm(track.reference_features)
                    )
                    
                    # If features are still similar, correct drift
                    if similarity > 0.8:
                        correction = -track.drift_accumulated * self.drift_correction_rate
                        track.bbox[:2] += correction
                        track.bbox[2:] += correction
                        track.drift_accumulated += correction
                        
                        # Update Kalman filter state
                        state = track.kalman.statePost.copy()
                        state[0] += correction[0]
                        state[1] += correction[1]
                        track.kalman.statePost = state
