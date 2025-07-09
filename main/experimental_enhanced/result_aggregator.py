"""
Result Aggregator
Collects landmarks from all workers, maps back to 4K coordinates,
maintains temporal consistency, and outputs unified face data
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import time
from collections import defaultdict, deque
import threading

class ResultAggregator:
    """Aggregates results from landmark workers and maintains consistency."""
    
    def __init__(self, max_history: int = 10):
        """
        Initialize result aggregator.
        
        Args:
            max_history: Number of frames to keep in history for temporal smoothing
        """
        self.max_history = max_history
        
        # Track ID to participant ID mapping
        self.track_to_participant = {}
        self.participant_counter = 0
        self.mapping_lock = threading.Lock()
        
        # Temporal history for smoothing
        self.landmark_history = defaultdict(lambda: deque(maxlen=max_history))
        self.confidence_history = defaultdict(lambda: deque(maxlen=max_history))
        
        # Latest results per track
        self.latest_results = {}
        self.results_lock = threading.Lock()
        
        # Frame synchronization
        self.frame_results = defaultdict(dict)
        self.completed_frames = set()
        
        # Performance tracking
        self.processing_times = deque(maxlen=100)
        
    def add_landmark_result(self, result, roi_manager=None):
        """
        Add landmark detection result.
        
        Args:
            result: LandmarkResult from worker
            roi_manager: ROIManager instance for coordinate transformation
        """
        start_time = time.time()
        
        with self.results_lock:
            track_id = result.track_id
            frame_id = result.frame_id
            
            # Assign participant ID if new track
            if track_id not in self.track_to_participant:
                with self.mapping_lock:
                    self.track_to_participant[track_id] = self.participant_counter
                    self.participant_counter += 1
            
            participant_id = self.track_to_participant[track_id]
            
            if result.success and result.landmarks is not None:
                # Transform landmarks to original coordinates if ROI manager available
                if roi_manager:
                    landmarks_original = roi_manager.transform_landmarks_to_original(
                        track_id, result.landmarks[:, :2]
                    )
                    if landmarks_original is not None:
                        # Combine with z coordinates
                        landmarks_3d = np.hstack([
                            landmarks_original,
                            result.landmarks[:, 2:3]
                        ])
                    else:
                        landmarks_3d = result.landmarks
                else:
                    landmarks_3d = result.landmarks
                
                # Add to history for temporal smoothing
                self.landmark_history[track_id].append(landmarks_3d)
                self.confidence_history[track_id].append(1.0)
                
                # Apply temporal smoothing
                smoothed_landmarks = self._smooth_landmarks(track_id)
                
                # Store result
                face_result = {
                    'track_id': track_id,
                    'participant_id': participant_id,
                    'frame_id': frame_id,
                    'landmarks': smoothed_landmarks,
                    'landmarks_raw': landmarks_3d,
                    'blendshapes': result.blendshapes,
                    'confidence': 1.0,
                    'processing_time': result.processing_time
                }
                
                self.latest_results[track_id] = face_result
                self.frame_results[frame_id][track_id] = face_result
                
            else:
                # Failed detection - use predicted position if available
                self.confidence_history[track_id].append(0.0)
                
                if track_id in self.latest_results:
                    # Decay confidence for missing detection
                    old_result = self.latest_results[track_id]
                    old_result['confidence'] *= 0.9
                    
                    # Store with lower confidence
                    self.frame_results[frame_id][track_id] = old_result
        
        self.processing_times.append(time.time() - start_time)
    
    def get_frame_results(self, frame_id: int) -> Dict[int, Dict]:
        """Get all results for a specific frame."""
        with self.results_lock:
            return self.frame_results.get(frame_id, {}).copy()
    
    def get_latest_results(self) -> Dict[int, Dict]:
        """Get latest results for all tracks."""
        with self.results_lock:
            return self.latest_results.copy()
    
    def get_unified_face_data(self, frame_id: int, full_resolution: Tuple[int, int]) -> List[Dict]:
        """
        Get unified face data with consistent IDs and normalized coordinates.
        
        Args:
            frame_id: Frame ID
            full_resolution: Full frame resolution (height, width)
            
        Returns:
            List of face data dictionaries
        """
        frame_results = self.get_frame_results(frame_id)
        face_data = []
        
        h, w = full_resolution
        
        for track_id, result in frame_results.items():
            if result['confidence'] > 0.5:  # Minimum confidence threshold
                # Normalize landmarks to [0, 1]
                landmarks_norm = result['landmarks'].copy()
                landmarks_norm[:, 0] /= w
                landmarks_norm[:, 1] /= h
                
                # Calculate centroid
                centroid = np.mean(landmarks_norm[:, :2], axis=0)
                
                face_data.append({
                    'track_id': track_id,
                    'participant_id': result['participant_id'],
                    'landmarks': landmarks_norm.tolist(),
                    'landmarks_3d': [(lm[0], lm[1], lm[2]) for lm in landmarks_norm],
                    'blendshapes': result.get('blendshapes', []),
                    'centroid': centroid.tolist(),
                    'confidence': result['confidence'],
                    'frame_id': frame_id
                })
        
        return face_data
    
    def _smooth_landmarks(self, track_id: int, alpha: float = 0.7) -> np.ndarray:
        """
        Apply temporal smoothing to landmarks.
        
        Args:
            track_id: Track ID
            alpha: Smoothing factor (0=no smoothing, 1=maximum smoothing)
            
        Returns:
            Smoothed landmarks
        """
        history = self.landmark_history[track_id]
        
        if len(history) == 0:
            return np.zeros((468, 3))
        elif len(history) == 1:
            return history[0]
        else:
            # Weighted average with more weight on recent frames
            weights = np.exp(np.linspace(-2, 0, len(history)))
            weights /= weights.sum()
            
            smoothed = np.zeros_like(history[0])
            for i, landmarks in enumerate(history):
                smoothed += landmarks * weights[i]
            
            # Blend with latest for responsiveness
            latest = history[-1]
            smoothed = alpha * smoothed + (1 - alpha) * latest
            
            return smoothed
    
    def cleanup_old_tracks(self, active_track_ids: List[int]):
        """Remove data for tracks that are no longer active."""
        with self.results_lock:
            # Find inactive tracks
            all_tracks = set(self.latest_results.keys())
            active_tracks = set(active_track_ids)
            inactive_tracks = all_tracks - active_tracks
            
            # Remove inactive tracks
            for track_id in inactive_tracks:
                if track_id in self.latest_results:
                    del self.latest_results[track_id]
                if track_id in self.landmark_history:
                    del self.landmark_history[track_id]
                if track_id in self.confidence_history:
                    del self.confidence_history[track_id]
    
    def mark_frame_complete(self, frame_id: int):
        """Mark a frame as complete for cleanup."""
        self.completed_frames.add(frame_id)
        
        # Clean up old frame data
        if len(self.completed_frames) > 100:
            old_frames = sorted(self.completed_frames)[:-50]
            for old_frame in old_frames:
                if old_frame in self.frame_results:
                    del self.frame_results[old_frame]
                self.completed_frames.remove(old_frame)
    
    def get_participant_mapping(self) -> Dict[int, int]:
        """Get track ID to participant ID mapping."""
        with self.mapping_lock:
            return self.track_to_participant.copy()
    
    def get_stats(self) -> Dict:
        """Get aggregator statistics."""
        with self.results_lock:
            avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0
            
            return {
                'active_tracks': len(self.latest_results),
                'total_participants': self.participant_counter,
                'avg_processing_time': avg_processing_time,
                'pending_frames': len(self.frame_results) - len(self.completed_frames)
            }