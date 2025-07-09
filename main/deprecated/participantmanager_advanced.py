# Enhanced participant manager with 4K face recognition integration

import numpy as np
from collections import defaultdict
import time
import math
import threading
from scipy.spatial import procrustes
from typing import Dict, Optional, Tuple, List

class GlobalParticipantManagerAdvanced:
    """
    Enhanced participant manager that integrates with face recognition.
    Manages participant IDs globally across all cameras using both
    Procrustes shape analysis and face recognition embeddings.
    """
    def __init__(self, max_participants=10, shape_weight=0.5, position_weight=0.2, 
                 recognition_weight=0.3, enable_recognition=True):
        # Core participant tracking
        self.participants = {}  # {global_id: ParticipantInfo}
        self.next_global_id = 1
        self.lock = threading.Lock()
        
        # Matching thresholds
        self.distance_threshold = 0.20  # 20% of screen for position
        self.procrustes_threshold = 0.02  # Shape matching
        self.recognition_threshold = 0.7  # Face recognition similarity
        
        # Weights for matching
        self.shape_weight = shape_weight
        self.position_weight = position_weight
        self.recognition_weight = recognition_weight if enable_recognition else 0
        self.enable_recognition = enable_recognition
        
        # Tracking parameters
        self.max_participants = max_participants
        self.recently_lost = {}  # For re-identification
        self.recently_lost_timeout = 10.0
        
        # Recognition integration
        self.track_to_participant = {}  # {(camera_idx, track_id): participant_id}
        self.participant_embeddings = {}  # {participant_id: [embeddings]}
        self.participant_names = {}
        
        # Performance tracking
        self.match_stats = {
            'position_matches': 0,
            'shape_matches': 0,
            'recognition_matches': 0,
            'new_participants': 0
        }
    
    class ParticipantInfo:
        def __init__(self, global_id, camera, centroid, shape=None):
            self.global_id = global_id
            self.camera = camera
            self.centroid = centroid
            self.shape = shape
            self.last_seen = time.time()
            self.track_ids = set()  # Track IDs associated with this participant
            self.confidence = 1.0
            self.embedding_mean = None  # Mean face embedding
            self.enrollment_status = 'unknown'
            self.appearance_count = 1
    
    def set_max_participants(self, n):
        with self.lock:
            self.max_participants = n
    
    def set_participant_names(self, names_dict):
        """Set custom participant names from GUI"""
        with self.lock:
            self.participant_names = names_dict.copy()
    
    def update_from_advanced_detection(self, camera_idx: int, face_data: List[Dict]) -> Dict[int, int]:
        """
        Update participants from advanced face detection results.
        
        Args:
            camera_idx: Camera index
            face_data: List of face detection results from parallelworker_advanced
            
        Returns:
            Mapping of track_id to global participant_id
        """
        with self.lock:
            current_time = time.time()
            track_to_participant = {}
            
            for face in face_data:
                track_id = face.get('track_id')
                participant_id = face.get('participant_id', -1)
                centroid = face.get('centroid')
                landmarks = face.get('landmarks')
                quality = face.get('quality_score', 0.5)
                
                # Convert landmarks to shape array for Procrustes
                shape = None
                if landmarks:
                    shape = np.array(landmarks)[:, :2]  # Use 2D coordinates
                
                # If face recognition assigned a participant ID
                if self.enable_recognition and participant_id >= 0:
                    # Check if this participant ID is already tracked
                    if participant_id in self.participants:
                        # Update existing participant
                        participant = self.participants[participant_id]
                        participant.camera = camera_idx
                        participant.centroid = centroid
                        participant.shape = shape
                        participant.last_seen = current_time
                        participant.track_ids.add((camera_idx, track_id))
                        participant.confidence = min(1.0, participant.confidence + 0.1)
                        
                        track_to_participant[track_id] = participant_id
                        self.match_stats['recognition_matches'] += 1
                    else:
                        # New participant from recognition
                        self._create_participant_from_recognition(
                            participant_id, camera_idx, track_id, centroid, shape
                        )
                        track_to_participant[track_id] = participant_id
                else:
                    # No recognition available, use shape/position matching
                    global_id = self._match_or_create_participant(
                        camera_idx, track_id, centroid, shape
                    )
                    track_to_participant[track_id] = global_id
            
            # Update track mapping
            for track_id, participant_id in track_to_participant.items():
                self.track_to_participant[(camera_idx, track_id)] = participant_id
            
            # Clean up old participants
            self._cleanup_lost_participants(current_time)
            
            return track_to_participant
    
    def _match_or_create_participant(self, camera_idx, track_id, centroid, shape):
        """Match using position and shape, or create new participant."""
        current_time = time.time()
        
        # Try to match with existing participants
        best_match = None
        best_score = float('inf')
        
        for pid, participant in self.participants.items():
            # Skip if from same camera (can't be same person)
            if participant.camera == camera_idx:
                continue
            
            # Calculate combined score
            score = self._combined_score(
                centroid, participant.centroid,
                shape, participant.shape
            )
            
            if score < best_score:
                best_score = score
                best_match = pid
        
        # Check matching thresholds
        if best_match and best_score < 1.0:  # Normalized threshold
            # Update existing participant
            participant = self.participants[best_match]
            participant.camera = camera_idx
            participant.centroid = centroid
            participant.shape = shape
            participant.last_seen = current_time
            participant.track_ids.add((camera_idx, track_id))
            
            if best_score < 0.5:
                self.match_stats['shape_matches'] += 1
            else:
                self.match_stats['position_matches'] += 1
            
            return best_match
        
        # Check recently lost participants
        for pid, lost_info in list(self.recently_lost.items()):
            score = self._combined_score(
                centroid, lost_info['centroid'],
                shape, lost_info['shape']
            )
            
            if score < 0.5:  # Stricter threshold for re-identification
                # Restore participant
                participant = self.ParticipantInfo(pid, camera_idx, centroid, shape)
                participant.confidence = 0.8
                participant.track_ids.add((camera_idx, track_id))
                self.participants[pid] = participant
                del self.recently_lost[pid]
                return pid
        
        # Create new participant if under limit
        if len(self.participants) < self.max_participants:
            new_id = self._get_next_available_id()
            participant = self.ParticipantInfo(new_id, camera_idx, centroid, shape)
            participant.track_ids.add((camera_idx, track_id))
            self.participants[new_id] = participant
            self.match_stats['new_participants'] += 1
            return new_id
        
        # Find least confident participant to replace
        if self.participants:
            least_confident = min(
                self.participants.items(),
                key=lambda x: (x[1].confidence, -x[1].last_seen)
            )
            pid = least_confident[0]
            
            # Move to recently lost
            self.recently_lost[pid] = {
                'centroid': least_confident[1].centroid,
                'shape': least_confident[1].shape,
                'last_seen': least_confident[1].last_seen
            }
            
            # Replace with new participant
            participant = self.ParticipantInfo(pid, camera_idx, centroid, shape)
            participant.confidence = 0.5
            participant.track_ids.add((camera_idx, track_id))
            self.participants[pid] = participant
            return pid
        
        return 1  # Default fallback
    
    def _create_participant_from_recognition(self, participant_id, camera_idx, 
                                           track_id, centroid, shape):
        """Create participant from face recognition result."""
        # Ensure we use consistent IDs (1-based for GUI compatibility)
        if participant_id not in self.participants:
            # Map recognition ID to our 1-based system
            if participant_id >= self.max_participants:
                participant_id = self._get_next_available_id()
            else:
                participant_id = participant_id + 1  # Convert 0-based to 1-based
            
            participant = self.ParticipantInfo(participant_id, camera_idx, centroid, shape)
            participant.track_ids.add((camera_idx, track_id))
            participant.enrollment_status = 'enrolled'
            participant.confidence = 0.9
            self.participants[participant_id] = participant
    
    def _combined_score(self, centroid1, centroid2, shape1, shape2):
        """Calculate combined matching score."""
        # Position distance
        pos_dist = math.sqrt((centroid1[0] - centroid2[0])**2 +
                           (centroid1[1] - centroid2[1])**2)
        pos_score = pos_dist / self.distance_threshold
        
        # Shape distance
        shape_score = 1.0  # Default if no shape
        if shape1 is not None and shape2 is not None:
            shape_score = self._procrustes_distance(shape1, shape2)
        
        # For now, no embedding score (handled by recognition)
        recognition_score = 1.0
        
        # Weighted combination
        total_weight = self.position_weight + self.shape_weight
        if total_weight > 0:
            combined = (self.position_weight * pos_score + 
                       self.shape_weight * shape_score) / total_weight
        else:
            combined = pos_score
        
        return combined
    
    def _procrustes_distance(self, shape1, shape2):
        """Calculate Procrustes distance between shapes."""
        try:
            if shape1 is None or shape2 is None:
                return float('inf')
            
            shape1 = np.array(shape1, dtype=np.float64)
            shape2 = np.array(shape2, dtype=np.float64)
            
            if shape1.shape != shape2.shape or shape1.size == 0:
                return float('inf')
            
            _, _, disparity = procrustes(shape1, shape2)
            normalized_distance = np.sqrt(disparity / len(shape1))
            
            return normalized_distance / self.procrustes_threshold  # Normalize
            
        except Exception as e:
            return float('inf')
    
    def _get_next_available_id(self):
        """Get next available participant ID starting from 1."""
        for i in range(1, self.max_participants + 1):
            if i not in self.participants:
                return i
        return self.next_global_id
    
    def _cleanup_lost_participants(self, current_time):
        """Clean up participants not seen recently."""
        # Move lost participants to recently_lost
        lost_timeout = 2.0  # Seconds before considering lost
        
        for pid in list(self.participants.keys()):
            participant = self.participants[pid]
            if current_time - participant.last_seen > lost_timeout:
                self.recently_lost[pid] = {
                    'centroid': participant.centroid,
                    'shape': participant.shape,
                    'last_seen': participant.last_seen,
                    'embedding_mean': participant.embedding_mean
                }
                del self.participants[pid]
        
        # Clean up old recently_lost entries
        for pid in list(self.recently_lost.keys()):
            if current_time - self.recently_lost[pid]['last_seen'] > self.recently_lost_timeout:
                del self.recently_lost[pid]
    
    def update_enrollment_status(self, enrollment_status: Dict):
        """Update enrollment status from face recognition."""
        with self.lock:
            for participant_id, status in enrollment_status.items():
                # Convert to 1-based ID
                display_id = participant_id + 1
                if display_id in self.participants:
                    self.participants[display_id].enrollment_status = status['state']
                    self.participants[display_id].confidence = max(
                        self.participants[display_id].confidence,
                        status.get('confidence_score', 0)
                    )
    
    def get_participant_name(self, global_id):
        """Get participant name for display."""
        with self.lock:
            if global_id in self.participant_names:
                return self.participant_names[global_id]
            return f"P{global_id}"
    
    def get_all_participants(self):
        """Get all active participants with their info."""
        with self.lock:
            result = {}
            for pid, participant in self.participants.items():
                result[pid] = {
                    'camera': participant.camera,
                    'centroid': participant.centroid,
                    'last_seen': participant.last_seen,
                    'confidence': participant.confidence,
                    'enrollment_status': participant.enrollment_status,
                    'name': self.get_participant_name(pid)
                }
            return result
    
    def get_stats(self):
        """Get matching statistics."""
        with self.lock:
            return {
                'active_participants': len(self.participants),
                'recently_lost': len(self.recently_lost),
                'match_stats': self.match_stats.copy()
            }
    
    # Backward compatibility methods
    def update_participant(self, camera_idx, local_tracker_id, centroid, shape=None):
        """Legacy method for compatibility with existing code."""
        return self._match_or_create_participant(camera_idx, local_tracker_id, centroid, shape)
    
    def cleanup_stale_participants(self):
        """Legacy cleanup method."""
        with self.lock:
            self._cleanup_lost_participants(time.time())
    
    def cleanup_old_participants(self):
        """Alias for cleanup_stale_participants for compatibility."""
        self.cleanup_stale_participants()