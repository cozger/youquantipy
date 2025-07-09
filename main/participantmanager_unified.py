# Unified participant manager supporting both standard and enhanced modes

import numpy as np
from collections import defaultdict
import time
import math
import threading
from scipy.spatial import procrustes
from typing import Dict, Optional, Tuple, List

class GlobalParticipantManager:
    """
    Unified participant manager that supports both standard and enhanced modes.
    Manages participant IDs globally across all cameras using Procrustes shape analysis
    and optionally face recognition embeddings.
    """
    def __init__(self, max_participants=2, shape_weight=0.7, position_weight=0.3,
                 recognition_weight=0.0, enable_recognition=False):
        # Core participant tracking
        self.participants = {}  # {global_id: ParticipantInfo or dict}
        self.next_global_id = 1
        self.lock = threading.Lock()
        
        # Mode configuration
        self.enable_recognition = enable_recognition
        self.use_advanced_features = enable_recognition
        
        # Matching thresholds
        self.distance_threshold = 0.20  # 20% of screen for position matching
        self.procrustes_threshold = 0.02  # Procrustes distance threshold
        self.recognition_threshold = 0.7  # Face recognition similarity (enhanced mode)
        
        # Weights for matching
        self.shape_weight = shape_weight if not enable_recognition else 0.5
        self.position_weight = position_weight if not enable_recognition else 0.2
        self.recognition_weight = recognition_weight if enable_recognition else 0.0
        
        # Tracking parameters
        self.max_participants = max_participants
        self.recently_lost = {}  # global_id -> participant info
        self.recently_lost_timeout = 10.0
        self.participant_names = {}
        
        # Enhanced mode specific
        if self.use_advanced_features:
            self.track_to_participant = {}  # {(camera_idx, track_id): participant_id}
            self.participant_embeddings = {}  # {participant_id: [embeddings]}
            self.match_stats = {
                'position_matches': 0,
                'shape_matches': 0,
                'recognition_matches': 0,
                'new_participants': 0
            }
    
    class ParticipantInfo:
        """Enhanced participant info for advanced mode"""
        def __init__(self, global_id, camera, centroid, shape=None, local_id=None):
            self.global_id = global_id
            self.camera = camera
            self.centroid = centroid
            self.shape = shape
            self.local_id = local_id
            self.last_seen = time.time()
            self.track_ids = set()  # Track IDs associated with this participant
            self.confidence = 1.0
            self.embedding_mean = None  # Mean face embedding
            self.enrollment_status = 'unknown'
            self.appearance_count = 1
            
        def to_dict(self):
            """Convert to dictionary for compatibility"""
            return {
                'camera': self.camera,
                'centroid': self.centroid,
                'shape': self.shape,
                'local_id': self.local_id,
                'last_seen': self.last_seen
            }
    
    def set_max_participants(self, n):
        with self.lock:
            self.max_participants = n
    
    def set_participant_names(self, names_dict):
        """Set custom participant names from GUI"""
        with self.lock:
            self.participant_names = names_dict.copy()
    
    def update_participant(self, camera_idx, local_tracker_id, centroid, shape=None):
        """
        Standard mode update - maintains backward compatibility.
        Returns the global participant ID (1, 2, 3, etc.)
        """
        with self.lock:
            if self.use_advanced_features:
                # In advanced mode, still support legacy calls
                return self._match_or_create_participant(camera_idx, local_tracker_id, centroid, shape)
            else:
                # Standard mode logic (original implementation)
                return self._update_participant_standard(camera_idx, local_tracker_id, centroid, shape)
    
    def update_from_advanced_detection(self, camera_idx: int, face_data: List[Dict]) -> Dict[int, int]:
        """
        Enhanced mode update - processes face detection results with recognition.
        Only available in enhanced mode.
        """
        if not self.use_advanced_features:
            # Fallback to standard processing
            track_to_participant = {}
            for face in face_data:
                track_id = face.get('track_id')
                centroid = face.get('centroid')
                landmarks = face.get('landmarks')
                shape = np.array(landmarks)[:, :2] if landmarks else None
                
                global_id = self.update_participant(camera_idx, track_id, centroid, shape)
                if global_id:
                    track_to_participant[track_id] = global_id
            return track_to_participant
        
        # Enhanced mode processing
        with self.lock:
            current_time = time.time()
            track_to_participant = {}
            
            for face in face_data:
                track_id = face.get('track_id')
                participant_id = face.get('participant_id', -1)
                centroid = face.get('centroid')
                landmarks = face.get('landmarks')
                quality = face.get('quality_score', 0.5)
                
                # Convert landmarks to shape array
                shape = None
                if landmarks:
                    shape = np.array(landmarks)[:, :2]
                
                # If face recognition assigned a participant ID
                if self.enable_recognition and participant_id >= 0:
                    # Check if this participant ID is already tracked
                    if participant_id in self.participants:
                        # Update existing participant
                        participant = self.participants[participant_id]
                        if isinstance(participant, dict):
                            # Convert old dict to ParticipantInfo
                            participant = self._dict_to_participant_info(participant_id, participant)
                            self.participants[participant_id] = participant
                        
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
                    if global_id:
                        track_to_participant[track_id] = global_id
            
            # Update track mapping
            for track_id, participant_id in track_to_participant.items():
                self.track_to_participant[(camera_idx, track_id)] = participant_id
            
            # Clean up old participants
            self._cleanup_lost_participants(current_time)
            
            return track_to_participant
    
    def _update_participant_standard(self, camera_idx, local_tracker_id, centroid, shape):
        """Standard mode participant update logic"""
        current_time = time.time()
        
        # First, try to match with existing active participants
        best_match = None
        best_score = float('inf')
        
        for global_id, info in self.participants.items():
            # Handle both dict and ParticipantInfo
            if isinstance(info, self.ParticipantInfo):
                info_dict = info.to_dict()
            else:
                info_dict = info
            
            # Skip if from different camera and too recent
            if info_dict['camera'] != camera_idx and current_time - info_dict['last_seen'] < 0.5:
                continue
            
            # Calculate combined score
            score = self._combined_score(
                centroid, info_dict['centroid'],
                shape, info_dict.get('shape')
            )
            
            # Apply thresholds based on context
            if info_dict['camera'] == camera_idx:
                # Same camera - use standard thresholds
                threshold = 0.5  # Combined score threshold
            else:
                # Different camera - be more strict with shape matching
                if shape is not None and info_dict.get('shape') is not None:
                    shape_dist = self._procrustes_distance(shape, info_dict['shape'])
                    # Only accept cross-camera match if shape is very similar
                    if shape_dist > self.procrustes_threshold:
                        continue
                threshold = 0.3
            
            if score < threshold and score < best_score:
                best_score = score
                best_match = global_id

        if best_match:
            # Update existing participant
            if isinstance(self.participants[best_match], self.ParticipantInfo):
                participant = self.participants[best_match]
                old_shape = participant.shape
                participant.camera = camera_idx
                participant.centroid = centroid
                participant.shape = shape if shape is not None else old_shape
                participant.last_seen = current_time
                participant.local_id = local_tracker_id
            else:
                old_shape = self.participants[best_match].get('shape')
                self.participants[best_match] = {
                    'camera': camera_idx,
                    'centroid': centroid,
                    'last_seen': current_time,
                    'local_id': local_tracker_id,
                    'shape': shape if shape is not None else old_shape
                }
            
            return best_match
        
        # No match in active participants - check recently lost
        best_lost_id = None
        best_lost_score = float('inf')
        
        for lost_id, lost_info in self.recently_lost.items():
            # Only consider IDs within our max range
            if lost_id > self.max_participants:
                continue
            
            # Calculate combined score
            score = self._combined_score(
                centroid, lost_info['centroid'],
                shape, lost_info.get('shape')
            )
            
            # For shape-based matching, prioritize shape over position
            if shape is not None and lost_info.get('shape') is not None:
                shape_dist = self._procrustes_distance(shape, lost_info['shape'])
                # Strong shape match can override position
                if shape_dist < self.procrustes_threshold:
                    score = shape_dist  # Use shape score directly
            
            if score < best_lost_score:
                best_lost_score = score
                best_lost_id = lost_id
        
        # Use recently lost participant if good match
        if best_lost_id is not None and best_lost_score < 0.3:  # Strict threshold
            # Reactivate lost participant
            lost_info = self.recently_lost[best_lost_id]
            if self.use_advanced_features:
                participant = self.ParticipantInfo(
                    best_lost_id, camera_idx, centroid,
                    shape if shape is not None else lost_info.get('shape'),
                    local_tracker_id
                )
                self.participants[best_lost_id] = participant
            else:
                self.participants[best_lost_id] = {
                    'camera': camera_idx,
                    'centroid': centroid,
                    'last_seen': current_time,
                    'local_id': local_tracker_id,
                    'shape': shape if shape is not None else lost_info.get('shape')
                }
            del self.recently_lost[best_lost_id]
            
            return best_lost_id
        
        # Count active participants
        active_count = len([p for p in self.participants.values() 
                          if current_time - (p.last_seen if isinstance(p, self.ParticipantInfo) 
                             else p['last_seen']) < 1.0])
        
        if active_count >= self.max_participants:
            print(f"[ParticipantManager] Max participants ({self.max_participants}) reached.")
            return None
        
        # Create new participant - use the lowest available ID
        new_id = None
        for potential_id in range(1, self.max_participants + 1):
            if (potential_id not in self.participants and 
                potential_id not in self.recently_lost):
                new_id = potential_id
                break
        
        if new_id is None:
            print(f"[ParticipantManager] Error: No available IDs within max_participants range")
            return None
        
        # Create the participant
        if self.use_advanced_features:
            participant = self.ParticipantInfo(new_id, camera_idx, centroid, shape, local_tracker_id)
            self.participants[new_id] = participant
        else:
            self.participants[new_id] = {
                'camera': camera_idx,
                'centroid': centroid,
                'last_seen': current_time,
                'local_id': local_tracker_id,
                'shape': shape
            }
        
        shape_info = "with shape" if shape is not None else "no shape"
        print(f"[ParticipantManager] Created participant {new_id} ({shape_info})")
        
        return new_id
    
    def _match_or_create_participant(self, camera_idx, track_id, centroid, shape):
        """Enhanced mode matching logic"""
        if not self.use_advanced_features:
            return self._update_participant_standard(camera_idx, track_id, centroid, shape)
        
        current_time = time.time()
        
        # Try to match with existing participants
        best_match = None
        best_score = float('inf')
        
        for pid, participant in self.participants.items():
            # Handle both dict and ParticipantInfo
            if isinstance(participant, dict):
                participant = self._dict_to_participant_info(pid, participant)
                self.participants[pid] = participant
            
            # Skip if from same camera (can't be same person in enhanced mode)
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
        
        # Find least confident participant to replace (enhanced mode only)
        if self.participants:
            least_confident_id = None
            min_confidence = float('inf')
            
            for pid, p in self.participants.items():
                if isinstance(p, self.ParticipantInfo):
                    if p.confidence < min_confidence:
                        min_confidence = p.confidence
                        least_confident_id = pid
            
            if least_confident_id:
                # Move to recently lost
                p = self.participants[least_confident_id]
                self.recently_lost[least_confident_id] = {
                    'centroid': p.centroid,
                    'shape': p.shape,
                    'last_seen': p.last_seen,
                    'embedding_mean': p.embedding_mean if hasattr(p, 'embedding_mean') else None
                }
                
                # Replace with new participant
                participant = self.ParticipantInfo(least_confident_id, camera_idx, centroid, shape)
                participant.confidence = 0.5
                participant.track_ids.add((camera_idx, track_id))
                self.participants[least_confident_id] = participant
                return least_confident_id
        
        return 1  # Default fallback
    
    def _create_participant_from_recognition(self, participant_id, camera_idx, 
                                           track_id, centroid, shape):
        """Create participant from face recognition result (enhanced mode)"""
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
    
    def _procrustes_distance(self, shape1, shape2):
        """Calculate Procrustes distance between two shapes."""
        if shape1 is None or shape2 is None:
            return float('inf')
            
        try:
            # Ensure numpy arrays
            shape1 = np.array(shape1, dtype=np.float64)
            shape2 = np.array(shape2, dtype=np.float64)
            
            # Check if shapes have same dimensions
            if shape1.shape != shape2.shape:
                return float('inf')
            
            # Check for degenerate shapes
            if shape1.size == 0 or shape2.size == 0:
                return float('inf')
            
            # Compute Procrustes distance
            _, _, disparity = procrustes(shape1, shape2)
            
            # Normalize by number of points
            normalized_distance = np.sqrt(disparity / len(shape1))
            
            # Further normalize for enhanced mode
            if self.use_advanced_features:
                return normalized_distance / self.procrustes_threshold
            
            return normalized_distance
            
        except Exception as e:
            print(f"[ParticipantManager] Procrustes error: {e}")
            return float('inf')
    
    def _combined_score(self, centroid1, centroid2, shape1, shape2):
        """Calculate combined matching score using position and shape."""
        # Position distance (normalized by threshold)
        pos_dist = math.sqrt((centroid1[0] - centroid2[0])**2 +
                           (centroid1[1] - centroid2[1])**2)
        pos_score = pos_dist / self.distance_threshold
        
        # Shape distance using Procrustes
        if shape1 is not None and shape2 is not None:
            shape_score = self._procrustes_distance(shape1, shape2)
        else:
            # If no shapes available, use position only
            return pos_score
        
        # Weighted combination
        if self.use_advanced_features:
            # Enhanced mode: different weighting
            total_weight = self.position_weight + self.shape_weight + self.recognition_weight
            if total_weight > 0:
                combined = (self.position_weight * pos_score + 
                           self.shape_weight * shape_score) / (self.position_weight + self.shape_weight)
            else:
                combined = pos_score
        else:
            # Standard mode: original weighting
            combined = (self.shape_weight * shape_score + 
                       self.position_weight * pos_score)
        
        return combined
    
    def get_participant_name(self, global_id):
        """Convert global ID to participant name, using custom names if available"""
        with self.lock:
            # Try to get custom name first (global_id is 1-based, dict is 0-based)
            custom_name = self.participant_names.get(global_id - 1)
            if custom_name and custom_name.strip() and not custom_name.strip().startswith('P'):
                return custom_name.strip()
            return f"P{global_id}"
    
    def get_all_shapes(self):
        """Get all participant shapes for Hungarian assignment in trackers"""
        with self.lock:
            shapes = {}
            current_time = time.time()
            
            # Include both active and recently lost participants
            all_participants = {}
            
            # Add active participants
            for global_id, info in self.participants.items():
                if isinstance(info, self.ParticipantInfo):
                    all_participants[global_id] = {
                        'shape': info.shape,
                        'centroid': info.centroid,
                        'last_seen': info.last_seen,
                        'source': 'active'
                    }
                else:
                    all_participants[global_id] = {
                        'shape': info.get('shape'),
                        'centroid': info['centroid'],
                        'last_seen': info['last_seen'],
                        'source': 'active'
                    }
            
            # Add recently lost participants
            for global_id, info in self.recently_lost.items():
                if global_id <= self.max_participants:
                    all_participants[global_id] = {
                        'shape': info.get('shape'),
                        'centroid': info['centroid'],
                        'last_seen': info['last_seen'],
                        'source': 'lost'
                    }
            
            # Build return dictionary
            for global_id, info in all_participants.items():
                if info.get('shape') is not None:
                    # Convert shape to list for JSON serialization
                    shape_list = info['shape'].tolist() if isinstance(info['shape'], np.ndarray) else info['shape']
                    
                    shapes[global_id] = {
                        'shape': shape_list,
                        'centroid': info['centroid'],
                        'frames_since_seen': int((current_time - info['last_seen']) * 30),
                        'source': info['source']
                    }
            
            return shapes
    
    def cleanup_old_participants(self, timeout=5.0):
        """Remove participants not seen for timeout seconds and manage recently lost."""
        with self.lock:
            current_time = time.time()
            to_remove = []
            
            # Move timed out participants to recently_lost
            for global_id, info in self.participants.items():
                if isinstance(info, self.ParticipantInfo):
                    last_seen = info.last_seen
                    shape = info.shape
                    centroid = info.centroid
                else:
                    last_seen = info['last_seen']
                    shape = info.get('shape')
                    centroid = info['centroid']
                
                if current_time - last_seen > timeout:
                    # Preserve the shape when moving to recently lost
                    self.recently_lost[global_id] = {
                        'centroid': centroid,
                        'last_seen': last_seen,
                        'shape': shape
                    }
                    if hasattr(info, 'embedding_mean'):
                        self.recently_lost[global_id]['embedding_mean'] = info.embedding_mean
                    
                    to_remove.append(global_id)
                    print(f"[ParticipantManager] Moving participant {global_id} to recently lost")
                    
            for global_id in to_remove:
                del self.participants[global_id]
                
            # Clean up recently lost that are too old
            to_prune = []
            for lost_id, info in self.recently_lost.items():
                if current_time - info['last_seen'] > self.recently_lost_timeout:
                    to_prune.append(lost_id)
                    
            for lost_id in to_prune:
                del self.recently_lost[lost_id]
                print(f"[ParticipantManager] Permanently removing participant {lost_id}")
    
    def reset(self):
        """Clear all participants and reset counter"""
        with self.lock:
            self.participants.clear()
            self.recently_lost.clear()
            self.next_global_id = 1
            if self.use_advanced_features:
                self.track_to_participant.clear()
                self.participant_embeddings.clear()
    
    def get_active_participants(self):
        """Get list of currently active participant IDs"""
        with self.lock:
            current_time = time.time()
            active = []
            for global_id, info in self.participants.items():
                if isinstance(info, self.ParticipantInfo):
                    last_seen = info.last_seen
                else:
                    last_seen = info['last_seen']
                
                if current_time - last_seen < 1.0:  # Active within last second
                    active.append(global_id)
            return sorted(active)
    
    # Enhanced mode specific methods
    def update_enrollment_status(self, enrollment_status: Dict):
        """Update enrollment status from face recognition (enhanced mode only)."""
        if not self.use_advanced_features:
            return
            
        with self.lock:
            for participant_id, status in enrollment_status.items():
                # Convert to 1-based ID
                display_id = participant_id + 1
                if display_id in self.participants:
                    participant = self.participants[display_id]
                    if isinstance(participant, self.ParticipantInfo):
                        participant.enrollment_status = status['state']
                        participant.confidence = max(
                            participant.confidence,
                            status.get('confidence_score', 0)
                        )
    
    def get_all_participants(self):
        """Get all active participants with their info (enhanced mode returns more details)."""
        with self.lock:
            result = {}
            for pid, participant in self.participants.items():
                if isinstance(participant, self.ParticipantInfo):
                    # Enhanced mode
                    result[pid] = {
                        'camera': participant.camera,
                        'centroid': participant.centroid,
                        'last_seen': participant.last_seen,
                        'confidence': participant.confidence,
                        'enrollment_status': participant.enrollment_status,
                        'name': self.get_participant_name(pid)
                    }
                else:
                    # Standard mode
                    result[pid] = {
                        'camera': participant['camera'],
                        'centroid': participant['centroid'],
                        'last_seen': participant['last_seen'],
                        'name': self.get_participant_name(pid)
                    }
            return result
    
    def get_stats(self):
        """Get matching statistics (enhanced mode only)."""
        with self.lock:
            if self.use_advanced_features:
                return {
                    'active_participants': len(self.participants),
                    'recently_lost': len(self.recently_lost),
                    'match_stats': self.match_stats.copy()
                }
            else:
                return {
                    'active_participants': len(self.participants),
                    'recently_lost': len(self.recently_lost)
                }
    
    # Backward compatibility methods
    def cleanup_stale_participants(self):
        """Legacy cleanup method for enhanced mode compatibility."""
        self.cleanup_old_participants(timeout=2.0 if self.use_advanced_features else 5.0)
    
    def _get_next_available_id(self):
        """Get next available participant ID starting from 1."""
        for i in range(1, self.max_participants + 1):
            if i not in self.participants and i not in self.recently_lost:
                return i
        return 1
    
    def _dict_to_participant_info(self, global_id, info_dict):
        """Convert dictionary participant to ParticipantInfo object"""
        participant = self.ParticipantInfo(
            global_id,
            info_dict['camera'],
            info_dict['centroid'],
            info_dict.get('shape'),
            info_dict.get('local_id')
        )
        participant.last_seen = info_dict['last_seen']
        return participant


# For backward compatibility, keep the advanced class as an alias
class GlobalParticipantManagerAdvanced(GlobalParticipantManager):
    """Alias for unified manager with enhanced mode enabled by default"""
    def __init__(self, max_participants=10, shape_weight=0.5, position_weight=0.2, 
                 recognition_weight=0.3, enable_recognition=True):
        super().__init__(
            max_participants=max_participants,
            shape_weight=shape_weight,
            position_weight=position_weight,
            recognition_weight=recognition_weight,
            enable_recognition=enable_recognition
        )