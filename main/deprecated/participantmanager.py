# In participantmanager.py - Complete updated GlobalParticipantManager

import numpy as np
from collections import defaultdict
import time
import math
import threading
from scipy.spatial import procrustes

class GlobalParticipantManager:
    """
    Manages participant IDs globally across all cameras using Procrustes shape analysis.
    Thread-safe version without multiprocessing.Manager for better compatibility.
    """
    def __init__(self, max_participants=2, shape_weight=0.7, position_weight=0.3):
        self.participants = {}  # {global_id: {'camera': cam_idx, 'centroid': (x,y), 'last_seen': timestamp, 'shape': np.array}}
        self.next_global_id = 1
        self.lock = threading.Lock()
        self.distance_threshold = 0.20  # 20% of screen for position matching
        self.procrustes_threshold = 0.02  # Procrustes distance threshold for shape matching
        self.max_participants = max_participants 
        self.recently_lost = {}  # global_id -> {'centroid': (x, y), 'last_seen': timestamp, 'shape': np.array}
        self.recently_lost_timeout = 10.0  # Keep shapes for 10 seconds
        self.participant_names = {}
        self.shape_weight = shape_weight
        self.position_weight = position_weight

    def set_max_participants(self, n):
        with self.lock:
            self.max_participants = n

    def set_participant_names(self, names_dict):
        """Set custom participant names from GUI"""
        with self.lock:
            self.participant_names = names_dict.copy()

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
            
            return normalized_distance
            
        except Exception as e:
            print(f"[ParticipantManager] Procrustes error: {e}")
            return float('inf')

    def _combined_score(self, centroid1, centroid2, shape1, shape2):
        """Calculate combined matching score using both position and shape."""
        # Position distance (normalized by threshold)
        pos_dist = math.sqrt((centroid1[0] - centroid2[0])**2 +
                           (centroid1[1] - centroid2[1])**2)
        pos_score = pos_dist / self.distance_threshold
        
        # Shape distance using Procrustes
        if shape1 is not None and shape2 is not None:
            shape_score = self._procrustes_distance(shape1, shape2)
            # # Debug print
            # print(f"[DEBUG] Procrustes distance: {shape_score:.4f}, Position distance: {pos_dist:.3f}")
        else:
            # If no shapes available, use position only
            return pos_score
        
        # Weighted combination
        combined = (self.shape_weight * shape_score + 
                   self.position_weight * pos_score)
        
        return combined

    def update_participant(self, camera_idx, local_tracker_id, centroid, shape=None):
        """
        Update or create a participant entry using both position and Procrustes shape.
        Returns the global participant ID (1, 2, 3, etc.)
        """
        with self.lock:
            current_time = time.time()
            
            # First, try to match with existing active participants
            best_match = None
            best_score = float('inf')
            
            for global_id, info in self.participants.items():
                # Skip if from different camera and too recent
                if info['camera'] != camera_idx and current_time - info['last_seen'] < 0.5:
                    continue
                
                # Calculate combined score
                score = self._combined_score(
                    centroid, info['centroid'],
                    shape, info.get('shape')
                )
                
                # Apply thresholds based on context
                if info['camera'] == camera_idx:
                    # Same camera - use standard thresholds
                    threshold = 0.5  # Combined score threshold
                else:
                    # Different camera - be more strict with shape matching
                    if shape is not None and info.get('shape') is not None:
                        shape_dist = self._procrustes_distance(shape, info['shape'])
                        # Only accept cross-camera match if shape is very similar
                        if shape_dist > self.procrustes_threshold:
                            continue
                    threshold = 0.3
                
                if score < threshold and score < best_score:
                    best_score = score
                    best_match = global_id

            if best_match:
                # Update existing participant
                old_shape = self.participants[best_match].get('shape')
                
                self.participants[best_match] = {
                    'camera': camera_idx,
                    'centroid': centroid,
                    'last_seen': current_time,
                    'local_id': local_tracker_id,
                    'shape': shape if shape is not None else old_shape
                }
                
                # No shape blending needed - Procrustes shapes are already normalized
                
                return best_match
            
            # No match in active participants - check recently lost
            best_lost_id = None
            best_lost_score = float('inf')
            
            # Try to match with recently lost participants
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
            if best_lost_id is not None and best_lost_score < 0.3:  # Strict threshold for reactivation
                # Reactivate lost participant
                lost_info = self.recently_lost[best_lost_id]
                self.participants[best_lost_id] = {
                    'camera': camera_idx,
                    'centroid': centroid,
                    'last_seen': current_time,
                    'local_id': local_tracker_id,
                    'shape': shape if shape is not None else lost_info.get('shape')
                }
                del self.recently_lost[best_lost_id]
                
                # Debug print
                shape_match = "shape" if shape is not None and lost_info.get('shape') is not None else "position"
                print(f"[ParticipantManager] Reused lost participant {best_lost_id} ({shape_match} match, score: {best_lost_score:.3f})")
                return best_lost_id
            
            # Count active participants
            active_count = len([p for p in self.participants.values() 
                              if current_time - p['last_seen'] < 1.0])
            
            if active_count >= self.max_participants:
                print(f"[ParticipantManager] Max participants ({self.max_participants}) reached. Not creating new participant.")
                return None
            
            # Create new participant - ALWAYS use the lowest available ID within max_participants
            new_id = None
            
            # Find the lowest available ID from 1 to max_participants
            for potential_id in range(1, self.max_participants + 1):
                if (potential_id not in self.participants and 
                    potential_id not in self.recently_lost):
                    new_id = potential_id
                    break
            
            if new_id is None:
                # This shouldn't happen if our counting is correct
                print(f"[ParticipantManager] Error: No available IDs within max_participants range")
                return None
            
            # Create the participant with the selected ID
            self.participants[new_id] = {
                'camera': camera_idx,
                'centroid': centroid,
                'last_seen': current_time,
                'local_id': local_tracker_id,
                'shape': shape
            }
            
            shape_info = "with shape" if shape is not None else "no shape"
            print(f"[ParticipantManager] Created participant {new_id} (lowest available, {shape_info})")
            
            return new_id
    
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
                all_participants[global_id] = {
                    'shape': info.get('shape'),
                    'centroid': info['centroid'],
                    'last_seen': info['last_seen'],
                    'source': 'active'
                }
            
            # Add recently lost participants
            for global_id, info in self.recently_lost.items():
                if global_id <= self.max_participants:  # Only include valid IDs
                    all_participants[global_id] = {
                        'shape': info.get('shape'),
                        'centroid': info['centroid'],
                        'last_seen': info['last_seen'],
                        'source': 'lost'
                    }
            
            # Build return dictionary
            for global_id, info in all_participants.items():
                if info.get('shape') is not None:
                    # Convert shape to list for JSON serialization over pipe
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
                if current_time - info['last_seen'] > timeout:
                    # Preserve the shape when moving to recently lost
                    self.recently_lost[global_id] = {
                        'centroid': info['centroid'],
                        'last_seen': info['last_seen'],
                        'shape': info.get('shape')  # Preserve shape
                    }
                    to_remove.append(global_id)
                    print(f"[ParticipantManager] Moving participant {global_id} to recently lost (preserving shape)")
                    
            for global_id in to_remove:
                del self.participants[global_id]
                
            # Clean up recently lost that are too old
            to_prune = []
            for lost_id, info in self.recently_lost.items():
                if current_time - info['last_seen'] > self.recently_lost_timeout:
                    to_prune.append(lost_id)
                    
            for lost_id in to_prune:
                del self.recently_lost[lost_id]
                print(f"[ParticipantManager] Permanently removing participant {lost_id} from recently lost")
                
    def reset(self):
        """Clear all participants and reset counter"""
        with self.lock:
            self.participants.clear()
            self.recently_lost.clear()
            self.next_global_id = 1
            
    def get_active_participants(self):
        """Get list of currently active participant IDs"""
        with self.lock:
            current_time = time.time()
            active = []
            for global_id, info in self.participants.items():
                if current_time - info['last_seen'] < 1.0:  # Active within last second
                    active.append(global_id)
            return sorted(active)