# tracker.py - Complete implementation with Procrustes + Hungarian
import math
import numpy as np
from collections import defaultdict
import time
from scipy.spatial import procrustes
from scipy.optimize import linear_sum_assignment

class UnifiedTracker:
    """
    Tracks participants using Procrustes shape analysis and Hungarian assignment.
    Associates faces with poses based on spatial proximity and shape matching.
    """
    
    # Stable landmarks that don't change much with expressions
    STABLE_LANDMARK_INDICES = [
        33, 263,          # Left and right eye outer corners
        133, 362,         # Left and right eye inner corners
        1,                # Nose tip
        168,              # Nose bridge upper
        197,              # Nose bridge middle
        5,                # Nose bridge lower
        234, 454,         # Left and right cheekbones
        152,              # Chin bottom
        377, 400,         # Left and right jaw angles
        10,               # Forehead center
        9,                # Forehead top
        127, 356,         # Left and right face sides (temple area)
        93, 323,          # Left and right lower cheek
    ]
    
    def __init__(self, face_threshold=0.1, pose_threshold=0.15, max_missed=60,
                 max_participants=2, recently_lost_frames=300,
                 procrustes_threshold=0.05, shape_weight=0.7, position_weight=0.3):
        """
        Initialize tracker with Procrustes-based shape matching.
        
        Args:
            face_threshold: Maximum distance for face position matching (normalized)
            pose_threshold: Maximum distance for pose association (normalized)
            max_missed: Number of frames a participant can be missed before removal
            recently_lost_frames: Number of frames to keep recently lost participants
            procrustes_threshold: Maximum Procrustes distance for valid match
            shape_weight: Weight for shape matching (0-1)
            position_weight: Weight for position matching (0-1)
        """
        self.face_threshold = face_threshold
        self.pose_threshold = pose_threshold
        self.max_missed = max_missed
        self.procrustes_threshold = procrustes_threshold
        self.shape_weight = shape_weight
        self.position_weight = position_weight
        
        self.next_id = 1
        self.participants = {}  # id -> {'face': {...}, 'pose': {...}, 'missed': int, 'shape': np.array}
        self.recently_lost = {}  # id -> {'centroid': (x,y), 'frames_missing': int, 'shape': np.array}
        self.max_participants = max_participants
        self.recently_lost_frames = recently_lost_frames
        
        # Track which IDs are preferred slots (1, 2, etc.) vs temporary IDs
        self.preferred_ids = set(range(1, max_participants + 1))
        self.id_generation = {}
        
        # Global shapes from participant manager
        self.global_shapes = {}
    
    def update_global_shapes(self, shapes_dict):
        """Receive global participant shapes from participant manager"""
        self.global_shapes = shapes_dict
    
    def _extract_stable_shape(self, landmarks):
        """
        Extract stable landmark points as a shape matrix for Procrustes analysis.
        Returns a normalized shape array.
        """
        if not landmarks or len(landmarks) < 468:
            return None
            
        try:
            # Extract stable points
            shape_points = []
            for idx in self.STABLE_LANDMARK_INDICES:
                if idx < len(landmarks):
                    # Use 2D coordinates (x, y) only for better stability
                    shape_points.append([landmarks[idx][0], landmarks[idx][1]])
                else:
                    return None
            
            shape = np.array(shape_points, dtype=np.float64)
            
            # Center the shape (remove translation)
            shape_centered = shape - np.mean(shape, axis=0)
            
            # Check validity
            scale = np.sqrt(np.sum(shape_centered**2))
            if scale < 0.001:
                return None
                
            # Return the centered shape (Procrustes will handle scale/rotation)
            return shape_centered
            
        except Exception as e:
            print(f"[Tracker] Error extracting shape: {e}")
            return None
    
    def _procrustes_distance(self, shape1, shape2):
        """
        Calculate Procrustes distance between two shapes.
        This removes differences due to translation, rotation, and scale.
        """
        if shape1 is None or shape2 is None:
            return float('inf')
            
        try:
            # Ensure arrays
            shape1 = np.array(shape1, dtype=np.float64)
            shape2 = np.array(shape2, dtype=np.float64)
            
            # Check shapes match
            if shape1.shape != shape2.shape:
                return float('inf')
            
            # Compute Procrustes distance
            # mtx1 and mtx2 are the aligned shapes, disparity is the sum of squared differences
            mtx1, mtx2, disparity = procrustes(shape1, shape2)
            
            # Normalize by number of points for consistent threshold
            normalized_distance = np.sqrt(disparity / len(shape1))
            
            return normalized_distance
            
        except Exception as e:
            print(f"[Tracker] Procrustes error: {e}")
            return float('inf')
    
    def _euclidean_distance(self, p1, p2):
        """Calculate Euclidean distance between two points"""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def _combined_cost(self, face, shape1, candidate_info, shape2):
        """
        Calculate combined cost using shape and position.
        Lower values indicate better matches.
        """
        # Shape cost (Procrustes distance)
        if shape1 is not None and shape2 is not None:
            shape_cost = self._procrustes_distance(shape1, shape2)
        else:
            shape_cost = 1.0  # Neutral cost if no shape available
        
        # Position cost (normalized)
        centroid1 = face['centroid']
        centroid2 = candidate_info.get('centroid', (0.5, 0.5))
        pos_cost = self._euclidean_distance(centroid1, centroid2) / self.face_threshold
        
        # Combined cost
        return self.shape_weight * shape_cost + self.position_weight * pos_cost
    
    def update(self, face_detections, pose_detections=None):
        """
        Update tracking with face and optional pose detections using Hungarian assignment.
        
        Args:
            face_detections: list of {'centroid': (x,y), 'landmarks': [...], 'blend': [...]}
            pose_detections: list of {'centroid': (x,y), 'landmarks': [...]} or None
            
        Returns:
            list of participant IDs matching face_detections order
        """
        # Use Hungarian assignment for faces
        face_assignments = self._assign_faces_hungarian(face_detections)
        
        # Associate poses if available
        if pose_detections:
            self._associate_poses(pose_detections)
            
        # Clean up missed participants
        self._cleanup_missed()
        
        return face_assignments
    
    def _assign_faces_hungarian(self, face_detections):
        """Assign faces using Hungarian algorithm with Procrustes distances"""
        n_detections = len(face_detections)
        
        if n_detections == 0:
            return []
        
        # Extract shapes for all detections
        detection_shapes = []
        for face in face_detections:
            shape = self._extract_stable_shape(face.get('landmarks', []))
            detection_shapes.append(shape)
        
        # Build list of all candidate participants (local + global + recently lost)
        candidates = []
        candidate_sources = []
        
        # Add active local participants
        for pid, participant in self.participants.items():
            if participant.get('shape') is not None:
                candidates.append({
                    'id': pid,
                    'shape': participant['shape'],
                    'centroid': participant['face']['centroid'] if participant.get('face') else (0.5, 0.5),
                    'missed': participant.get('missed', 0),
                })
                candidate_sources.append(('local', pid))
        
        # Add recently lost participants
        for pid, info in self.recently_lost.items():
            if info.get('shape') is not None and pid <= self.max_participants:
                candidates.append({
                    'id': pid,
                    'shape': info['shape'],
                    'centroid': info.get('centroid', (0.5, 0.5)),
                    'missed': info.get('frames_missing', 0),
                })
                candidate_sources.append(('lost', pid))
        
        # Add global participants from other cameras
        for pid, shape_info in self.global_shapes.items():
            # Don't add if already in local or recently lost
            if not any(src[1] == pid for src in candidate_sources):
                candidates.append({
                    'id': pid,
                    'shape': np.array(shape_info['shape']),
                    'centroid': shape_info.get('centroid', (0.5, 0.5)),
                    'missed': shape_info.get('frames_since_seen', 999),
                })
                candidate_sources.append(('global', pid))
        
        # If no candidates, create new IDs
        if not candidates:
            assignments = []
            used_ids = set()
            for i in range(n_detections):
                new_id = self._get_next_available_id(used_ids)
                if new_id and new_id <= self.max_participants:
                    self.participants[new_id] = {
                        'face': face_detections[i],
                        'pose': None,
                        'missed': 0,
                        'shape': detection_shapes[i]
                    }
                    assignments.append(new_id)
                    used_ids.add(new_id)
                    print(f"[Tracker] Created new participant {new_id} (no candidates)")
                else:
                    assignments.append(None)
            return assignments
        
        # Build cost matrix
        n_candidates = len(candidates)
        cost_matrix = np.full((n_detections, n_candidates), 10.0)  # High default cost
        
        for i, (face, det_shape) in enumerate(zip(face_detections, detection_shapes)):
            for j, candidate in enumerate(candidates):
                # Calculate combined cost
                cost = self._combined_cost(face, det_shape, candidate, candidate['shape'])
                
                # Add penalties based on source
                source_type, source_id = candidate_sources[j]
                if source_type == 'lost':
                    cost += 0.1  # Small penalty for recently lost
                elif source_type == 'global':
                    cost += 0.2  # Slightly higher penalty for global participants
                
                # Add penalty for participants missed many frames
                cost += min(candidate['missed'] / 100.0, 0.5)
                
                cost_matrix[i, j] = cost
        
        # Run Hungarian algorithm
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        # Process assignments
        assignments = [None] * n_detections
        used_candidate_indices = set()
        used_ids = set()
        
        # First pass - apply valid assignments
        for det_idx, cand_idx in zip(row_indices, col_indices):
            cost = cost_matrix[det_idx, cand_idx]
            
            # Check if assignment is valid
            if cost < 1.0:  # Threshold for valid match
                candidate = candidates[cand_idx]
                source_type, source_id = candidate_sources[cand_idx]
                participant_id = candidate['id']
                
                # Update or create local participant
                self.participants[participant_id] = {
                    'face': face_detections[det_idx],
                    'shape': detection_shapes[det_idx],
                    'missed': 0,
                    'pose': self.participants.get(participant_id, {}).get('pose')
                }
                
                # Remove from recently lost if it was there
                if participant_id in self.recently_lost:
                    del self.recently_lost[participant_id]
                    print(f"[Tracker] Recovered lost participant {participant_id}")
                
                assignments[det_idx] = participant_id
                used_candidate_indices.add(cand_idx)
                used_ids.add(participant_id)
        
        # Second pass - create new participants for unassigned detections
        for i, pid in enumerate(assignments):
            if pid is None and detection_shapes[i] is not None:
                new_id = self._get_next_available_id(used_ids)
                if new_id and new_id <= self.max_participants:
                    self.participants[new_id] = {
                        'face': face_detections[i],
                        'pose': None,
                        'missed': 0,
                        'shape': detection_shapes[i]
                    }
                    assignments[i] = new_id
                    used_ids.add(new_id)
                    print(f"[Tracker] Created new participant {new_id}")
        
        # Mark unused participants as missed
        for pid in list(self.participants.keys()):
            if pid not in used_ids:
                self.participants[pid]['missed'] += 1
                if self.participants[pid]['missed'] > 2:
                    self.participants[pid]['pose'] = None
        
        return assignments
    
    def _get_next_available_id(self, used_ids):
        """Get the next available participant ID, preferring low numbers"""
        # First try to use preferred IDs (1, 2, 3, ...)
        for pid in range(1, self.max_participants + 1):
            if pid not in used_ids and pid not in self.participants and pid not in self.recently_lost:
                return pid
        return None
    
    def _associate_poses(self, pose_detections):
        """Associate pose detections with participants based on proximity"""
        used_poses = set()
        
        # For each participant with a face, find the closest pose
        for pid, participant in self.participants.items():
            if participant['missed'] > 0:  # Skip missed participants
                continue
                
            if 'face' in participant and participant['face']:
                face_centroid = participant['face']['centroid']
                best_pose_idx = None
                best_dist = float('inf')
                
                for pose_idx, pose in enumerate(pose_detections):
                    if pose_idx in used_poses:
                        continue
                    
                    # Use nose position (landmark 0) if available for better accuracy
                    if 'landmarks' in pose and len(pose['landmarks']) > 0:
                        nose_pos = pose['landmarks'][0]
                        dist = self._euclidean_distance(nose_pos[:2], face_centroid)
                    else:
                        dist = self._euclidean_distance(pose['centroid'], face_centroid)
                    
                    if dist < best_dist and dist < self.pose_threshold:
                        best_dist = dist
                        best_pose_idx = pose_idx
                
                if best_pose_idx is not None:
                    self.participants[pid]['pose'] = pose_detections[best_pose_idx]
                    used_poses.add(best_pose_idx)
    
    def _cleanup_missed(self):
        """Remove participants that have been missed too long, and keep recently lost"""
        to_remove = []
        for pid, participant in self.participants.items():
            if participant['missed'] > self.max_missed:
                # Add to recently lost with shape
                self.recently_lost[pid] = {
                    'centroid': participant['face']['centroid'] if participant.get('face') else (0.5, 0.5),
                    'frames_missing': 0,
                    'shape': participant.get('shape')
                }
                to_remove.append(pid)
                print(f"[Tracker] Moving participant {pid} to recently lost after {participant['missed']} missed frames")
        
        for pid in to_remove:
            del self.participants[pid]
        
        # Update recently lost
        lost_to_delete = []
        for pid, info in self.recently_lost.items():
            info['frames_missing'] += 1
            if info['frames_missing'] > self.recently_lost_frames:
                lost_to_delete.append(pid)
                
        for pid in lost_to_delete:
            del self.recently_lost[pid]
            print(f"[Tracker] Permanently removing participant {pid} from recently lost")
    
    def get_participant(self, pid):
        """Get participant data by ID"""
        return self.participants.get(pid, None)
    
    def get_active_participants(self):
        """Get all participants that haven't been missed"""
        return {pid: p for pid, p in self.participants.items() if p['missed'] == 0}
    
    def reset(self):
        """Clear all tracking data"""
        self.next_id = 1
        self.participants.clear()
        self.recently_lost.clear()
        self.global_shapes.clear()