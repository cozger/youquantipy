"""
Canvas Drawing Module for YouQuantiPy
Handles all canvas rendering operations with optimizations for face and pose tracking.
"""

import time
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image, ImageTk
import tkinter as tk
from typing import Dict, List, Tuple, Optional, Any

try:
    RESAMPLE = Image.Resampling.LANCZOS
except AttributeError:
    RESAMPLE = Image.LANCZOS


class CanvasDrawingManager:
    """
    Manages efficient canvas drawing for face and pose tracking.
    Implements dirty rectangle tracking, object pooling, and smart updates.
    """
    
    def __init__(self):
        # Canvas object caches
        self.canvas_objects = {}  # {canvas_idx: {face_lines: {}, face_labels: {}, ...}}
        self.transform_cache = {}  # {canvas_idx: {video_bounds: (x, y, w, h)}}
        
        # Performance tracking
        self.last_frame_time = {}  # {canvas_idx: timestamp}
        self.last_face_state = {}  # {canvas_idx: {face_id: (centroid, landmark_count)}}
        self.last_face_ids = {}  # {canvas_idx: tuple(face_ids)}
        
        # Dirty tracking
        self.dirty_regions = {}  # {canvas_idx: [(x, y, w, h), ...]}
        
        # Frame rate limiting
        self.min_frame_interval = 0.033  # 30 FPS max per canvas
        
        # MediaPipe connections
        self.face_connections = mp.solutions.face_mesh.FACEMESH_CONTOURS
        self.pose_connections = mp.solutions.pose.POSE_CONNECTIONS
        
        # Photo image cache to prevent garbage collection
        self._photo_images = {}
        
    def should_skip_frame(self, canvas_idx: int) -> bool:
        """Check if we should skip this frame based on time elapsed."""
        current_time = time.time()
        last_time = self.last_frame_time.get(canvas_idx, 0)
        
        if current_time - last_time < self.min_frame_interval:
            return True
            
        self.last_frame_time[canvas_idx] = current_time
        return False
    
    def render_frame_to_canvas(self, frame_bgr: np.ndarray, canvas: tk.Canvas, 
                               canvas_idx: int) -> Optional[ImageTk.PhotoImage]:
        """
        Render a frame to canvas with proper scaling and caching.
        Returns PhotoImage or None if canvas not ready.
        """
        try:
            # Get canvas dimensions
            canvas.update_idletasks()
            W, H = canvas.winfo_width(), canvas.winfo_height()
            if W <= 1 or H <= 1:
                return None
            
            # Calculate scaling
            frame_h, frame_w = frame_bgr.shape[:2]
            scale = min(W / frame_w, H / frame_h)
            scaled_w, scaled_h = int(frame_w * scale), int(frame_h * scale)
            
            # Calculate centering offsets
            x_offset = (W - scaled_w) // 2
            y_offset = (H - scaled_h) // 2
            
            # Store transform for overlay calculations
            if canvas_idx not in self.transform_cache:
                self.transform_cache[canvas_idx] = {}
            self.transform_cache[canvas_idx]['video_bounds'] = (x_offset, y_offset, scaled_w, scaled_h)
            
            # Resize frame if needed
            if abs(scale - 1.0) > 0.01:
                frame_bgr = cv2.resize(frame_bgr, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
            
            # Create canvas-sized image with black background
            canvas_img = np.zeros((H, W, 3), dtype=np.uint8)
            
            # Place frame in center
            y_end = min(y_offset + scaled_h, H)
            x_end = min(x_offset + scaled_w, W)
            canvas_img[y_offset:y_end, x_offset:x_end] = frame_bgr[:y_end-y_offset, :x_end-x_offset]
            
            # Convert to RGB and create PhotoImage
            rgb = cv2.cvtColor(canvas_img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            photo = ImageTk.PhotoImage(pil_img)
            
            # Update canvas image
            if hasattr(canvas, '_image_id'):
                canvas.itemconfig(canvas._image_id, image=photo)
            else:
                canvas._image_id = canvas.create_image(W // 2, H // 2, image=photo, anchor='center')
            
            # Store reference to prevent garbage collection
            self._photo_images[canvas_idx] = photo
            
            return photo
            
        except Exception as e:
            print(f"[CanvasDrawing] Error rendering frame: {e}")
            return None
    
    def draw_faces_optimized(self, canvas: tk.Canvas, faces: List[Dict], 
                            canvas_idx: int, labels: Optional[Dict] = None,
                            participant_count: int = 2, participant_names: Dict = None) -> None:
        """
        Draw face overlays with smart updates and dirty rectangle tracking.
        Only updates what changed since last frame.
        """
        if not canvas.winfo_exists():
            return
            
        # Initialize cache for this canvas
        if canvas_idx not in self.canvas_objects:
            self.canvas_objects[canvas_idx] = {
                'face_lines': {},
                'face_labels': {},
                'last_seen': {}
            }
        cache = self.canvas_objects[canvas_idx]
        
        # Ensure all required keys exist (for backward compatibility)
        if 'last_seen' not in cache:
            cache['last_seen'] = {}
        if 'face_lines' not in cache:
            cache['face_lines'] = {}
        if 'face_labels' not in cache:
            cache['face_labels'] = {}
        
        # Get video bounds for coordinate transformation
        video_bounds = self.transform_cache.get(canvas_idx, {}).get('video_bounds')
        if not video_bounds:
            return
        x_offset, y_offset, video_w, video_h = video_bounds
        
        # Filter valid faces (respect participant count)
        valid_faces = []
        for face in faces:
            fid = face['id']
            
            # Handle temporary local IDs
            if isinstance(fid, str) and fid.startswith('local_'):
                if participant_count == 1:
                    valid_faces.append(face)
                continue
            
            # Only include faces within participant count
            if isinstance(fid, int) and 1 <= fid <= participant_count:
                valid_faces.append(face)
        
        # Quick check: did face set change?
        current_face_ids = tuple(sorted(f['id'] for f in valid_faces))
        last_ids = self.last_face_ids.get(canvas_idx)
        
        faces_changed = current_face_ids != last_ids
        self.last_face_ids[canvas_idx] = current_face_ids
        
        # Determine which faces need updates
        current_time = time.time()
        active_face_ids = set()
        faces_to_update = {}  # {face_id: 'create'|'update'|'recreate'}
        
        for face in valid_faces:
            fid = face['id']
            cache_key = fid
            
            # Map local IDs in single participant mode
            if isinstance(fid, str) and fid.startswith('local_') and participant_count == 1:
                cache_key = 1
            
            active_face_ids.add(cache_key)
            
            # Check if face needs recreation
            if cache_key not in cache['last_seen']:
                faces_to_update[cache_key] = 'create'
            else:
                time_since = current_time - cache['last_seen'][cache_key]
                if time_since > 2.0:
                    faces_to_update[cache_key] = 'recreate'
                elif not faces_changed:
                    # Just update position
                    faces_to_update[cache_key] = 'update'
                else:
                    faces_to_update[cache_key] = 'update'
            
            cache['last_seen'][cache_key] = current_time
        
        # Hide inactive faces
        self._hide_inactive_faces(canvas, cache, active_face_ids)
        
        # Process each face based on update type
        for face in valid_faces:
            fid = face['id']
            cache_key = 1 if (isinstance(fid, str) and fid.startswith('local_') and participant_count == 1) else fid
            
            if isinstance(cache_key, str) and cache_key.startswith('local_'):
                continue
            
            update_type = faces_to_update.get(cache_key, 'update')
            
            # Get display label
            if labels and cache_key in labels:
                display_label = labels[cache_key]
            elif participant_names and isinstance(cache_key, int):
                display_label = participant_names.get(cache_key - 1, f"P{cache_key}")
            else:
                display_label = f"P{cache_key}"
            
            # Transform landmarks to canvas coordinates
            lm_xyz = face.get('landmarks', [])
            if not lm_xyz:
                continue
                
            canvas_landmarks = [(x * video_w + x_offset, y * video_h + y_offset, z) 
                              for x, y, z in lm_xyz]
            cx_norm, cy_norm = face.get('centroid', (0.5, 0.5))
            label_x = int(cx_norm * video_w + x_offset)
            label_y = int(cy_norm * video_h + y_offset) - 120
            
            if update_type in ['create', 'recreate']:
                # Delete old objects if recreating
                if update_type == 'recreate':
                    self._delete_face_objects(canvas, cache, cache_key)
                
                # Create new objects
                self._create_face_objects(canvas, cache, cache_key, canvas_landmarks, 
                                        label_x, label_y, display_label)
            else:
                # Just update positions
                self._update_face_positions(canvas, cache, cache_key, canvas_landmarks,
                                          label_x, label_y, display_label)
        
        # Ensure overlays are on top
        try:
            canvas.tag_raise('overlay')
        except:
            pass
    
    def draw_poses_optimized(self, canvas: tk.Canvas, poses: List[Dict], 
                            canvas_idx: int, enabled: bool = True) -> None:
        """Draw pose overlays with caching and smart updates."""
        if not enabled or not poses:
            # Hide all pose overlays
            if canvas_idx in self.canvas_objects:
                self._hide_all_poses(canvas, canvas_idx)
            return
        
        # Initialize cache
        if canvas_idx not in self.canvas_objects:
            self.canvas_objects[canvas_idx] = {}
        cache = self.canvas_objects[canvas_idx]
        
        # Get video bounds
        video_bounds = self.transform_cache.get(canvas_idx, {}).get('video_bounds')
        if not video_bounds:
            return
        x_offset, y_offset, video_w, video_h = video_bounds
        
        # Process each pose
        for pose_idx, pose in enumerate(poses):
            if not pose or 'landmarks' not in pose:
                continue
            
            pose_key = f'pose_lines_{pose_idx}'
            if pose_key not in cache:
                cache[pose_key] = []
            
            # Transform coordinates
            pose_coords = [(x * video_w + x_offset, y * video_h + y_offset, z) 
                          for x, y, z in pose['landmarks']]
            
            # Draw or update pose
            self._draw_connections_cached(canvas, pose_coords, self.pose_connections,
                                        cache[pose_key], '#40FFFF', 2)
        
        # Hide unused pose overlays
        self._cleanup_unused_poses(canvas, cache, len(poses))
    
    def draw_debug_detections(self, canvas: tk.Canvas, debug_data: Dict, 
                             canvas_idx: int) -> None:
        """DEBUG_RETINAFACE: Draw raw RetinaFace detections for debugging."""
        if not debug_data:
            return
        
        # Get video bounds
        video_bounds = self.transform_cache.get(canvas_idx, {}).get('video_bounds')
        if not video_bounds:
            return
        x_offset, y_offset, video_w, video_h = video_bounds
        
        # Get raw detections
        raw_detections = debug_data.get('raw_detections', [])
        if not raw_detections:
            return
        
        # Get frame dimensions from debug data
        frame_shape = debug_data.get('frame_shape', (1080, 1920, 3))
        frame_h, frame_w = frame_shape[:2]
        
        # Clear previous debug overlays
        try:
            canvas.delete('debug_detection')
        except:
            pass
        
        # Draw each raw detection
        for i, det in enumerate(raw_detections):
            bbox = det.get('bbox', [])
            if len(bbox) != 4:
                continue
                
            confidence = det.get('confidence', 0)
            
            # Transform coordinates (bbox is already in frame coordinates)
            # Scale to canvas coordinates using actual frame dimensions
            x1 = int(bbox[0] * video_w / frame_w + x_offset)
            y1 = int(bbox[1] * video_h / frame_h + y_offset)
            x2 = int(bbox[2] * video_w / frame_w + x_offset)
            y2 = int(bbox[3] * video_h / frame_h + y_offset)
            
            # Draw rectangle
            color = '#FF0000' if confidence > 0.98 else '#FFFF00'  # Red for high conf, yellow for lower
            canvas.create_rectangle(x1, y1, x2, y2, 
                                  outline=color, width=2, 
                                  tags=('debug_detection', 'overlay'))
            
            # Draw confidence text
            text = f"{confidence:.3f}"
            canvas.create_text(x1, y1-5, text=text, 
                             fill=color, anchor='sw',
                             font=('Arial', 10, 'bold'),
                             tags=('debug_detection', 'overlay'))
        
        # Log debug info
        print(f"[DEBUG_RETINAFACE] Drew {len(raw_detections)} raw detections on canvas {canvas_idx}")
    
    def _hide_inactive_faces(self, canvas: tk.Canvas, cache: Dict, active_ids: set) -> None:
        """Hide faces that are no longer active."""
        # Ensure face_lines exists
        if 'face_lines' not in cache:
            return
            
        for face_id in list(cache['face_lines'].keys()):
            if face_id not in active_ids:
                # Hide lines
                for line_id in cache['face_lines'].get(face_id, []):
                    try:
                        canvas.itemconfig(line_id, state='hidden')
                    except:
                        pass
                
                # Hide label
                if 'face_labels' in cache:
                    label_id = cache['face_labels'].get(face_id)
                    if label_id:
                        try:
                            canvas.itemconfig(label_id, state='hidden')
                        except:
                            pass
    
    def _delete_face_objects(self, canvas: tk.Canvas, cache: Dict, face_id: Any) -> None:
        """Delete all canvas objects for a face."""
        # Delete lines
        for line_id in cache['face_lines'].get(face_id, []):
            try:
                canvas.delete(line_id)
            except:
                pass
        cache['face_lines'][face_id] = []
        
        # Delete label
        label_id = cache['face_labels'].get(face_id)
        if label_id:
            try:
                canvas.delete(label_id)
            except:
                pass
            cache['face_labels'][face_id] = None
    
    def _create_face_objects(self, canvas: tk.Canvas, cache: Dict, face_id: Any,
                           landmarks: List[Tuple], label_x: int, label_y: int, 
                           label_text: str) -> None:
        """Create all canvas objects for a face."""
        # Create lines
        cache['face_lines'][face_id] = []
        for start_idx, end_idx in self.face_connections:
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                x1, y1, _ = landmarks[start_idx]
                x2, y2, _ = landmarks[end_idx]
                
                line_id = canvas.create_line(
                    int(x1), int(y1), int(x2), int(y2),
                    fill='#40FF40', width=1, state='normal',
                    tags=('overlay', f'face_{face_id}', 'face_line')
                )
                cache['face_lines'][face_id].append(line_id)
        
        # Create label
        cache['face_labels'][face_id] = canvas.create_text(
            label_x, label_y, text=label_text,
            fill='yellow', font=('Arial', 14, 'bold'),
            anchor='center', state='normal',
            tags=('overlay', f'face_{face_id}_label', 'face_label')
        )
    
    def _update_face_positions(self, canvas: tk.Canvas, cache: Dict, face_id: Any,
                             landmarks: List[Tuple], label_x: int, label_y: int,
                             label_text: str) -> None:
        """Update positions of existing face objects."""
        # Update lines
        lines = cache['face_lines'].get(face_id, [])
        for i, (start_idx, end_idx) in enumerate(self.face_connections):
            if i >= len(lines):
                break
            
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                x1, y1, _ = landmarks[start_idx]
                x2, y2, _ = landmarks[end_idx]
                
                try:
                    canvas.coords(lines[i], int(x1), int(y1), int(x2), int(y2))
                    canvas.itemconfig(lines[i], state='normal')
                except:
                    # Line is invalid, recreate
                    self._create_face_objects(canvas, cache, face_id, landmarks,
                                            label_x, label_y, label_text)
                    return
        
        # Update label
        label_id = cache['face_labels'].get(face_id)
        if label_id:
            try:
                canvas.coords(label_id, label_x, label_y)
                canvas.itemconfig(label_id, text=label_text, state='normal')
            except:
                # Label is invalid, recreate
                cache['face_labels'][face_id] = canvas.create_text(
                    label_x, label_y, text=label_text,
                    fill='yellow', font=('Arial', 14, 'bold'),
                    anchor='center', state='normal',
                    tags=('overlay', f'face_{face_id}_label', 'face_label')
                )
    
    def _draw_connections_cached(self, canvas: tk.Canvas, landmarks: List[Tuple],
                               connections: List[Tuple], line_cache: List,
                               color: str, width: int) -> None:
        """Draw connections with object pooling."""
        connection_list = list(connections)
        
        # Ensure we have enough lines
        while len(line_cache) < len(connection_list):
            line_id = canvas.create_line(0, 0, 0, 0, fill=color, width=width, 
                                       tags=('overlay',))
            line_cache.append(line_id)
        
        # Update lines
        for i, (start_idx, end_idx) in enumerate(connection_list):
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                x1, y1, _ = landmarks[start_idx]
                x2, y2, _ = landmarks[end_idx]
                
                canvas.coords(line_cache[i], int(x1), int(y1), int(x2), int(y2))
                canvas.itemconfig(line_cache[i], state='normal')
            else:
                canvas.itemconfig(line_cache[i], state='hidden')
        
        # Hide excess lines
        for i in range(len(connection_list), len(line_cache)):
            canvas.itemconfig(line_cache[i], state='hidden')
    
    def _hide_all_poses(self, canvas: tk.Canvas, canvas_idx: int) -> None:
        """Hide all pose overlays for a canvas."""
        cache = self.canvas_objects.get(canvas_idx, {})
        for key in list(cache.keys()):
            if key.startswith('pose_lines'):
                for line_id in cache[key]:
                    try:
                        canvas.itemconfig(line_id, state='hidden')
                    except:
                        pass
    
    def _cleanup_unused_poses(self, canvas: tk.Canvas, cache: Dict, active_count: int) -> None:
        """Hide pose overlays beyond active count."""
        for key in list(cache.keys()):
            if key.startswith('pose_lines_'):
                pose_idx = int(key.split('_')[2])
                if pose_idx >= active_count:
                    for line_id in cache[key]:
                        try:
                            canvas.itemconfig(line_id, state='hidden')
                        except:
                            pass
    
    def cleanup_canvas(self, canvas_idx: int) -> None:
        """Clean up all cached objects for a canvas."""
        if canvas_idx in self.canvas_objects:
            del self.canvas_objects[canvas_idx]
        if canvas_idx in self.transform_cache:
            del self.transform_cache[canvas_idx]
        if canvas_idx in self.last_frame_time:
            del self.last_frame_time[canvas_idx]
        if canvas_idx in self.last_face_state:
            del self.last_face_state[canvas_idx]
        if canvas_idx in self.last_face_ids:
            del self.last_face_ids[canvas_idx]
        if canvas_idx in self._photo_images:
            del self._photo_images[canvas_idx]
    
    def get_stats(self, canvas_idx: int) -> Dict[str, Any]:
        """Get performance statistics for a canvas."""
        stats = {
            'cached_faces': 0,
            'cached_lines': 0,
            'last_update': 0
        }
        
        if canvas_idx in self.canvas_objects:
            cache = self.canvas_objects[canvas_idx]
            stats['cached_faces'] = len(cache.get('face_lines', {}))
            stats['cached_lines'] = sum(len(lines) for lines in cache.get('face_lines', {}).values())
        
        if canvas_idx in self.last_frame_time:
            stats['last_update'] = time.time() - self.last_frame_time[canvas_idx]
        
        return stats


# Standalone utility functions

def draw_overlays_combined(frame_bgr: np.ndarray, faces: List[Dict] = None, 
                          pose_landmarks: List = None, labels: Dict = None,
                          face_mesh: bool = True, face_contours: bool = True,
                          face_points: bool = True, pose_lines: bool = True) -> np.ndarray:
    """
    Draw face and pose overlays on a frame.
    Used for video recording with overlays.
    """
    frame = frame_bgr.copy()
    h, w = frame.shape[:2]

    # Draw face overlays
    if faces:
        for idx, face in enumerate(faces):
            landmarks = face["landmarks"]
            fid = face.get("id", idx+1)
            face_landmarks_px = [(int(x * w), int(y * h), z) for x, y, z in landmarks]
            
            if face_mesh:
                for conn in mp.solutions.face_mesh.FACEMESH_TESSELATION:
                    i, j = conn
                    cv2.line(frame, face_landmarks_px[i][:2], face_landmarks_px[j][:2], 
                            (64,255,64), 1, cv2.LINE_AA)
            
            if face_contours:
                for conn in mp.solutions.face_mesh.FACEMESH_CONTOURS:
                    i, j = conn
                    cv2.line(frame, face_landmarks_px[i][:2], face_landmarks_px[j][:2], 
                            (0,255,0), 2, cv2.LINE_AA)
            
            if face_points:
                for pt in face_landmarks_px:
                    cv2.circle(frame, pt[:2], 1, (0,200,255), -1)
            
            # Draw label
            label_text = None
            if labels and fid in labels:
                label_text = labels[fid]
            elif labels and idx in labels:
                label_text = labels[idx]
            elif not labels:
                label_text = f"Face {fid}"
            
            if label_text:
                if "centroid" in face:
                    cx, cy = int(face["centroid"][0] * w), int(face["centroid"][1] * h) - 50
                else:
                    cx, cy = face_landmarks_px[10][0], face_landmarks_px[10][1] - 50
                cv2.putText(frame, label_text, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (0,255,255), 2, cv2.LINE_AA)

    # Draw pose overlays
    if pose_landmarks is not None and len(pose_landmarks) > 0 and pose_lines:
        pose_landmarks_px = [(int(x * w), int(y * h), z) for x, y, z in pose_landmarks]
        for conn in mp.solutions.holistic.POSE_CONNECTIONS:
            i, j = conn
            if i < len(pose_landmarks_px) and j < len(pose_landmarks_px):
                cv2.line(frame, pose_landmarks_px[i][:2], pose_landmarks_px[j][:2], 
                        (64,255,255), 2, cv2.LINE_AA)

    return frame