"""
Canvas Drawing Module for YouQuantiPy
Handles all canvas rendering operations with optimizations for face and pose tracking.
FIXED: Bbox coordinate transformation for 720p+ resolutions
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
    Implements proper layering and coordinate transformation.
    """
    
    def __init__(self):
        # Canvas object caches
        self.canvas_objects = {}  # {canvas_idx: {face_lines: {}, face_labels: {}, ...}}
        self.transform_cache = {}  # {canvas_idx: {video_bounds: (x, y, w, h), frame_size: (w, h)}}
        
        # Performance tracking
        self.last_frame_time = {}  # {canvas_idx: timestamp}
        self.last_face_state = {}  # {canvas_idx: {face_id: (centroid, landmark_count)}}
        self.last_face_ids = {}  # {canvas_idx: tuple(face_ids)}
        
        # Frame rate limiting
        self.min_frame_interval = 0.033  # 30 FPS max per canvas
        
        # MediaPipe connections
        self.face_connections = mp.solutions.face_mesh.FACEMESH_CONTOURS
        self.pose_connections = mp.solutions.pose.POSE_CONNECTIONS
        
        # Photo image cache to prevent garbage collection
        self._photo_images = {}
        
        # Simple landmark cache for smooth drawing when processing lags
        self.landmark_cache = {}  # {cache_key: (landmarks, timestamp)}
        self.landmark_cache_max_age = 1.0  # seconds
        
        # Debug mode
        self.debug_mode = True  # Set to False to disable debug overlays
        
    def initialize_canvas(self, canvas: tk.Canvas, canvas_idx: int):
        """Initialize canvas with default transform and ensure proper layering"""
        if canvas_idx not in self.transform_cache:
            self.transform_cache[canvas_idx] = {}
        
        # Set default video bounds based on canvas size
        try:
            canvas.update_idletasks()
            W, H = canvas.winfo_width(), canvas.winfo_height()
            if W > 1 and H > 1:
                self.transform_cache[canvas_idx]['video_bounds'] = (0, 0, W, H)
                self.transform_cache[canvas_idx]['canvas_size'] = (W, H)
                print(f"[CanvasDrawing] Initialized canvas {canvas_idx} with size {W}x{H}")
        except:
            pass
    
    def should_skip_frame(self, canvas_idx: int) -> bool:
        """Check if we should skip this frame based on time elapsed."""
        current_time = time.time()
        last_time = self.last_frame_time.get(canvas_idx, 0)
        
        if current_time - last_time < self.min_frame_interval:
            return True
            
        self.last_frame_time[canvas_idx] = current_time
        return False
    
    def render_frame_to_canvas(self, frame_bgr: np.ndarray, canvas: tk.Canvas, 
                            canvas_idx: int, original_resolution: tuple = None) -> Optional[ImageTk.PhotoImage]:
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
            
            # ALWAYS downsample to 480p max for GUI display
            frame_h, frame_w = frame_bgr.shape[:2]
            max_display_width = 640
            max_display_height = 480
            
            # Calculate if we need to downsample
            scale_factor = min(max_display_width / frame_w, max_display_height / frame_h, 1.0)
            if scale_factor < 1.0:
                display_w = int(frame_w * scale_factor)
                display_h = int(frame_h * scale_factor)
                frame_bgr = cv2.resize(frame_bgr, (display_w, display_h), interpolation=cv2.INTER_LINEAR)
                #print(f"[CanvasDrawing] Downsampled from {frame_w}x{frame_h} to {display_w}x{display_h} for display")
            else:
                display_w, display_h = frame_w, frame_h
            
            # Now calculate scaling for canvas
            scale = min(W / display_w, H / display_h)
            scaled_w, scaled_h = int(display_w * scale), int(display_h * scale)
            
            # Calculate centering offsets
            x_offset = (W - scaled_w) // 2
            y_offset = (H - scaled_h) // 2
            
            # Store transform for overlay calculations - CRITICAL FIX
            if canvas_idx not in self.transform_cache:
                self.transform_cache[canvas_idx] = {}
            
            # Store BOTH original capture size and display size
            self.transform_cache[canvas_idx]['video_bounds'] = (x_offset, y_offset, scaled_w, scaled_h)
            if original_resolution:
                self.transform_cache[canvas_idx]['frame_size'] = original_resolution
            else:
                print(f"[CanvasDrawing] WARNING: No original_resolution provided, using frame size {frame_w}x{frame_h}")
                self.transform_cache[canvas_idx]['frame_size'] = (frame_w, frame_h)
                
            self.transform_cache[canvas_idx]['display_size'] = (display_w, display_h)
            self.transform_cache[canvas_idx]['canvas_size'] = (W, H)
            self.transform_cache[canvas_idx]['scale'] = scale
            self.transform_cache[canvas_idx]['downsample_scale'] = scale_factor
            
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
            
            # Update canvas image - use lower tag to ensure overlays appear on top
            if hasattr(canvas, '_image_id'):
                canvas.itemconfig(canvas._image_id, image=photo)
            else:
                canvas._image_id = canvas.create_image(W // 2, H // 2, image=photo, anchor='center', tags=('background',))
                canvas.tag_lower('background')  # Ensure background is at bottom
            
            # Store reference to prevent garbage collection
            self._photo_images[canvas_idx] = photo
            
            return photo
            
        except Exception as e:
            print(f"[CanvasDrawing] Error rendering frame: {e}")
            import traceback
            traceback.print_exc()
            return None

    def convert_bbox_to_canvas(self, bbox, transform_data):
        frame_w, frame_h = transform_data['frame_size']   # Original frame resolution
        x_offset, y_offset, video_w, video_h = transform_data['video_bounds']

        # Compute scaling from original frame to GUI canvas explicitly
        scale_x = video_w / frame_w
        scale_y = video_h / frame_h

        # Correctly transform bbox coordinates from capture to GUI
        x1 = bbox[0] * scale_x + x_offset
        y1 = bbox[1] * scale_y + y_offset
        x2 = bbox[2] * scale_x + x_offset
        y2 = bbox[3] * scale_y + y_offset

        #print(f"Original bbox: {bbox}, Transformed bbox: {(x1, y1, x2, y2)}")

        return [x1, y1, x2, y2]


    def draw_faces_optimized(self, canvas: tk.Canvas, faces: List[Dict], 
                            canvas_idx: int, labels: Optional[Dict] = None,
                            participant_count: int = 2, participant_names: Dict = None) -> None:
        """
        Draw face overlays with proper coordinate transformation.
        """
        if not canvas.winfo_exists():
            return
        
        # Initialize cache for this canvas
        if canvas_idx not in self.canvas_objects:
            self.canvas_objects[canvas_idx] = {
                'face_lines': {},
                'face_labels': {},
                'face_bboxes': {},  # Add bbox tracking
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
        if 'face_bboxes' not in cache:
            cache['face_bboxes'] = {}
        
        # Get video bounds and frame size for coordinate transformation
        transform_data = self.transform_cache.get(canvas_idx, {})
        video_bounds = transform_data.get('video_bounds')
        frame_size = transform_data.get('frame_size')
        
        # Get display size (after downsampling) - with fallback to frame_size
        display_size = transform_data.get('display_size', frame_size)
        downsample_scale = transform_data.get('downsample_scale', 1.0)
        
        if not video_bounds or not frame_size:
            print(f"[CanvasDrawing] No transform data for canvas {canvas_idx}")
            return
            
        x_offset, y_offset, video_w, video_h = video_bounds
        frame_w, frame_h = frame_size
        
        # Extract display dimensions with fallback
        if display_size:
            display_w, display_h = display_size
        else:
            # Fallback: use frame size
            display_w, display_h = frame_w, frame_h
        
        # Filter valid faces (existing code)
        valid_faces = []
        for face in faces:
            fid = face.get('id', face.get('global_id', face.get('track_id')))
            
            # Handle temporary local IDs
            if isinstance(fid, str) and fid.startswith('local_'):
                if participant_count == 1:
                    face['id'] = 1
                    valid_faces.append(face)
                continue
            
            # Accept face IDs
            if isinstance(fid, int):
                if fid <= participant_count or fid == 0:
                    valid_faces.append(face)
        
        # Track active faces
        current_time = time.time()
        active_face_ids = set()
        
        # Process each face
        for face in valid_faces:
            fid = face.get('id', 1)
            active_face_ids.add(fid)
            cache['last_seen'][fid] = current_time
            
            # Get display label
            if labels and fid in labels:
                display_label = labels[fid]
            elif participant_names and isinstance(fid, int):
                display_label = participant_names.get(fid - 1, f"P{fid}")
            else:
                display_label = f"P{fid}"
            
           # Draw bounding box if available
            if 'bbox' in face and self.debug_mode:
                bbox = face['bbox']
                
                # Use the new conversion method
                canvas_bbox = self.convert_bbox_to_canvas(bbox, transform_data)
                x1, y1, x2, y2 = [int(coord) for coord in canvas_bbox]
                
                # Ensure coordinates stay within canvas bounds
                canvas_w, canvas_h = self.transform_cache[canvas_idx].get('canvas_size', (9999, 9999))
                x1 = max(0, min(x1, canvas_w - 1))
                y1 = max(0, min(y1, canvas_h - 1))
                x2 = max(0, min(x2, canvas_w - 1))
                y2 = max(0, min(y2, canvas_h - 1))
                
                # Debug print occasionally
                if canvas_idx == 0 and not hasattr(self, '_bbox_debug_count'):
                    self._bbox_debug_count = 0
                self._bbox_debug_count = getattr(self, '_bbox_debug_count', 0) + 1
                
                if self._bbox_debug_count % 300 == 0:
                    print(f"\n[CanvasDrawing] Bbox Conversion Debug:")
                    print(f"  Original bbox: {bbox}")
                    print(f"  Canvas bbox: ({x1}, {y1}) - ({x2}, {y2})")
                    print(f"  Transform used: scale_x={video_w/frame_w:.3f}, scale_y={video_h/frame_h:.3f}")
                
                # Create or update bbox rectangle
                bbox_key = f'bbox_{fid}'
                if bbox_key in cache.get('face_bboxes', {}):
                    bbox_id = cache['face_bboxes'][bbox_key]
                    canvas.coords(bbox_id, x1, y1, x2, y2)
                    canvas.itemconfig(bbox_id, state='normal')
                else:
                    bbox_id = canvas.create_rectangle(
                        x1, y1, x2, y2,
                        outline='#00FF00', width=2,
                        tags=('overlay', f'face_bbox_{fid}')
                    )
                    if 'face_bboxes' not in cache:
                        cache['face_bboxes'] = {}
                    cache['face_bboxes'][bbox_key] = bbox_id

                # print("---- DEBUG INFO ----")
                # print(f"Original bbox coords: {bbox}")
                # print(f"Original frame size: {transform_data['frame_size']}")
                # print(f"Canvas video bounds: {transform_data['video_bounds']}")
                # converted_bbox = self.convert_bbox_to_canvas(bbox, transform_data)
                # print(f"Converted bbox coords: {converted_bbox}")
                # print("---------------------")

            
            # Draw face landmarks if available (rest of the method remains the same)
            lm_xyz = face.get('landmarks', [])
            if lm_xyz and len(lm_xyz) > 0:
                # Transform landmarks from normalized to canvas coordinates
                canvas_landmarks = []
                for x, y, z in lm_xyz:
                    canvas_x = x * video_w + x_offset
                    canvas_y = y * video_h + y_offset
                    canvas_landmarks.append((canvas_x, canvas_y, z))
                
                # Initialize face lines if needed
                if fid not in cache['face_lines']:
                    cache['face_lines'][fid] = []
                
                # Draw or update face contours
                self._update_face_contours(canvas, cache, fid, canvas_landmarks, display_label)
            
            # Draw centroid if available (for debugging)
            if 'centroid' in face and self.debug_mode:
                cx_norm, cy_norm = face['centroid']
                # Centroid is already normalized (0-1), so just scale to display
                cx = int(cx_norm * video_w + x_offset)
                cy = int(cy_norm * video_h + y_offset)
                
                centroid_key = f'centroid_{fid}'
                if centroid_key in cache.get('face_bboxes', {}):
                    centroid_id = cache['face_bboxes'][centroid_key]
                    canvas.coords(centroid_id, cx-5, cy-5, cx+5, cy+5)
                    canvas.itemconfig(centroid_id, state='normal')
                else:
                    centroid_id = canvas.create_oval(
                        cx-5, cy-5, cx+5, cy+5,
                        fill='red', outline='yellow', width=2,
                        tags=('overlay', f'face_centroid_{fid}')
                    )
                    cache['face_bboxes'][centroid_key] = centroid_id
        
        # Hide inactive faces
        self._hide_inactive_faces(canvas, cache, active_face_ids)
        
        # Ensure overlays are on top
        canvas.tag_raise('overlay')

    def _update_face_contours(self, canvas: tk.Canvas, cache: Dict, face_id: int,
                             landmarks: List[Tuple], label_text: str) -> None:
        """Update or create face contour lines."""
        lines = cache['face_lines'].get(face_id, [])
        
        # Ensure we have enough line objects
        connection_list = list(self.face_connections)
        while len(lines) < len(connection_list):
            line_id = canvas.create_line(
                0, 0, 0, 0,
                fill='#40FF40', width=2,
                tags=('overlay', f'face_{face_id}', 'face_line')
            )
            lines.append(line_id)
        
        cache['face_lines'][face_id] = lines
        
        # Update line positions
        for i, (start_idx, end_idx) in enumerate(connection_list):
            if i >= len(lines):
                break
                
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                x1, y1, _ = landmarks[start_idx]
                x2, y2, _ = landmarks[end_idx]
                
                try:
                    canvas.coords(lines[i], int(x1), int(y1), int(x2), int(y2))
                    canvas.itemconfig(lines[i], state='normal')
                except:
                    pass
            else:
                canvas.itemconfig(lines[i], state='hidden')
        
        # Hide excess lines
        for i in range(len(connection_list), len(lines)):
            canvas.itemconfig(lines[i], state='hidden')
    
    def _hide_inactive_faces(self, canvas: tk.Canvas, cache: Dict, active_ids: set) -> None:
        """Hide faces that are no longer active."""
        current_time = time.time()
        
        # Check all tracked faces
        for face_id in list(cache.get('last_seen', {}).keys()):
            if face_id not in active_ids:
                # Hide after timeout
                last_seen = cache['last_seen'].get(face_id, 0)
                if current_time - last_seen > 0.5:  # 500ms timeout
                    # Hide face lines
                    for line_id in cache.get('face_lines', {}).get(face_id, []):
                        try:
                            canvas.itemconfig(line_id, state='hidden')
                        except:
                            pass
                    
                    # Hide face bboxes and labels
                    for key in [f'bbox_{face_id}', f'bbox_label_{face_id}', f'centroid_{face_id}']:
                        if key in cache.get('face_bboxes', {}):
                            try:
                                canvas.itemconfig(cache['face_bboxes'][key], state='hidden')
                            except:
                                pass
    
    def draw_poses_optimized(self, canvas: tk.Canvas, poses: List[Dict], 
                            canvas_idx: int, enabled: bool = True) -> None:
        """Draw pose overlays with proper coordinate transformation."""
        if not enabled or not poses:
            # Hide all pose overlays
            if canvas_idx in self.canvas_objects:
                self._hide_all_poses(canvas, canvas_idx)
            return
        
        # Initialize cache
        if canvas_idx not in self.canvas_objects:
            self.canvas_objects[canvas_idx] = {}
        cache = self.canvas_objects[canvas_idx]
        
        # Get transform data
        transform_data = self.transform_cache.get(canvas_idx, {})
        video_bounds = transform_data.get('video_bounds')
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
        
        # Ensure overlays are on top
        canvas.tag_raise('overlay')
    
    def draw_debug_detections(self, canvas: tk.Canvas, debug_data: Dict, 
                            canvas_idx: int) -> None:
        """Draw raw RetinaFace detections for debugging."""
        if not debug_data or not self.debug_mode:
            return
        
        # Get transform data
        transform_data = self.transform_cache.get(canvas_idx, {})
        if not transform_data.get('video_bounds') or not transform_data.get('frame_size'):
            return
        
        # Get raw detections
        raw_detections = debug_data.get('raw_detections', [])
        if not raw_detections:
            return
        
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
            
            # Use the convert_bbox_to_canvas method
            canvas_bbox = self.convert_bbox_to_canvas(bbox, transform_data)
            x1, y1, x2, y2 = [int(coord) for coord in canvas_bbox]
            
            # Ensure coordinates stay within canvas bounds
            canvas_w, canvas_h = self.transform_cache.get(canvas_idx, {}).get('canvas_size', (9999, 9999))
            x1 = max(0, min(x1, canvas_w - 1))
            y1 = max(0, min(y1, canvas_h - 1))
            x2 = max(0, min(x2, canvas_w - 1))
            y2 = max(0, min(y2, canvas_h - 1))
            
            # Draw rectangle
            color = '#FF0000' if confidence > 0.98 else '#FFFF00'
            canvas.create_rectangle(x1, y1, x2, y2, 
                                outline=color, width=2, 
                                tags=('debug_detection', 'overlay'))
            
            # Draw confidence text
            text = f"{confidence:.3f}"
            label_y = max(5, y1 - 5)  # Ensure label stays on screen
            canvas.create_text(x1, label_y, text=text, 
                            fill=color, anchor='sw',
                            font=('Arial', 10, 'bold'),
                            tags=('debug_detection', 'overlay'))
        
        # Ensure debug overlays are on top
        canvas.tag_raise('overlay')
    
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
            'last_update': 0,
            'transform_data': None
        }
        
        if canvas_idx in self.canvas_objects:
            cache = self.canvas_objects[canvas_idx]
            stats['cached_faces'] = len(cache.get('face_lines', {}))
            stats['cached_lines'] = sum(len(lines) for lines in cache.get('face_lines', {}).values())
        
        if canvas_idx in self.last_frame_time:
            stats['last_update'] = time.time() - self.last_frame_time[canvas_idx]
        
        if canvas_idx in self.transform_cache:
            stats['transform_data'] = self.transform_cache[canvas_idx]
        
        return stats


# Standalone utility function remains the same
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
            landmarks = face.get("landmarks", [])
            fid = face.get("id", idx+1)
            
            if landmarks:
                face_landmarks_px = [(int(x * w), int(y * h), z) for x, y, z in landmarks]
                
                if face_mesh:
                    for conn in mp.solutions.face_mesh.FACEMESH_TESSELATION:
                        i, j = conn
                        if i < len(face_landmarks_px) and j < len(face_landmarks_px):
                            cv2.line(frame, face_landmarks_px[i][:2], face_landmarks_px[j][:2], 
                                    (64,255,64), 1, cv2.LINE_AA)
                
                if face_contours:
                    for conn in mp.solutions.face_mesh.FACEMESH_CONTOURS:
                        i, j = conn
                        if i < len(face_landmarks_px) and j < len(face_landmarks_px):
                            cv2.line(frame, face_landmarks_px[i][:2], face_landmarks_px[j][:2], 
                                    (0,255,0), 2, cv2.LINE_AA)
                
                if face_points:
                    for pt in face_landmarks_px:
                        cv2.circle(frame, pt[:2], 1, (0,200,255), -1)
            
            # Draw bounding box if available
            if 'bbox' in face:
                bbox = face['bbox']
                # Bbox coordinates are in pixel coordinates for the frame
                x1, y1, x2, y2 = [int(v) for v in bbox]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label_text = None
            if labels and fid in labels:
                label_text = labels[fid]
            elif not labels:
                label_text = f"Face {fid}"
            
            if label_text:
                if "centroid" in face and face["centroid"]:
                    cx, cy = int(face["centroid"][0] * w), int(face["centroid"][1] * h) - 50
                elif landmarks and len(face_landmarks_px) > 10:
                    cx, cy = face_landmarks_px[10][0], face_landmarks_px[10][1] - 50
                elif 'bbox' in face:
                    bbox = face['bbox']
                    # Bbox is in pixel coordinates
                    cx = int((bbox[0] + bbox[2]) / 2)
                    cy = int(bbox[1] - 10)
                else:
                    cx, cy = w // 2, 50
                    
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