# Phase 3 GUI Integration Guide

## Overview
This guide provides concrete code examples and patterns for integrating the GPU pipeline camera workers into the existing GUI system.

## Key Integration Points

### 1. Replace Direct Camera Capture

**Current Pattern (to remove):**
```python
# In parallelworker_unified.py or gui.py
self.cap = cv2.VideoCapture(camera_index)
ret, frame = self.cap.read()
# Direct GPU processing...
```

**New Pattern (to implement):**
```python
# In gui.py initialization
from camera_worker_integration import CameraWorkerManager

class YouQuantiPyGUI:
    def __init__(self):
        # ... existing init code ...
        
        # Create camera worker manager
        self.camera_manager = CameraWorkerManager(self.config)
        
        # Track active cameras
        self.active_cameras = {}
        
    def start_camera(self, camera_index):
        """Start a camera worker."""
        if self.camera_manager.start_camera(camera_index):
            self.active_cameras[camera_index] = {
                'connected': True,
                'last_frame': None,
                'face_count': 0
            }
            return True
        return False
```

### 2. Update Main Processing Loop

**Replace the current frame processing with:**
```python
def update_frame_processing(self):
    """Main GUI update loop for frame processing."""
    # Process metadata from all cameras
    metadata_list = self.camera_manager.process_metadata()
    for metadata in metadata_list:
        cam_idx = metadata['camera_index']
        
        # Update camera info
        if cam_idx in self.active_cameras:
            self.active_cameras[cam_idx]['face_count'] = metadata['n_faces']
            self.active_cameras[cam_idx]['last_frame_id'] = metadata['frame_id']
            
        # Update performance stats
        self.update_performance_display(cam_idx, metadata)
    
    # Process status updates
    status_list = self.camera_manager.process_status_updates()
    for status in status_list:
        self.handle_status_update(status)
    
    # Get preview frames for display
    for cam_idx in self.active_cameras:
        frame = self.camera_manager.get_preview_frame(cam_idx)
        if frame is not None:
            self.display_camera_frame(cam_idx, frame)
            
        # Get landmarks if needed
        landmarks = self.camera_manager.get_landmarks(cam_idx)
        if landmarks is not None:
            self.process_landmarks(cam_idx, landmarks)
```

### 3. Canvas Drawing Integration

**Update canvasdrawing.py to work with shared memory frames:**
```python
class CanvasDrawing:
    def update_frame(self, camera_index, frame, faces=None):
        """Update canvas with new frame from shared memory."""
        # Frame is already in preview resolution (960x540)
        # No need for GPU processing - just display
        
        # Convert BGR to RGB for display
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Draw face overlays if provided
        if faces:
            for face in faces:
                self.draw_face_bbox(rgb_frame, face['bbox'])
                if 'landmarks' in face:
                    self.draw_landmarks(rgb_frame, face['landmarks'])
        
        # Update canvas
        self.update_canvas_image(camera_index, rgb_frame)
```

### 4. Coordinate Transformation

**Important: Coordinates are already in preview space (960x540):**
```python
def transform_coordinates(self, cam_idx, faces):
    """Transform face coordinates for display."""
    # Get display dimensions
    display_width, display_height = self.get_display_size(cam_idx)
    
    # Preview is always 960x540
    preview_width, preview_height = 960, 540
    
    # Simple scaling from preview to display
    scale_x = display_width / preview_width
    scale_y = display_height / preview_height
    
    for face in faces:
        # Transform bbox
        bbox = face['bbox']
        face['display_bbox'] = [
            bbox[0] * scale_x,
            bbox[1] * scale_y,
            bbox[2] * scale_x,
            bbox[3] * scale_y
        ]
        
        # Transform landmarks if present
        if 'landmarks' in face:
            face['display_landmarks'] = face['landmarks'].copy()
            face['display_landmarks'][:, 0] *= scale_x
            face['display_landmarks'][:, 1] *= scale_y
```

### 5. LSL Integration Updates

**Modify LSLHelper.py to read from metadata queue:**
```python
class LSLHelper:
    def process_camera_data(self, metadata, landmarks_data):
        """Process camera data for LSL output."""
        # Create LSL sample
        sample = []
        
        # Add timestamp
        sample.append(metadata['timestamp'])
        
        # Add face count
        sample.append(metadata['n_faces'])
        
        # Add face data if available
        if landmarks_data:
            for i in range(landmarks_data['n_faces']):
                face_landmarks = landmarks_data['landmarks'][i]
                # Flatten landmarks for LSL
                sample.extend(face_landmarks.flatten())
        
        # Send via LSL
        self.outlet.push_sample(sample)
```

### 6. Error Handling and Recovery

**Add camera health monitoring:**
```python
def handle_status_update(self, status):
    """Handle status updates from camera workers."""
    cam_idx = status['camera_index']
    
    if status['type'] == 'error':
        # Log error
        logger.error(f"Camera {cam_idx}: {status['data']['message']}")
        
        # Update UI to show error
        self.show_camera_error(cam_idx, status['data']['message'])
        
        # Attempt recovery
        if 'disconnected' in status['data']['message']:
            self.schedule_camera_restart(cam_idx)
            
    elif status['type'] == 'warning':
        # Show warning in UI
        self.show_camera_warning(cam_idx, status['data']['message'])
        
    elif status['type'] == 'heartbeat':
        # Update health indicator
        self.update_camera_health(cam_idx, status['data'])
```

### 7. GUI Control Integration

**Add camera control buttons:**
```python
def create_camera_controls(self, camera_index):
    """Create control buttons for camera."""
    controls_frame = tk.Frame(self.camera_frame)
    
    # Pause/Resume button
    self.pause_btn = tk.Button(
        controls_frame, 
        text="Pause",
        command=lambda: self.toggle_camera_pause(camera_index)
    )
    
    # Reconnect button
    self.reconnect_btn = tk.Button(
        controls_frame,
        text="Reconnect",
        command=lambda: self.reconnect_camera(camera_index)
    )
    
    # Stats button
    self.stats_btn = tk.Button(
        controls_frame,
        text="Stats",
        command=lambda: self.show_camera_stats(camera_index)
    )

def toggle_camera_pause(self, camera_index):
    """Toggle camera pause state."""
    if self.active_cameras[camera_index].get('paused', False):
        self.camera_manager.resume_camera(camera_index)
        self.pause_btn.config(text="Pause")
        self.active_cameras[camera_index]['paused'] = False
    else:
        self.camera_manager.pause_camera(camera_index)
        self.pause_btn.config(text="Resume")
        self.active_cameras[camera_index]['paused'] = True
```

### 8. Cleanup and Shutdown

**Proper cleanup on application exit:**
```python
def on_closing(self):
    """Handle application shutdown."""
    logger.info("Shutting down YouQuantiPy...")
    
    # Stop all camera workers
    logger.info("Stopping camera workers...")
    self.camera_manager.stop_all()
    
    # Stop other processes (LSL, recording, etc.)
    self.stop_all_processes()
    
    # Destroy GUI
    self.root.destroy()
```

## Migration Checklist

### Remove from GUI:
- [ ] Direct cv2.VideoCapture calls
- [ ] GPU context creation in GUI process
- [ ] Direct GPU memory operations
- [ ] Frame distributor initialization
- [ ] GPU face processor calls

### Add to GUI:
- [ ] CameraWorkerManager initialization
- [ ] Shared memory readers
- [ ] Metadata queue processing
- [ ] Status update handling
- [ ] Camera health monitoring
- [ ] Error recovery UI

### Update in GUI:
- [ ] Frame display to use shared memory
- [ ] Coordinate transformations for preview space
- [ ] LSL output to use metadata
- [ ] Performance monitoring
- [ ] Control buttons for cameras

## Testing Strategy

1. **Single Camera Test**
   ```bash
   python test_camera_worker.py --camera 0 --duration 60
   ```

2. **GUI Integration Test**
   - Start with one camera
   - Verify frame display
   - Test pause/resume
   - Disconnect camera and verify recovery

3. **Multi-Camera Test**
   - Start 2-4 cameras
   - Verify round-robin GPU assignment
   - Monitor memory usage
   - Test simultaneous operations

4. **Performance Validation**
   - Measure end-to-end latency
   - Verify FPS targets are met
   - Check GPU memory usage
   - Monitor CPU usage

## Common Issues and Solutions

### Issue: Shared memory connection fails
**Solution:** Ensure camera worker is fully started before connecting
```python
# Wait for ready status before connecting
status = wait_for_status(status_queue, 'ready', timeout=10)
if status:
    connect_shared_memory(status['data']['shared_memory'])
```

### Issue: Frame lag or stuttering
**Solution:** Check queue sizes and processing times
```python
# Monitor queue depths
if metadata_queue.qsize() > 10:
    logger.warning("Metadata queue backing up")
    # Skip some frames or reduce processing
```

### Issue: Memory growth over time
**Solution:** Ensure proper cleanup of frames
```python
# Always copy data from shared memory
frame = preview_array.copy()  # Don't hold references
# Process frame...
del frame  # Explicit cleanup
```

## Final Integration Example

```python
# Complete GUI update method
def update(self):
    """Main GUI update loop."""
    try:
        # Process all camera data
        self.update_frame_processing()
        
        # Update UI elements
        self.update_status_display()
        
        # Schedule next update
        self.root.after(33, self.update)  # ~30 FPS
        
    except Exception as e:
        logger.error(f"GUI update error: {e}")
        # Continue running despite errors
        self.root.after(100, self.update)
```

This guide provides the key patterns and code snippets needed for Phase 3 GUI integration. Follow these examples to successfully migrate the GUI to use the new GPU pipeline architecture.