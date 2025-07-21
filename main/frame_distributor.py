"""
Frame Distributor Module for YouQuantiPy
Handles camera capture and frame distribution to multiple consumers.
"""

import cv2
import time
import threading
import numpy as np
from typing import Dict, List, Tuple, Optional
from multiprocessing import Queue as MPQueue
import queue

class FrameDistributor:
    """
    Captures frames from a camera and distributes them to multiple consumer queues.
    Handles different resolution requirements and format conversions.
    """
    
    def __init__(self, camera_index: int, resolution: Tuple[int, int], fps: int):
        """
        Initialize frame distributor.
        
        Args:
            camera_index: Camera device index
            resolution: Target resolution (width, height)
            fps: Target frame rate
        """
        self.camera_index = camera_index
        self.resolution = resolution
        self.fps = fps
        self.cap = None
        self.subscribers = []
        self.is_running = False
        self.thread = None
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.actual_fps = 0
        
        # Error tracking
        self.consecutive_failures = 0
        self.max_failures = 30
        
        # Performance monitoring
        self.stats = {
            'frames_captured': 0,
            'frames_distributed': 0,
            'queue_drops': 0,
            'capture_failures': 0
        }
    
    def initialize_camera(self):
        """Initialize camera with robust settings"""
        if self.cap is not None:
            self.cap.release()
            
        self.cap = self._robust_camera_init(self.camera_index, self.fps, self.resolution)
        if self.cap is None:
            raise RuntimeError(f"Failed to initialize camera {self.camera_index}")
    
    def _robust_camera_init(self, camera_index: int, fps: int, resolution: Tuple[int, int]):
        """Initialize camera with platform-specific optimizations"""
        # Try different backends for compatibility
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        cap = None
        
        for backend in backends:
            test_cap = cv2.VideoCapture(camera_index, backend)
            if test_cap.isOpened():
                # Test if we can grab a frame
                ret, _ = test_cap.read()
                if ret:
                    cap = test_cap
                    print(f"[FrameDistributor] Using backend: {backend}")
                    break
                else:
                    test_cap.release()
        
        if cap is None:
            return None
        
        # Configure camera
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        cap.set(cv2.CAP_PROP_FPS, fps)
        
        # Verify settings
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"[FrameDistributor] Camera {camera_index} initialized:")
        print(f"  Resolution: {actual_width}x{actual_height} (requested: {resolution})")
        print(f"  FPS: {actual_fps} (requested: {fps})")
        
        # Warm up camera
        for _ in range(10):
            cap.read()
            
        return cap
    
    def add_subscriber(self, subscriber_info: Dict):
        """
        Add a subscriber queue.
        
        Args:
            subscriber_info: Dictionary containing:
                - 'name': Subscriber identifier
                - 'queue': MPQueue object
                - 'full_res': Whether to send full resolution
                - 'include_bgr': Whether to include BGR format
                - 'downsample_to': Optional (width, height) for downsampling
        """
        self.subscribers.append(subscriber_info)
        print(f"[FrameDistributor] Added subscriber: {subscriber_info['name']}")
    
    def start(self):
        """Start the frame distribution thread"""
        if self.is_running:
            return
        
        # Initialize camera if not already done
        if self.cap is None:
            self.initialize_camera()
        
        self.is_running = True
        self.thread = threading.Thread(
            target=self._distribution_loop,
            name=f"FrameDistributor-Cam{self.camera_index}",
            daemon=True
        )
        self.thread.start()
        print(f"[FrameDistributor] Started for camera {self.camera_index}")
    
    def stop(self):
        """Stop the frame distribution thread"""
        print(f"[FrameDistributor] Stopping camera {self.camera_index}...")
        self.is_running = False
        
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        # Clear all queues
        for sub in self.subscribers:
            try:
                while not sub['queue'].empty():
                    sub['queue'].get_nowait()
            except:
                pass
        
        print(f"[FrameDistributor] Stopped camera {self.camera_index}")
    
    def _distribution_loop(self):
        """Main loop that captures and distributes frames"""
        frame_interval = 1.0 / self.fps
        last_frame_time = time.time()
        
        # Detection frame size
        detection_size = (640, 640)
        
        while self.is_running:
            try:
                # Frame rate limiting
                current_time = time.time()
                elapsed = current_time - last_frame_time
                if elapsed < frame_interval:
                    time.sleep(frame_interval - elapsed)
                    continue
                
                # Capture frame
                ret, frame_bgr = self.cap.read()
                if not ret:
                    self.consecutive_failures += 1
                    self.stats['capture_failures'] += 1
                    
                    if self.consecutive_failures > self.max_failures:
                        print(f"[FrameDistributor] Too many failures, stopping camera {self.camera_index}")
                        self.is_running = False
                        break
                    
                    time.sleep(0.1)
                    continue
                
                # Reset failure counter on successful capture
                self.consecutive_failures = 0
                
                # Update stats
                self.frame_count += 1
                self.stats['frames_captured'] += 1
                timestamp = current_time
                
                # Calculate actual FPS
                if self.frame_count % 30 == 0:
                    fps_elapsed = current_time - self.last_fps_time
                    if fps_elapsed > 0:
                        self.actual_fps = 30 / fps_elapsed
                        self.last_fps_time = current_time
                
                # Convert to RGB once
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                
                # Create detection frame (640x640 with padding)
                h, w = frame_bgr.shape[:2]
                scale = min(detection_size[0] / w, detection_size[1] / h)
                new_w = int(w * scale)
                new_h = int(h * scale)
                
                # Resize for detection
                resized = cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                
                # Create padded detection frame
                rgb_detection = np.zeros((detection_size[1], detection_size[0], 3), dtype=np.uint8)
                y_offset = (detection_size[1] - new_h) // 2
                x_offset = (detection_size[0] - new_w) // 2
                rgb_detection[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
                
                # Calculate detection transform parameters
                detection_scale_x = w / new_w
                detection_scale_y = h / new_h
                detection_offset_x = x_offset
                detection_offset_y = y_offset
                
                # Distribute to subscribers
                for sub in self.subscribers:
                    if not sub['queue'].full():
                        # Build frame data based on subscriber needs
                        frame_data = {
                            'frame_id': self.frame_count,
                            'timestamp': timestamp,
                            'camera_index': self.camera_index,
                            'resolution': (w, h)
                        }
                        
                        if sub.get('full_res', True):
                            # Full resolution RGB
                            frame_data['rgb'] = frame_rgb
                            
                            # Add detection frame info
                            frame_data['rgb_detection'] = rgb_detection
                            frame_data['detection_scale_x'] = detection_scale_x
                            frame_data['detection_scale_y'] = detection_scale_y
                            frame_data['detection_offset_x'] = detection_offset_x
                            frame_data['detection_offset_y'] = detection_offset_y
                        else:
                            # Downsampled version
                            downsample_to = sub.get('downsample_to', (640, 480))
                            if downsample_to != (w, h):
                                frame_data['rgb'] = cv2.resize(
                                    frame_rgb, downsample_to, 
                                    interpolation=cv2.INTER_LINEAR
                                )
                            else:
                                frame_data['rgb'] = frame_rgb
                        
                        # Include BGR if requested
                        if sub.get('include_bgr', False):
                            frame_data['bgr'] = frame_bgr
                        
                        # Send to queue
                        try:
                            sub['queue'].put_nowait(frame_data)
                            self.stats['frames_distributed'] += 1
                        except queue.Full:
                            self.stats['queue_drops'] += 1
                    else:
                        self.stats['queue_drops'] += 1
                
                last_frame_time = current_time
                
            except Exception as e:
                print(f"[FrameDistributor] Error in distribution loop: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.1)
        
        print(f"[FrameDistributor] Distribution loop ended for camera {self.camera_index}")
    
    def get_stats(self) -> Dict:
        """Get performance statistics"""
        return {
            'camera_index': self.camera_index,
            'is_running': self.is_running,
            'actual_fps': self.actual_fps,
            'frame_count': self.frame_count,
            'stats': self.stats.copy(),
            'subscribers': len(self.subscribers),
            'consecutive_failures': self.consecutive_failures
        }
    
    def is_healthy(self) -> bool:
        """Check if distributor is operating normally"""
        return (self.is_running and 
                self.consecutive_failures < self.max_failures and
                self.cap is not None)