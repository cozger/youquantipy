"""
Camera Worker Process - GPU Pipeline Phase 2

Multiprocessing Process that owns a GPU pipeline for a single camera.
Captures frames, processes through GPU pipeline, and communicates via IPC.

Key Features:
- Single CUDA context ownership per process
- Pinned memory frame capture
- Shared memory output (preview + landmarks)
- Non-blocking queue communication
- Automatic camera reconnection
- Comprehensive error handling
"""

import multiprocessing as mp
from multiprocessing import Process, Queue, Value, shared_memory
import cv2
import numpy as np
import time
import logging
import atexit
from typing import Dict, Any, Optional, Tuple, List
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gpu_pipeline import GPUPipeline
from confighandler import ConfigHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(f'CameraWorker')


class CameraWorker(Process):
    """
    Process that owns GPU pipeline for a single camera.
    
    Architecture:
    - Captures frames using OpenCV with optimal settings
    - Processes through gpu_pipeline.py (single CUDA context)
    - Outputs to shared memory (preview/landmarks) with ready flag protocol
    - Sends metadata via multiprocessing queues
    - Handles camera disconnection/reconnection
    - Provides heartbeat for monitoring
    """
    
    def __init__(self,
                 camera_index: int,
                 gpu_device_id: int,
                 config: Dict[str, Any],
                 control_queue: Queue,
                 status_queue: Queue,
                 metadata_queue: Queue,
                 shared_memory_names: Optional[Dict[str, str]] = None):
        """
        Initialize camera worker process.
        
        Args:
            camera_index: Camera device index (0, 1, etc.)
            gpu_device_id: GPU to use (supports multi-GPU round-robin)
            config: Configuration dict from ConfigHandler
            control_queue: Receives commands (start, stop, pause, resume, ping)
            status_queue: Sends status updates (heartbeat, errors, ready)
            metadata_queue: Sends frame metadata (id, timestamp, face count)
            shared_memory_names: Optional pre-created shared memory names
        """
        super().__init__()
        self.camera_index = camera_index
        self.gpu_device_id = gpu_device_id
        self.config = config
        self.control_queue = control_queue
        self.status_queue = status_queue
        self.metadata_queue = metadata_queue
        self.shared_memory_names = shared_memory_names
        
        # Camera state
        self.cap = None
        self.actual_resolution = None
        self.target_resolution = None
        self.target_fps = None
        
        # GPU pipeline (created in run() to ensure correct process context)
        self.gpu_pipeline = None
        
        # Control flags (shared between threads if needed)
        self.running = Value('b', True)
        self.paused = Value('b', False)
        
        # Frame tracking
        self.frame_id = 0
        self.frames_captured = 0
        self.frames_dropped = 0
        self.frames_processed = 0
        
        # Performance tracking
        self.last_fps_time = time.time()
        self.last_fps_frames = 0
        
        # Camera reconnection state
        self.reconnect_attempts = 0
        self.last_reconnect_time = 0
        self.reconnect_backoff = 1.0  # Start with 1 second
        
        # Set process name
        self.name = f'CameraWorker-{camera_index}'
        
        logger.info(f"Camera worker {camera_index} initialized for GPU {gpu_device_id}")
    
    def run(self):
        """Main process entry point - runs in separate process context."""
        try:
            # Update logger for this process
            global logger
            logger = logging.getLogger(f'CameraWorker-{self.camera_index}')
            
            logger.info(f"Camera worker {self.camera_index} starting on GPU {self.gpu_device_id}")
            
            # Initialize camera
            if not self._init_camera():
                self._send_status('error', {
                    'message': 'Camera initialization failed',
                    'camera_index': self.camera_index
                })
                return
            
            # Initialize GPU pipeline (creates CUDA context in this process)
            try:
                gpu_config = self._build_gpu_config()
                # Add actual camera resolution to config
                gpu_config['capture_resolution'] = self.actual_resolution
                self.gpu_pipeline = GPUPipeline(
                    gpu_device_id=self.gpu_device_id,
                    config=gpu_config
                )
                logger.info(f"GPU pipeline initialized for camera {self.camera_index} with resolution {self.actual_resolution}")
                
                # Get shared memory names
                if not self.shared_memory_names:
                    self.shared_memory_names = self.gpu_pipeline.get_shared_memory_names()
                
            except Exception as e:
                logger.error(f"Failed to initialize GPU pipeline: {e}")
                self._send_status('error', {
                    'message': f'GPU pipeline initialization failed: {str(e)}',
                    'camera_index': self.camera_index
                })
                return
            
            # Register cleanup
            atexit.register(self._cleanup)
            
            # Send ready status with shared memory info
            self._send_status('ready', {
                'camera_index': self.camera_index,
                'gpu_device_id': self.gpu_device_id,
                'resolution': self.actual_resolution,
                'shared_memory': self.shared_memory_names,
                'pid': os.getpid()
            })
            
            # Main capture loop
            self._capture_loop()
            
        except Exception as e:
            logger.error(f"Camera {self.camera_index} fatal error: {e}")
            import traceback
            traceback.print_exc()
            self._send_status('error', {
                'message': str(e),
                'camera_index': self.camera_index,
                'traceback': traceback.format_exc()
            })
        finally:
            self._cleanup()
    
    def _init_camera(self) -> bool:
        """
        Initialize camera with optimal settings.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Extract settings from config
            camera_settings = self.config.get('camera_settings', {})
            resolution_name = camera_settings.get('resolution', '1080p')
            resolution_map = self.config.get('camera_resolutions', {
                '480p': [640, 480],
                '720p': [1280, 720],
                '1080p': [1920, 1080],
                '4K': [3840, 2160]
            })
            self.target_resolution = resolution_map.get(resolution_name, [1280, 720])
            self.target_fps = camera_settings.get('target_fps', 30)
            
            logger.info(f"Initializing camera {self.camera_index} with target {self.target_resolution} @ {self.target_fps}fps")
            
            # Try multiple backends for best compatibility
            if sys.platform == 'win32':
                backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
            else:
                backends = [cv2.CAP_V4L2, cv2.CAP_ANY]
            
            for backend in backends:
                self.cap = cv2.VideoCapture(self.camera_index, backend)
                if self.cap.isOpened():
                    logger.info(f"Camera {self.camera_index} opened with backend {backend}")
                    break
            
            if not self.cap or not self.cap.isOpened():
                logger.error(f"Failed to open camera {self.camera_index}")
                return False
            
            # Configure camera for optimal performance
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_resolution[1])
            self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency
            
            # Some cameras need FOURCC set for proper format
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            self.cap.set(cv2.CAP_PROP_FOURCC, fourcc)
            
            # Read test frame to get actual dimensions
            ret, test_frame = self.cap.read()
            if ret and test_frame is not None:
                h, w = test_frame.shape[:2]
                self.actual_resolution = (w, h)
                logger.info(f"Camera {self.camera_index} actual resolution: {w}x{h}")
                
                # Verify frame is valid
                if w == 0 or h == 0:
                    logger.error(f"Invalid frame dimensions: {w}x{h}")
                    return False
            else:
                # Fallback to camera properties
                w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if w > 0 and h > 0:
                    self.actual_resolution = (w, h)
                    logger.warning(f"Using camera properties for resolution: {w}x{h}")
                else:
                    logger.error("Could not determine camera resolution")
                    return False
            
            # Log actual camera settings
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            logger.info(f"Camera {self.camera_index} initialized: {self.actual_resolution} @ {actual_fps}fps")
            
            return True
            
        except Exception as e:
            logger.error(f"Camera initialization error: {e}")
            return False
    
    def _capture_loop(self):
        """Main frame capture and processing loop."""
        last_heartbeat = time.time()
        last_stats_report = time.time()
        frame_times = []  # For FPS calculation
        
        logger.info(f"Starting capture loop for camera {self.camera_index}")
        
        while self.running.value:
            try:
                # Process control messages (non-blocking)
                self._process_control_messages()
                
                # Send periodic heartbeat
                current_time = time.time()
                if current_time - last_heartbeat > 1.0:
                    self._send_heartbeat()
                    last_heartbeat = current_time
                
                # Report detailed stats every 5 seconds
                if current_time - last_stats_report > 5.0:
                    self._report_stats(frame_times)
                    last_stats_report = current_time
                    frame_times = []  # Reset for next interval
                
                # Skip processing if paused
                if self.paused.value:
                    time.sleep(0.01)
                    continue
                
                # Handle camera reconnection backoff
                if self.reconnect_attempts > 0:
                    if current_time - self.last_reconnect_time < self.reconnect_backoff:
                        time.sleep(0.1)
                        continue
                
                # Capture frame
                frame_start = time.perf_counter()
                ret, frame = self.cap.read()
                
                if not ret or frame is None:
                    # Handle camera disconnection
                    self._handle_capture_failure()
                    continue
                
                # Reset reconnection state on successful capture
                if self.reconnect_attempts > 0:
                    logger.info(f"Camera {self.camera_index} reconnected successfully")
                    self.reconnect_attempts = 0
                    self.reconnect_backoff = 1.0
                
                self.frames_captured += 1
                
                # Process frame through GPU pipeline
                processing_start = time.perf_counter()
                results = self.gpu_pipeline.process_frame(frame, self.frame_id)
                processing_time = (time.perf_counter() - processing_start) * 1000
                
                # Track frame timing
                total_frame_time = (time.perf_counter() - frame_start) * 1000
                frame_times.append(total_frame_time)
                
                # Prepare metadata
                metadata = {
                    'frame_id': self.frame_id,
                    'timestamp': time.time(),
                    'camera_index': self.camera_index,
                    'processing_time_ms': processing_time,
                    'total_time_ms': total_frame_time,
                    'n_faces': len(results.get('faces', [])),
                    'preview_ready': results.get('preview_ready', False),
                    'capture_resolution': self.actual_resolution
                }
                
                # Send metadata via queue (non-blocking)
                try:
                    self.metadata_queue.put(metadata, timeout=0.001)
                    self.frames_processed += 1
                except:
                    self.frames_dropped += 1
                    if self.frames_dropped % 100 == 0:
                        logger.warning(f"Camera {self.camera_index} dropped {self.frames_dropped} frames")
                
                self.frame_id += 1
                
                # Rate limiting if processing is too fast
                if total_frame_time < (1000.0 / self.target_fps):
                    sleep_time = (1000.0 / self.target_fps - total_frame_time) / 1000.0
                    time.sleep(sleep_time)
                
            except KeyboardInterrupt:
                logger.info(f"Camera {self.camera_index} interrupted by user")
                self.running.value = False
            except Exception as e:
                logger.error(f"Camera {self.camera_index} capture error: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.1)
    
    def _handle_capture_failure(self):
        """Handle camera capture failure with reconnection logic."""
        self.reconnect_attempts += 1
        
        if self.reconnect_attempts == 1:
            logger.warning(f"Camera {self.camera_index} capture failed, attempting reconnect...")
        
        # Try immediate reconnect for first few attempts
        if self.reconnect_attempts <= 3:
            if self._reconnect_camera():
                return
        
        # Use exponential backoff for subsequent attempts
        self.last_reconnect_time = time.time()
        self.reconnect_backoff = min(self.reconnect_backoff * 2, 30.0)  # Max 30 seconds
        
        if self.reconnect_attempts % 10 == 0:
            logger.error(f"Camera {self.camera_index} still disconnected after {self.reconnect_attempts} attempts")
            self._send_status('warning', {
                'message': f'Camera disconnected, retrying every {self.reconnect_backoff:.1f}s',
                'camera_index': self.camera_index,
                'attempts': self.reconnect_attempts
            })
    
    def _reconnect_camera(self) -> bool:
        """
        Attempt to reconnect to camera.
        
        Returns:
            bool: True if successful, False otherwise
        """
        logger.info(f"Attempting to reconnect camera {self.camera_index}")
        
        # Release old camera
        if self.cap:
            try:
                self.cap.release()
            except:
                pass
            self.cap = None
        
        # Small delay before reconnection
        time.sleep(0.5)
        
        # Try to reinitialize
        return self._init_camera()
    
    def _process_control_messages(self):
        """Process control messages without blocking."""
        try:
            # Process all available messages
            while not self.control_queue.empty():
                try:
                    msg = self.control_queue.get_nowait()
                    
                    if isinstance(msg, dict):
                        command = msg.get('command', '')
                        
                        if command == 'stop':
                            logger.info(f"Camera {self.camera_index} received stop command")
                            self.running.value = False
                        elif command == 'pause':
                            logger.info(f"Camera {self.camera_index} paused")
                            self.paused.value = True
                        elif command == 'resume':
                            logger.info(f"Camera {self.camera_index} resumed")
                            self.paused.value = False
                        elif command == 'ping':
                            self._send_status('pong', {
                                'timestamp': time.time(),
                                'camera_index': self.camera_index
                            })
                        elif command == 'get_stats':
                            self._send_detailed_stats()
                        elif command == 'reset_stats':
                            self._reset_stats()
                        else:
                            logger.warning(f"Unknown command: {command}")
                    
                except:
                    pass
        except:
            pass
    
    def _build_gpu_config(self) -> Dict[str, Any]:
        """Build GPU pipeline config from main config."""
        # Get GPU-specific settings
        advanced_detection = self.config.get('advanced_detection', {})
        gpu_settings = advanced_detection.get('gpu_settings', {})
        
        # Build configuration for GPU pipeline
        gpu_config = {
            'max_batch_size': gpu_settings.get('max_batch_size', 8),
            'buffer_size': 32,  # Ring buffer size (power of 2)
            'enable_fp16': gpu_settings.get('enable_fp16', False),
            'model_paths': {
                'retinaface': advanced_detection.get('retinaface_trt_path', 'D:/Projects/youquantipy/retinaface.trt'),
                'landmarks': advanced_detection.get('landmark_trt_path', 'D:/Projects/youquantipy/landmark.trt'),
                'blendshape': advanced_detection.get('blendshape_trt_path', 'D:/Projects/youquantipy/blendshape.trt')
            },
            'memory_pool_size': gpu_settings.get('memory_pool_size', 2 * 1024 * 1024 * 1024),  # 2GB default
            'detection_confidence': advanced_detection.get('detection_confidence', 0.5),
            'roi_size': tuple(advanced_detection.get('roi_settings', {}).get('target_size', [256, 256])),
            'preview_size': (960, 540),
            'detection_size': gpu_settings.get('detection_size', (640, 640))  # Will be overridden by engine
        }
        
        # Validate model paths
        for model_name, model_path in gpu_config['model_paths'].items():
            if model_path and not os.path.exists(model_path):
                logger.warning(f"Model file not found: {model_name} at {model_path}")
                # Remove missing models
                gpu_config['model_paths'][model_name] = None
        
        return gpu_config
    
    def _send_status(self, status_type: str, data: Any):
        """Send status update via status queue."""
        try:
            status_msg = {
                'type': status_type,
                'camera_index': self.camera_index,
                'timestamp': time.time(),
                'data': data
            }
            self.status_queue.put_nowait(status_msg)
        except:
            # Queue full, skip
            pass
    
    def _send_heartbeat(self):
        """Send heartbeat with basic statistics."""
        # Calculate FPS
        current_time = time.time()
        time_delta = current_time - self.last_fps_time
        if time_delta > 0:
            frames_delta = self.frames_processed - self.last_fps_frames
            current_fps = frames_delta / time_delta
            self.last_fps_time = current_time
            self.last_fps_frames = self.frames_processed
        else:
            current_fps = 0
        
        # Get GPU memory usage if available
        gpu_memory_used = 0
        if self.gpu_pipeline and hasattr(self.gpu_pipeline, 'mempool'):
            try:
                gpu_memory_used = self.gpu_pipeline.mempool.used_bytes()
            except:
                pass
        
        heartbeat_data = {
            'camera_index': self.camera_index,
            'frames_captured': self.frames_captured,
            'frames_processed': self.frames_processed,
            'frames_dropped': self.frames_dropped,
            'frame_id': self.frame_id,
            'current_fps': current_fps,
            'gpu_memory_used': gpu_memory_used,
            'is_paused': self.paused.value,
            'is_connected': self.reconnect_attempts == 0
        }
        
        self._send_status('heartbeat', heartbeat_data)
    
    def _send_detailed_stats(self):
        """Send detailed statistics."""
        stats = {
            'camera_index': self.camera_index,
            'frames_captured': self.frames_captured,
            'frames_processed': self.frames_processed,
            'frames_dropped': self.frames_dropped,
            'drop_rate': self.frames_dropped / max(self.frames_captured, 1),
            'frame_id': self.frame_id,
            'reconnect_attempts': self.reconnect_attempts,
            'actual_resolution': self.actual_resolution,
            'target_resolution': self.target_resolution,
            'is_paused': self.paused.value,
            'uptime': time.time() - self.last_fps_time
        }
        
        # Add GPU pipeline stats if available
        if self.gpu_pipeline and hasattr(self.gpu_pipeline, 'diagnostics'):
            stats['gpu_stats'] = self.gpu_pipeline.diagnostics.copy()
        
        self._send_status('stats', stats)
    
    def _report_stats(self, frame_times: List[float]):
        """Report performance statistics."""
        if not frame_times:
            return
        
        # Calculate statistics
        avg_time = sum(frame_times) / len(frame_times)
        min_time = min(frame_times)
        max_time = max(frame_times)
        
        # Calculate actual FPS based on frame times
        if len(frame_times) > 1:
            actual_fps = 1000.0 / avg_time
        else:
            actual_fps = 0
        
        logger.info(f"Camera {self.camera_index} stats: "
                   f"{len(frame_times)} frames, "
                   f"{actual_fps:.1f} FPS, "
                   f"frame time: {avg_time:.1f}ms avg "
                   f"({min_time:.1f}-{max_time:.1f}ms range), "
                   f"dropped: {self.frames_dropped}")
    
    def _reset_stats(self):
        """Reset statistics counters."""
        self.frames_captured = 0
        self.frames_processed = 0
        self.frames_dropped = 0
        self.last_fps_time = time.time()
        self.last_fps_frames = 0
        logger.info(f"Camera {self.camera_index} stats reset")
    
    def _cleanup(self):
        """Clean up resources."""
        logger.info(f"Camera {self.camera_index} cleaning up...")
        
        # Release camera
        if self.cap:
            try:
                self.cap.release()
                logger.info(f"Camera {self.camera_index} released")
            except Exception as e:
                logger.error(f"Error releasing camera: {e}")
        
        # GPU pipeline cleanup (handled by its own atexit)
        if self.gpu_pipeline:
            try:
                self.gpu_pipeline.cleanup()
            except:
                pass
        
        # Send final status
        self._send_status('stopped', {
            'camera_index': self.camera_index,
            'frames_processed': self.frames_processed,
            'frames_dropped': self.frames_dropped
        })
        
        logger.info(f"Camera {self.camera_index} cleanup complete")


# Helper functions for testing
def test_single_camera():
    """Test function for single camera worker."""
    import multiprocessing as mp
    
    # Load config
    config = ConfigHandler().config
    
    # Create queues
    control_queue = mp.Queue()
    status_queue = mp.Queue()
    metadata_queue = mp.Queue()
    
    # Create and start worker
    worker = CameraWorker(
        camera_index=0,
        gpu_device_id=0,
        config=config,
        control_queue=control_queue,
        status_queue=status_queue,
        metadata_queue=metadata_queue
    )
    
    worker.start()
    
    # Monitor status
    try:
        while True:
            # Check status
            try:
                status = status_queue.get_nowait()
                print(f"Status: {status['type']} - {status.get('data', {})}")
                
                if status['type'] == 'ready':
                    print(f"Camera ready with shared memory: {status['data']['shared_memory']}")
                elif status['type'] == 'error':
                    print(f"Error: {status['data']}")
                    break
            except:
                pass
            
            # Check metadata
            try:
                metadata = metadata_queue.get_nowait()
                print(f"Frame {metadata['frame_id']}: "
                      f"{metadata['n_faces']} faces, "
                      f"{metadata['processing_time_ms']:.1f}ms")
            except:
                pass
            
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("Stopping camera worker...")
        control_queue.put({'command': 'stop'})
        worker.join(timeout=5)
    
    print("Test complete")


if __name__ == '__main__':
    # Run test
    test_single_camera()