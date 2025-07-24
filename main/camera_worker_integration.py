"""
Camera Worker Integration Example

Shows how to integrate the new camera_worker.py with the existing system.
This example demonstrates:
- Starting multiple camera workers
- Connecting to shared memory
- Processing metadata and results
- Proper cleanup and error handling
"""

import multiprocessing as mp
from multiprocessing import Queue, shared_memory
import time
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

from camera_worker import CameraWorker
from confighandler import ConfigHandler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('CameraIntegration')


class CameraWorkerManager:
    """Manages multiple camera workers for the application."""
    
    def __init__(self, config: Dict = None):
        """Initialize camera worker manager."""
        self.config = config or ConfigHandler().config
        self.workers = {}  # camera_index -> worker process
        self.control_queues = {}  # camera_index -> control queue
        self.status_queue = mp.Queue()  # Shared status queue for all cameras
        self.metadata_queue = mp.Queue()  # Shared metadata queue
        self.shared_memories = {}  # camera_index -> {'preview': shm, 'landmarks': shm}
        
        # Get number of GPUs available
        self.num_gpus = self._get_num_gpus()
        logger.info(f"Camera manager initialized with {self.num_gpus} GPU(s)")
    
    def _get_num_gpus(self) -> int:
        """Get number of available GPUs."""
        try:
            import cupy as cp
            return cp.cuda.runtime.getDeviceCount()
        except:
            return 1  # Default to 1 if can't detect
    
    def start_camera(self, camera_index: int) -> bool:
        """
        Start a camera worker process.
        
        Args:
            camera_index: Camera device index
            
        Returns:
            bool: True if started successfully
        """
        if camera_index in self.workers:
            logger.warning(f"Camera {camera_index} already started")
            return False
        
        try:
            # Create control queue for this camera
            control_queue = mp.Queue()
            self.control_queues[camera_index] = control_queue
            
            # Determine GPU assignment (round-robin)
            gpu_id = camera_index % self.num_gpus
            
            # Create worker process
            worker = CameraWorker(
                camera_index=camera_index,
                gpu_device_id=gpu_id,
                config=self.config,
                control_queue=control_queue,
                status_queue=self.status_queue,
                metadata_queue=self.metadata_queue
            )
            
            # Start the worker
            worker.start()
            self.workers[camera_index] = worker
            
            # Wait for ready status
            timeout = time.time() + 10  # 10 second timeout
            while time.time() < timeout:
                try:
                    status = self.status_queue.get(timeout=0.1)
                    if status['camera_index'] == camera_index:
                        if status['type'] == 'ready':
                            # Connect to shared memory
                            self._connect_shared_memory(camera_index, status['data']['shared_memory'])
                            logger.info(f"Camera {camera_index} started successfully on GPU {gpu_id}")
                            return True
                        elif status['type'] == 'error':
                            logger.error(f"Camera {camera_index} error: {status['data']}")
                            return False
                except:
                    continue
            
            logger.error(f"Camera {camera_index} startup timeout")
            return False
            
        except Exception as e:
            logger.error(f"Failed to start camera {camera_index}: {e}")
            return False
    
    def _connect_shared_memory(self, camera_index: int, shm_names: Dict[str, str]):
        """Connect to shared memory created by camera worker."""
        try:
            self.shared_memories[camera_index] = {}
            
            # Connect to preview shared memory
            if 'preview' in shm_names:
                preview_shm = shared_memory.SharedMemory(name=shm_names['preview'])
                self.shared_memories[camera_index]['preview'] = {
                    'shm': preview_shm,
                    'array': np.ndarray((540, 960, 3), dtype=np.uint8, 
                                       buffer=preview_shm.buf[:540*960*3]),
                    'metadata': self._create_metadata_view(preview_shm.buf[540*960*3:])
                }
            
            # Connect to landmarks shared memory
            if 'landmarks' in shm_names:
                landmark_shm = shared_memory.SharedMemory(name=shm_names['landmarks'])
                self.shared_memories[camera_index]['landmarks'] = {
                    'shm': landmark_shm,
                    'array': np.ndarray((10, 478, 3), dtype=np.float32,
                                       buffer=landmark_shm.buf[:10*478*3*4]),
                    'metadata': self._create_metadata_view(landmark_shm.buf[10*478*3*4:])
                }
            
            logger.info(f"Connected to shared memory for camera {camera_index}")
            
        except Exception as e:
            logger.error(f"Failed to connect shared memory for camera {camera_index}: {e}")
    
    def _create_metadata_view(self, buffer):
        """Create structured metadata view."""
        metadata_dtype = np.dtype([
            ('frame_id', 'int32'),
            ('timestamp_ms', 'int64'),
            ('n_faces', 'int32'),
            ('ready', 'int8'),
        ])
        return np.ndarray(1, dtype=metadata_dtype, buffer=buffer)[0]
    
    def stop_camera(self, camera_index: int):
        """Stop a camera worker."""
        if camera_index not in self.workers:
            return
        
        # Send stop command
        if camera_index in self.control_queues:
            self.control_queues[camera_index].put({'command': 'stop'})
        
        # Wait for worker to finish
        worker = self.workers[camera_index]
        worker.join(timeout=5)
        
        if worker.is_alive():
            logger.warning(f"Camera {camera_index} did not stop gracefully, terminating...")
            worker.terminate()
            worker.join()
        
        # Cleanup shared memory
        if camera_index in self.shared_memories:
            for shm_type, shm_data in self.shared_memories[camera_index].items():
                try:
                    shm_data['shm'].close()
                except:
                    pass
            del self.shared_memories[camera_index]
        
        # Remove from tracking
        del self.workers[camera_index]
        if camera_index in self.control_queues:
            del self.control_queues[camera_index]
        
        logger.info(f"Camera {camera_index} stopped")
    
    def stop_all(self):
        """Stop all camera workers."""
        camera_indices = list(self.workers.keys())
        for camera_index in camera_indices:
            self.stop_camera(camera_index)
    
    def get_preview_frame(self, camera_index: int) -> Optional[np.ndarray]:
        """
        Get preview frame from shared memory if available.
        
        Returns:
            np.ndarray or None: Preview frame if ready
        """
        if camera_index not in self.shared_memories:
            return None
        
        if 'preview' not in self.shared_memories[camera_index]:
            return None
        
        preview_data = self.shared_memories[camera_index]['preview']
        
        # Check if frame is ready
        if preview_data['metadata']['ready'] == 1:
            # Copy frame data
            frame = preview_data['array'].copy()
            
            # Reset ready flag (acknowledge read)
            preview_data['metadata']['ready'] = 0
            
            return frame
        
        return None
    
    def get_landmarks(self, camera_index: int) -> Optional[Dict]:
        """
        Get landmark data from shared memory if available.
        
        Returns:
            Dict or None: Landmark data with metadata
        """
        if camera_index not in self.shared_memories:
            return None
        
        if 'landmarks' not in self.shared_memories[camera_index]:
            return None
        
        landmark_data = self.shared_memories[camera_index]['landmarks']
        
        # Check if data is ready
        if landmark_data['metadata']['ready'] == 1:
            # Get metadata
            n_faces = landmark_data['metadata']['n_faces']
            frame_id = landmark_data['metadata']['frame_id']
            
            # Copy landmark data
            landmarks = landmark_data['array'][:n_faces].copy()
            
            # Reset ready flag
            landmark_data['metadata']['ready'] = 0
            
            return {
                'frame_id': frame_id,
                'n_faces': n_faces,
                'landmarks': landmarks
            }
        
        return None
    
    def process_metadata(self) -> List[Dict]:
        """Process all available metadata from the queue."""
        metadata_list = []
        
        # Get all available metadata
        while not self.metadata_queue.empty():
            try:
                metadata = self.metadata_queue.get_nowait()
                metadata_list.append(metadata)
            except:
                break
        
        return metadata_list
    
    def process_status_updates(self) -> List[Dict]:
        """Process all available status updates."""
        status_list = []
        
        # Get all available status updates
        while not self.status_queue.empty():
            try:
                status = self.status_queue.get_nowait()
                status_list.append(status)
            except:
                break
        
        return status_list
    
    def send_command(self, camera_index: int, command: str, data: Dict = None):
        """Send command to specific camera."""
        if camera_index not in self.control_queues:
            logger.warning(f"Camera {camera_index} not found")
            return
        
        msg = {'command': command}
        if data:
            msg.update(data)
        
        try:
            self.control_queues[camera_index].put_nowait(msg)
        except:
            logger.warning(f"Control queue full for camera {camera_index}")
    
    def pause_camera(self, camera_index: int):
        """Pause a camera."""
        self.send_command(camera_index, 'pause')
    
    def resume_camera(self, camera_index: int):
        """Resume a camera."""
        self.send_command(camera_index, 'resume')
    
    def get_stats(self, camera_index: int = None):
        """Request statistics from camera(s)."""
        if camera_index is not None:
            self.send_command(camera_index, 'get_stats')
        else:
            # Request from all cameras
            for cam_idx in self.control_queues:
                self.send_command(cam_idx, 'get_stats')


def example_usage():
    """Example of using the camera worker manager."""
    # Create manager
    manager = CameraWorkerManager()
    
    try:
        # Start cameras
        num_cameras = 2
        for i in range(num_cameras):
            if manager.start_camera(i):
                print(f"Camera {i} started successfully")
            else:
                print(f"Failed to start camera {i}")
        
        # Main loop
        frame_count = 0
        start_time = time.time()
        
        while True:
            # Process metadata
            metadata_list = manager.process_metadata()
            for metadata in metadata_list:
                print(f"Camera {metadata['camera_index']} - "
                      f"Frame {metadata['frame_id']}: "
                      f"{metadata['n_faces']} faces, "
                      f"{metadata['processing_time_ms']:.1f}ms")
            
            # Process status updates
            status_list = manager.process_status_updates()
            for status in status_list:
                if status['type'] == 'heartbeat':
                    data = status['data']
                    print(f"Heartbeat from camera {status['camera_index']}: "
                          f"{data['current_fps']:.1f} FPS, "
                          f"{data['frames_processed']} processed")
            
            # Get preview frames
            for cam_idx in range(num_cameras):
                frame = manager.get_preview_frame(cam_idx)
                if frame is not None:
                    frame_count += 1
                    # Here you would display or process the frame
                    print(f"Got preview frame from camera {cam_idx}")
            
            # Get landmarks
            for cam_idx in range(num_cameras):
                landmarks = manager.get_landmarks(cam_idx)
                if landmarks is not None:
                    print(f"Got landmarks from camera {cam_idx}: "
                          f"{landmarks['n_faces']} faces")
            
            # Print overall stats every 5 seconds
            if time.time() - start_time > 5:
                print(f"\nOverall: {frame_count} frames in 5 seconds")
                frame_count = 0
                start_time = time.time()
                
                # Request detailed stats
                manager.get_stats()
            
            time.sleep(0.01)  # Small delay
            
    except KeyboardInterrupt:
        print("\nStopping cameras...")
    finally:
        # Clean shutdown
        manager.stop_all()
        print("All cameras stopped")


if __name__ == '__main__':
    example_usage()