#!/usr/bin/env python3
"""
Test script for camera worker implementation.
Tests single camera capture and GPU pipeline integration.
"""

import multiprocessing as mp
import time
import numpy as np
from multiprocessing import shared_memory
import logging
import sys

from camera_worker import CameraWorker
from confighandler import ConfigHandler

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger('CameraWorkerTest')


def test_camera_worker(camera_index=0, duration=30):
    """Test camera worker for specified duration."""
    logger.info(f"Starting camera worker test for camera {camera_index}")
    
    # Load configuration
    config = ConfigHandler().config
    
    # Create communication queues
    control_queue = mp.Queue()
    status_queue = mp.Queue()
    metadata_queue = mp.Queue()
    
    # Create camera worker
    worker = CameraWorker(
        camera_index=camera_index,
        gpu_device_id=0,
        config=config,
        control_queue=control_queue,
        status_queue=status_queue,
        metadata_queue=metadata_queue
    )
    
    # Start worker
    worker.start()
    logger.info(f"Camera worker process started (PID: {worker.pid})")
    
    # Variables for tracking
    shared_memory_names = None
    preview_shm = None
    landmark_shm = None
    start_time = time.time()
    frame_count = 0
    last_report_time = start_time
    
    # Metadata structure
    metadata_dtype = np.dtype([
        ('frame_id', 'int32'),
        ('timestamp_ms', 'int64'),
        ('n_faces', 'int32'),
        ('ready', 'int8'),
    ])
    
    try:
        while time.time() - start_time < duration:
            # Process status messages
            while not status_queue.empty():
                try:
                    status = status_queue.get_nowait()
                    
                    if status['type'] == 'ready':
                        logger.info(f"Camera ready! Resolution: {status['data']['resolution']}")
                        shared_memory_names = status['data']['shared_memory']
                        
                        # Connect to shared memory
                        try:
                            # Preview shared memory
                            preview_shm = shared_memory.SharedMemory(
                                name=shared_memory_names['preview']
                            )
                            logger.info(f"Connected to preview shared memory: {preview_shm.name}")
                            
                            # Landmark shared memory
                            landmark_shm = shared_memory.SharedMemory(
                                name=shared_memory_names['landmarks']
                            )
                            logger.info(f"Connected to landmark shared memory: {landmark_shm.name}")
                            
                        except Exception as e:
                            logger.error(f"Failed to connect to shared memory: {e}")
                    
                    elif status['type'] == 'heartbeat':
                        data = status['data']
                        logger.debug(f"Heartbeat: {data['current_fps']:.1f} FPS, "
                                   f"{data['frames_processed']} processed, "
                                   f"{data['frames_dropped']} dropped")
                    
                    elif status['type'] == 'error':
                        logger.error(f"Camera error: {status['data']}")
                        return False
                    
                    elif status['type'] == 'warning':
                        logger.warning(f"Camera warning: {status['data']}")
                    
                except:
                    pass
            
            # Process metadata
            while not metadata_queue.empty():
                try:
                    metadata = metadata_queue.get_nowait()
                    logger.debug(f"Frame {metadata['frame_id']}: "
                               f"{metadata['n_faces']} faces, "
                               f"{metadata['processing_time_ms']:.1f}ms")
                    frame_count += 1
                except:
                    pass
            
            # Try to read from shared memory if connected
            if preview_shm:
                try:
                    # Create views
                    preview_array = np.ndarray(
                        (540, 960, 3), dtype=np.uint8,
                        buffer=preview_shm.buf[:540*960*3]
                    )
                    preview_metadata = np.ndarray(
                        1, dtype=metadata_dtype,
                        buffer=preview_shm.buf[540*960*3:]
                    )[0]
                    
                    # Check if frame is ready
                    if preview_metadata['ready'] == 1:
                        # Would display frame here
                        logger.debug(f"Preview frame ready: {preview_metadata['frame_id']}")
                        
                        # Acknowledge read
                        preview_metadata['ready'] = 0
                        
                except Exception as e:
                    logger.error(f"Error reading shared memory: {e}")
            
            # Periodic reporting
            current_time = time.time()
            elapsed = current_time - start_time
            
            if current_time - last_report_time > 5.0:
                fps = frame_count / elapsed
                logger.info(f"Test progress: {elapsed:.1f}s, "
                          f"{frame_count} frames, "
                          f"{fps:.1f} FPS average")
                last_report_time = current_time
                
                # Request stats
                control_queue.put({'command': 'get_stats'})
            
            # Small sleep to prevent CPU spinning
            time.sleep(0.01)
            
            # Test pause/resume at halfway point
            if elapsed > duration/2 and elapsed < duration/2 + 1:
                logger.info("Testing pause/resume...")
                control_queue.put({'command': 'pause'})
                time.sleep(2)
                control_queue.put({'command': 'resume'})
        
        # Test completed successfully
        logger.info(f"Test completed: {frame_count} frames in {duration}s")
        return True
        
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        return False
        
    except Exception as e:
        logger.error(f"Test error: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup
        logger.info("Cleaning up...")
        
        # Send stop command
        control_queue.put({'command': 'stop'})
        
        # Wait for worker to finish
        worker.join(timeout=5)
        
        if worker.is_alive():
            logger.warning("Worker did not stop gracefully, terminating...")
            worker.terminate()
            worker.join()
        
        # Close shared memory
        if preview_shm:
            preview_shm.close()
        if landmark_shm:
            landmark_shm.close()
        
        logger.info("Cleanup complete")


def main():
    """Main test entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test camera worker')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera index to test (default: 0)')
    parser.add_argument('--duration', type=int, default=30,
                       help='Test duration in seconds (default: 30)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run test
    success = test_camera_worker(args.camera, args.duration)
    
    if success:
        logger.info("✅ Camera worker test PASSED")
        sys.exit(0)
    else:
        logger.error("❌ Camera worker test FAILED")
        sys.exit(1)


if __name__ == '__main__':
    main()