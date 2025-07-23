"""
Test script for GPU Inter-Process Communication (IPC) implementation
"""

import numpy as np
import time
import multiprocessing as mp
from multiprocessing import Process, Queue
import cv2

# Import our GPU memory manager
from gpu_memory_manager import (
    GPUMemoryPool, GPUMemoryClient, GPUMemoryHandle,
    create_gpu_memory_manager, create_gpu_memory_client,
    CUDA_AVAILABLE
)

try:
    import cupy as cp
except ImportError:
    cp = None
    print("WARNING: CuPy not installed, GPU IPC tests will be skipped")


def producer_process(queue: Queue, enable_gpu_ipc: bool = True):
    """Producer process that writes frames to GPU and sends handles."""
    print("[PRODUCER] Starting...")
    
    if not enable_gpu_ipc or not CUDA_AVAILABLE:
        print("[PRODUCER] GPU IPC disabled or not available")
        return
        
    # Create GPU memory pool
    gpu_pool = create_gpu_memory_manager(pool_size=10, device_id=0)
    if not gpu_pool:
        print("[PRODUCER] Failed to create GPU memory pool")
        return
        
    # Create test frames
    frame_count = 10
    frame_shape = (480, 640, 3)  # 640x480 RGB
    
    for i in range(frame_count):
        # Create test frame with pattern
        frame = np.ones(frame_shape, dtype=np.uint8) * (i * 25)
        frame[:100, :100] = 255  # White square in corner
        frame[i*40:(i+1)*40, :, 0] = 255  # Red stripe
        
        # Write to GPU memory
        frame_id = f"test_frame_{i}"
        gpu_handle = gpu_pool.write(frame_id, frame)
        
        if gpu_handle:
            print(f"[PRODUCER] Sent GPU handle for frame {i}")
            queue.put((frame_id, gpu_handle, frame.shape))
        else:
            print(f"[PRODUCER] Failed to create GPU handle for frame {i}")
            
        time.sleep(0.1)  # Simulate frame rate
        
    # Send stop signal
    queue.put(("stop", None, None))
    print("[PRODUCER] Finished")
    
    # Cleanup
    gpu_pool.cleanup()


def consumer_process(queue: Queue, enable_gpu_ipc: bool = True):
    """Consumer process that reads frames from GPU handles."""
    print("[CONSUMER] Starting...")
    
    if not enable_gpu_ipc or not CUDA_AVAILABLE:
        print("[CONSUMER] GPU IPC disabled or not available")
        return
        
    # Create GPU memory client
    gpu_client = create_gpu_memory_client(device_id=0)
    if not gpu_client:
        print("[CONSUMER] Failed to create GPU memory client")
        return
        
    frames_received = 0
    
    while True:
        try:
            frame_id, gpu_handle, shape = queue.get(timeout=5.0)
            
            if frame_id == "stop":
                break
                
            # Get GPU array from handle
            gpu_array = gpu_client.get_array(gpu_handle)
            if gpu_array is not None:
                # Verify shape
                assert gpu_array.shape == shape, f"Shape mismatch: {gpu_array.shape} != {shape}"
                
                # Do some GPU processing (example: mean calculation)
                mean_val = cp.mean(gpu_array).get()
                
                # Extract a region for verification
                corner_region = gpu_array[:100, :100].get()
                corner_mean = np.mean(corner_region)
                
                print(f"[CONSUMER] Received frame {frame_id}: shape={shape}, "
                      f"mean={mean_val:.1f}, corner_mean={corner_mean:.1f}")
                
                frames_received += 1
            else:
                print(f"[CONSUMER] Failed to get GPU array from handle")
                
        except mp.queues.Empty:
            print("[CONSUMER] Timeout waiting for frame")
            break
            
    print(f"[CONSUMER] Finished. Received {frames_received} frames")
    
    # Cleanup
    gpu_client.cleanup()


def test_frame_distributor_integration():
    """Test FrameDistributor with GPU IPC."""
    print("\n=== Testing FrameDistributor with GPU IPC ===")
    
    from frame_distributor import FrameDistributor
    
    # Create frame distributor with GPU sharing
    distributor = FrameDistributor(
        camera_index=0,
        resolution=(640, 480),
        fps=30,
        enable_gpu_sharing=True,
        gpu_device_id=0
    )
    
    # Create test queue
    test_queue = Queue(maxsize=5)
    
    # Add GPU-enabled subscriber
    distributor.add_subscriber({
        'name': 'test_gpu',
        'queue': test_queue,
        'full_res': True,
        'include_bgr': False,
        'gpu_enabled': True
    })
    
    print("FrameDistributor configured with GPU IPC")
    print("Subscribers:", len(distributor.subscribers))
    
    # Note: Actual camera test would require a camera to be connected
    

def test_roi_processor_gpu_handles():
    """Test ROI processor with GPU memory handles."""
    print("\n=== Testing ROI Processor with GPU Handles ===")
    
    from roi_processor import ROIProcessor
    
    # Create ROI processor with GPU IPC
    roi_processor = ROIProcessor(
        target_size=(256, 256),
        padding_ratio=0.3,
        min_quality_score=0.5,
        enable_gpu_ipc=True,
        gpu_device_id=0
    )
    
    if roi_processor.enable_gpu_ipc:
        print("ROI Processor initialized with GPU IPC support")
        
        # Create test frame
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Create GPU memory pool and write frame
        gpu_pool = create_gpu_memory_manager(pool_size=5, device_id=0)
        if gpu_pool:
            gpu_handle = gpu_pool.write("test_roi_frame", test_frame)
            
            if gpu_handle:
                # Test ROI extraction with GPU handle
                bbox = [100, 100, 200, 200]  # x1, y1, x2, y2
                roi_result = roi_processor.extract_roi(
                    gpu_handle,  # Pass GPU handle instead of numpy array
                    bbox,
                    track_id=1,
                    timestamp=time.time()
                )
                
                if roi_result:
                    print(f"ROI extraction successful: shape={roi_result['roi'].shape}")
                    print(f"Quality score: {roi_result['quality_score']:.3f}")
                else:
                    print("ROI extraction failed")
            else:
                print("Failed to create GPU handle")
                
            gpu_pool.cleanup()
    else:
        print("ROI Processor GPU IPC not available")


def main():
    """Run GPU IPC tests."""
    print("GPU IPC Implementation Test")
    print(f"CUDA Available: {CUDA_AVAILABLE}")
    print(f"CuPy Available: {cp is not None}")
    
    if not CUDA_AVAILABLE:
        print("\nGPU/CUDA not available. Skipping tests.")
        return
        
    # Test 1: Basic IPC communication
    print("\n=== Test 1: Basic GPU IPC Communication ===")
    queue = mp.Queue()
    
    producer = Process(target=producer_process, args=(queue, True))
    consumer = Process(target=consumer_process, args=(queue, True))
    
    producer.start()
    consumer.start()
    
    producer.join()
    consumer.join()
    
    print("\nBasic IPC test completed")
    
    # Test 2: FrameDistributor integration
    test_frame_distributor_integration()
    
    # Test 3: ROI Processor with GPU handles
    test_roi_processor_gpu_handles()
    
    print("\n=== All tests completed ===")


if __name__ == "__main__":
    # Required for multiprocessing on Windows
    mp.set_start_method('spawn', force=True)
    main()