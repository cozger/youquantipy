#!/usr/bin/env python3
"""Test script to verify TensorRT landmark integration"""

import numpy as np
import cv2
import time
import sys
import os

# Add project directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
except ImportError:
    print("ERROR: TensorRT and PyCUDA required!")
    print("Install with: pip install tensorrt pycuda")
    sys.exit(1)

from confighandler import ConfigHandler
from retinaface_detector_gpu import RetinaFaceDetectorGPU
from roi_processor import ROIProcessor

def test_landmark_models():
    """Test TensorRT landmark and blendshape models"""
    print("=== Testing TensorRT Landmark Integration ===\n")
    
    # Load config
    config = ConfigHandler().config
    landmark_trt_path = config['advanced_detection']['landmark_trt_path']
    blendshape_trt_path = config['advanced_detection']['blendshape_trt_path']
    
    print(f"Landmark model: {landmark_trt_path}")
    print(f"Blendshape model: {blendshape_trt_path}")
    
    # Check if models exist
    if not os.path.exists(landmark_trt_path):
        print(f"ERROR: Landmark model not found at {landmark_trt_path}")
        return False
    
    if not os.path.exists(blendshape_trt_path):
        print(f"ERROR: Blendshape model not found at {blendshape_trt_path}")
        return False
    
    print("\n✓ Model files exist")
    
    # Initialize CUDA
    cuda.init()
    cuda_device = cuda.Device(0)
    cuda_context = cuda_device.make_context()
    
    try:
        # TensorRT logger
        trt_logger = trt.Logger(trt.Logger.INFO)
        
        # Load engines
        print("\nLoading TensorRT engines...")
        
        with open(landmark_trt_path, 'rb') as f:
            landmark_engine = trt.Runtime(trt_logger).deserialize_cuda_engine(f.read())
        print("✓ Landmark engine loaded")
        
        with open(blendshape_trt_path, 'rb') as f:
            blendshape_engine = trt.Runtime(trt_logger).deserialize_cuda_engine(f.read())
        print("✓ Blendshape engine loaded")
        
        # Print engine info
        print(f"\nLandmark engine:")
        print(f"  - Bindings: {landmark_engine.num_bindings}")
        for i in range(landmark_engine.num_bindings):
            name = landmark_engine.get_binding_name(i)
            shape = landmark_engine.get_binding_shape(i)
            dtype = landmark_engine.get_binding_dtype(i)
            is_input = landmark_engine.binding_is_input(i)
            print(f"  - {name}: {shape}, {dtype}, {'input' if is_input else 'output'}")
        
        print(f"\nBlendshape engine:")
        print(f"  - Bindings: {blendshape_engine.num_bindings}")
        for i in range(blendshape_engine.num_bindings):
            name = blendshape_engine.get_binding_name(i)
            shape = blendshape_engine.get_binding_shape(i)
            dtype = blendshape_engine.get_binding_dtype(i)
            is_input = blendshape_engine.binding_is_input(i)
            print(f"  - {name}: {shape}, {dtype}, {'input' if is_input else 'output'}")
        
        # Test with dummy input
        print("\nTesting inference with dummy input...")
        
        # Create contexts
        landmark_context = landmark_engine.create_execution_context()
        blendshape_context = blendshape_engine.create_execution_context()
        
        # Create dummy 256x256 RGB image (normalized)
        dummy_roi = np.random.randn(1, 256, 256, 3).astype(np.float32)
        
        # Allocate buffers
        stream = cuda.Stream()
        
        # Landmark buffers
        landmark_input_shape = (1, 256, 256, 3)
        landmark_output_shape = (1, 468, 3)
        
        landmark_input_host = cuda.pagelocked_empty(np.prod(landmark_input_shape), np.float32)
        landmark_output_host = cuda.pagelocked_empty(np.prod(landmark_output_shape), np.float32)
        
        landmark_input_device = cuda.mem_alloc(landmark_input_host.nbytes)
        landmark_output_device = cuda.mem_alloc(landmark_output_host.nbytes)
        
        # Copy input
        np.copyto(landmark_input_host, dummy_roi.ravel())
        cuda.memcpy_htod_async(landmark_input_device, landmark_input_host, stream)
        
        # Run landmark inference
        bindings = [int(landmark_input_device), int(landmark_output_device)]
        landmark_context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        
        # Get output
        cuda.memcpy_dtoh_async(landmark_output_host, landmark_output_device, stream)
        stream.synchronize()
        
        landmarks = landmark_output_host.reshape(landmark_output_shape)
        print(f"✓ Landmark inference successful: {landmarks.shape}")
        print(f"  - Sample landmark: x={landmarks[0,0,0]:.3f}, y={landmarks[0,0,1]:.3f}, z={landmarks[0,0,2]:.3f}")
        
        # Test blendshape with subset of landmarks
        blendshape_indices = [
            0, 1, 4, 5, 6, 10, 12, 13, 14, 17, 18, 21, 33, 37, 39, 40, 46, 52, 53, 54, 
            55, 58, 61, 63, 65, 66, 67, 70, 78, 80, 81, 82, 84, 87, 88, 91, 93, 95, 103, 
            105, 107, 109, 127, 132, 133, 136, 144, 145, 146, 148, 149, 150, 152, 153, 
            154, 155, 157, 158, 159, 160, 161, 162, 163, 168, 169, 170, 171, 172, 173, 
            174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 191, 195, 197, 
            234, 246, 249, 251, 263, 267, 269, 270, 271, 272, 276, 282, 283, 284, 285, 
            288, 291, 293, 295, 296, 297, 300, 308, 310, 311, 312, 314, 317, 318, 321, 
            323, 324, 332, 334, 336, 338, 356, 361, 362, 365, 373, 374, 375, 377, 378, 
            379, 380, 381, 382, 384, 385, 386, 387, 388, 389, 390, 397, 398, 400, 402, 
            405, 409, 415
        ]
        
        selected_points = landmarks[0, blendshape_indices, :2]  # Only x,y
        
        # Blendshape buffers
        blendshape_input_shape = (146, 2)
        blendshape_output_shape = (52,)
        
        blendshape_input_host = cuda.pagelocked_empty(np.prod(blendshape_input_shape), np.float32)
        blendshape_output_host = cuda.pagelocked_empty(np.prod(blendshape_output_shape), np.float32)
        
        blendshape_input_device = cuda.mem_alloc(blendshape_input_host.nbytes)
        blendshape_output_device = cuda.mem_alloc(blendshape_output_host.nbytes)
        
        # Copy input
        np.copyto(blendshape_input_host, selected_points.ravel())
        cuda.memcpy_htod_async(blendshape_input_device, blendshape_input_host, stream)
        
        # Run blendshape inference
        bindings = [int(blendshape_input_device), int(blendshape_output_device)]
        blendshape_context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        
        # Get output
        cuda.memcpy_dtoh_async(blendshape_output_host, blendshape_output_device, stream)
        stream.synchronize()
        
        blendshapes = blendshape_output_host
        print(f"✓ Blendshape inference successful: {blendshapes.shape}")
        print(f"  - Sample scores: {blendshapes[:5]}")
        
        print("\n✓ All tests passed!")
        return True
        
    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        cuda_context.pop()
        cuda_context.detach()


def test_full_pipeline():
    """Test the full pipeline with real camera/video"""
    print("\n=== Testing Full Pipeline ===\n")
    
    # Use test video or camera
    cap = cv2.VideoCapture(0)  # Use camera 0
    if not cap.isOpened():
        print("Failed to open camera, trying test video...")
        # Try a test video if available
        test_videos = [
            "test_video.mp4",
            "../test_video.mp4",
            "../../test_video.mp4"
        ]
        for video_path in test_videos:
            if os.path.exists(video_path):
                cap = cv2.VideoCapture(video_path)
                if cap.isOpened():
                    print(f"Using test video: {video_path}")
                    break
        
        if not cap.isOpened():
            print("No camera or test video available")
            return False
    
    # Initialize components
    config = ConfigHandler().config
    detector = RetinaFaceDetectorGPU(
        model_path=config['advanced_detection']['retinaface_model'],
        confidence_threshold=config['advanced_detection']['detection_confidence']
    )
    
    roi_processor = ROIProcessor(
        target_size=(256, 256),
        use_gpu=True
    )
    
    # Start detector
    detector.start()
    roi_processor.start()
    
    try:
        # Process a few frames
        print("Processing frames...")
        frame_count = 0
        max_frames = 30
        
        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Submit frame for detection
            detector.input_queue.put({
                'frame': frame,
                'frame_id': frame_count
            })
            
            # Get detection results
            try:
                results = detector.output_queue.get(timeout=0.1)
                detections = results['detections']
                
                if detections:
                    print(f"Frame {frame_count}: {len(detections)} faces detected")
                    
                    # Extract ROI for first detection
                    det = detections[0]
                    roi_result = roi_processor.extract_roi(
                        frame,
                        det['bbox'],
                        track_id=0,
                        timestamp=time.time()
                    )
                    
                    if roi_result:
                        roi = roi_result['roi']
                        print(f"  - ROI extracted: {roi.shape}, dtype={roi.dtype}")
                        
                        # Show ROI
                        if roi.dtype == np.float32:
                            # Denormalize for display
                            display_roi = ((roi + 1) * 127.5).astype(np.uint8)
                        else:
                            display_roi = roi
                        
                        cv2.imshow('ROI', display_roi)
                        
            except:
                pass
            
            # Display frame
            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            frame_count += 1
        
        print(f"\nProcessed {frame_count} frames")
        
    finally:
        # Cleanup
        detector.stop()
        roi_processor.stop()
        cap.release()
        cv2.destroyAllWindows()
    
    return True


if __name__ == "__main__":
    print("TensorRT Landmark Integration Test\n")
    
    # Test 1: Model loading and basic inference
    if not test_landmark_models():
        print("\nModel test failed!")
        sys.exit(1)
    
    # Test 2: Full pipeline (optional)
    print("\nPress Enter to test full pipeline with camera/video (or Ctrl+C to skip)...")
    try:
        input()
        test_full_pipeline()
    except KeyboardInterrupt:
        print("\nSkipping full pipeline test")
    
    print("\nAll tests completed!")