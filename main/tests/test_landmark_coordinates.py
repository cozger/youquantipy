#!/usr/bin/env python3
"""
Test script to determine the coordinate system of landmark model outputs.
This will help identify if landmarks are:
1. Normalized (0-1) within the ROI
2. Pixel coordinates (0-256) within the ROI
"""

import numpy as np
import cv2
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

def test_landmark_range():
    """Test the output range of the landmark model"""
    print("=== Testing Landmark Model Output Range ===\n")
    
    # Load config
    config = ConfigHandler().config
    landmark_trt_path = config['advanced_detection']['landmark_trt_path']
    
    print(f"Landmark model: {landmark_trt_path}")
    
    if not os.path.exists(landmark_trt_path):
        print(f"ERROR: Landmark model not found at {landmark_trt_path}")
        return
    
    # Initialize CUDA
    cuda.init()
    cuda_device = cuda.Device(0)
    cuda_context = cuda_device.make_context()
    
    try:
        # TensorRT logger
        trt_logger = trt.Logger(trt.Logger.INFO)
        
        # Load engine
        with open(landmark_trt_path, 'rb') as f:
            landmark_engine = trt.Runtime(trt_logger).deserialize_cuda_engine(f.read())
        print("✓ Landmark engine loaded")
        
        # Create context
        landmark_context = landmark_engine.create_execution_context()
        
        # Test with different input patterns
        test_patterns = [
            ("white_image", np.ones((1, 256, 256, 3), dtype=np.float32)),
            ("black_image", np.zeros((1, 256, 256, 3), dtype=np.float32)),
            ("gray_image", np.ones((1, 256, 256, 3), dtype=np.float32) * 0.5),
            ("random_image", np.random.rand(1, 256, 256, 3).astype(np.float32)),
            ("face_like_pattern", create_face_pattern())
        ]
        
        # Allocate buffers
        stream = cuda.Stream()
        landmark_input_shape = (1, 256, 256, 3)
        landmark_output_shape = (1, 468, 3)
        
        landmark_input_host = cuda.pagelocked_empty(np.prod(landmark_input_shape), np.float32)
        landmark_output_host = cuda.pagelocked_empty(np.prod(landmark_output_shape), np.float32)
        
        landmark_input_device = cuda.mem_alloc(landmark_input_host.nbytes)
        landmark_output_device = cuda.mem_alloc(landmark_output_host.nbytes)
        
        print("\nTesting different input patterns:")
        print("-" * 60)
        
        for pattern_name, pattern in test_patterns:
            # Copy input
            np.copyto(landmark_input_host, pattern.ravel())
            cuda.memcpy_htod_async(landmark_input_device, landmark_input_host, stream)
            
            # Run inference
            bindings = [int(landmark_input_device), int(landmark_output_device)]
            landmark_context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
            
            # Get output
            cuda.memcpy_dtoh_async(landmark_output_host, landmark_output_device, stream)
            stream.synchronize()
            
            landmarks = landmark_output_host.reshape(landmark_output_shape)
            
            # Analyze output range
            x_coords = landmarks[0, :, 0]
            y_coords = landmarks[0, :, 1]
            z_coords = landmarks[0, :, 2]
            
            print(f"\n{pattern_name}:")
            print(f"  X range: [{x_coords.min():.3f}, {x_coords.max():.3f}] (mean: {x_coords.mean():.3f})")
            print(f"  Y range: [{y_coords.min():.3f}, {y_coords.max():.3f}] (mean: {y_coords.mean():.3f})")
            print(f"  Z range: [{z_coords.min():.3f}, {z_coords.max():.3f}] (mean: {z_coords.mean():.3f})")
            
            # Check if values are normalized or pixel coordinates
            if x_coords.max() <= 1.0 and y_coords.max() <= 1.0:
                print("  → Appears to be NORMALIZED coordinates (0-1)")
            elif x_coords.max() > 200 or y_coords.max() > 200:
                print("  → Appears to be PIXEL coordinates (0-256)")
            else:
                print("  → UNCLEAR coordinate system")
            
            # Sample some specific landmarks
            key_landmarks = {
                0: "nose tip",
                17: "right eye",
                133: "left eye", 
                13: "upper lip",
                14: "lower lip"
            }
            
            print("  Key landmarks:")
            for idx, name in key_landmarks.items():
                if idx < len(landmarks[0]):
                    x, y, z = landmarks[0, idx]
                    print(f"    {name}: ({x:.3f}, {y:.3f}, {z:.3f})")
        
        print("\n" + "="*60)
        print("CONCLUSION:")
        
        # Make a conclusion based on the tests
        if all(landmarks[0, :, 0].max() <= 1.0 and landmarks[0, :, 1].max() <= 1.0 
               for _, pattern in test_patterns):
            print("The landmark model outputs NORMALIZED coordinates (0-1) within the ROI.")
            print("These need to be scaled by 256 to get pixel coordinates in ROI space.")
        else:
            print("The landmark model outputs PIXEL coordinates (0-256) within the ROI.")
            print("These can be used directly without scaling.")
            
    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        cuda_context.pop()
        cuda_context.detach()


def create_face_pattern():
    """Create a simple face-like pattern for testing"""
    img = np.ones((1, 256, 256, 3), dtype=np.float32) * 0.7
    
    # Add some darker regions for eyes
    img[0, 80:100, 60:80, :] = 0.3
    img[0, 80:100, 176:196, :] = 0.3
    
    # Add mouth region
    img[0, 160:180, 100:156, :] = 0.4
    
    return img


if __name__ == "__main__":
    print("Landmark Coordinate System Test\n")
    test_landmark_range()