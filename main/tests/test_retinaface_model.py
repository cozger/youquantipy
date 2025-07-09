#!/usr/bin/env python3
"""Test script to debug RetinaFace model initialization"""

import os
import numpy as np
import cv2
from retinaface_detector import RetinaFaceDetector

def test_retinaface_model():
    # Get model path from config
    model_path = "D:/Projects/youquantipy/retinaface.onnx"
    
    print("Testing RetinaFace model initialization...")
    print(f"Model path: {model_path}")
    print(f"Model exists: {os.path.exists(model_path)}")
    
    # Create detector
    detector = RetinaFaceDetector(
        model_path=model_path,
        tile_size=640,
        overlap=0.2,
        confidence_threshold=0.5,
        max_workers=2
    )
    
    # Test with a dummy frame
    dummy_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    print(f"\nTesting with dummy frame: {dummy_frame.shape}")
    
    # Start detector
    detector.start()
    
    # Submit a test frame
    success = detector.submit_frame(dummy_frame, 1)
    print(f"Frame submission: {success}")
    
    # Wait a bit for processing
    import time
    time.sleep(2)
    
    # Check for results
    result = detector.get_detections(timeout=1.0)
    if result:
        frame_id, detections = result
        print(f"Got detection result for frame {frame_id}: {len(detections)} detections")
    else:
        print("No detection results received")
    
    # Clean up
    detector.stop()
    print("Test completed")

if __name__ == "__main__":
    test_retinaface_model()