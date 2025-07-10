#!/usr/bin/env python3
"""
Simple RetinaFace Detection Test
================================

A minimal test script to verify RetinaFace detection is working correctly.
Tests with a generated test image containing faces.
"""

import cv2
import numpy as np
import time
import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from retinaface_detector import RetinaFaceDetector


def create_test_image():
    """Create a test image with drawn faces for testing"""
    # Create a white image
    img = np.ones((720, 1280, 3), dtype=np.uint8) * 255
    
    # Draw some face-like rectangles and circles
    faces = [
        {"center": (320, 240), "size": 150},
        {"center": (640, 240), "size": 120},
        {"center": (960, 240), "size": 100},
        {"center": (480, 480), "size": 180},
    ]
    
    for face in faces:
        cx, cy = face["center"]
        size = face["size"]
        
        # Face outline (ellipse)
        cv2.ellipse(img, (cx, cy), (size//2, int(size*0.6)), 0, 0, 360, (200, 180, 160), -1)
        
        # Eyes
        eye_y = cy - size//5
        eye_offset = size//4
        cv2.circle(img, (cx - eye_offset, eye_y), size//10, (50, 50, 50), -1)
        cv2.circle(img, (cx + eye_offset, eye_y), size//10, (50, 50, 50), -1)
        
        # Nose
        cv2.circle(img, (cx, cy), size//15, (150, 130, 110), -1)
        
        # Mouth
        mouth_y = cy + size//4
        cv2.ellipse(img, (cx, mouth_y), (size//3, size//6), 0, 0, 180, (100, 80, 80), 2)
    
    # Add some text
    cv2.putText(img, "RetinaFace Test Image", (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)
    cv2.putText(img, "4 synthetic faces for detection", (20, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 1)
    
    return img


def test_detection():
    """Run a simple detection test"""
    print("=== Simple RetinaFace Detection Test ===\n")
    
    # Model path
    model_path = "D:/Projects/youquantipy/retinaface.onnx"
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please ensure the RetinaFace ONNX model is available")
        return
    
    print(f"Model path: {model_path}")
    print("Creating detector...")
    
    # Create detector with simple settings
    detector = RetinaFaceDetector(
        model_path=model_path,
        tile_size=640,
        overlap=0.1,
        confidence_threshold=0.5,  # Lower threshold for synthetic faces
        nms_threshold=0.4,
        max_workers=2
    )
    
    print("Starting detector...")
    detector.start()
    
    # Create or load test image
    print("\nCreating test image with synthetic faces...")
    test_img = create_test_image()
    
    # Save test image
    cv2.imwrite("test_input.jpg", test_img)
    print("Test image saved as 'test_input.jpg'")
    
    # Convert to RGB
    rgb_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
    
    # Submit for detection
    print("\nSubmitting image for detection...")
    start_time = time.time()
    
    success = detector.submit_frame(rgb_img, frame_id=1)
    if not success:
        print("Failed to submit frame!")
        detector.stop()
        return
    
    print("Frame submitted, waiting for results...")
    
    # Wait for results
    result = None
    timeout = 10.0
    
    while time.time() - start_time < timeout:
        result = detector.get_detections(timeout=0.1)
        if result:
            break
        print(".", end="", flush=True)
    
    elapsed = time.time() - start_time
    print(f"\nDetection took {elapsed:.3f} seconds")
    
    if result:
        frame_id, detections = result
        print(f"\nResults for frame {frame_id}:")
        print(f"Found {len(detections)} detections")
        
        # Draw results
        output_img = test_img.copy()
        
        for i, det in enumerate(detections):
            bbox = det['bbox']
            conf = det['confidence']
            
            # Draw bounding box
            x1, y1, x2, y2 = [int(v) for v in bbox]
            cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"Face {i+1}: {conf:.2f}"
            cv2.putText(output_img, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Print details
            print(f"  Detection {i+1}:")
            print(f"    BBox: [{x1}, {y1}, {x2}, {y2}]")
            print(f"    Confidence: {conf:.3f}")
            print(f"    Size: {x2-x1}x{y2-y1}")
        
        # Save output
        cv2.imwrite("test_output.jpg", output_img)
        print(f"\nOutput saved as 'test_output.jpg'")
        
        # Display if possible
        print("\nPress any key to close display window...")
        cv2.imshow("Detection Results", output_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    else:
        print(f"\nNo results received within {timeout} seconds!")
        print("Possible issues:")
        print("- Model loading failed")
        print("- Detection threshold too high")
        print("- Processing error")
    
    # Stop detector
    print("\nStopping detector...")
    detector.stop()
    print("Test complete!")


def test_real_image():
    """Test with a real image if available"""
    print("\n=== Testing with Real Image ===\n")
    
    # Common test image paths
    test_images = [
        "test_face.jpg",
        "test_faces.jpg", 
        "group_photo.jpg",
        "D:/test_images/faces.jpg"
    ]
    
    # Find first available image
    test_image = None
    for img_path in test_images:
        if os.path.exists(img_path):
            test_image = img_path
            break
    
    if not test_image:
        print("No real test image found. Skipping real image test.")
        print("To test with a real image, place an image file named 'test_face.jpg' in the current directory")
        return
    
    print(f"Found test image: {test_image}")
    
    # Load image
    img = cv2.imread(test_image)
    if img is None:
        print(f"Failed to load image: {test_image}")
        return
    
    print(f"Image shape: {img.shape}")
    
    # Create detector
    model_path = "D:/Projects/youquantipy/retinaface.onnx"
    detector = RetinaFaceDetector(
        model_path=model_path,
        confidence_threshold=0.9,  # Higher threshold for real faces
        nms_threshold=0.4
    )
    
    detector.start()
    
    # Detect
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    detector.submit_frame(rgb_img, 1)
    
    # Wait for results
    start = time.time()
    result = None
    
    while time.time() - start < 5.0:
        result = detector.get_detections(timeout=0.1)
        if result:
            break
    
    if result:
        _, detections = result
        print(f"Found {len(detections)} faces in real image")
        
        # Draw and save
        for det in detections:
            bbox = det['bbox']
            x1, y1, x2, y2 = [int(v) for v in bbox]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        output_path = test_image.replace('.jpg', '_detected.jpg')
        cv2.imwrite(output_path, img)
        print(f"Result saved to: {output_path}")
    
    detector.stop()


if __name__ == "__main__":
    # Run simple test first
    test_detection()
    
    # Try real image test
    test_real_image()
    
    print("\n=== All tests complete ===")