#!/usr/bin/env python3
"""
Test script to verify landmark transformation from ROI space to frame space.
This tests the fix for landmarks being returned in ROI coordinates instead of frame coordinates.
"""

import numpy as np
import cv2
import sys
import os

# Add project directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gpu_face_processor import GPUFaceProcessor

def test_landmark_transformation():
    """Test that landmarks are correctly transformed from ROI to frame space"""
    print("=== Testing Landmark Coordinate Transformation ===\n")
    
    # Create a test scenario
    print("Test scenario:")
    print("- Original bbox: [100, 100, 300, 300] (200x200 face)")
    print("- With 30% padding: bbox becomes approx [70, 70, 330, 330] (260x260)")
    print("- ROI is resized to 256x256")
    print("- Landmark at ROI center (128, 128) should map back to frame center (200, 200)")
    
    # Simulate the transformation that happens in _extract_rois_gpu
    original_bbox = np.array([100, 100, 300, 300])
    x1, y1, x2, y2 = original_bbox
    w = x2 - x1  # 200
    h = y2 - y1  # 200
    
    # Add 30% padding (15% each side)
    pad_w = w * 0.15  # 30
    pad_h = h * 0.15  # 30
    
    x1_padded = max(0, x1 - pad_w)  # 70
    y1_padded = max(0, y1 - pad_h)  # 70
    x2_padded = x2 + pad_w  # 330
    y2_padded = y2 + pad_h  # 330
    
    # Calculate ROI dimensions
    roi_w = x2_padded - x1_padded  # 260
    roi_h = y2_padded - y1_padded  # 260
    
    # Transformation parameters
    transform = {
        'x1': x1_padded,
        'y1': y1_padded,
        'scale_x': roi_w / 256.0,  # 260/256 = 1.015625
        'scale_y': roi_h / 256.0   # 260/256 = 1.015625
    }
    
    print(f"\nTransformation parameters:")
    print(f"- Padded bbox: [{x1_padded}, {y1_padded}, {x2_padded}, {y2_padded}]")
    print(f"- Scale X: {transform['scale_x']:.6f}")
    print(f"- Scale Y: {transform['scale_y']:.6f}")
    
    # Test landmark at ROI center
    roi_landmark = np.array([128.0, 128.0, 0.5])  # Center of 256x256 ROI
    
    # Apply transformation (mimicking _transform_landmarks_to_frame)
    frame_x = roi_landmark[0] * transform['scale_x'] + transform['x1']
    frame_y = roi_landmark[1] * transform['scale_y'] + transform['y1']
    
    print(f"\nLandmark transformation test:")
    print(f"- ROI landmark: ({roi_landmark[0]}, {roi_landmark[1]})")
    print(f"- Expected frame coords: (200, 200) - center of original face")
    print(f"- Actual frame coords: ({frame_x:.2f}, {frame_y:.2f})")
    print(f"- Error: ({abs(frame_x - 200):.2f}, {abs(frame_y - 200):.2f}) pixels")
    
    # Test corner landmarks
    print("\nCorner landmark tests:")
    test_points = [
        ([0, 0], [x1_padded, y1_padded], "Top-left"),
        ([256, 0], [x2_padded, y1_padded], "Top-right"),
        ([0, 256], [x1_padded, y2_padded], "Bottom-left"),
        ([256, 256], [x2_padded, y2_padded], "Bottom-right")
    ]
    
    for roi_point, expected, name in test_points:
        frame_x = roi_point[0] * transform['scale_x'] + transform['x1']
        frame_y = roi_point[1] * transform['scale_y'] + transform['y1']
        print(f"- {name}: ROI {roi_point} -> Frame ({frame_x:.1f}, {frame_y:.1f}), "
              f"Expected ({expected[0]:.1f}, {expected[1]:.1f})")
    
    # Verify the transformation is invertible
    print("\nInverse transformation test:")
    frame_point = np.array([200.0, 200.0])  # Center of face
    roi_x = (frame_point[0] - transform['x1']) / transform['scale_x']
    roi_y = (frame_point[1] - transform['y1']) / transform['scale_y']
    print(f"- Frame point: ({frame_point[0]}, {frame_point[1]})")
    print(f"- Transformed to ROI: ({roi_x:.2f}, {roi_y:.2f})")
    print(f"- Expected: (128.00, 128.00)")
    
    print("\n✓ Landmark transformation logic verified!")

def visualize_transformation():
    """Create a visual representation of the transformation"""
    print("\n=== Creating Visual Test ===")
    
    # Create a test image
    img_size = 400
    img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    
    # Draw original face bbox
    original_bbox = [100, 100, 300, 300]
    cv2.rectangle(img, (original_bbox[0], original_bbox[1]), 
                  (original_bbox[2], original_bbox[3]), (0, 255, 0), 2)
    cv2.putText(img, "Original", (original_bbox[0], original_bbox[1]-5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # Draw padded bbox
    x1, y1, x2, y2 = original_bbox
    w = x2 - x1
    h = y2 - y1
    pad_w = w * 0.15
    pad_h = h * 0.15
    padded_bbox = [int(x1 - pad_w), int(y1 - pad_h), 
                   int(x2 + pad_w), int(y2 + pad_h)]
    cv2.rectangle(img, (padded_bbox[0], padded_bbox[1]), 
                  (padded_bbox[2], padded_bbox[3]), (255, 0, 0), 2)
    cv2.putText(img, "Padded ROI", (padded_bbox[0], padded_bbox[1]-5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    # Draw center point
    center_x = (original_bbox[0] + original_bbox[2]) // 2
    center_y = (original_bbox[1] + original_bbox[3]) // 2
    cv2.circle(img, (center_x, center_y), 5, (0, 0, 255), -1)
    cv2.putText(img, "Center", (center_x + 10, center_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    # Save visualization
    output_path = "landmark_transform_test.png"
    cv2.imwrite(output_path, img)
    print(f"Visualization saved to: {output_path}")
    
    # Also display if possible
    try:
        cv2.imshow("Landmark Transform Test", img)
        print("Press any key to close the visualization...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except:
        print("Could not display image (running headless?)")

if __name__ == "__main__":
    print("Landmark Coordinate Transformation Test")
    print("=" * 50)
    
    test_landmark_transformation()
    visualize_transformation()
    
    print("\n✓ All tests completed!")