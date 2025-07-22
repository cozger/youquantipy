#!/usr/bin/env python3
"""
Test script to verify face landmark drawing with MediaPipe connections.
This will help debug the green line issue in canvas drawing.
"""

import cv2
import mediapipe as mp
import numpy as np

def test_mediapipe_connections():
    """Test and visualize MediaPipe face connections."""
    mp_face_mesh = mp.solutions.face_mesh
    
    # Print connection info
    print(f"Total FACEMESH_CONTOURS connections: {len(mp_face_mesh.FACEMESH_CONTOURS)}")
    print(f"Total FACEMESH_TESSELATION connections: {len(mp_face_mesh.FACEMESH_TESSELATION)}")
    
    # Print first few connections
    print("\nFirst 10 FACEMESH_CONTOURS connections:")
    for i, (start, end) in enumerate(list(mp_face_mesh.FACEMESH_CONTOURS)[:10]):
        print(f"  Connection {i}: {start} -> {end}")
    
    # Check which landmarks form the jaw line
    print("\nJaw line landmarks (first 17):")
    jaw_connections = []
    for i in range(16):  # Connect 0-1, 1-2, ..., 15-16
        jaw_connections.append((i, i+1))
    print(f"Sequential jaw connections: {jaw_connections}")
    
    # Create a test image with face landmarks
    print("\nCreating test visualization...")
    img = np.zeros((600, 800, 3), dtype=np.uint8)
    
    # Simulate face landmarks in a circle (468 points)
    center_x, center_y = 400, 300
    radius = 200
    landmarks = []
    
    # Generate 468 landmarks in an elliptical pattern
    for i in range(468):
        angle = (i / 468) * 2 * np.pi
        # Make it more face-like with varying radius
        if i < 17:  # Jaw line
            r = radius * 0.9
            y_offset = 50  # Lower position for jaw
        elif i < 100:  # Lower face
            r = radius * 0.8
            y_offset = 0
        else:  # Upper face
            r = radius * 0.7
            y_offset = -30
            
        x = int(center_x + r * np.cos(angle))
        y = int(center_y + r * np.sin(angle) * 0.7 + y_offset)  # Flatten vertically
        landmarks.append((x, y))
    
    # Draw all landmarks as small circles
    for i, (x, y) in enumerate(landmarks):
        color = (0, 255, 255) if i < 17 else (100, 100, 100)  # Highlight jaw
        cv2.circle(img, (x, y), 2, color, -1)
        if i < 20:  # Label first 20 points
            cv2.putText(img, str(i), (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    
    # Draw FACEMESH_CONTOURS connections
    for start_idx, end_idx in mp_face_mesh.FACEMESH_CONTOURS:
        if start_idx < len(landmarks) and end_idx < len(landmarks):
            cv2.line(img, landmarks[start_idx], landmarks[end_idx], (0, 255, 0), 1)
    
    # Save the test image
    cv2.imwrite('test_face_connections.png', img)
    print("Saved test visualization to test_face_connections.png")
    
    # Create another image showing just jaw connections
    img_jaw = np.zeros((600, 800, 3), dtype=np.uint8)
    
    # Draw jaw line only (first 17 points connected sequentially)
    for i in range(17):
        x, y = landmarks[i]
        cv2.circle(img_jaw, (x, y), 3, (0, 255, 255), -1)
        cv2.putText(img_jaw, str(i), (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        if i < 16:  # Connect to next point
            x2, y2 = landmarks[i+1]
            cv2.line(img_jaw, (x, y), (x2, y2), (0, 255, 0), 2)
    
    cv2.imwrite('test_jaw_only.png', img_jaw)
    print("Saved jaw-only visualization to test_jaw_only.png")

if __name__ == "__main__":
    test_mediapipe_connections()