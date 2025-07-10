#!/usr/bin/env python3
"""Direct test of RetinaFace detector to debug anchor initialization"""

import sys
import os
import numpy as np

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Testing RetinaFace detector initialization...")

try:
    from retinaface_detector import RetinaFaceDetector
    print("✓ Successfully imported RetinaFaceDetector")
except Exception as e:
    print(f"✗ Failed to import: {e}")
    sys.exit(1)

# Test detector creation
print("\nCreating detector...")
try:
    detector = RetinaFaceDetector(
        model_path="D:/Projects/youquantipy/retinaface.onnx",
        tile_size=640,
        overlap=0.2,
        confidence_threshold=0.98
    )
    print("✓ Detector created successfully")
    
    # Check anchors
    if detector.anchors is not None:
        print(f"✓ Anchors initialized: shape={detector.anchors.shape}")
        print(f"  First anchor: {detector.anchors[0]}")
        print(f"  Last anchor: {detector.anchors[-1]}")
    else:
        print("✗ Anchors are None after initialization!")
        
except Exception as e:
    print(f"✗ Failed to create detector: {e}")
    import traceback
    traceback.print_exc()

print("\nTest complete.")