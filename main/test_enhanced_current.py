#!/usr/bin/env python3
"""Test if enhanced architecture loads correctly with current module structure"""

import os
import sys

print("Testing enhanced architecture loading...")
print(f"Current directory: {os.getcwd()}")
print(f"Python path: {sys.path[:3]}...")

# Test imports that GUI actually uses
try:
    from advanced_detection_integration import parallel_participant_worker_auto, should_use_advanced_detection
    print("✓ Advanced detection integration imported successfully")
    
    # Check which mode would be used
    uses_advanced = should_use_advanced_detection()
    print(f"should_use_advanced_detection() = {uses_advanced}")
    
    if not uses_advanced:
        print("\nChecking why enhanced mode is not available:")
        
        # Check config
        try:
            from confighandler import ConfigHandler
            config = ConfigHandler()
            face_recog_enabled = config.get('startup_mode.enable_face_recognition', False)
            print(f"  - Face recognition enabled in config: {face_recog_enabled}")
            
            if face_recog_enabled:
                # Check model paths
                retinaface_model = config.get('advanced_detection.retinaface_model')
                arcface_model = config.get('advanced_detection.arcface_model')
                
                print(f"  - RetinaFace model path: {retinaface_model}")
                print(f"  - ArcFace model path: {arcface_model}")
                
                if retinaface_model:
                    exists = os.path.exists(retinaface_model)
                    print(f"  - RetinaFace model exists: {exists}")
                    
                if arcface_model:
                    exists = os.path.exists(arcface_model)
                    print(f"  - ArcFace model exists: {exists}")
                    
        except Exception as e:
            print(f"  - Error checking config: {e}")
            
except ImportError as e:
    print(f"✗ Failed to import advanced detection integration: {e}")

# Check if unified modules exist
try:
    from parallelworker_unified import parallel_participant_worker
    print("✓ Unified parallel worker imported successfully")
except ImportError as e:
    print(f"✗ Failed to import unified parallel worker: {e}")

# Check if enhanced components are available
try:
    from retinaface_detector import RetinaFaceDetector
    print("✓ RetinaFace detector imported successfully")
except ImportError as e:
    print(f"✗ Failed to import RetinaFace detector: {e}")

try:
    from lightweight_tracker import LightweightTracker
    print("✓ Lightweight tracker imported successfully")
except ImportError as e:
    print(f"✗ Failed to import lightweight tracker: {e}")

try:
    from roi_processor import ROIProcessor
    print("✓ ROI processor imported successfully")
except ImportError as e:
    print(f"✗ Failed to import ROI processor: {e}")

try:
    from face_recognition_process import FaceRecognitionProcess
    print("✓ Face recognition process imported successfully")
except ImportError as e:
    print(f"✗ Failed to import face recognition process: {e}")

print("\nDone.")