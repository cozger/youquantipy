#!/usr/bin/env python3
"""Analyze enhanced architecture implementation without importing modules"""

import os
import json

print("Enhanced Architecture Analysis")
print("="*60)

# Check configuration
config_path = "youquantipy_config.json"
if os.path.exists(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print("\nConfiguration:")
    face_recognition = config.get('startup_mode', {}).get('enable_face_recognition', False)
    print(f"  - Face recognition enabled: {face_recognition}")
    
    if face_recognition:
        print("\nModel paths configured:")
        retinaface = config.get('advanced_detection', {}).get('retinaface_model', 'Not set')
        arcface = config.get('advanced_detection', {}).get('arcface_model', 'Not set')
        print(f"  - RetinaFace: {retinaface}")
        print(f"  - ArcFace: {arcface}")
        
        print("\nModel files exist:")
        print(f"  - RetinaFace: {os.path.exists(retinaface) if retinaface != 'Not set' else 'N/A'}")
        print(f"  - ArcFace: {os.path.exists(arcface) if arcface != 'Not set' else 'N/A'}")

# Check required files
print("\nRequired Enhanced Architecture Files:")
enhanced_files = {
    "Core Integration": [
        "advanced_detection_integration.py",
        "parallelworker_unified.py",
        "participantmanager_unified.py"
    ],
    "Detection & Tracking": [
        "retinaface_detector.py",
        "lightweight_tracker.py",
        "roi_processor.py"
    ],
    "Recognition": [
        "face_recognition_process.py",
        "enrollment_manager.py"
    ],
    "Experimental Enhanced": [
        "experimental_enhanced/frame_router.py",
        "experimental_enhanced/roi_manager.py",
        "experimental_enhanced/result_aggregator.py",
        "experimental_enhanced/camera_worker_enhanced.py"
    ]
}

for category, files in enhanced_files.items():
    print(f"\n{category}:")
    for file in files:
        exists = os.path.exists(file)
        status = "✓" if exists else "✗"
        print(f"  {status} {file}")

# Check deprecated files
print("\nDeprecated Files (should exist in deprecated/):")
deprecated_files = [
    "deprecated/parallelworker_enhanced_integration.py",
    "deprecated/parallelworker_advanced_enhanced.py",
    "deprecated/landmark_worker_pool_adaptive.py"
]

for file in deprecated_files:
    exists = os.path.exists(file)
    status = "✓" if exists else "✗"
    print(f"  {status} {file}")

print("\nSummary:")
print("-"*60)

# Determine if enhanced mode would be activated
face_recog_enabled = config.get('startup_mode', {}).get('enable_face_recognition', False) if 'config' in locals() else False
models_exist = False

if face_recog_enabled and 'config' in locals():
    retinaface = config.get('advanced_detection', {}).get('retinaface_model')
    if retinaface and os.path.exists(retinaface):
        models_exist = True

if face_recog_enabled and not models_exist:
    print("⚠️  Enhanced mode is ENABLED in config but model files are MISSING")
    print("   The system will fall back to standard mode")
    print("\nTo use enhanced mode:")
    print("   1. Obtain retinaface.onnx and arcface.onnx model files")
    print("   2. Place them in D:/Projects/youquantipy/")
    print("   3. Restart the application")
elif face_recog_enabled and models_exist:
    print("✓ Enhanced mode is ENABLED and models are available")
    print("  The system should use enhanced architecture")
else:
    print("ℹ️  Enhanced mode is DISABLED in config")
    print("  The system will use standard mode")
    print("\nTo enable enhanced mode:")
    print("   1. Set startup_mode.enable_face_recognition to true in config")
    print("   2. Ensure model files exist at configured paths")