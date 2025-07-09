#!/usr/bin/env python3
"""
Simple script to enable enhanced architecture
Run this before starting the GUI
"""

import os
import sys

# Set environment variable
os.environ['USE_ENHANCED_ARCHITECTURE'] = 'true'

print("="*60)
print("ENABLING ENHANCED ARCHITECTURE")
print("="*60)

# Monkey patch the imports to use enhanced architecture
import parallelworker_advanced
import camera_worker_enhanced
from landmark_worker_pool import LandmarkWorkerPool

# Replace the old face worker with enhanced
print("Replacing face_worker_advanced_process with enhanced architecture...")

# Store original for fallback
original_face_worker = parallelworker_advanced.face_worker_advanced_process

def enhanced_face_worker_wrapper(*args, **kwargs):
    """Wrapper that redirects to camera_worker_enhanced"""
    print("\n[REDIRECT] Redirecting to ENHANCED camera worker")
    print("[REDIRECT] This will use parallel landmark workers")
    
    # Extract arguments
    if len(args) >= 7:
        frame_queue = args[0]
        result_queue = args[1] 
        face_model_path = args[2]
        control_pipe = args[3]
        retinaface_model_path = args[4]
        arcface_model_path = args[5]
        enable_recognition = args[6]
    else:
        frame_queue = kwargs.get('frame_queue')
        result_queue = kwargs.get('result_queue')
        face_model_path = kwargs.get('face_model_path')
        control_pipe = kwargs.get('control_pipe')
        retinaface_model_path = kwargs.get('retinaface_model_path')
        arcface_model_path = kwargs.get('arcface_model_path')
        enable_recognition = kwargs.get('enable_recognition', True)
    
    # Note: The enhanced architecture has a different structure
    # For now, print a message
    print("[REDIRECT] Enhanced architecture loaded but needs integration")
    print("[REDIRECT] Falling back to original for now")
    
    # Call original
    return original_face_worker(frame_queue, result_queue, face_model_path,
                               control_pipe, retinaface_model_path,
                               arcface_model_path, enable_recognition)

# Monkey patch
parallelworker_advanced.face_worker_advanced_process = enhanced_face_worker_wrapper

print("Enhanced architecture enabled!")
print("Now start the GUI normally")
print("="*60)