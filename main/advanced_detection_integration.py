"""
Seamless integration of advanced detection capabilities using unified modules.
No GUI modifications required - just import and use.
"""

import os
import json
from confighandler import ConfigHandler

# Import unified modules
from participantmanager_unified import GlobalParticipantManager
from parallelworker_unified import create_parallel_worker, parallel_participant_worker

# Pre-declare for forward reference
parallel_participant_worker_enhanced = None

def create_participant_manager(max_participants=10):
    """
    Create appropriate participant manager based on configuration.
    Uses unified manager that automatically adapts to mode.
    """
    try:
        # Use ConfigHandler to ensure consistent config loading
        config_handler = ConfigHandler()
        
        # Check if face recognition is enabled
        face_recognition_enabled = config_handler.get('startup_mode.enable_face_recognition', False)
        if face_recognition_enabled:
            # Check if models exist
            arcface_model = config_handler.get('advanced_detection.arcface_model')
            
            if arcface_model and os.path.exists(arcface_model):
                print("[Integration] Using unified participant manager with face recognition")
                return GlobalParticipantManager(
                    max_participants=max_participants,
                    enable_recognition=True,
                    shape_weight=0.5,
                    position_weight=0.2,
                    recognition_weight=0.3
                )
    except:
        pass
    
    # Default to standard mode
    print("[Integration] Using unified participant manager in standard mode")
    return GlobalParticipantManager(max_participants=max_participants)


def should_use_advanced_detection():
    """Check if advanced detection should be used based on configuration."""
    try:
        # Use ConfigHandler to ensure consistent config loading
        config_handler = ConfigHandler()
        
        # Check if face recognition is enabled
        face_recognition_enabled = config_handler.get('startup_mode.enable_face_recognition', False)
        
        if not face_recognition_enabled:
            print("[CONFIG] Face recognition disabled in config")
            return False
            
        # Check if models exist
        retinaface_model = config_handler.get('advanced_detection.retinaface_model')
        
        if retinaface_model:
            model_exists = os.path.exists(retinaface_model)
            if not model_exists:
                print(f"[CONFIG] RetinaFace model not found at: {retinaface_model}")
            return model_exists
        else:
            print("[CONFIG] No RetinaFace model path configured")
            return False
    except Exception as e:
        print(f"[CONFIG CHECK] Error reading config: {e}")
    
    return False


# For backward compatibility - these can be imported by GUI without changes
ParallelWorker = create_parallel_worker  # Factory function acts as class

# Check if enhanced architecture should be used
use_enhanced = os.environ.get('USE_ENHANCED_ARCHITECTURE', 'true').lower() == 'true'
if use_enhanced and should_use_advanced_detection():
    print("[INTEGRATION] Enhanced architecture is available via unified modules")
    # The unified worker handles both modes internally
    ParallelWorker = lambda: create_parallel_worker(enhanced_mode=True)


def parallel_participant_worker_auto(cam_idx, face_model_path, pose_model_path,
                                    fps, enable_mesh, enable_pose,
                                    preview_queue, score_buffer_name, result_pipe,
                                    recording_queue, lsl_queue,
                                    participant_update_queue,
                                    worker_pipe, correlation_queue,
                                    max_participants,
                                    resolution):
    """
    Drop-in replacement for parallel_participant_worker that uses unified module
    with appropriate mode based on configuration.
    """
    print(f"\n[INTEGRATION] Starting worker for camera {cam_idx}")
    print(f"[INTEGRATION] Resolution: {resolution}")
    print(f"[INTEGRATION] Checking if advanced detection should be used...")
    
    # Get configuration
    config_handler = ConfigHandler()
    enhanced_mode = should_use_advanced_detection()
    
    if enhanced_mode:
        print(f"[INTEGRATION] ✓ ENHANCED MODE ENABLED - Using high-resolution pipeline")
        print(f"[INTEGRATION] Loading RetinaFace + ArcFace models...")
        
        # Get model paths from config
        retinaface_model = config_handler.get('advanced_detection.retinaface_model')
        arcface_model = config_handler.get('advanced_detection.arcface_model')
        
        # Use unified worker in enhanced mode
        return parallel_participant_worker(
            cam_idx, face_model_path, pose_model_path,
            fps, enable_mesh, enable_pose,
            preview_queue, score_buffer_name, result_pipe,
            recording_queue, lsl_queue,
            participant_update_queue,
            worker_pipe, correlation_queue,
            max_participants,
            resolution,
            enhanced_mode=True,
            retinaface_model_path=retinaface_model,
            arcface_model_path=arcface_model,
            enable_recognition=True
        )
    else:
        print(f"[INTEGRATION] Using standard detection pipeline")
        
        # Use unified worker in standard mode
        return parallel_participant_worker(
            cam_idx, face_model_path, pose_model_path,
            fps, enable_mesh, enable_pose,
            preview_queue, score_buffer_name, result_pipe,
            recording_queue, lsl_queue,
            participant_update_queue,
            worker_pipe, correlation_queue,
            max_participants,
            resolution,
            enhanced_mode=False
        )