"""
Integration wrapper for enhanced architecture
Provides backward compatibility with existing system
"""

from camera_worker_enhanced import camera_worker_enhanced
from parallelworker_advanced_wrapper import parallel_participant_worker_advanced
from advanced_detection_integration import should_use_advanced_detection
import os

def parallel_participant_worker_enhanced(cam_idx, face_model_path, pose_model_path,
                                       fps, enable_mesh, enable_pose,
                                       preview_queue, score_buffer_name, result_pipe,
                                       recording_queue, lsl_queue,
                                       participant_update_queue,
                                       worker_pipe, correlation_queue,
                                       max_participants,
                                       resolution):
    """
    Enhanced worker that matches the original parallel_participant_worker interface.
    Automatically selects between old advanced detection or new enhanced architecture.
    """
    print("\n" + "="*80)
    print("[ENHANCED WORKER] parallel_participant_worker_enhanced called!")
    print(f"[ENHANCED WORKER] Camera {cam_idx}, Resolution: {resolution}")
    print("="*80)
    
    # Check if we should use enhanced architecture
    use_enhanced = os.environ.get('USE_ENHANCED_ARCHITECTURE', 'true').lower() == 'true'
    
    print(f"[ENHANCED INTEGRATION] use_enhanced={use_enhanced}, should_use_advanced={should_use_advanced_detection()}")
    
    if use_enhanced and should_use_advanced_detection():
        print(f"[WORKER] Using ENHANCED architecture for camera {cam_idx}")
        print(f"[WORKER] This uses the new parallel landmark worker architecture")
        
        # Get advanced detection config
        try:
            from confighandler import ConfigHandler
            config_handler = ConfigHandler()
            advanced_config = config_handler.config.get('advanced_detection', {})
            
            # Use enhanced camera worker
            camera_worker_enhanced(
                cam_idx=cam_idx,
                face_model_path=face_model_path,
                pose_model_path=pose_model_path,
                fps=fps,
                resolution=resolution,
                preview_queue=preview_queue,
                result_pipe=result_pipe,
                recording_queue=recording_queue,
                lsl_queue=lsl_queue,
                participant_update_queue=participant_update_queue,
                worker_pipe=worker_pipe,
                correlation_queue=correlation_queue,
                max_participants=max_participants,
                retinaface_model_path=advanced_config.get('retinaface_model'),
                arcface_model_path=advanced_config.get('arcface_model'),
                enable_recognition=True,
                enable_mesh=enable_mesh,
                enable_pose=enable_pose
            )
        except Exception as e:
            import traceback
            print(f"[WORKER] Error using enhanced architecture: {e}")
            print(f"[WORKER] Error type: {type(e).__name__}")
            print(f"[WORKER] Traceback:")
            traceback.print_exc()
            print(f"[WORKER] Falling back to advanced detection")
            # Fall back to advanced detection
            parallel_participant_worker_advanced(
                cam_idx, face_model_path, pose_model_path,
                fps, enable_mesh, enable_pose,
                preview_queue, score_buffer_name, result_pipe,
                recording_queue, lsl_queue,
                participant_update_queue,
                worker_pipe, correlation_queue,
                max_participants,
                resolution
            )
    else:
        # Use existing advanced detection
        print(f"[WORKER] Using advanced detection for camera {cam_idx}")
        parallel_participant_worker_advanced(
            cam_idx, face_model_path, pose_model_path,
            fps, enable_mesh, enable_pose,
            preview_queue, score_buffer_name, result_pipe,
            recording_queue, lsl_queue,
            participant_update_queue,
            worker_pipe, correlation_queue,
            max_participants,
            resolution
        )

# Update the create_parallel_worker function to use enhanced architecture
def create_parallel_worker_enhanced():
    """
    Factory function that returns the appropriate worker based on configuration.
    """
    use_enhanced = os.environ.get('USE_ENHANCED_ARCHITECTURE', 'true').lower() == 'true'
    
    if use_enhanced:
        return parallel_participant_worker_enhanced
    else:
        # Import original if needed
        from parallelworker_integration import create_parallel_worker
        return create_parallel_worker()