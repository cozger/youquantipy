"""
Wrapper to make advanced detection work with the existing parallel_participant_worker interface.
"""

import cv2
import numpy as np
import time
import json
from sharedbuffer import NumpySharedBuffer
from participantmanager_advanced import GlobalParticipantManagerAdvanced
from confighandler import ConfigHandler

# Try to use enhanced version if available
try:
    import os
    if os.environ.get('USE_ENHANCED_ARCHITECTURE', 'true').lower() == 'true':
        print("[WRAPPER] Loading ENHANCED architecture with parallel landmark workers")
        import parallelworker_advanced_enhanced  # This monkey patches the original
        from parallelworker_advanced import ParallelWorkerAdvanced
    else:
        from parallelworker_advanced import ParallelWorkerAdvanced
except ImportError:
    print("[WRAPPER] Enhanced architecture not available, using standard")
    from parallelworker_advanced import ParallelWorkerAdvanced

def parallel_participant_worker_advanced(cam_idx, face_model_path, pose_model_path,
                                       fps, enable_mesh, enable_pose,
                                       preview_queue, score_buffer_name, result_pipe,
                                       recording_queue, lsl_queue,
                                       participant_update_queue,
                                       worker_pipe, correlation_queue,
                                       max_participants,
                                       resolution):
    """
    Advanced detection worker that matches the original parallel_participant_worker interface.
    """
    print(f"\n[ADVANCED WORKER] Initializing for camera {cam_idx}")
    print(f"[ADVANCED WORKER] Resolution: {resolution[0]}x{resolution[1]}")
    
    # Load configuration for advanced detection
    try:
        config_handler = ConfigHandler()
        advanced_config = config_handler.config.get('advanced_detection', {})
    except:
        advanced_config = {}
    
    # Initialize camera using the robust initialization
    from parallelworker import robust_initialize_camera
    cap = robust_initialize_camera(cam_idx, fps, resolution)
    
    # Score buffer - connect to existing shared memory
    score_buffer = NumpySharedBuffer(size=104, name=score_buffer_name)
    score_buffer_array = score_buffer.arr  # Direct access to numpy array
    
    # Determine downscale resolution (typically 480p for downstream processing)
    downscale_resolution = (640, 480)  # Default 480p
    if resolution[1] <= 480:  # If already 480p or lower, don't downscale
        downscale_resolution = resolution
    
    # Create advanced worker
    worker = ParallelWorkerAdvanced(
        participant_index=cam_idx,
        face_model_path=face_model_path,
        pose_model_path=pose_model_path,
        retinaface_model_path=advanced_config.get('retinaface_model'),
        arcface_model_path=advanced_config.get('arcface_model'),
        enable_recognition=True,
        detection_interval=7,  # ~4Hz detection at 30fps
        downscale_resolution=downscale_resolution
    )
    
    # Set initial states
    worker.set_mesh_enabled(enable_mesh)
    worker.set_pose_enabled(enable_pose)
    
    # Start the worker
    worker.start()
    print(f"[ADVANCED WORKER] Started advanced detection pipeline")
    
    frame_count = 0
    last_fps_time = time.time()
    fps_interval = 5.0
    
    try:
        while True:
            # Check for control commands
            if worker_pipe.poll():
                cmd = worker_pipe.recv()
                if cmd == 'stop':
                    print(f"[ADVANCED WORKER] Received stop command")
                    break
                elif cmd == 'toggle_mesh':
                    enable_mesh = not enable_mesh
                    worker.set_mesh_enabled(enable_mesh)
                elif cmd == 'toggle_pose':
                    enable_pose = not enable_pose
                    worker.set_pose_enabled(enable_pose)
            
            # Get frame from camera
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
            timestamp = time.time()
            
            # Create downscaled frame for display/preview
            frame_downscaled = cv2.resize(frame, downscale_resolution) if frame.shape[:2][::-1] != downscale_resolution else frame
            
            # Prepare frame data (keep full res for detection)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_data = {
                'rgb': rgb,
                'timestamp': timestamp
            }
            
            # Submit to advanced worker
            if worker.submit_frame(frame_data):
                frame_count += 1
            
            # Get results
            results = worker.get_results(timeout=0.001)
            
            for result in results:
                if result['type'] == 'face_advanced':
                    num_faces = len(result['data'])
                    if num_faces > 0:
                        print(f"[ADVANCED WORKER] Detected {num_faces} faces with landmarks")
                    
                    # Process face data
                    face_data = result['data']
                    
                    # Convert to format expected by GUI
                    for face in face_data:
                        # Update participant manager
                        if participant_update_queue:
                            # Convert landmarks to shape for Procrustes
                            shape = None
                            if face.get('landmarks'):
                                shape = [(lm[0], lm[1]) for lm in face['landmarks'][:468]]  # Use face mesh points
                            
                            update_data = {
                                'camera_idx': cam_idx,
                                'local_id': face['track_id'],  # Use track_id as local_id
                                'centroid': face['centroid'],
                                'shape': shape,
                                'bbox': face.get('bbox'),  # Include bbox for advanced features
                            }
                            participant_update_queue.put(update_data)
                        
                        # Send to result pipe with bbox
                        if result_pipe:
                            result_data = {
                                'camera_idx': cam_idx,
                                'face_data': face,
                                'timestamp': timestamp,
                                'bbox': face.get('bbox')  # Include bbox
                            }
                            result_pipe.send(result_data)
                        
                        # Update score buffer (blendshapes)
                        if face.get('blend'):
                            blend_scores = face['blend']
                            score_buffer_array[:len(blend_scores)] = blend_scores
                        
                        # Send to LSL
                        if lsl_queue and face.get('landmarks'):
                            lsl_data = {
                                'type': 'face',
                                'camera_idx': cam_idx,
                                'participant_id': face.get('participant_id', face['track_id']),
                                'landmarks': face['landmarks'],
                                'blend': face.get('blend', []),
                                'timestamp': timestamp
                            }
                            try:
                                lsl_queue.put_nowait(lsl_data)
                            except:
                                pass
                    
                    # Update enrollment status in GUI
                    if result.get('enrollment_status'):
                        print(f"[ADVANCED WORKER] Enrollment status: {result['enrollment_status']}")
                
                elif result['type'] == 'pose':
                    # Handle pose data if needed
                    pass
            
            # Send frame to preview with bboxes drawn
            if preview_queue:
                # Draw bboxes on downscaled frame
                display_frame = frame_downscaled.copy()
                
                # Get latest detection results
                if results:
                    for result in results:
                        if result['type'] == 'face_advanced':
                            for face in result['data']:
                                if 'bbox' in face:
                                    bbox = face['bbox']
                                    # Scale bbox coordinates to match downscaled frame
                                    scale_x = downscale_resolution[0] / resolution[0]
                                    scale_y = downscale_resolution[1] / resolution[1]
                                    x1 = int(bbox[0] * scale_x)
                                    y1 = int(bbox[1] * scale_y)
                                    x2 = int(bbox[2] * scale_x)
                                    y2 = int(bbox[3] * scale_y)
                                    
                                    # Draw bbox
                                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                    
                                    # Draw participant ID
                                    pid = face.get('participant_id', face['track_id'])
                                    label = f"P{pid}"
                                    cv2.putText(display_frame, label, (x1, y1-10),
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                
                                # Draw face landmarks if available
                                if 'landmarks' in face and face['landmarks']:
                                    # Draw face contour points (indices 0-16 for jaw line)
                                    landmarks = face['landmarks']
                                    for i in range(min(17, len(landmarks))):  # Jaw line
                                        x = int(landmarks[i][0] * display_frame.shape[1])
                                        y = int(landmarks[i][1] * display_frame.shape[0])
                                        cv2.circle(display_frame, (x, y), 2, (0, 255, 0), -1)
                                    
                                    # Draw some key facial points
                                    key_points = [
                                        30,   # Nose tip
                                        8,    # Chin
                                        36,   # Left eye corner
                                        45,   # Right eye corner
                                        48,   # Left mouth corner
                                        54    # Right mouth corner
                                    ]
                                    for idx in key_points:
                                        if idx < len(landmarks):
                                            x = int(landmarks[idx][0] * display_frame.shape[1])
                                            y = int(landmarks[idx][1] * display_frame.shape[0])
                                            cv2.circle(display_frame, (x, y), 3, (255, 0, 0), -1)
                
                # Send to preview in expected format
                try:
                    preview_msg = {
                        'frame_bgr': display_frame,
                        'timestamp': timestamp,
                        'camera_idx': cam_idx
                    }
                    preview_queue.put_nowait(preview_msg)
                except:
                    pass
            
            # Send to recording queue
            if recording_queue:
                try:
                    recording_queue.put_nowait((frame, timestamp))
                except:
                    pass
            
            # Performance reporting
            current_time = time.time()
            if current_time - last_fps_time >= fps_interval:
                actual_fps = frame_count / fps_interval
                print(f"[ADVANCED WORKER] Camera {cam_idx}: {actual_fps:.1f} FPS")
                
                # Get worker stats
                stats = worker.get_stats()
                if stats:
                    print(f"[ADVANCED WORKER] Stats: {stats}")
                
                frame_count = 0
                last_fps_time = current_time
                
    except Exception as e:
        print(f"[ADVANCED WORKER] Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        print(f"[ADVANCED WORKER] Shutting down...")
        worker.stop()
        cap.release()
        print(f"[ADVANCED WORKER] Shutdown complete")