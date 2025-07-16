"""
Enhanced Camera Worker Process
Implements the proposed architecture with RetinaFace detector
"""

import cv2
import numpy as np
import time
from multiprocessing import Process, Queue, Pipe
import queue
from typing import Tuple

# Import new modules
from frame_router import FrameRouterBuffer
from roi_manager import ROIManager
from landmark_worker_pool_adaptive import LandmarkWorkerPoolAdaptive as LandmarkWorkerPool
from result_aggregator import ResultAggregator

# Import existing modules
from retinaface_detector import RetinaFaceDetector
from lightweight_tracker import LightweightTracker
from enrollment_manager import EnrollmentManager
from face_recognition_process import FaceRecognitionProcess
from parallelworker import pose_worker_process, robust_initialize_camera

def camera_worker_enhanced(cam_idx: int,
                         face_model_path: str,
                         pose_model_path: str,
                         fps: int,
                         resolution: Tuple[int, int],
                         preview_queue: Queue,
                         result_pipe,
                         recording_queue: Queue,
                         lsl_queue: Queue,
                         participant_update_queue: Queue,
                         worker_pipe,
                         correlation_queue: Queue,
                         max_participants: int,
                         retinaface_model_path: str = None,
                         arcface_model_path: str = None,
                         enable_recognition: bool = False,
                         enable_mesh: bool = False,
                         enable_pose: bool = True):
    """
    Enhanced camera worker implementing the proposed architecture.
    """
    print(f"\n" + "="*80)
    print(f"[CAMERA WORKER ENHANCED] Starting ENHANCED ARCHITECTURE for camera {cam_idx}")
    print(f"[CAMERA WORKER ENHANCED] Resolution: {resolution}")
    print(f"[CAMERA WORKER ENHANCED] Face recognition: {enable_recognition}")
    print(f"[CAMERA WORKER ENHANCED] Using parallel landmark workers")
    print("="*80)
    
    # Initialize camera
    cap = robust_initialize_camera(cam_idx, fps, resolution)
    if cap is None:
        print(f"[CAMERA WORKER ENHANCED] Failed to initialize camera {cam_idx}")
        return
    
    # Initialize components
    frame_router = FrameRouterBuffer(buffer_size=30, detection_interval=7)
    roi_manager = ROIManager(target_size=(256, 256), padding_ratio=1.5)
    landmark_pool = LandmarkWorkerPool(face_model_path, num_workers=4, enable_mesh=enable_mesh)
    result_aggregator = ResultAggregator(max_history=10)
    
    # Set frame buffer reference
    roi_manager.set_frame_buffer(frame_router)
    
    # Initialize face detector (RetinaFace)
    detector = RetinaFaceDetector(
        model_path=retinaface_model_path or "retinaface.onnx",
        tile_size=640,
        overlap=0.15,
        confidence_threshold=0.3,  # Lowered from 0.7 to match standalone
        nms_threshold=0.4,  # Added NMS threshold
        max_workers=4
    )
    
    # Initialize tracker
    tracker = LightweightTracker(
        max_age=10,
        min_hits=3,
        iou_threshold=0.3
    )
    
    # Initialize face recognition if enabled
    face_recognition = None
    enrollment_manager = None
    if enable_recognition and arcface_model_path:
        face_recognition = FaceRecognitionProcess(
            model_path=arcface_model_path,
            embedding_dim=512,
            similarity_threshold=0.5
        )
        enrollment_manager = EnrollmentManager(
            min_samples_for_enrollment=10,
            min_quality_score=0.7
        )
        face_recognition.start()
    
    # Initialize pose worker if enabled
    pose_process = None
    pose_queue = Queue(maxsize=5)
    pose_result_queue = Queue(maxsize=10)
    pose_control_send, pose_control_recv = Pipe()
    
    if enable_pose and pose_model_path:
        pose_process = Process(
            target=pose_worker_process,
            args=(pose_queue, pose_result_queue, pose_model_path, pose_control_recv)
        )
        pose_process.daemon = False  # Changed to match non-daemon pattern
        pose_process.start()
    
    # Start all components
    roi_manager.start()
    landmark_pool.start()
    detector.start()
    
    # Processing loop
    frame_count = 0
    running = True
    last_fps_time = time.time()
    fps_frame_count = 0
    
    try:
        while running:
            # Check for control commands
            if worker_pipe.poll():
                cmd = worker_pipe.recv()
                if cmd == 'stop':
                    running = False
                    break
                elif cmd == 'toggle_mesh':
                    enable_mesh = not enable_mesh
                    landmark_pool.set_mesh_enabled(enable_mesh)
                elif cmd == 'toggle_pose':
                    enable_pose = not enable_pose
                    if pose_control_send:
                        pose_control_send.send('toggle')
            
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                continue
            
            timestamp = time.time()
            frame_count += 1
            fps_frame_count += 1
            
            # RGB conversion
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Initialize variables for this frame
            pose_results = []
            
            # Add to frame router
            should_detect, should_track, frame_downscaled = frame_router.add_frame(rgb_frame, timestamp)
            
            # Submit to detector if needed
            if should_detect:
                detector.submit_frame(rgb_frame, frame_count)
            
            # Get detection results
            detection_result = detector.get_detections(timeout=0.001)
            detections = []
            if detection_result:
                _, detections = detection_result
            
            # Update tracker (always, for optical flow)
            # Scale detections to downscaled resolution
            scale_x = frame_downscaled.shape[1] / rgb_frame.shape[1]
            scale_y = frame_downscaled.shape[0] / rgb_frame.shape[0]
            
            scaled_detections = []
            for det in detections:
                scaled_det = det.copy()
                scaled_det['bbox'] = [
                    det['bbox'][0] * scale_x,
                    det['bbox'][1] * scale_y,
                    det['bbox'][2] * scale_x,
                    det['bbox'][3] * scale_y
                ]
                if det.get('landmarks') is not None:
                    scaled_det['landmarks'] = det['landmarks'] * np.array([scale_x, scale_y])
                scaled_detections.append(scaled_det)
            
            # Update tracker
            tracked_faces = tracker.update(frame_downscaled, scaled_detections)
            
            # Limit number of tracks to prevent explosion
            if len(tracked_faces) > max_participants * 2:  # Allow some buffer
                print(f"[CAMERA WORKER ENHANCED] Too many tracks ({len(tracked_faces)}), limiting to {max_participants * 2}")
                # Keep only the most confident/recent tracks
                tracked_faces = sorted(tracked_faces, key=lambda x: x.get('confidence', 0), reverse=True)[:max_participants * 2]
                # Clean up tracker
                active_ids = [t['track_id'] for t in tracked_faces]
                tracker.cleanup_inactive_tracks(active_ids)
            
            # Adjust worker pool based on number of tracked faces (no-op in current design)
            num_faces = len(tracked_faces) if tracked_faces else 0
            landmark_pool.adjust_workers(num_faces)
            
            # Submit ROI requests for tracked faces
            if tracked_faces:
                # Scale back to original resolution
                original_tracks = []
                for track in tracked_faces:
                    orig_track = track.copy()
                    orig_track['bbox'] = [
                        track['bbox'][0] / scale_x,
                        track['bbox'][1] / scale_y,
                        track['bbox'][2] / scale_x,
                        track['bbox'][3] / scale_y
                    ]
                    original_tracks.append(orig_track)
                
                # Submit to ROI manager
                roi_manager.submit_batch_requests(frame_count, original_tracks)
            
            # Process ROIs
            roi_results = roi_manager.get_roi_results(timeout=0.001)
            for roi_result in roi_results:
                # Submit to landmark pool
                landmark_pool.submit_roi(roi_result.roi_image, roi_result.track_id, roi_result.frame_id)
                
                # Face recognition if enabled
                if face_recognition and enrollment_manager:
                    face_recognition.submit_face(
                        roi_result.roi_image,
                        roi_result.track_id,
                        roi_result.frame_id,
                        0.8  # Quality score
                    )
            
            # Collect landmark results - increase timeout to prevent queue overflow
            landmark_results = landmark_pool.get_results(timeout=0.01)  # 10ms instead of 1ms
            if len(landmark_results) > 10:  # Too many results, something is wrong
                print(f"[CAMERA WORKER ENHANCED] Warning: {len(landmark_results)} landmark results in one frame")
            for result in landmark_results:
                result_aggregator.add_landmark_result(result, roi_manager)
            
            # Process face recognition results
            if face_recognition:
                rec_results = face_recognition.get_recognition_result(timeout=0.001)
                if rec_results:
                    # Debug logging
                    if frame_count % 100 == 0:  # Log every 100 frames
                        print(f"[CAMERA WORKER ENHANCED] rec_results type: {type(rec_results)}, value: {rec_results[:100] if isinstance(rec_results, str) else rec_results}")
                    
                    # Ensure rec_results is a list
                    if not isinstance(rec_results, list):
                        rec_results = [rec_results] if rec_results else []
                    
                    for rec_result in rec_results:
                        # Ensure rec_result is a dict
                        if isinstance(rec_result, dict) and 'track_id' in rec_result:
                            assigned_id = enrollment_manager.process_recognition_result(rec_result)
                            if assigned_id is not None:
                                # Update track to participant mapping
                                with result_aggregator.mapping_lock:
                                    result_aggregator.track_to_participant[rec_result['track_id']] = assigned_id
            
            # Get unified face data
            face_data = result_aggregator.get_unified_face_data(frame_count, rgb_frame.shape[:2])
            
            # Send results
            if face_data and isinstance(face_data, list):
                result = {
                    'type': 'face',
                    'camera_idx': cam_idx,
                    'data': face_data,
                    'timestamp': timestamp,
                    'frame_id': frame_count
                }
                
                if result_pipe:
                    result_pipe.send(result)
                
                # Send to participant update queue
                for face in face_data:
                    if participant_update_queue and isinstance(face, dict):
                        update_data = {
                            'camera_idx': cam_idx,
                            'local_id': face.get('track_id', -1),
                            'participant_id': face.get('participant_id', -1),
                            'centroid': face.get('centroid', [0, 0]),
                            'landmarks': face.get('landmarks_3d', [])
                        }
                        try:
                            participant_update_queue.put_nowait(update_data)
                        except queue.Full:
                            pass
                
                # Send to LSL
                if lsl_queue:
                    for face in face_data:
                        if isinstance(face, dict):
                            lsl_data = {
                                'type': 'face',
                                'camera_idx': cam_idx,
                                'participant_id': face.get('participant_id', -1),
                                'landmarks': face.get('landmarks_3d', []),
                                'blend': face.get('blendshapes', []),
                                'timestamp': timestamp
                            }
                            try:
                                lsl_queue.put_nowait(lsl_data)
                            except queue.Full:
                                pass
            
            # Submit frame for pose detection
            if enable_pose and pose_queue:
                try:
                    pose_queue.put_nowait({
                        'rgb': frame_downscaled if frame_downscaled.shape[0] < 720 else 
                               cv2.resize(rgb_frame, (640, 480)),
                        'timestamp': timestamp,
                        'frame_id': frame_count
                    })
                except queue.Full:
                    pass
            
            # Collect pose results
            pose_results = []
            if pose_result_queue:
                try:
                    while True:
                        pose_result = pose_result_queue.get_nowait()
                        pose_results.append(pose_result)
                        if result_pipe:
                            result_pipe.send({
                                'type': 'pose',
                                'camera_idx': cam_idx,
                                'data': pose_result,
                                'timestamp': timestamp
                            })
                except queue.Empty:
                    pass
            
            # Send to preview queue in unified format expected by GUI
            if preview_queue and frame_count % 2 == 0:  # Skip every other frame for performance
                # Convert face data to GUI format
                faces_to_draw = []
                for face in face_data:
                    if isinstance(face, dict):
                        face_info = {
                            'id': face.get('participant_id', face.get('track_id', -1)),
                            'local_id': face.get('track_id', -1),
                            'landmarks': face.get('landmarks_3d', []),
                            'centroid': face.get('centroid', [0.5, 0.5]),
                            'blend': face.get('blendshapes', []),
                            'mesh': None  # TODO: Add mesh data if needed
                        }
                        faces_to_draw.append(face_info)
                
                # Get pose data
                pose_data = None
                all_poses = []
                if pose_results:
                    for idx, pose_result in enumerate(pose_results):
                        if pose_result and 'landmarks' in pose_result:
                            pose_info = {
                                'id': idx + 1,
                                'local_id': idx,
                                'landmarks': pose_result['landmarks'],
                                'centroid': pose_result.get('centroid', [0.5, 0.5])
                            }
                            all_poses.append(pose_info)
                            if pose_data is None:
                                pose_data = pose_info
                
                # Create preview data in unified format
                preview_data = {
                    'mode': 'unified',
                    'faces': faces_to_draw,
                    'primary_face': faces_to_draw[0] if faces_to_draw else None,
                    'all_faces': face_data if face_data else [],
                    'pose': pose_data,
                    'all_poses': all_poses,
                    'frame_bgr': frame,
                    'camera_index': cam_idx,
                    'tracker': None  # Enhanced architecture doesn't use UnifiedTracker
                }
                
                # Debug logging
                if frame_count % 30 == 0 and (faces_to_draw or all_poses):
                    print(f"[CAMERA WORKER ENHANCED] Sending preview: {len(faces_to_draw)} faces, {len(all_poses)} poses")
                
                try:
                    preview_queue.put_nowait(preview_data)
                except queue.Full:
                    # Clear old data if queue is full
                    try:
                        preview_queue.get_nowait()  # Remove old frame
                        preview_queue.put_nowait(preview_data)  # Add new frame
                    except:
                        pass
            
            # Send to recording queue
            if recording_queue:
                try:
                    recording_queue.put_nowait((frame, timestamp))
                except queue.Full:
                    pass
            
            # FPS reporting
            current_time = time.time()
            if current_time - last_fps_time >= 2.0:
                fps = fps_frame_count / (current_time - last_fps_time)
                
                # Get pool stats for logging
                pool_stats = landmark_pool.get_stats()
                tracked_count = len(tracker.tracks) if tracker.tracks else 0
                
                print(f"[CAMERA WORKER ENHANCED] Camera {cam_idx}: {fps:.1f} FPS, " +
                      f"Tracked: {tracked_count}, " +
                      f"Workers: {pool_stats.get('workers_alive', 0)}/{pool_stats.get('num_workers', 0)}, " +
                      f"Tasks: {pool_stats.get('active_tasks', 0)} active, " +
                      f"{pool_stats.get('completed_tasks', 0)} done")
                
                # Send stats
                if worker_pipe:
                    stats = {
                        'fps': fps,
                        'detector_stats': detector.get_stats(),
                        'tracked_faces': tracked_count,
                        'landmark_pool_stats': pool_stats,
                        'aggregator_stats': result_aggregator.get_stats()
                    }
                    worker_pipe.send(('stats', stats))
                
                fps_frame_count = 0
                last_fps_time = current_time
            
            # Cleanup old tracks
            active_track_ids = [t.track_id for t in tracker.tracks]
            result_aggregator.cleanup_old_tracks(active_track_ids)
            
            # Mark frame complete
            result_aggregator.mark_frame_complete(frame_count)
    
    finally:
        # Cleanup
        print(f"[CAMERA WORKER ENHANCED] Stopping camera {cam_idx}")
        
        cap.release()
        roi_manager.stop()
        landmark_pool.stop()
        detector.stop()
        
        if face_recognition:
            face_recognition.stop()
        
        if pose_process:
            pose_control_send.send('stop')
            pose_process.join(timeout=2.0)
            if pose_process.is_alive():
                pose_process.terminate()
        
        print(f"[CAMERA WORKER ENHANCED] Camera {cam_idx} stopped")