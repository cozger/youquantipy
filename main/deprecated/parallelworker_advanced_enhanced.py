"""
Enhanced version of parallelworker_advanced using the new architecture
Drop-in replacement that uses landmark worker pool
"""

import cv2
import time
import threading
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from collections import deque
from multiprocessing import Process, Queue as MPQueue, Pipe
import queue
from sharedbuffer import NumpySharedBuffer
from tracker import UnifiedTracker
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Import new architecture components
from frame_router import FrameRouterBuffer
from roi_manager import ROIManager
from landmark_worker_pool_adaptive import LandmarkWorkerPoolAdaptive as LandmarkWorkerPool
from result_aggregator import ResultAggregator
from retinaface_detector import RetinaFaceDetector
from lightweight_tracker import LightweightTracker
from roi_processor import ROIProcessor
from face_recognition_process import FaceRecognitionProcess
from enrollment_manager import EnrollmentManager

def face_worker_advanced_process_enhanced(frame_queue: MPQueue,
                          result_queue: MPQueue,
                          face_model_path: str,
                          control_pipe,
                          retinaface_model_path: str = None,
                          arcface_model_path: str = None,
                          enable_recognition: bool = True,
                          detection_interval: int = 7,
                          downscale_resolution: tuple = (640, 480)):
    """Enhanced face detection worker using parallel landmark workers"""
    print("\n" + "="*60)
    print("[ENHANCED FACE WORKER] Starting with PARALLEL LANDMARK WORKERS")
    print(f"[ENHANCED FACE WORKER] RetinaFace model: {retinaface_model_path}")
    print(f"[ENHANCED FACE WORKER] ArcFace model: {arcface_model_path}")
    print(f"[ENHANCED FACE WORKER] Recognition enabled: {enable_recognition}")
    print(f"[ENHANCED FACE WORKER] Using {4} landmark worker processes")
    print("="*60 + "\n")
    
    # Initialize new architecture components
    frame_router = FrameRouterBuffer(buffer_size=30, detection_interval=detection_interval)
    roi_manager = ROIManager(target_size=(256, 256), padding_ratio=1.5)
    landmark_pool = LandmarkWorkerPool(face_model_path, num_workers=4, enable_mesh=False)
    result_aggregator = ResultAggregator(max_history=10)
    
    # Set frame buffer reference
    roi_manager.set_frame_buffer(frame_router)
    
    # Initialize detector
    detector = RetinaFaceDetector(
        model_path=retinaface_model_path or "retinaface.onnx",
        tile_size=640,
        overlap=0.15,
        confidence_threshold=0.7,
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
        face_recognition.start()
        
        enrollment_manager = EnrollmentManager(
            min_samples_for_enrollment=10,
            min_quality_score=0.7
        )
    
    # Start all components
    roi_manager.start()
    landmark_pool.start()
    detector.start()
    
    print("[ENHANCED FACE WORKER] All components started")
    print(f"[ENHANCED FACE WORKER] ROI Manager: Started")
    print(f"[ENHANCED FACE WORKER] Landmark Pool: {landmark_pool.num_workers} workers")
    print(f"[ENHANCED FACE WORKER] Detector: Started with {detector.max_workers} workers")
    
    # Control flags
    enable_mesh = False
    
    # Performance tracking
    frame_count = 0
    detection_count = 0
    last_fps_report = time.time()
    fps_report_interval = 10.0
    
    try:
        while True:
            # Check control pipe
            if control_pipe.poll():
                cmd = control_pipe.recv()
                if cmd == 'stop':
                    break
                elif cmd == 'toggle_mesh':
                    enable_mesh = not enable_mesh
                    landmark_pool.set_mesh_enabled(enable_mesh)
                    print(f"[ENHANCED FACE WORKER] Mesh data {'enabled' if enable_mesh else 'disabled'}")
            
            # Report stats periodically
            if time.time() - last_fps_report > fps_report_interval:
                stats = {
                    'type': 'face_stats_enhanced',
                    'fps': frame_count / (time.time() - last_fps_report) if time.time() > last_fps_report else 0,
                    'detection_rate': detection_count / frame_count if frame_count > 0 else 0,
                    'detector_stats': detector.get_stats(),
                    'landmark_pool_stats': landmark_pool.get_stats(),
                    'aggregator_stats': result_aggregator.get_stats()
                }
                control_pipe.send(stats)
            
            # Get frame
            try:
                frame_data = frame_queue.get(timeout=0.1)
            except:
                continue
            
            rgb = frame_data['rgb']
            timestamp = frame_data['timestamp']
            frame_count += 1
            
            # Add to frame router
            should_detect, should_track, frame_downscaled = frame_router.add_frame(rgb, timestamp)
            
            # Submit to detector if needed
            if should_detect:
                detector.submit_frame(rgb, frame_count)
                if frame_count % 30 == 0:
                    print(f"[ENHANCED FACE WORKER] Submitted frame {frame_count} to detector")
            
            # Get detection results
            detection_result = detector.get_detections(timeout=0.001)
            detections = []
            if detection_result:
                _, detections = detection_result
                if len(detections) > 0:
                    print(f"[ENHANCED FACE WORKER] Frame {frame_count}: {len(detections)} detections")
            
            # Scale detections for tracker
            scale_x = downscale_resolution[0] / rgb.shape[1]
            scale_y = downscale_resolution[1] / rgb.shape[0]
            
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
            
            # Submit ROI requests
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
                
                roi_manager.submit_batch_requests(frame_count, original_tracks)
            
            # Process ROIs with landmark pool
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
                        0.8
                    )
            
            # Collect landmark results
            landmark_results = landmark_pool.get_results(timeout=0.001)
            if landmark_results and frame_count % 30 == 0:
                print(f"[ENHANCED FACE WORKER] Got {len(landmark_results)} landmark results")
            
            for result in landmark_results:
                result_aggregator.add_landmark_result(result, roi_manager)
                if result.success:
                    detection_count += 1
            
            # Get unified face data
            face_data = result_aggregator.get_unified_face_data(frame_count, rgb.shape[:2])
            
            # Send results
            if face_data:
                result = {
                    'type': 'face_advanced',
                    'data': face_data,
                    'timestamp': timestamp,
                    'frame_id': frame_count
                }
                
                try:
                    result_queue.put_nowait(result)
                except:
                    pass
        
        # Performance reporting
        current_time = time.time()
        if current_time - last_fps_report >= fps_report_interval:
            if frame_count > 0:
                fps = frame_count / fps_report_interval
                detection_rate = (detection_count / frame_count) * 100
                pool_stats = landmark_pool.get_stats()
                print(f"[ENHANCED FACE WORKER] Performance: {fps:.1f} FPS, "
                      f"{detection_rate:.1f}% detection rate, "
                      f"Landmark pool: {pool_stats['pending_tasks']} pending")
            
            frame_count = 0
            detection_count = 0
            last_fps_report = current_time
    
    finally:
        # Cleanup
        print("[ENHANCED FACE WORKER] Stopping...")
        detector.stop()
        roi_manager.stop()
        landmark_pool.stop()
        if face_recognition:
            face_recognition.stop()
        print("[ENHANCED FACE WORKER] Stopped")

# Monkey patch the original to use enhanced version
print("[MODULE] Replacing face_worker_advanced_process with ENHANCED version")
import parallelworker_advanced
parallelworker_advanced.face_worker_advanced_process = face_worker_advanced_process_enhanced