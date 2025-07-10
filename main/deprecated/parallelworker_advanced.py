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

# Import our new 4K modules
from retinaface_detector import RetinaFaceDetector
from lightweight_tracker import LightweightTracker
from roi_processor import ROIProcessor
from face_recognition_process import FaceRecognitionProcess
from enrollment_manager import EnrollmentManager

def face_worker_advanced_process(frame_queue: MPQueue,
                          result_queue: MPQueue,
                          face_model_path: str,
                          control_pipe,
                          retinaface_model_path: str = None,
                          arcface_model_path: str = None,
                          enable_recognition: bool = True,
                          detection_interval: int = 7,
                          downscale_resolution: tuple = (640, 480)):
    """Advanced face detection worker with dynamic enrollment for high-resolution video"""
    print("\n" + "="*60)
    print("[ADVANCED FACE WORKER] Starting with dynamic face recognition")
    print(f"[ADVANCED FACE WORKER] RetinaFace model: {retinaface_model_path}")
    print(f"[ADVANCED FACE WORKER] ArcFace model: {arcface_model_path}")
    print(f"[ADVANCED FACE WORKER] Recognition enabled: {enable_recognition}")
    print("="*60 + "\n")
    
    # Initialize components
    detector = RetinaFaceDetector(
        model_path=retinaface_model_path or "retinaface.onnx",
        tile_size=640,
        overlap=0.15,  # Reduced overlap for fewer tiles
        confidence_threshold=0.7,  # Moderate threshold to balance detections
        max_workers=4  # Increased workers for better parallelism
    )
    detector.start()
    
    tracker = LightweightTracker(
        max_age=21,  # Keep tracks for 3 detection cycles (7 frames × 3)
        min_hits=1,  # Accept tracks immediately instead of waiting
        iou_threshold=0.3
    )
    
    roi_processor = ROIProcessor(
        target_size=(256, 256),
        padding_ratio=0.3,
        min_quality_score=0.5
    )
    roi_processor.start()
    
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
        
        # Set up enrollment callbacks
        def on_enrollment_complete(participant_id, enrollment):
            print(f"[4K FACE WORKER] Participant {participant_id} enrolled successfully")
            # Send embeddings to face recognition
            embeddings = enrollment_manager.get_enrolled_embeddings(participant_id)
            if embeddings:
                face_recognition.enroll_participant(participant_id, [])
                for emb in embeddings:
                    face_recognition.update_participant(participant_id, emb, 0.9)
        
        enrollment_manager.on_enrollment_complete = on_enrollment_complete
    
    # Initialize MediaPipe for landmark detection
    face_base_opts = python.BaseOptions(
        model_asset_path=face_model_path
    )
    face_options = vision.FaceLandmarkerOptions(
        base_options=face_base_opts,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=10,  # Support more faces for 4K
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=True
    )
    face_landmarker = vision.FaceLandmarker.create_from_options(face_options)
    
    # State
    enable_mesh = False
    frame_id = 0
    
    # Performance monitoring
    frame_count = 0
    detection_count = 0
    last_fps_report = time.time()
    fps_report_interval = 10.0
    
    # Monotonic timestamp for MediaPipe
    class MonotonicTS:
        def __init__(self):
            self.ts = int(time.monotonic() * 1000)
            self.lock = threading.Lock()
        def next(self):
            with self.lock:
                now = int(time.monotonic() * 1000)
                if now <= self.ts:
                    self.ts += 1
                else:
                    self.ts = now
                return self.ts
    
    ts_gen = MonotonicTS()
    
    while True:
        # Check for control messages
        if control_pipe.poll():
            msg = control_pipe.recv()
            if msg == 'stop':
                break
            elif isinstance(msg, tuple) and msg[0] == 'set_mesh':
                enable_mesh = msg[1]
                print(f"[ADVANCED FACE WORKER] Mesh data {'enabled' if enable_mesh else 'disabled'}")
            elif msg == 'get_stats':
                stats = {
                    'type': 'face_stats_advanced',
                    'fps': frame_count / (time.time() - last_fps_report) if time.time() > last_fps_report else 0,
                    'detection_rate': detection_count / frame_count if frame_count > 0 else 0,
                    'detector_stats': detector.get_stats(),
                    'tracked_faces': len(tracker.tracks)
                }
                control_pipe.send(stats)
        
        # Get frame
        try:
            frame_data = frame_queue.get(timeout=0.1)
        except:
            continue
        
        rgb = frame_data['rgb']
        bgr = frame_data.get('bgr')  # Get BGR frame for preview
        timestamp = frame_data['timestamp']
        frame_id += 1
        
        # Debug BGR frame availability
        if frame_id % 100 == 0:
            print(f"[ADVANCED FACE WORKER] Frame {frame_id}: RGB shape={rgb.shape}, BGR available={bgr is not None}, BGR shape={bgr.shape if bgr is not None else 'None'}")
        
        # Create downscaled version for everything except face detection
        rgb_downscaled = cv2.resize(rgb, downscale_resolution) if rgb.shape[:2] != downscale_resolution else rgb
        
        # Log resolution info once
        if frame_id == 1:
            print(f"[ADVANCED FACE WORKER] Input resolution: {rgb.shape[:2]}, Downscaled: {rgb_downscaled.shape[:2]}")
            print(f"[ADVANCED FACE WORKER] Detection interval: every {detection_interval} frames (~{30/detection_interval:.1f}Hz)")
        
        # Submit to RetinaFace detector only at specified interval (for ~4Hz detection)
        # Skip detection if we have stable tracks to save processing
        should_detect = frame_id % detection_interval == 0
        if should_detect and len(tracker.tracks) > 0:
            # Check if all tracks are stable (high confidence, recently updated)
            stable_tracks = all(t.hit_streak > 5 and t.confidence > 0.8 for t in tracker.tracks)
            if stable_tracks:
                should_detect = frame_id % (detection_interval * 3) == 0  # Reduce frequency further
        
        if should_detect:
            if detector.submit_frame(rgb, frame_id):
                if frame_id % 30 == 0:  # Every 30 frames
                    print(f"[ADVANCED FACE WORKER] Submitted frame {frame_id} to detector")
        
        # Calculate scale factors for coordinate transformation
        scale_x = downscale_resolution[0] / rgb.shape[1]
        scale_y = downscale_resolution[1] / rgb.shape[0]
        
        # Get detection results
        detection_result = detector.get_detections(timeout=0.001)
        
        # Process detections if available
        scaled_detections = []
        if detection_result:
            det_frame_id, detections = detection_result
            
            if len(detections) > 0:
                print(f"[ADVANCED FACE WORKER] Frame {det_frame_id}: {len(detections)} detections")
            
            # Scale detections to downscaled resolution for tracker
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
        
        # Always update tracker (with or without detections) for optical flow
        tracked_faces = tracker.update(rgb_downscaled, scaled_detections)
        
        # Cap the number of tracks to prevent runaway accumulation
        max_tracks = 10  # Reasonable maximum for face tracking
        if len(tracker.tracks) > max_tracks:
            # Sort by confidence and age, keep best tracks
            tracker.tracks.sort(key=lambda t: (-t.confidence, -t.hit_streak, t.age))
            tracker.tracks = tracker.tracks[:max_tracks]
        
        # Submit tracks for ROI processing
        if tracked_faces:
            if frame_id % 30 == 0:
                print(f"[ADVANCED FACE WORKER] Tracked faces: {len(tracked_faces)}")
            
            # Scale tracked faces back to original resolution for ROI extraction
            upscaled_tracks = []
            for track in tracked_faces:
                upscaled_track = track.copy()
                upscaled_track['bbox'] = [
                    track['bbox'][0] / scale_x,
                    track['bbox'][1] / scale_y,
                    track['bbox'][2] / scale_x,
                    track['bbox'][3] / scale_y
                ]
                if track.get('landmarks') is not None:
                    upscaled_track['landmarks'] = [[lm[0] / scale_x, lm[1] / scale_y] for lm in track['landmarks']]
                upscaled_tracks.append(upscaled_track)
            
            roi_processor.submit_frame_tracks(rgb, upscaled_tracks, frame_id)
            
            # Process ROIs with MediaPipe
            processed_rois = roi_processor.get_processed_rois(timeout=0.001)
            
            face_data = []
            
            if processed_rois:
                if frame_id % 30 == 0:
                    print(f"[ADVANCED FACE WORKER] Processing {len(processed_rois)} ROIs")
                for roi_data in processed_rois:
                    # Run MediaPipe on ROI
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=roi_data.roi_image)
                    timestamp_ms = ts_gen.next()
                    
                    try:
                        face_result = face_landmarker.detect_for_video(mp_image, timestamp_ms)
                        
                        if face_result.face_landmarks and len(face_result.face_landmarks) > 0:
                            # Take first face in ROI (should be the main one)
                            face_landmarks = face_result.face_landmarks[0]
                            blendshapes = face_result.face_blendshapes[0] if face_result.face_blendshapes else []
                            
                            # Transform landmarks back to original coordinates
                            landmarks_roi = np.array([(lm.x * roi_data.target_size[0], 
                                                      lm.y * roi_data.target_size[1]) 
                                                     for lm in face_landmarks])
                            landmarks_original = roi_data.transform_points(landmarks_roi)
                            
                            # Normalize to 0-1 range based on original frame size
                            h, w = rgb.shape[:2]
                            landmarks_normalized = landmarks_original / np.array([w, h])
                            
                            # Face recognition
                            participant_id = roi_data.track_id  # Default to track_id
                            
                            if face_recognition and enrollment_manager:
                                # Get face for recognition
                                face_roi_recognition = roi_processor.get_roi_for_recognition(roi_data)
                                
                                # Submit for recognition
                                face_recognition.submit_face(
                                    face_roi_recognition,
                                    roi_data.track_id,
                                    roi_data.frame_id,
                                    roi_data.quality_score
                                )
                                
                                # Get recognition result
                                rec_result = face_recognition.get_recognition_result(timeout=0.001)
                                if rec_result:
                                    # Process through enrollment manager
                                    assigned_id = enrollment_manager.process_recognition_result(rec_result)
                                    if assigned_id is not None:
                                        participant_id = assigned_id
                            
                            # Calculate centroid in normalized coordinates
                            face_centroid = np.mean(landmarks_normalized, axis=0)
                            
                            # Blend scores
                            blend_scores = [b.score for b in blendshapes][:52]
                            blend_scores += [0.0] * (52 - len(blend_scores))
                            
                            # Mesh data
                            mesh_data = []
                            if enable_mesh:
                                for i, lm in enumerate(face_landmarks):
                                    x_norm = landmarks_normalized[i, 0]
                                    y_norm = landmarks_normalized[i, 1]
                                    mesh_data.extend([x_norm, y_norm, lm.z])
                            
                            face_data.append({
                                'track_id': roi_data.track_id,
                                'participant_id': participant_id,
                                'bbox': roi_data.original_bbox.tolist(),
                                'landmarks': [(landmarks_normalized[i, 0], 
                                             landmarks_normalized[i, 1], 
                                             face_landmarks[i].z) 
                                            for i in range(len(face_landmarks))],
                                'blend': blend_scores,
                                'centroid': face_centroid.tolist(),
                                'mesh': mesh_data if enable_mesh else None,
                                'timestamp': timestamp,
                                'quality_score': roi_data.quality_score
                            })
                            
                            detection_count += 1
                            
                    except Exception as e:
                        print(f"[ADVANCED FACE WORKER] MediaPipe error: {e}")
            
            # Create face data from tracked objects even without landmarks
            if not face_data and tracked_faces:
                # Send tracked faces without landmarks for immediate overlay
                for track in tracked_faces:
                    face_data.append({
                        'track_id': track['track_id'],
                        'participant_id': -1,  # Not enrolled yet
                        'bbox': track['bbox'].tolist(),
                        'landmarks': [],  # No landmarks yet
                        'blend': [0.0] * 52,
                        'centroid': [(track['bbox'][0] + track['bbox'][2]) / 2,
                                   (track['bbox'][1] + track['bbox'][3]) / 2],
                        'mesh': None,
                        'timestamp': timestamp,
                        'quality_score': 0.0
                    })
            
            # Always send results to ensure frame is displayed
            result = {
                'type': 'face_advanced',
                'data': face_data,
                'timestamp': timestamp,
                'enrollment_status': enrollment_manager.get_all_enrollments() if enrollment_manager else {},
                'frame_bgr': bgr  # Include BGR frame for preview
            }
            
            try:
                result_queue.put_nowait(result)
            except:
                pass
        
        frame_count += 1
        
        # Performance reporting
        current_time = time.time()
        if current_time - last_fps_report >= fps_report_interval:
            if frame_count > 0:
                fps = frame_count / fps_report_interval
                detection_rate = (detection_count / frame_count) * 100
                detector_stats = detector.get_stats()
                print(f"[ADVANCED FACE WORKER] Performance: {fps:.1f} FPS, "
                      f"{detection_rate:.1f}% detection rate, "
                      f"Detector: {detector_stats['avg_fps']:.1f} FPS, "
                      f"Tracking: {len(tracker.tracks)} faces")
            
            frame_count = 0
            detection_count = 0
            last_fps_report = current_time
    
    # Cleanup
    detector.stop()
    roi_processor.stop()
    if face_recognition:
        face_recognition.stop()
    face_landmarker.close()
    print("[ADVANCED FACE WORKER] Stopped")


class ParallelWorkerAdvanced:
    """Advanced parallel worker with integrated face recognition for high-resolution video"""
    
    def __init__(self, participant_index, face_model_path, pose_model_path=None,
                 retinaface_model_path=None, arcface_model_path=None,
                 enable_recognition=True, detection_interval=7, 
                 downscale_resolution=(640, 480)):
        self.participant_index = participant_index
        self.face_model_path = face_model_path
        self.pose_model_path = pose_model_path
        self.retinaface_model_path = retinaface_model_path
        self.arcface_model_path = arcface_model_path
        self.enable_recognition = enable_recognition
        self.detection_interval = detection_interval
        self.downscale_resolution = downscale_resolution
        
        # Queues
        self.frame_queue = MPQueue(maxsize=5)
        self.result_queue = MPQueue(maxsize=10)
        
        # Control
        self.face_control_send, self.face_control_recv = Pipe()
        self.pose_control_send, self.pose_control_recv = Pipe()
        
        # Processes
        self.face_process = None
        self.pose_process = None
        
        # State
        self.running = False
        self.enable_mesh = False
        self.enable_pose = True
        
    def start(self):
        """Start worker processes"""
        if not self.running:
            self.running = True
            
            # Start advanced face worker
            self.face_process = Process(
                target=face_worker_advanced_process,
                args=(self.frame_queue, self.result_queue, self.face_model_path,
                      self.face_control_recv, self.retinaface_model_path,
                      self.arcface_model_path, self.enable_recognition,
                      self.detection_interval, self.downscale_resolution)
            )
            self.face_process.daemon = False  # Changed to allow spawning child processes
            self.face_process.start()
            
            # Start pose worker if enabled
            if self.pose_model_path and self.enable_pose:
                from parallelworker import pose_worker_process
                self.pose_process = Process(
                    target=pose_worker_process,
                    args=(self.frame_queue, self.result_queue, self.pose_model_path,
                          self.pose_control_recv)
                )
                self.pose_process.daemon = False  # Changed to allow proper cleanup
                self.pose_process.start()
            
            print(f"[ParallelWorkerAdvanced] Started for participant {self.participant_index}")
    
    def stop(self):
        """Stop worker processes"""
        if self.running:
            self.running = False
            
            # Send stop signals
            self.face_control_send.send('stop')
            if self.pose_process:
                self.pose_control_send.send('stop')
            
            # Wait for processes
            if self.face_process:
                self.face_process.join(timeout=2.0)
                if self.face_process.is_alive():
                    self.face_process.terminate()
            
            if self.pose_process:
                self.pose_process.join(timeout=2.0)
                if self.pose_process.is_alive():
                    self.pose_process.terminate()
            
            print(f"[ParallelWorkerAdvanced] Stopped for participant {self.participant_index}")
    
    def submit_frame(self, frame_data):
        """Submit frame for processing"""
        if self.running:
            try:
                self.frame_queue.put_nowait(frame_data)
                return True
            except:
                return False
        return False
    
    def get_results(self, timeout=0.001):
        """Get processing results"""
        results = []
        deadline = time.time() + timeout
        
        while time.time() < deadline:
            try:
                result = self.result_queue.get_nowait()
                results.append(result)
            except:
                break
        
        return results
    
    def set_mesh_enabled(self, enabled):
        """Enable/disable mesh data"""
        self.enable_mesh = enabled
        self.face_control_send.send(('set_mesh', enabled))
    
    def set_pose_enabled(self, enabled):
        """Enable/disable pose detection"""
        self.enable_pose = enabled
        if self.pose_process:
            self.pose_control_send.send(('enable_pose', enabled))
    
    def get_stats(self):
        """Get performance statistics"""
        stats = {}
        
        # Request stats from workers
        self.face_control_send.send('get_stats')
        if self.pose_process:
            self.pose_control_send.send('get_stats')
        
        # Collect responses
        deadline = time.time() + 0.1
        while time.time() < deadline:
            if self.face_control_recv.poll(0.01):
                face_stats = self.face_control_recv.recv()
                if isinstance(face_stats, dict):
                    stats['face'] = face_stats
            
            if self.pose_control_recv.poll(0.01):
                pose_stats = self.pose_control_recv.recv()
                if isinstance(pose_stats, dict):
                    stats['pose'] = pose_stats
        
        return stats