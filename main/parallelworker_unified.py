# Unified parallel worker supporting both standard and enhanced modes

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
import os
warnings.filterwarnings("ignore", category=UserWarning)

# Import enhanced mode components conditionally
try:
    from retinaface_detector import RetinaFaceDetector
    from lightweight_tracker import LightweightTracker
    from roi_processor import ROIProcessor
    from face_recognition_process import FaceRecognitionProcess
    from enrollment_manager import EnrollmentManager
    ENHANCED_COMPONENTS_AVAILABLE = True
except ImportError:
    ENHANCED_COMPONENTS_AVAILABLE = False
    print("[ParallelWorker] Enhanced components not available, will use standard mode only")


def robust_initialize_camera(camera_index, fps, resolution, settle_time=2.0, target_brightness_range=(30, 200), max_attempts=1):
    """Initialize camera with robust settings and FPS validation"""
    # Try different backends for max compatibility
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    cap = None

    for b in backends:
        test_cap = cv2.VideoCapture(camera_index, b)
        if test_cap.isOpened():
            # Optional: open camera settings dialog if supported
            try:
                test_cap.set(cv2.CAP_PROP_SETTINGS, 1)
                print("[Camera] Opened camera settings dialog (if supported).")
            except Exception as e:
                print(f"[Camera] Settings dialog not supported: {e}")
            # Check if we can actually grab a frame
            ret, _ = test_cap.read()
            if ret:
                cap = test_cap
                print(f"[Camera] Using backend: {b}")
                break
            else:
                test_cap.release()
    if cap is None:
        raise RuntimeError(f"Cannot open camera {camera_index!r}")

    print(f"[Camera] Initializing camera {camera_index} for {resolution[0]}x{resolution[1]} @ {fps} FPS...")

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])

    # Check actual resolution
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if (actual_width, actual_height) != resolution:
        print(f"[Camera] Warning: Resolution {resolution} not supported, using {actual_width}x{actual_height}")
        resolution = (actual_width, actual_height)

    cap.set(cv2.CAP_PROP_FPS, fps)
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    if actual_fps > 0 and actual_fps < fps * 0.8:
        print(f"[Camera] Warning: Camera cannot achieve {fps} FPS, actual max is {actual_fps:.1f} FPS")
        fps = int(actual_fps)

    # Shorter settle time for integrated cameras
    if camera_index == 0:  # usually integrated
        settle_time = min(settle_time, 1.5)

    print(f"[Camera] Settling for {settle_time} seconds...")
    
    # Warm-up phase
    warmup_start = time.time()
    while time.time() - warmup_start < 0.5:
        cap.read()

    # Exposure/brightness monitoring
    exposure_readings = []
    brightness_readings = []
    gain_readings = []
    frame_brightnesses = []
    fps_measurements = []

    start_time = time.time()
    frame_count = 0
    last_frame_time = start_time

    while time.time() - start_time < settle_time:
        ret, frame = cap.read()
        if not ret:
            continue

        current_time = time.time()
        frame_count += 1

        # Measure FPS
        if frame_count > 1:
            frame_interval = current_time - last_frame_time
            if frame_interval > 0:
                instant_fps = 1.0 / frame_interval
                fps_measurements.append(instant_fps)
        last_frame_time = current_time

        # Frame brightness
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        frame_brightnesses.append(mean_brightness)

        # Sample camera values periodically
        if frame_count % 10 == 0:
            exposure = cap.get(cv2.CAP_PROP_EXPOSURE)
            brightness = cap.get(cv2.CAP_PROP_BRIGHTNESS)
            gain = cap.get(cv2.CAP_PROP_GAIN)
            if exposure not in [0, -1]:
                exposure_readings.append(exposure)
            if brightness not in [0, -1]:
                brightness_readings.append(brightness)
            if gain not in [0, -1]:
                gain_readings.append(gain)

    # Analyze and print settings
    recent_brightnesses = frame_brightnesses[-max(1, len(frame_brightnesses)//3):]
    mean_brightness = np.mean(recent_brightnesses) if recent_brightnesses else 128
    brightness_std = np.std(recent_brightnesses) if len(recent_brightnesses) > 1 else 0

    if fps_measurements:
        actual_measured_fps = np.median(fps_measurements)
        print(f"[Camera] Measured FPS: {actual_measured_fps:.1f} (requested: {fps})")
        
        # Add warning for low FPS
        if actual_measured_fps < fps * 0.5:  # Less than 50% of requested
            print(f"\n[Camera] ⚠️  WARNING: Low FPS detected!")
            print(f"[Camera] Camera {camera_index} only achieving {actual_measured_fps:.1f} FPS (requested {fps})")
            print(f"[Camera] This is likely a hardware/driver limitation.")
    else:
        actual_measured_fps = fps

    print("\n[Camera] Final settings:")
    settings = {
        'camera_index': camera_index,
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'fps_setting': cap.get(cv2.CAP_PROP_FPS),
        'fps_measured': actual_measured_fps,
        'auto_exposure': cap.get(cv2.CAP_PROP_AUTO_EXPOSURE),
        'exposure': cap.get(cv2.CAP_PROP_EXPOSURE),
        'brightness': cap.get(cv2.CAP_PROP_BRIGHTNESS),
        'gain': cap.get(cv2.CAP_PROP_GAIN),
        'mean_frame_brightness': mean_brightness,
        'brightness_stability': brightness_std,
        'backend': cap.getBackendName()
    }
    for key, value in settings.items():
        print(f"  {key}: {value}")

    return cap


class FrameDistributor:
    """Distributes frames to multiple processing pipelines with dual-stream output"""
    def __init__(self, camera_index, resolution, fps):
        self.camera_index = camera_index
        self.resolution = resolution
        self.fps = fps
        self.subscribers = []
        self.running = False
        self.target_interval = 1.0 / 30.0  # 30Hz distribution
        
    def add_subscriber(self, info):
        """Add a subscriber with its configuration
        info = {
            'queue': queue object,
            'full_res': bool,  # True for full resolution, False for downsampled
            'name': str  # For debugging
        }
        """
        self.subscribers.append(info)
        
    def start(self):
        """Start frame distribution in a separate thread"""
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        
    def stop(self):
        """Stop frame distribution"""
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=2.0)
            
    def _capture_loop(self):
        """Capture and distribute frames to all subscribers"""
        cap = robust_initialize_camera(self.camera_index, self.fps, self.resolution)
        
        # Downsampled resolution for pose and preview
        downscale_width = 640
        downscale_height = 480
        
        frame_count = 0
        last_frame_bgr = None
        
        while self.running:
            ret, frame_bgr = cap.read()
            if ret:
                # Only process new frames (avoid processing duplicates)
                if last_frame_bgr is None or not np.array_equal(frame_bgr, last_frame_bgr):
                    # Convert once
                    rgb_full = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    
                    # Create downsampled version
                    h, w = frame_bgr.shape[:2]
                    if w > downscale_width or h > downscale_height:
                        scale = min(downscale_width / w, downscale_height / h)
                        new_w = int(w * scale)
                        new_h = int(h * scale)
                        bgr_downsampled = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                        rgb_downsampled = cv2.cvtColor(bgr_downsampled, cv2.COLOR_BGR2RGB)
                    else:
                        bgr_downsampled = frame_bgr
                        rgb_downsampled = rgb_full
                    
                    last_frame_bgr = frame_bgr.copy()
                    
                    current_time = time.time()
                    
                    # Prepare frame data
                    frame_data_full = {
                        'bgr': frame_bgr,
                        'rgb': rgb_full,
                        'timestamp': current_time,
                        'frame_id': frame_count
                    }
                    
                    frame_data_downsampled = {
                        'bgr': bgr_downsampled,
                        'rgb': rgb_downsampled,
                        'timestamp': current_time,
                        'frame_id': frame_count
                    }
                    
                    # Send to all subscribers based on their needs
                    for sub in self.subscribers:
                        if not sub['queue'].full():
                            try:
                                if sub['full_res']:
                                    sub['queue'].put_nowait(frame_data_full)
                                else:
                                    sub['queue'].put_nowait(frame_data_downsampled)
                            except:
                                pass
                    
                    frame_count += 1
                    
                    if frame_count % 30 == 0:
                        print(f"[Frame Distributor] Distributed frame {frame_count} to {len(self.subscribers)} subscribers")
            else:
                time.sleep(0.001)
                
        cap.release()


def face_detection_process(detection_queue: MPQueue,
                          detection_result_queue: MPQueue,
                          retinaface_model_path: str,
                          control_pipe):
    """Dedicated process for face detection using RetinaFace"""
    print("[FACE DETECTOR] Starting RetinaFace detection process")
    
    # Initialize detector
    detector = RetinaFaceDetector(
        model_path=retinaface_model_path,
        tile_size=640,
        overlap=0.15,
        confidence_threshold=0.7,
        max_workers=4
    )
    detector.start()
    
    frame_count = 0
    detection_interval = 7  # Detect every N frames
    
    while True:
        # Check for control messages
        if control_pipe.poll():
            msg = control_pipe.recv()
            if msg == 'stop':
                break
        
        # Get frame for detection
        try:
            frame_data = detection_queue.get(timeout=0.1)
        except:
            continue
        
        rgb = frame_data['rgb']
        frame_id = frame_data['frame_id']
        timestamp = frame_data['timestamp']
        
        # Submit for detection at intervals
        if frame_id % detection_interval == 0:
            if detector.submit_frame(rgb, frame_id):
                if frame_id % 30 == 0:
                    print(f"[FACE DETECTOR] Submitted frame {frame_id} for detection")
        
        # Check for detection results
        detection_result = detector.get_detections(timeout=0.001)
        if detection_result:
            det_frame_id, detections = detection_result
            if len(detections) > 0:
                print(f"[FACE DETECTOR] Frame {det_frame_id}: Found {len(detections)} faces")
                
                # Send detection results
                try:
                    detection_result_queue.put_nowait({
                        'frame_id': det_frame_id,
                        'detections': detections,
                        'timestamp': timestamp
                    })
                except:
                    pass
        
        frame_count += 1
    
    # Cleanup
    print("[FACE DETECTOR] Stopping...")
    detector.stop()


def face_landmark_process(roi_queue: MPQueue,
                         landmark_result_queue: MPQueue,
                         model_path: str,
                         enable_mesh: bool,
                         control_pipe):
    """Dedicated process for face landmark extraction from ROIs"""
    print("[FACE LANDMARK] Starting landmark extraction process")
    
    # Initialize MediaPipe
    with open(model_path, 'rb') as f:
        model_buffer = f.read()
    base_opts = python.BaseOptions(model_asset_buffer=model_buffer)
    
    face_options = vision.FaceLandmarkerOptions(
        base_options=base_opts,
        output_face_blendshapes=True,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,  # One face per ROI
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    face_landmarker = vision.FaceLandmarker.create_from_options(face_options)
    
    # Use real-time based timestamp generation for ROIs
    start_time = time.time()
    
    processed_count = 0
    last_debug_time = time.time()
    
    while True:
        # Check for control messages
        if control_pipe.poll():
            msg = control_pipe.recv()
            if msg == 'stop':
                break
            elif isinstance(msg, tuple) and msg[0] == 'set_mesh':
                enable_mesh = msg[1]
                print(f"[FACE LANDMARK] Mesh data {'enabled' if enable_mesh else 'disabled'}")
        
        # Get ROI
        try:
            roi_data = roi_queue.get(timeout=0.1)
        except:
            continue
        
        roi = roi_data['roi']
        track_id = roi_data['track_id']
        roi_timestamp = roi_data['timestamp']
        transform = roi_data['transform']
        quality_score = roi_data.get('quality_score', 0.5)
        
        # Generate timestamp based on when we process the ROI
        current_time = time.time()
        timestamp_ms = int((current_time - start_time) * 1000)
        
        # Process ROI
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=roi)
        face_result = face_landmarker.detect_for_video(mp_image, timestamp_ms)
        
        processed_count += 1
        
        # Debug output
        if processed_count % 30 == 0:
            print(f"[FACE LANDMARK] Processed {processed_count} ROIs, landmarks found: {len(face_result.face_landmarks) if face_result.face_landmarks else 0}")
        
        if face_result.face_landmarks:
            # Process first face in ROI
            face_landmarks = face_result.face_landmarks[0]
            blendshapes = face_result.face_blendshapes[0] if face_result.face_blendshapes else []
            
            # Transform landmarks back to original coordinates
            landmarks = []
            for lm in face_landmarks:
                x = lm.x * transform['scale'] + transform['offset_x']
                y = lm.y * transform['scale'] + transform['offset_y']
                landmarks.append((x, y, lm.z))
            
            # Calculate centroid
            x_coords = [l[0] for l in landmarks]
            y_coords = [l[1] for l in landmarks]
            centroid = (np.mean(x_coords), np.mean(y_coords))
            
            # Blend scores
            blend_scores = [b.score for b in blendshapes][:52]
            blend_scores += [0.0] * (52 - len(blend_scores))
            
            # Mesh data
            mesh_data = []
            if enable_mesh:
                for l in landmarks:
                    mesh_data.extend(l)
            
            # Send result
            result = {
                'track_id': track_id,
                'landmarks': landmarks,
                'blend': blend_scores,
                'centroid': centroid,
                'mesh': mesh_data if enable_mesh else None,
                'timestamp': roi_timestamp,
                'quality_score': quality_score
            }
            
            try:
                landmark_result_queue.put_nowait(result)
                processed_count += 1
                
                if processed_count % 30 == 0:
                    print(f"[FACE LANDMARK] Processed {processed_count} ROIs")
            except:
                pass
    
    print("[FACE LANDMARK] Stopping...")


def face_worker_process(frame_queue: MPQueue, 
                       result_queue: MPQueue,
                       model_path: str,
                       enable_mesh: bool,
                       control_pipe,
                       enhanced_mode: bool = False,
                       retinaface_model_path: str = None,
                       arcface_model_path: str = None,
                       enable_recognition: bool = False,
                       detection_interval: int = 7,
                       downscale_resolution: tuple = (640, 480)):
    """
    Unified face worker process supporting both standard and enhanced modes.
    
    In enhanced mode: Manages parallel detection and landmark extraction
    """
    mode = "ENHANCED" if enhanced_mode else "STANDARD"
    print(f"[FACE WORKER] Starting in {mode} mode")
    
    if enhanced_mode and ENHANCED_COMPONENTS_AVAILABLE:
        # Enhanced mode - parallel architecture
        print(f"[FACE WORKER] Initializing enhanced parallel pipeline...")
        
        # Create internal queues
        detection_queue = MPQueue(maxsize=2)
        detection_result_queue = MPQueue(maxsize=5)
        roi_queue = MPQueue(maxsize=10)
        landmark_result_queue = MPQueue(maxsize=10)
        
        # Create control pipes for sub-processes
        detector_parent, detector_child = Pipe()
        landmark_parent, landmark_child = Pipe()
        
        # Start detection process
        detector_proc = Process(
            target=face_detection_process,
            args=(detection_queue, detection_result_queue, retinaface_model_path, detector_child)
        )
        detector_proc.start()
        
        # Start landmark process
        landmark_proc = Process(
            target=face_landmark_process,
            args=(roi_queue, landmark_result_queue, model_path, enable_mesh, landmark_child)
        )
        landmark_proc.start()
        
        # Initialize tracker and ROI processor
        # Adjusted parameters for detection every 7 frames:
        # - min_hits=1: Accept tracks immediately
        # - max_age=21: Keep tracks for 3 detection cycles (7*3)
        # - iou_threshold=0.3: Standard IOU threshold
        tracker = LightweightTracker(
            max_age=21,
            min_hits=1,
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
        if enable_recognition and arcface_model_path and os.path.exists(arcface_model_path):
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
        
        frame_count = 0
        latest_detections = {}
        
        while True:
            # Check for control messages
            if control_pipe.poll():
                msg = control_pipe.recv()
                if msg == 'stop':
                    # Forward stop to sub-processes
                    detector_parent.send('stop')
                    landmark_parent.send('stop')
                    break
                elif isinstance(msg, tuple) and msg[0] == 'set_mesh':
                    # Forward to landmark process
                    landmark_parent.send(msg)
            
            # Get frame
            try:
                frame_data = frame_queue.get(timeout=0.1)
            except:
                continue
            
            rgb = frame_data['rgb']
            timestamp = frame_data['timestamp']
            frame_id = frame_data.get('frame_id', frame_count)
            
            # Send frame to detector (it will handle detection interval)
            try:
                detection_queue.put_nowait(frame_data)
            except:
                pass
            
            # Check for detection results
            try:
                det_result = detection_result_queue.get_nowait()
                latest_detections[det_result['frame_id']] = det_result['detections']
                print(f"[FACE WORKER] Received {len(det_result['detections'])} detections for frame {det_result['frame_id']}")
            except:
                pass
            
            # Get detections for current frame or use empty list
            detections = latest_detections.get(frame_id, [])
            
            # If no detections for this frame, use empty list but still update tracker
            # This allows tracking to continue between detection frames
            tracked_objects = tracker.update(rgb, detections)
            
            if frame_count % 30 == 0:
                print(f"[FACE WORKER] Frame {frame_id}: {len(detections)} detections -> {len(tracked_objects)} tracked objects")
            
            # Extract ROIs for tracked objects
            rois_extracted = 0
            for track_dict in tracked_objects:
                # Get ROI
                roi_result = roi_processor.extract_roi(
                    rgb,
                    track_dict['bbox'],
                    track_dict['track_id'],
                    timestamp
                )
                
                if roi_result and roi_result.get('roi') is not None:
                    rois_extracted += 1
                    # Send ROI to landmark process
                    roi_data = {
                        'roi': roi_result['roi'],
                        'track_id': track_dict['track_id'],
                        'timestamp': timestamp,
                        'transform': roi_result['transform'],
                        'quality_score': roi_result.get('quality_score', 0.5)
                    }
                    
                    try:
                        roi_queue.put_nowait(roi_data)
                    except:
                        pass
                    
                    # Also send to face recognition if enabled
                    if face_recognition:
                        face_recognition.submit_face(
                            roi_result['roi'],
                            track_dict['track_id'],
                            roi_result.get('quality_score', 0.5),
                            timestamp
                        )
            
            if frame_count % 30 == 0 and rois_extracted > 0:
                print(f"[FACE WORKER] Extracted {rois_extracted} ROIs for landmarks")
            
            # Collect landmark results
            face_data = []
            landmarks_collected = 0
            while True:
                try:
                    landmark_result = landmark_result_queue.get_nowait()
                    landmarks_collected += 1
                    
                    # Find corresponding track
                    track_info = None
                    for track in tracked_objects:
                        if track['track_id'] == landmark_result['track_id']:
                            track_info = track
                            break
                    
                    if track_info:
                        face_info = {
                            'track_id': landmark_result['track_id'],
                            'landmarks': landmark_result['landmarks'],
                            'blend': landmark_result['blend'],
                            'centroid': landmark_result['centroid'],
                            'mesh': landmark_result['mesh'],
                            'timestamp': timestamp,
                            'bbox': track_info['bbox'],
                            'confidence': track_info['confidence'],
                            'quality_score': landmark_result['quality_score']
                        }
                        
                        # Add face recognition results if available
                        if face_recognition:
                            recog_result = face_recognition.get_result(timeout=0.001)
                            if recog_result and recog_result['track_id'] == landmark_result['track_id']:
                                face_info['embedding'] = recog_result['embedding']
                                face_info['participant_id'] = recog_result.get('participant_id', -1)
                                
                                # Update enrollment if available
                                if enrollment_manager and recog_result.get('embedding') is not None:
                                    enrollment_manager.add_face_sample(
                                        landmark_result['track_id'],
                                        recog_result['embedding'],
                                        landmark_result['quality_score'],
                                        timestamp
                                    )
                        
                        face_data.append(face_info)
                except:
                    break
            
            if frame_count % 30 == 0:
                print(f"[FACE WORKER] Frame {frame_id}: {len(tracked_objects)} tracked objects, {landmarks_collected} landmark results -> {len(face_data)} faces")
            
            # Create face data from tracked objects even if no landmarks yet
            # This ensures faces are shown immediately while landmarks are processed
            if not face_data and tracked_objects:
                for track in tracked_objects:
                    face_info = {
                        'track_id': track['track_id'],
                        'bbox': track['bbox'],
                        'confidence': track['confidence'],
                        'landmarks': [],  # Will be filled by landmark process later
                        'blend': [0.0] * 52,  # Empty blend scores initially
                        'centroid': ((track['bbox'][0] + track['bbox'][2]) / 2, 
                                   (track['bbox'][1] + track['bbox'][3]) / 2),
                        'timestamp': timestamp
                    }
                    face_data.append(face_info)
            
            # Send results with frame (always send if we have tracked objects)
            if face_data or tracked_objects:
                result = {
                    'type': 'face',
                    'mode': mode,
                    'data': face_data,
                    'timestamp': timestamp,
                    'frame_bgr': frame_data.get('bgr')
                }
                
                try:
                    result_queue.put_nowait(result)
                except:
                    pass
            
            frame_count += 1
            
            # Clean up old detections
            if len(latest_detections) > 100:
                # Keep only recent detections
                sorted_frames = sorted(latest_detections.keys())
                for old_frame in sorted_frames[:-50]:
                    del latest_detections[old_frame]
        
        # Cleanup enhanced mode
        print("[FACE WORKER] Stopping enhanced components...")
        detector_proc.join(timeout=2.0)
        landmark_proc.join(timeout=2.0)
        roi_processor.stop()
        if face_recognition:
            face_recognition.stop()
    
    else:
        # Standard mode processing - direct MediaPipe detection
        # Initialize MediaPipe
        with open(model_path, 'rb') as f:
            model_buffer = f.read()
        base_opts = python.BaseOptions(model_asset_buffer=model_buffer)
        
        face_options = vision.FaceLandmarkerOptions(
            base_options=base_opts,
            output_face_blendshapes=True,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=2,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        face_landmarker = vision.FaceLandmarker.create_from_options(face_options)
        
        # Monotonic timestamp
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
        frame_count = 0
        
        while True:
            # Check for control messages
            if control_pipe.poll():
                msg = control_pipe.recv()
                if msg == 'stop':
                    break
                elif isinstance(msg, tuple) and msg[0] == 'set_mesh':
                    enable_mesh = msg[1]
                    print(f"[FACE WORKER] Mesh data {'enabled' if enable_mesh else 'disabled'}")
            
            # Get frame
            try:
                frame_data = frame_queue.get(timeout=0.1)
            except:
                continue
            
            rgb = frame_data['rgb']
            timestamp = frame_data['timestamp']
            timestamp_ms = ts_gen.next()
            
            # Process
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            face_result = face_landmarker.detect_for_video(mp_image, timestamp_ms)
            
            # Extract data
            face_data = []
            if face_result.face_landmarks:
                for i, face_landmarks in enumerate(face_result.face_landmarks):
                    blendshapes = face_result.face_blendshapes[i] if face_result.face_blendshapes and i < len(face_result.face_blendshapes) else []
                    
                    # Calculate centroid
                    x_coords = [lm.x for lm in face_landmarks]
                    y_coords = [lm.y for lm in face_landmarks]
                    face_centroid = (np.mean(x_coords), np.mean(y_coords))
                    
                    # Blend scores
                    blend_scores = [b.score for b in blendshapes][:52]
                    blend_scores += [0.0] * (52 - len(blend_scores))
                    
                    # Mesh data
                    mesh_data = []
                    if enable_mesh:
                        for lm in face_landmarks:
                            mesh_data.extend([lm.x, lm.y, lm.z])
                    
                    face_data.append({
                        'landmarks': [(lm.x, lm.y, lm.z) for lm in face_landmarks],
                        'blend': blend_scores,
                        'centroid': face_centroid,
                        'mesh': mesh_data if enable_mesh else None,
                        'timestamp': timestamp
                    })
            
            # Send result with frame
            result = {
                'type': 'face',
                'mode': mode,
                'data': face_data,
                'timestamp': timestamp,
                'frame_bgr': frame_data.get('bgr')
            }
            
            try:
                result_queue.put_nowait(result)
            except:
                pass
            
            frame_count += 1
    
    print(f"[FACE WORKER] Stopping {mode} mode...")


def pose_worker_process(frame_queue: MPQueue,
                       result_queue: MPQueue,
                       model_path: str,
                       control_pipe):
    """Dedicated process for pose detection"""
    print("[POSE WORKER] Starting")
    
    # Initialize pose landmarker
    with open(model_path, 'rb') as f:
        model_buffer = f.read()
    base_opts = python.BaseOptions(model_asset_buffer=model_buffer)
    
    pose_options = vision.PoseLandmarkerOptions(
        base_options=base_opts,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=2,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False
    )
    pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)
    
    # Monotonic timestamp
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
    
    # Performance monitoring
    frame_count = 0
    detection_count = 0
    last_fps_report = time.time()
    
    while True:
        # Check for control messages
        if control_pipe.poll():
            msg = control_pipe.recv()
            if msg == 'stop':
                break
            elif msg == 'get_stats':
                elapsed = time.time() - last_fps_report
                if elapsed > 0:
                    stats = {
                        'type': 'pose_stats',
                        'fps': frame_count / elapsed,
                        'detection_rate': detection_count / frame_count if frame_count > 0 else 0
                    }
                    control_pipe.send(stats)
        
        # Get frame
        try:
            frame_data = frame_queue.get(timeout=0.1)
        except:
            continue
        
        rgb = frame_data['rgb']  # Already downsampled by distributor
        timestamp_ms = ts_gen.next()
        
        # Process
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        pose_result = pose_landmarker.detect_for_video(mp_image, timestamp_ms)
        
        # Extract data
        pose_data = []
        if pose_result.pose_landmarks:
            for pose_landmarks in pose_result.pose_landmarks:
                # Calculate centroid from key points
                key_indices = [11, 12, 23, 24]  # shoulders and hips
                key_points = [pose_landmarks[i] for i in key_indices if i < len(pose_landmarks)]
                if key_points:
                    x_coords = [p.x for p in key_points]
                    y_coords = [p.y for p in key_points]
                    pose_centroid = (np.mean(x_coords), np.mean(y_coords))
                else:
                    pose_centroid = (0.5, 0.5)
                
                # Create pose values array with x,y,z,visibility for each landmark
                pose_vals = []
                for lm in pose_landmarks:
                    pose_vals.extend([lm.x, lm.y, lm.z, lm.visibility])
                
                pose_data.append({
                    'landmarks': [(lm.x, lm.y, lm.z) for lm in pose_landmarks],
                    'centroid': pose_centroid,
                    'values': pose_vals,
                    'timestamp': frame_data['timestamp']
                })
            
            detection_count += len(pose_data)
        
        # Send result
        result = {
            'type': 'pose',
            'data': pose_data,
            'timestamp': frame_data['timestamp']
        }
        
        try:
            result_queue.put_nowait(result)
        except:
            pass
        
        frame_count += 1
        
        # Report FPS periodically
        current_time = time.time()
        if current_time - last_fps_report > 10.0:
            elapsed = current_time - last_fps_report
            fps = frame_count / elapsed
            print(f"[POSE WORKER] FPS: {fps:.1f}, Detection rate: {detection_count/frame_count:.2f}")
            frame_count = 0
            detection_count = 0
            last_fps_report = current_time
    
    print("[POSE WORKER] Stopping...")


def fusion_process(face_result_queue: MPQueue,
                  pose_result_queue: MPQueue,
                  preview_queue: queue.Queue,
                  score_buffer: NumpySharedBuffer,
                  result_pipe,
                  recording_queue: queue.Queue,
                  lsl_queue: queue.Queue,
                  participant_update_queue: queue.Queue,
                  worker_pipe,
                  correlation_queue: queue.Queue,
                  cam_idx: int,
                  enable_pose: bool,
                  resolution: tuple):
    """Fuses face and pose results, manages preview and output"""
    print(f"[FUSION] Starting for camera {cam_idx}")
    
    # Debug tracking
    frame_count = 0
    
    # Initialize trackers
    face_tracker = UnifiedTracker(max_participants=10)
    pose_tracker = UnifiedTracker(max_participants=10) if enable_pose else None
    
    # State
    latest_face_data = []
    latest_pose_data = []
    latest_frame_bgr = None
    last_preview_time = 0
    last_lsl_time = 0
    preview_interval = 1.0 / 15  # 15 FPS preview
    lsl_interval = 1.0 / 30  # 30 FPS LSL data (throttled)
    enable_mesh = False  # Track mesh state
    
    # Result collection with timeout
    def collect_results(timeout=0.05):
        """Collect all available results within timeout"""
        face_results = []
        pose_results = []
        
        # Collect face results
        deadline = time.time() + timeout/2
        while time.time() < deadline:
            try:
                result = face_result_queue.get_nowait()
                if result['type'] == 'face':
                    face_results.append(result)
            except:
                break
        
        # Collect pose results if enabled
        if enable_pose:
            deadline = time.time() + timeout/2
            while time.time() < deadline:
                try:
                    result = pose_result_queue.get_nowait()
                    if result['type'] == 'pose':
                        pose_results.append(result)
                except:
                    break
        
        return face_results, pose_results
    
    print("[FUSION] Ready, processing results...")
    
    while True:
        # Check for control messages
        if worker_pipe.poll():
            msg = worker_pipe.recv()
            if msg == 'stop':
                break
            elif isinstance(msg, tuple) and msg[0] == 'set_mesh':
                enable_mesh = msg[1]
                print(f"[FUSION] Mesh data {'enabled' if enable_mesh else 'disabled'}")
        
        # Collect results
        face_results, pose_results = collect_results()
        
        frame_count += 1  # Increment frame counter
        
        # Process face results
        for result in face_results:
            face_data = result['data']
            timestamp = result['timestamp']
            
            # Extract frame if available
            if 'frame_bgr' in result and result['frame_bgr'] is not None:
                latest_frame_bgr = result['frame_bgr']
            
            # Check if this is enhanced mode data
            is_enhanced = any('track_id' in face for face in face_data)
            
            if is_enhanced:
                # Enhanced mode - data already has track IDs
                tracked_faces = []
                for face in face_data:
                    participant_id = face.get('participant_id', -1)
                    if participant_id < 0:
                        # Not yet enrolled, use tracker
                        tracked_face = face_tracker.update_single_detection({
                            'centroid': face['centroid'],
                            'landmarks': face['landmarks'],
                            'local_id': face.get('track_id', -1)
                        })
                        if tracked_face:
                            face['global_id'] = tracked_face['global_id']
                            face['id'] = tracked_face['global_id']  # Add 'id' field for GUI compatibility
                    else:
                        # Use participant ID from face recognition
                        face['global_id'] = participant_id + 1  # Convert to 1-based
                        face['id'] = participant_id + 1  # Add 'id' field for GUI compatibility
                    
                    tracked_faces.append(face)
                
                latest_face_data = tracked_faces
                
                if frame_count % 30 == 0 and len(tracked_faces) > 0:
                    print(f"[FUSION DEBUG] Enhanced mode: {len(tracked_faces)} faces processed")
            else:
                # Standard mode - need tracking
                detections = []
                for i, face in enumerate(face_data):
                    detections.append({
                        'centroid': face['centroid'],
                        'landmarks': face['landmarks'],
                        'local_id': i
                    })
                
                # For now, bypass the UnifiedTracker and send to participant manager directly
                latest_face_data = []
                for i, face in enumerate(face_data):
                    # Send to participant manager for global ID assignment
                    if participant_update_queue:
                        try:
                            update_msg = {
                                'camera_idx': cam_idx,
                                'local_id': i,
                                'centroid': face['centroid'],
                                'landmarks': face['landmarks']
                            }
                            participant_update_queue.put_nowait(update_msg)
                        except:
                            pass
                    
                    # For now, assign simple IDs
                    face['global_id'] = i + 1
                    face['id'] = i + 1  # Add 'id' field for GUI compatibility
                    latest_face_data.append(face)
                
                if frame_count % 30 == 0 and len(latest_face_data) > 0:
                    print(f"[FUSION DEBUG] Standard mode: {len(latest_face_data)} faces processed")
        
        # Process pose results
        if enable_pose:
            for result in pose_results:
                pose_data = result['data']
                
                detections = []
                for i, pose in enumerate(pose_data):
                    detections.append({
                        'centroid': pose['centroid'],
                        'landmarks': pose['landmarks'],
                        'local_id': i
                    })
                
                # For now, bypass tracker for poses too
                latest_pose_data = []
                for i, pose in enumerate(pose_data):
                    # Simple ID assignment
                    pose['global_id'] = i + 1
                    pose['id'] = i + 1  # Add 'id' field for GUI compatibility
                    latest_pose_data.append(pose)
        
        # Send preview at limited rate
        current_time = time.time()
        if current_time - last_preview_time > preview_interval:
            preview_data = {
                'cam_idx': cam_idx,
                'faces': latest_face_data,  # Changed from 'face_data' to 'faces' to match GUI
                'all_poses': latest_pose_data if enable_pose else [],  # Changed from 'pose_data' to 'all_poses'
                'timestamp': current_time,
                'frame_bgr': latest_frame_bgr  # Already downsampled by distributor
            }
            
            if frame_count % 60 == 0:  # Reduce debug frequency
                print(f"[FUSION DEBUG] Sending preview: {len(latest_face_data)} faces, {len(latest_pose_data)} poses, frame: {latest_frame_bgr is not None}")
            
            try:
                preview_queue.put_nowait(preview_data)
                last_preview_time = current_time
            except:
                pass
        
        # Send to other outputs (recording, LSL, correlation) at limited rate
        if (latest_face_data or latest_pose_data) and current_time - last_lsl_time > lsl_interval:
            output_data = {
                'cam_idx': cam_idx,
                'face_data': latest_face_data,
                'pose_data': latest_pose_data,
                'timestamp': current_time
            }
            
            # Recording
            try:
                recording_queue.put_nowait(output_data)
            except:
                pass
            
            # LSL - send individual participant data in expected format
            if latest_face_data:
                for face in latest_face_data:
                    if 'global_id' in face and 'blend' in face:
                        lsl_data = {
                            'type': 'participant_data',
                            'participant_id': f"P{face['global_id']}",
                            'global_id': face['global_id'],
                            'blend_scores': face['blend'],
                            'mesh_data': face.get('mesh', None) if enable_mesh else None
                        }
                        try:
                            lsl_queue.put_nowait(lsl_data)
                        except:
                            pass
            
            # LSL - send pose data if enabled
            if enable_pose and latest_pose_data:
                for pose in latest_pose_data:
                    if 'global_id' in pose and 'values' in pose:
                        pose_lsl_data = {
                            'type': 'pose_data',
                            'participant_id': f"P{pose['global_id']}",
                            'global_id': pose['global_id'],
                            'pose_data': pose['values']
                        }
                        try:
                            lsl_queue.put_nowait(pose_lsl_data)
                        except:
                            pass
            
            # Correlation - send individual face data in expected format
            if latest_face_data:
                for face in latest_face_data:
                    if 'global_id' in face and 'blend' in face:
                        correlation_data = {
                            'participant_id': f"P{face['global_id']}",
                            'blend_scores': face['blend'],
                            'camera_index': cam_idx
                        }
                        try:
                            correlation_queue.put_nowait(correlation_data)
                        except:
                            pass
            
            # Update LSL time after sending
            last_lsl_time = current_time
            
            # Score buffer update
            if latest_face_data:
                # Update blend scores in shared buffer
                for face in latest_face_data:
                    if 'global_id' in face and 'blend' in face:
                        participant_idx = face['global_id'] - 1
                        if hasattr(score_buffer, 'shape') and len(score_buffer.shape) == 2:
                            # Multi-participant buffer
                            if 0 <= participant_idx < score_buffer.shape[1]:
                                blend_scores = np.array(face['blend'][:52])
                                score_buffer.update_column(participant_idx, blend_scores)
                        else:
                            # Single buffer - write first face only
                            score_buffer.write(face['blend'])
                            break
    
    print(f"[FUSION] Stopping for camera {cam_idx}...")


def parallel_participant_worker(cam_idx, face_model_path, pose_model_path,
                               fps, enable_mesh, enable_pose,
                               preview_queue, score_buffer_name, result_pipe,
                               recording_queue, lsl_queue,
                               participant_update_queue,
                               worker_pipe, correlation_queue,
                               max_participants,
                               resolution,
                               enhanced_mode=False,
                               retinaface_model_path=None,
                               arcface_model_path=None,
                               enable_recognition=False):
    """
    Unified parallel participant worker supporting both standard and enhanced modes.
    
    This is the main entry point that spawns face/pose workers and fusion process.
    """
    mode = "ENHANCED" if enhanced_mode else "STANDARD"
    print(f"\n{'='*60}")
    print(f"[PARALLEL WORKER] Camera {cam_idx} starting in {mode} mode")
    print(f"[PARALLEL WORKER] Resolution: {resolution}")
    print(f"[PARALLEL WORKER] FPS: {fps}")
    print(f"[PARALLEL WORKER] Face: {'Enabled' if True else 'Disabled'}")
    print(f"[PARALLEL WORKER] Pose: {'Enabled' if enable_pose else 'Disabled'}")
    if enhanced_mode:
        print(f"[PARALLEL WORKER] Face Recognition: {'Enabled' if enable_recognition else 'Disabled'}")
    print(f"{'='*60}\n")
    
    # Initialize shared buffer
    score_buffer = NumpySharedBuffer(name=score_buffer_name)
    
    # Create queues
    face_frame_queue = MPQueue(maxsize=2)
    pose_frame_queue = MPQueue(maxsize=2) if enable_pose else None
    face_result_queue = MPQueue(maxsize=10)
    pose_result_queue = MPQueue(maxsize=10) if enable_pose else None
    
    # Create control pipes
    face_control_parent, face_control_child = Pipe()
    pose_control_parent, pose_control_child = Pipe() if enable_pose else (None, None)
    
    # Start frame distributor with dual-stream configuration
    distributor = FrameDistributor(cam_idx, resolution, fps)
    
    # Face gets full resolution in enhanced mode, downsampled in standard
    distributor.add_subscriber({
        'queue': face_frame_queue,
        'full_res': enhanced_mode,
        'name': 'face'
    })
    
    # Pose always gets downsampled
    if enable_pose:
        distributor.add_subscriber({
            'queue': pose_frame_queue,
            'full_res': False,
            'name': 'pose'
        })
    
    distributor.start()
    
    # Start worker processes
    processes = []
    
    # Face worker
    face_proc = Process(
        target=face_worker_process,
        args=(face_frame_queue, face_result_queue, face_model_path, enable_mesh, 
              face_control_child, enhanced_mode, retinaface_model_path, 
              arcface_model_path, enable_recognition),
        kwargs={'downscale_resolution': (640, 480)}
    )
    face_proc.start()
    processes.append(face_proc)
    
    # Pose worker
    if enable_pose:
        pose_proc = Process(
            target=pose_worker_process,
            args=(pose_frame_queue, pose_result_queue, pose_model_path, pose_control_child)
        )
        pose_proc.start()
        processes.append(pose_proc)
    
    # Fusion process
    fusion_proc = Process(
        target=fusion_process,
        args=(face_result_queue, pose_result_queue, preview_queue, score_buffer,
              result_pipe, recording_queue, lsl_queue, participant_update_queue,
              worker_pipe, correlation_queue, cam_idx, enable_pose, resolution)
    )
    fusion_proc.start()
    processes.append(fusion_proc)
    
    print(f"[PARALLEL WORKER] All processes started for camera {cam_idx}")
    
    # Monitor loop
    try:
        while True:
            # Check both control pipes (GUI sends on result_pipe)
            if result_pipe.poll():
                msg = result_pipe.recv()
                if msg == 'stop':
                    print(f"[PARALLEL WORKER] Received stop signal for camera {cam_idx}")
                    break
                elif isinstance(msg, tuple) and msg[0] == 'set_mesh':
                    # Forward to face worker
                    enable_mesh = msg[1]
                    face_control_parent.send(msg)
                    print(f"[PARALLEL WORKER] Camera {cam_idx} mesh {'enabled' if enable_mesh else 'disabled'}")
                    
                    # Send config update to LSL
                    lsl_queue.put({
                        'type': 'config_update',
                        'camera_index': cam_idx,
                        'mesh_enabled': enable_mesh
                    })
                elif isinstance(msg, tuple) and msg[0] == 'enable_pose':
                    # Forward to pose worker if it exists
                    if pose_control_parent:
                        pose_control_parent.send(msg)
                elif isinstance(msg, tuple) and msg[0] == 'streaming_state':
                    # When streaming starts, notify LSL about current mesh state
                    streaming_active = msg[1]
                    if streaming_active:
                        lsl_queue.put({
                            'type': 'config_update',
                            'camera_index': cam_idx,
                            'mesh_enabled': enable_mesh
                        })
            
            # Also check worker pipe for participant updates
            if worker_pipe.poll(timeout=0.1):
                msg = worker_pipe.recv()
                # Handle participant updates from GUI
            
            # Check if processes are alive
            for proc in processes:
                if not proc.is_alive():
                    print(f"[PARALLEL WORKER] Process {proc.name} died unexpectedly!")
                    break
    
    except KeyboardInterrupt:
        print(f"[PARALLEL WORKER] Interrupted for camera {cam_idx}")
    
    # Cleanup
    print(f"[PARALLEL WORKER] Stopping camera {cam_idx}...")
    
    # Stop distributor
    distributor.stop()
    
    # Send stop signals
    face_control_parent.send('stop')
    if enable_pose:
        pose_control_parent.send('stop')
    
    # Wait for processes
    for proc in processes:
        proc.join(timeout=2.0)
        if proc.is_alive():
            print(f"[PARALLEL WORKER] Force terminating {proc.name}")
            proc.terminate()
            proc.join()
    
    print(f"[PARALLEL WORKER] Camera {cam_idx} stopped")


# Factory function for backward compatibility
def create_parallel_worker(enhanced_mode=False):
    """Factory function to create appropriate worker based on mode"""
    def worker(*args, **kwargs):
        # Add enhanced mode parameters if not present
        if 'enhanced_mode' not in kwargs:
            kwargs['enhanced_mode'] = enhanced_mode
        return parallel_participant_worker(*args, **kwargs)
    return worker


# Aliases for compatibility
ParallelWorker = create_parallel_worker
parallel_participant_worker_standard = parallel_participant_worker
parallel_participant_worker_advanced = lambda *args, **kwargs: parallel_participant_worker(*args, enhanced_mode=True, **kwargs)