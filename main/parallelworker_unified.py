# Parallel worker for face detection and processing

import cv2
import time
import threading
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from collections import deque
from multiprocessing import Process, Queue as MPQueue, Pipe, SimpleQueue
import queue
from sharedbuffer import NumpySharedBuffer
from tracker import UnifiedTracker
# Removed queue_manager imports - using native queue methods instead
import warnings
import os
warnings.filterwarnings("ignore", category=UserWarning)

# Enable enhanced queue debugging
DEBUG_QUEUES = True
QUEUE_STATS_INTERVAL = 5.0  # Print queue stats every 5 seconds

# Import detection components
from retinaface_detector import RetinaFaceDetector
from lightweight_tracker import LightweightTracker
from roi_processor import ROIProcessor
from face_recognition_process import FaceRecognitionProcess
from enrollment_manager import EnrollmentManager

# Import participant manager for direct access
from participantmanager_unified import GlobalParticipantManager
from confighandler import ConfigHandler

class LatestOnlyQueue:
    """Queue that only keeps the latest item"""
    def __init__(self):
        self._queue = MPQueue(maxsize=1)
        self._lock = threading.Lock()
    
    def put(self, item, block=True, timeout=None):
        """Put item, dropping previous if full"""
        with self._lock:
            # Try to remove old item first
            try:
                self._queue.get_nowait()
            except:
                pass
            # Put new item
            self._queue.put(item, block=False)
    
    def put_nowait(self, item):
        """Put without blocking"""
        self.put(item, block=False)
    
    def get(self, block=True, timeout=None):
        """Get item"""
        return self._queue.get(block=block, timeout=timeout)
    
    def get_nowait(self):
        """Get without blocking"""
        return self._queue.get(block=False, timeout=0)
    
    def qsize(self):
        """Get queue size"""
        return self._queue.qsize()
    
    def empty(self):
        """Check if empty"""
        return self._queue.empty()
    
    def full(self):
        """Check if full"""
        return self._queue.full()


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
        if frame_count % 240 == 0:
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
    """Distributes frames to multiple processing pipelines - LATEST ONLY"""
    def __init__(self, camera_index, resolution, fps):
        self.camera_index = camera_index
        self.resolution = resolution
        self.fps = fps
        self.subscribers = []
        self.running = False
        
    def add_subscriber(self, info):
        """Add a subscriber with its configuration"""
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
            
# In parallelworker_unified.py, update the FrameDistributor._capture_loop method:

    def _capture_loop(self):
        """Capture and distribute frames - LATEST ONLY"""
        cap = robust_initialize_camera(self.camera_index, self.fps, self.resolution)
        
        frame_count = 0
        
        while self.running:
            ret, frame_bgr = cap.read()
            if ret:
                # Convert once
                rgb_full = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                
                # ALWAYS create 480p version for GUI preview
                h, w = frame_bgr.shape[:2]
                gui_max_width = 640
                gui_max_height = 480
                scale = min(gui_max_width / w, gui_max_height / h, 1.0)
                
                if scale < 1.0:
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    bgr_gui = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                    rgb_gui = cv2.cvtColor(bgr_gui, cv2.COLOR_BGR2RGB)
                else:
                    bgr_gui = frame_bgr
                    rgb_gui = rgb_full
                
                current_time = time.time()
                
                # Prepare frame data
                frame_data_full = {
                    'bgr': frame_bgr,
                    'rgb': rgb_full,
                    'timestamp': current_time,
                    'frame_id': frame_count,
                    'original_resolution': (w, h) 
                }
                
                frame_data_gui = {
                    'bgr': bgr_gui,  # Always 480p max
                    'rgb': rgb_gui,  # Always 480p max
                    'timestamp': current_time,
                    'frame_id': frame_count,
                    'original_resolution': (w, h) 
                }
                
                # Send to subscribers
                for sub in self.subscribers:
                    # Fusion/preview always gets GUI resolution
                    if sub['name'] == 'fusion':
                        data = frame_data_gui
                    else:
                        # Face detection gets full resolution
                        data = frame_data_full if sub['full_res'] else frame_data_gui
                    
                    # Always try to send latest frame
                    try:
                        # First try to clear old frame
                        try:
                            sub['queue'].get_nowait()
                        except:
                            pass
                        
                        # Then put new frame
                        sub['queue'].put_nowait(data)
                        
                    except Exception as e:
                        # Queue is busy, skip this frame
                        pass
                
                frame_count += 1
                
                if frame_count % 240 == 0:
                    print(f"[Frame Distributor] Distributed frame {frame_count}")
        
        cap.release()

def face_detection_process(detection_queue, detection_result_queue, 
                          retinaface_model_path: str, control_pipe, debug_queue=None):
    """Face detection - LATEST FRAME ONLY"""
    print("[FACE DETECTOR] Starting with LATEST FRAME ONLY approach")
    
    # Get configuration
    config = ConfigHandler()
    confidence_threshold = config.get('advanced_detection.detection_confidence', 0.3)
    nms_threshold = config.get('advanced_detection.nms_threshold', 0.4)
    
    # Initialize detector
    detector = RetinaFaceDetector(
        model_path=retinaface_model_path,
        confidence_threshold=confidence_threshold,
        nms_threshold=nms_threshold,
        debug_queue=debug_queue
    )
    detector.start()
    
    frame_count = 0
    detection_interval = 7
    
    while True:
        # Check for control messages
        if control_pipe.poll():
            msg = control_pipe.recv()
            if msg == 'stop':
                break
        
        # Always get LATEST frame only
        frame_data = None
        dropped = 0
        
        # Drain queue to get latest
        while True:
            try:
                frame_data = detection_queue.get_nowait()
                dropped += 1
            except:
                break
        
        if frame_data is None:
            # No frame available, wait for one
            try:
                frame_data = detection_queue.get(timeout=0.1)
            except:
                continue
        
        if dropped > 1:
            print(f"[FACE DETECTOR] Dropped {dropped-1} old frames")
        
        rgb = frame_data['rgb']
        frame_id = frame_data['frame_id']
        timestamp = frame_data['timestamp']
        
        # Submit for detection at intervals
        if frame_id % detection_interval == 0:
            detector.submit_frame(rgb, frame_id)
        
        # Check for results
        detection_result = detector.get_detections(timeout=0.001)
        if detection_result:
            det_frame_id, detections = detection_result
            
            # Send result - drop old if queue full
            try:
                # Clear old result
                try:
                    detection_result_queue.get_nowait()
                except:
                    pass
                    
                # Put new result
                detection_result_queue.put_nowait({
                    'frame_id': det_frame_id,
                    'detections': detections,
                    'timestamp': timestamp
                })
            except:
                pass
        
        frame_count += 1
    
    detector.stop()


def face_landmark_process(roi_queue, landmark_result_queue, model_path: str,
                         enable_mesh: bool, control_pipe):
    """Face landmark extraction - fixed queue handling"""
    print("[FACE LANDMARK] Starting landmark extraction process")
    
    # Initialize MediaPipe
    with open(model_path, 'rb') as f:
        model_buffer = f.read()
    base_opts = python.BaseOptions(model_asset_buffer=model_buffer)
    
    face_options = vision.FaceLandmarkerOptions(
        base_options=base_opts,
        output_face_blendshapes=True,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    face_landmarker = vision.FaceLandmarker.create_from_options(face_options)
    
    start_time = time.time()
    processed_count = 0
    failed_count = 0
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
            # Debug output
            current_time = time.time()
            if current_time - last_debug_time > 5.0:
                print(f"[FACE LANDMARK] Processed {processed_count} ROIs, {failed_count} failed")
                last_debug_time = current_time
            continue
        
        roi = roi_data['roi']
        track_id = roi_data['track_id']
        roi_timestamp = roi_data['timestamp']
        transform = roi_data['transform']
        quality_score = roi_data.get('quality_score', 0.5)
        frame_id = roi_data.get('frame_id', -1)
        
        # Generate timestamp
        current_time = time.time()
        timestamp_ms = int((current_time - start_time) * 1000)
        
        try:
            # Process ROI
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=roi)
            face_result = face_landmarker.detect_for_video(mp_image, timestamp_ms)
            
            if face_result.face_landmarks and len(face_result.face_landmarks) > 0:
                # Process first face in ROI
                face_landmarks = face_result.face_landmarks[0]
                blendshapes = face_result.face_blendshapes[0] if face_result.face_blendshapes else []
                
                # Transform landmarks back to original coordinates
                landmarks = []
                for lm in face_landmarks:
                    # Apply transform
                    x = lm.x * roi.shape[1]  # Convert from normalized to pixel coords
                    y = lm.y * roi.shape[0]
                    
                    # Transform to original frame coordinates
                    x_orig = x * transform['scale'] + transform['offset_x']
                    y_orig = y * transform['scale'] + transform['offset_y']
                    
                    # Normalize back to 0-1 range for original frame
                    x_norm = x_orig / transform.get('frame_width', 1280)
                    y_norm = y_orig / transform.get('frame_height', 720) #Defaults for 720p
                    
                    landmarks.append((x_norm, y_norm, lm.z))
                
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
                    'quality_score': quality_score,
                    'frame_id': frame_id
                }
                
                # Always try to send, drop old if needed
                if landmark_result_queue.full():
                    try:
                        landmark_result_queue.get_nowait()  # Drop oldest
                    except:
                        pass
                
                try:
                    landmark_result_queue.put_nowait(result)
                    processed_count += 1
                    
                    if processed_count % 30 == 0:
                        print(f"[FACE LANDMARK] Processed {processed_count} ROIs successfully")
                except Exception as e:
                    print(f"[FACE LANDMARK] Failed to send result: {e}")
            else:
                failed_count += 1
                if failed_count % 30 == 0:
                    print(f"[FACE LANDMARK] No landmarks found in ROI (total failed: {failed_count})")
                    
        except Exception as e:
            print(f"[FACE LANDMARK] Error processing ROI: {e}")
            failed_count += 1
    
    print(f"[FACE LANDMARK] Stopping... Processed {processed_count} ROIs")

def face_worker_process(frame_queue, result_queue, model_path: str,
                       enable_mesh: bool, control_pipe,
                       retinaface_model_path: str = None,
                       arcface_model_path: str = None,
                       enable_recognition: bool = False,
                       detection_interval: int = 7,
                       downscale_resolution: tuple = (640, 480),
                       max_participants: int = 10):
    """Face worker process - fixed ROI extraction"""
    print(f"[FACE WORKER] Starting with max_participants={max_participants}")
    
    # Create internal queues with appropriate sizes
    detection_queue = MPQueue(maxsize=10)
    detection_result_queue = MPQueue(maxsize=10)
    roi_queue = MPQueue(maxsize=10)
    landmark_result_queue = MPQueue(maxsize=10)
    debug_detection_queue = MPQueue(maxsize=10)
    
    # Create control pipes for sub-processes
    detector_parent, detector_child = Pipe()
    landmark_parent, landmark_child = Pipe()
    
    # Start detection process
    detector_proc = Process(
        target=face_detection_process,
        args=(detection_queue, detection_result_queue, retinaface_model_path, detector_child, debug_detection_queue)
    )
    detector_proc.start()
    
    # Start landmark process
    landmark_proc = Process(
        target=face_landmark_process,
        args=(roi_queue, landmark_result_queue, model_path, enable_mesh, landmark_child)
    )
    landmark_proc.start()
    
    # Initialize tracker - THIS WAS MISSING!
    tracker = LightweightTracker(
        max_age=14,
        min_hits=1,
        iou_threshold=0.4,
        detection_interval=detection_interval,
        min_hits_in_window=3,
        window_cycles=4,
        max_tracks=max_participants  # Use the actual max_participants value
    )
    print(f"[FACE WORKER] Tracker initialized with max_tracks={max_participants}")
    
    # Initialize ROI processor
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
    latest_detection_time = 0
    latest_detection_data = []
    
    # Track landmark results
    landmark_results_cache = {}  # track_id -> landmark_data
    
    while True:
        # Check for control messages
        if control_pipe.poll():
            msg = control_pipe.recv()
            if msg == 'stop':
                detector_parent.send('stop')
                landmark_parent.send('stop')
                break
            elif isinstance(msg, tuple) and msg[0] == 'set_mesh':
                landmark_parent.send(msg)
                enable_mesh = msg[1]
        
        # Get frame
        try:
            frame_data = frame_queue.get(timeout=0.1)
        except Exception as e:
            continue
        
        rgb = frame_data['rgb']
        timestamp = frame_data['timestamp']
        frame_id = frame_data.get('frame_id', frame_count)
        
        # Send frame to detector
        if not detection_queue.full():
            try:
                detection_queue.put_nowait(frame_data)
            except:
                pass
        
        # Check for detection results
        try:
            det_result = detection_result_queue.get_nowait()
            latest_detections[det_result['frame_id']] = det_result['detections']
            latest_detection_time = time.time()
            latest_detection_data = det_result['detections']
            print(f"[FACE WORKER] Got {len(det_result['detections'])} detections for frame {det_result['frame_id']}")
        except:
            pass
        
        # Use most recent detections
        current_time = time.time()
        if latest_detection_time > 0 and current_time - latest_detection_time < 0.5 and latest_detection_data:
            detections = latest_detection_data
        else:
            detections = []
        
        # Update tracker
        tracked_objects = tracker.update(rgb, detections)
        
        if tracked_objects and frame_count % 60 == 0:
            print(f"[FACE WORKER] Frame {frame_id}: {len(tracked_objects)} tracked objects")
        
        # Extract ROIs for tracked objects
        rois_extracted = 0
        for track_dict in tracked_objects:
            # Skip if track is too new (not confirmed)
            if track_dict.get('age', 0) < 2:
                continue
                
            # Get ROI
            roi_result = roi_processor.extract_roi(
                rgb,
                track_dict['bbox'],
                track_dict['track_id'],
                timestamp
            )
            
            if roi_result and roi_result.get('roi') is not None:
                rois_extracted += 1

                # DEBUG: Verify ROI dimensions
                roi_shape = roi_result['roi'].shape
                if frame_count % 60 == 0:
                    print(f"[FACE WORKER] ROI extracted: shape={roi_shape}, track_id={track_dict['track_id']}, "
                        f"original_bbox={track_dict['bbox']}, quality={roi_result.get('quality_score', 0):.2f}")

                # Send ROI to landmark process
                roi_data = {
                    'roi': roi_result['roi'],
                    'track_id': track_dict['track_id'],
                    'timestamp': timestamp,
                    'transform': roi_result['transform'],
                    'quality_score': roi_result.get('quality_score', 0.5),
                    'frame_id': frame_id  # Add frame_id for debugging
                }
                
                # Always try to send, drop old if needed
                if roi_queue.full():
                    try:
                        roi_queue.get_nowait()  # Drop oldest
                    except:
                        pass
                
                try:
                    roi_queue.put_nowait(roi_data)
                    if frame_count % 60 == 0:
                        print(f"[FACE WORKER] Sent ROI for track {track_dict['track_id']}")
                except Exception as e:
                    print(f"[FACE WORKER] Failed to send ROI: {e}")
        
        if rois_extracted > 0 and frame_count % 60 == 0:
            print(f"[FACE WORKER] Extracted {rois_extracted} ROIs")
        
        # Collect landmark results
        landmarks_collected = 0
        while True:
            try:
                landmark_result = landmark_result_queue.get_nowait()
                landmarks_collected += 1
                
                # Cache the landmark result
                track_id = landmark_result['track_id']
                landmark_results_cache[track_id] = landmark_result
                
                if frame_count % 60 == 0:
                    print(f"[FACE WORKER] Got landmarks for track {track_id}")
            except:
                break
        
        # Build face data from tracked objects with cached landmarks
        face_data = []
        for track in tracked_objects:
            track_id = track['track_id']
            
            # Get cached landmark data if available
            landmark_data = landmark_results_cache.get(track_id, {})
            
            face_info = {
                'track_id': track_id,
                'bbox': track['bbox'],
                'confidence': track['confidence'],
                'landmarks': landmark_data.get('landmarks', []),
                'blend': landmark_data.get('blend', [0.0] * 52),
                'centroid': landmark_data.get('centroid', 
                           ((track['bbox'][0] + track['bbox'][2]) / 2,
                            (track['bbox'][1] + track['bbox'][3]) / 2)),
                'mesh': landmark_data.get('mesh') if enable_mesh else None,
                'timestamp': timestamp
            }
            face_data.append(face_info)

        if face_data and frame_count % 60 == 0:
            for face in face_data:
                has_landmarks = len(face.get('landmarks', [])) > 0
                print(f"[FACE WORKER] Sending face track_id={face['track_id']}, "
                    f"has_landmarks={has_landmarks}, centroid={face.get('centroid')}")
        
        # Get debug detections
        debug_detections = None
        try:
            debug_data = debug_detection_queue.get_nowait()
            debug_detections = debug_data
        except:
            pass
        
        # Always send results
        bgr_frame = frame_data.get('bgr')
        result = {
            'type': 'face',
            'data': face_data,
            'timestamp': timestamp,
            'frame_bgr': bgr_frame,
            'debug_detections': debug_detections
        }
        
        try:
            result_queue.put_nowait(result)
            if frame_count % 60 == 0:
                print(f"[FACE WORKER] Sent {len(face_data)} faces to fusion")
        except Exception as e:
            if frame_count % 60 == 0:
                print(f"[FACE WORKER] Failed to send result: {e}")
        
        frame_count += 1
    
    # Cleanup
    print("[FACE WORKER] Stopping components...")
    detector_proc.join(timeout=2.0)
    landmark_proc.join(timeout=2.0)
    roi_processor.stop()
    if face_recognition:
        face_recognition.stop()
    
    print("[FACE WORKER] Stopping...")


def pose_worker_process(frame_queue,  # Can be MPQueue or RobustQueue
                       result_queue,  # Can be MPQueue or RobustQueue
                       model_path: str,
                       control_pipe):
    """Dedicated process for pose detection"""
    print("[POSE WORKER] Starting")
    print(f"[POSE WORKER] frame_queue type: {type(frame_queue)}")
    print(f"[POSE WORKER] has get method: {hasattr(frame_queue, 'get')}")
    
    # Initialize pose landmarker
    with open(model_path, 'rb') as f:
        model_buffer = f.read()
    base_opts = python.BaseOptions(model_asset_buffer=model_buffer)
    
    pose_options = vision.PoseLandmarkerOptions(
        base_options=base_opts,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=5,  # Increased from 2 to support more participants
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
            if frame_count == 0:
                print(f"[POSE WORKER] Successfully got first frame")
        except Exception as e:
            if frame_count == 0:
                print(f"[POSE WORKER] Error getting frame: {type(e).__name__}: {e}")
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
        if current_time - last_fps_report > 240.0:
            elapsed = current_time - last_fps_report
            fps = frame_count / elapsed
            print(f"[POSE WORKER] FPS: {fps:.1f}, Detection rate: {detection_count/frame_count:.2f}")
            frame_count = 0
            detection_count = 0
            last_fps_report = current_time
    
    print("[POSE WORKER] Stopping...")


def fusion_process(face_result_queue, pose_result_queue, frame_queue,
                  preview_queue, score_buffer: NumpySharedBuffer,
                  result_pipe, recording_queue, lsl_queue,
                  participant_update_queue, worker_pipe, correlation_queue,
                  cam_idx: int, enable_pose: bool, resolution: tuple,
                  max_participants: int = 10):
    """Fusion with LATEST DATA ONLY approach"""
    print(f"[FUSION] Starting LATEST ONLY for camera {cam_idx}")
    
    # Initialize participant manager
    global_participant_manager = GlobalParticipantManager(max_participants=max_participants)
    
    # State - only keep latest
    latest_face_data = []
    latest_pose_data = []
    latest_frame_bgr = None
    latest_timestamp = 0
    frame_count = 0
    original_resolution = resolution 
    

    
    while True:
        current_time = time.time()
        
        # Check control messages
        if worker_pipe and worker_pipe.poll(timeout=0.001):
            try:
                msg = worker_pipe.recv()
                if msg == 'stop':
                    break
            except:
                pass
        
        # Get LATEST frame
        frame_data = None
        while True:
            try:
                frame_data = frame_queue.get_nowait()
            except:
                break
        
        if frame_data and 'bgr' in frame_data:
            latest_frame_bgr = frame_data['bgr']
            latest_timestamp = frame_data['timestamp']
        
        # Get LATEST face result
        face_result = None
        while True:
            try:
                face_result = face_result_queue.get_nowait()
            except:
                break
                
        if face_result and face_result['type'] == 'face':
            # Process faces with participant manager
            face_data = face_result['data']
            tracked_faces = []
            
            for face in face_data:
                # Your existing face processing logic
                track_id = face.get('track_id', -1)
                
                # Calculate shape from landmarks
                shape = None
                if 'landmarks' in face and face['landmarks']:
                    landmarks = np.array(face['landmarks'])
                    if len(landmarks) > 0:
                        shape = landmarks[:, :2]
                
                # Get global ID
                global_id = global_participant_manager.update_participant(
                    cam_idx, track_id, face['centroid'], shape=shape
                )
                
                face['id'] = global_id
                face['global_id'] = global_id
                tracked_faces.append(face)
            
            latest_face_data = tracked_faces
        
        # Get LATEST pose result if enabled
        if enable_pose:
            pose_result = None
            while True:
                try:
                    pose_result = pose_result_queue.get_nowait()
                except:
                    break
                    
            if pose_result and pose_result['type'] == 'pose':
                # Process poses
                pose_data = pose_result['data']
                latest_pose_data = []
                
                for i, pose in enumerate(pose_data):
                    # Your existing pose processing logic
                    shape = None
                    if 'landmarks' in pose and pose['landmarks']:
                        landmarks = np.array(pose['landmarks'])
                        if len(landmarks) >= 33:
                            key_points = landmarks[[0, 11, 12, 23, 24], :2]
                            shape = key_points
                    
                    global_id = global_participant_manager.update_participant(
                        cam_idx, i, pose['centroid'], shape=shape
                    )
                    
                    pose['id'] = global_id
                    pose['global_id'] = global_id
                    latest_pose_data.append(pose)
        
        # Send preview - ALWAYS LATEST ONLY
        if latest_frame_bgr is not None:
            preview_data = {
                'cam_idx': cam_idx,
                'faces': latest_face_data,
                'all_poses': latest_pose_data if enable_pose else [],
                'timestamp': latest_timestamp,
                'frame_bgr': latest_frame_bgr,
                'original_resolution': original_resolution
            }
            
            # Drop old preview if queue full
            try:
                try:
                    preview_queue.get_nowait()
                except:
                    pass
                preview_queue.put_nowait(preview_data)
            except:
                pass
        
        # Send to other outputs (simplified)
        if latest_face_data:
            # LSL
            for face in latest_face_data:
                if 'global_id' in face and 'blend' in face:
                    lsl_data = {
                        'type': 'participant_data',
                        'participant_id': f"P{face['global_id']}",
                        'global_id': face['global_id'],
                        'blend_scores': face['blend']
                    }
                    try:
                        lsl_queue.put_nowait(lsl_data)
                    except:
                        pass

                if 'landmarks' not in face or not face['landmarks']:
                    print(f"[FUSION] WARNING: Face {face.get('id', 'unknown')} missing landmarks")
                
                # Ensure centroid is present
                if 'centroid' not in face:
                    if 'bbox' in face:
                        bbox = face['bbox']
                        face['centroid'] = ((bbox[0] + bbox[2]) / 2 / resolution[0], 
                                        (bbox[1] + bbox[3]) / 2 / resolution[1])
                    else:
                        face['centroid'] = (0.5, 0.5)
            # Update score buffer
            for face in latest_face_data:
                if 'global_id' in face and 'blend' in face:
                    try:
                        score_buffer.write(face['blend'])
                        break  # Only write first face
                    except:
                        pass
        
        frame_count += 1
        
        if frame_count % 240 == 0:
            print(f"[FUSION] Frame {frame_count}: {len(latest_face_data)} faces, {len(latest_pose_data)} poses")
    
    print(f"[FUSION] Stopping for camera {cam_idx}")

def parallel_participant_worker(cam_idx, face_model_path, pose_model_path,
                               fps, enable_mesh, enable_pose,
                               preview_queue, score_buffer_name, result_pipe,
                               recording_queue, lsl_queue,
                               participant_update_queue,
                               worker_pipe, correlation_queue,
                               max_participants,
                               resolution,
                               retinaface_model_path=None,
                               arcface_model_path=None,
                               enable_recognition=False):
    """
    Parallel participant worker with latest-frame-only approach
    """
    print(f"\n{'='*60}")
    print(f"[PARALLEL WORKER] Camera {cam_idx} starting with LATEST FRAME ONLY")
    print(f"{'='*60}\n")
    
    # Initialize shared buffer
    score_buffer = NumpySharedBuffer(name=score_buffer_name)
    
    # Use queue size 1 for all queues - latest frame only
    face_frame_queue = MPQueue(maxsize=1)
    pose_frame_queue = MPQueue(maxsize=1) if enable_pose else None
    fusion_frame_queue = MPQueue(maxsize=1)
    face_result_queue = MPQueue(maxsize=1)
    pose_result_queue = MPQueue(maxsize=1) if enable_pose else None
        
    # No throttling - process frames at camera framerate
    face_throttler = None
    pose_throttler = None
    
    # Create control pipes
    face_control_parent, face_control_child = Pipe()
    pose_control_parent, pose_control_child = Pipe() if enable_pose else (None, None)
    
    # Start frame distributor with dual-stream configuration
    distributor = FrameDistributor(cam_idx, resolution, fps)
    
    # Face gets full resolution (from spinbox) with no throttling
    distributor.add_subscriber({
    'queue': face_frame_queue,
    'full_res': True,  # Always full resolution from spinbox
    'name': 'face',
    'throttler': None  # No throttling for face detection
})
    # Pose always gets downsampled to 480p max
    if enable_pose:
        distributor.add_subscriber({
            'queue': pose_frame_queue,
            'full_res': False,  # Always downsampled to 480p
            'name': 'pose',
            'throttler': None
        })

    # Fusion/GUI always gets downsampled to 480p for preview
    distributor.add_subscriber({
        'queue': fusion_frame_queue,
        'full_res': False,  # Always downsampled for GUI preview
        'name': 'fusion'
    })
    
    distributor.start()
    
    # Start worker processes
    processes = []
    
    # Face worker
    face_proc = Process(
        target=face_worker_process,
        args=(face_frame_queue, face_result_queue, face_model_path, enable_mesh, 
              face_control_child, retinaface_model_path, 
              arcface_model_path, enable_recognition),
        kwargs={'downscale_resolution': (1920, 1080), 'max_participants': max_participants}
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
        args=(face_result_queue, pose_result_queue, fusion_frame_queue, preview_queue, score_buffer,
              result_pipe, recording_queue, lsl_queue, participant_update_queue,
              worker_pipe, correlation_queue, cam_idx, enable_pose, resolution, max_participants)
    )
    fusion_proc.start()
    processes.append(fusion_proc)
    
    print(f"[PARALLEL WORKER] All processes started for camera {cam_idx}")
    
    # Monitor loop with queue statistics
    last_stats_time = time.time()
    try:
        while True:
            # Print queue statistics periodically
            current_time = time.time()
            if DEBUG_QUEUES and current_time - last_stats_time > QUEUE_STATS_INTERVAL:
                print(f"\n[QUEUE STATS] Camera {cam_idx}:")
                for name, queue in [("face_frame", face_frame_queue),
                                   ("face_result", face_result_queue),
                                   ("fusion_frame", fusion_frame_queue)]:
                    if queue:
                        try:
                            qsize = queue.qsize()
                            full = queue.full()
                            print(f"  {name}: size={qsize}, full={full}")
                        except:
                            print(f"  {name}: status unknown")
                if enable_pose:
                    for name, queue in [("pose_frame", pose_frame_queue),
                                       ("pose_result", pose_result_queue)]:
                        if queue:
                            try:
                                qsize = queue.qsize()
                                full = queue.full()
                                print(f"  {name}: size={qsize}, full={full}")
                            except:
                                print(f"  {name}: status unknown")
                last_stats_time = current_time
            
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


# Aliases for compatibility
ParallelWorker = parallel_participant_worker
parallel_participant_worker_standard = parallel_participant_worker
parallel_participant_worker_advanced = parallel_participant_worker