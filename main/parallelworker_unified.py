# Parallel worker for face detection and processing - GPU ONLY VERSION

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
from frame_distributor import FrameDistributor

import warnings
import os
warnings.filterwarnings("ignore", category=UserWarning)

# Enable enhanced queue debugging
DEBUG_QUEUES = True
QUEUE_STATS_INTERVAL = 5.0  # Print queue stats every 5 seconds

# Import detection components - GPU ONLY
from retinaface_detector_gpu import RetinaFaceDetectorGPU
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


def face_detection_process(detection_queue, detection_result_queue, 
                          retinaface_model_path: str, control_pipe, debug_queue=None):
    """Face detection - GPU ONLY VERSION with frame ID synchronization"""
    print("[FACE DETECTOR] Starting GPU-ONLY face detection")
    
    # Get configuration
    config = ConfigHandler()
    confidence_threshold = config.get('advanced_detection.detection_confidence', 0.3)
    nms_threshold = config.get('advanced_detection.nms_threshold', 0.4)
    
    # Initialize GPU detector - NO FALLBACK
    try:
        detector = RetinaFaceDetectorGPU(
            model_path=retinaface_model_path,
            confidence_threshold=confidence_threshold,
            nms_threshold=nms_threshold,
            debug_queue=debug_queue,
            max_batch_size=4
        )
        print("[FACE DETECTOR] GPU detector initialized successfully")
    except Exception as e:
        print(f"[FACE DETECTOR] FATAL ERROR: GPU detector initialization failed: {e}")
        print("[FACE DETECTOR] This version requires GPU acceleration. Please ensure:")
        print("  1. NVIDIA GPU with CUDA support is available")
        print("  2. TensorRT/ONNX Runtime GPU is properly installed")
        print("  3. CUDA drivers are up to date")
        raise RuntimeError("GPU detector initialization failed - cannot continue")
    
    detector.start()
    
    frame_count = 0
    detection_interval = 3
    latest_frame_id = 0  # Track the actual latest frame ID received
    
    # FIX: Track pending detections to match with results
    pending_detections = {}  # frame_id -> frame_data
    MAX_PENDING = 20  # Don't keep too many pending frames
    
    while True:
        # Check for control messages
        if control_pipe.poll():
            msg = control_pipe.recv()
            if msg == 'stop':
                break
        
        # Get frame - try to get the most recent one by dropping old frames
        frame_data = None
        frames_dropped = 0
        
        # Drop old frames if queue has multiple items
        while True:
            try:
                # Try to get a frame without blocking
                new_frame = detection_queue.get_nowait()
                if frame_data is not None:
                    # We already had a frame, so the previous one was old
                    frames_dropped += 1
                frame_data = new_frame
            except:
                # Queue is empty, use the last frame we got (if any)
                break
        
        # If we still don't have a frame, wait for one
        if frame_data is None:
            try:
                frame_data = detection_queue.get(timeout=0.1)
            except:
                # No frame available, check for detection results
                pass
        
        if frames_dropped > 0 and frame_count % 30 == 0:
            print(f"[FACE DETECTOR] Dropped {frames_dropped} old frames to catch up")
        
        if frame_data is not None:
            frame_count += 1
            
            # Extract frame ID first to track actual frame numbers
            frame_id = frame_data['frame_id']
            
            # Validate frame ID to detect wraparound or reset
            if latest_frame_id > 0 and frame_id < latest_frame_id - 1000:
                print(f"[FACE DETECTOR] WARNING: Frame ID jumped backwards: {latest_frame_id} -> {frame_id}")
                print(f"[FACE DETECTOR] This may indicate frame ID wraparound or reset")
            elif frame_id > latest_frame_id + 1000:
                print(f"[FACE DETECTOR] WARNING: Large frame ID jump: {latest_frame_id} -> {frame_id}")
                print(f"[FACE DETECTOR] Possible frame drops or sync issue")
            
            latest_frame_id = frame_id  # Update latest frame ID
            
            # Extract all necessary data
            if 'rgb_detection' in frame_data:
                rgb_detection = frame_data['rgb_detection']
                detection_scale_x = frame_data['detection_scale_x']
                detection_scale_y = frame_data['detection_scale_y']
                detection_offset_x = frame_data['detection_offset_x']
                detection_offset_y = frame_data['detection_offset_y']
            else:
                # Use full resolution frame
                rgb_detection = frame_data['rgb']
                detection_scale_x = 1.0
                detection_scale_y = 1.0
                detection_offset_x = 0
                detection_offset_y = 0
            
            timestamp = frame_data['timestamp']
            
            # Store frame data for later coordinate transformation
            pending_detections[frame_id] = {
                'timestamp': timestamp,
                'detection_scale_x': detection_scale_x,
                'detection_scale_y': detection_scale_y,
                'detection_offset_x': detection_offset_x,
                'detection_offset_y': detection_offset_y,
                'submitted_at': time.time()
            }
            
            # Clean old pending detections
            if len(pending_detections) > MAX_PENDING:
                oldest_ids = sorted(pending_detections.keys())[:len(pending_detections) - MAX_PENDING]
                for old_id in oldest_ids:
                    del pending_detections[old_id]
                    print(f"[FACE DETECTOR] Dropped pending detection for old frame {old_id}")
            
            # Submit for detection at intervals
            if frame_id % detection_interval == 0:
                detector.submit_frame(rgb_detection, frame_id)
                if frame_id % 60 == 0:
                    print(f"[FACE DETECTOR] Submitted frame {frame_id} for detection: shape={rgb_detection.shape}, " +
                          f"scale=({detection_scale_x:.3f}, {detection_scale_y:.3f})")
        
        # Always check for detection results (even if no new frame)
        detection_result = detector.get_detections(timeout=0.001)
        if detection_result:
            det_frame_id, detections = detection_result
            
            # Look up the pending frame data
            if det_frame_id in pending_detections:
                frame_info = pending_detections[det_frame_id]
                
                # Transform detection coordinates from detection space to original resolution
                for detection in detections:
                    bbox = detection['bbox']
                    # Remove padding offset and scale to original resolution
                    bbox[0] = (bbox[0] - frame_info['detection_offset_x']) * frame_info['detection_scale_x']
                    bbox[1] = (bbox[1] - frame_info['detection_offset_y']) * frame_info['detection_scale_y']
                    bbox[2] = (bbox[2] - frame_info['detection_offset_x']) * frame_info['detection_scale_x']
                    bbox[3] = (bbox[3] - frame_info['detection_offset_y']) * frame_info['detection_scale_y']
                
                # Calculate detection latency
                detection_latency = time.time() - frame_info['submitted_at']
                
                # Send result with proper frame ID and metadata
                result = {
                    'frame_id': det_frame_id,
                    'detections': detections,
                    'timestamp': frame_info['timestamp'],
                    'detection_latency': detection_latency,
                    'current_frame': latest_frame_id  # Use actual frame ID for debugging
                }
                
                # Send to result queue - always send the latest result
                try:
                    # Clear old results first
                    old_results_dropped = 0
                    while not detection_result_queue.empty():
                        try:
                            old_result = detection_result_queue.get_nowait()
                            old_results_dropped += 1
                        except:
                            break
                    
                    if old_results_dropped > 0:
                        print(f"[FACE DETECTOR] Dropped {old_results_dropped} old detection results")
                    
                    detection_result_queue.put_nowait(result)
                    
                    print(f"[FACE DETECTOR] Sent {len(detections)} detections for frame {det_frame_id} " +
                          f"(current frame: {latest_frame_id}, delay: {latest_frame_id - det_frame_id} frames, " +
                          f"latency: {detection_latency*1000:.1f}ms)")
                    
                    # Debug info if requested
                    if debug_queue and len(detections) > 0:
                        debug_info = {
                            'frame_id': det_frame_id,
                            'raw_detections': detections,
                            'detection_delay': latest_frame_id - det_frame_id,
                            'detection_latency_ms': detection_latency * 1000,
                            'current_frame': latest_frame_id
                        }
                        try:
                            debug_queue.put(debug_info, timeout=0.001)
                        except:
                            pass
                    
                except queue.Full:
                    print(f"[FACE DETECTOR] Failed to send detection result - queue full")
                
                # Remove from pending
                del pending_detections[det_frame_id]
                
            else:
                print(f"[FACE DETECTOR] WARNING: Received detection for unknown frame {det_frame_id} " +
                      f"(latest frame: {latest_frame_id}, processed: {frame_count})")
        
        # Periodic status update
        if frame_count % 300 == 0 and frame_count > 0:
            print(f"[FACE DETECTOR] Status: processed {frame_count} frames (latest frame_id: {latest_frame_id}), " +
                  f"{len(pending_detections)} pending detections")
            
            # Check for stale pending detections
            current_time = time.time()
            for fid, info in list(pending_detections.items()):
                age = current_time - info['submitted_at']
                if age > 2.0:  # 2 seconds is too old
                    print(f"[FACE DETECTOR] WARNING: Frame {fid} detection pending for {age:.1f}s")
    
    detector.stop()
    print("[FACE DETECTOR] Stopped")

def face_landmark_process(roi_queue, landmark_result_queue, model_path: str,
                         enable_mesh: bool, control_pipe):
    """Face landmark extraction - fixed for CPU MediaPipe"""
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
        
        # IMPORTANT: Ensure ROI is CPU array and correct format
        if not isinstance(roi, np.ndarray):
            print(f"[FACE LANDMARK] Warning: ROI is not numpy array: {type(roi)}")
            continue
        
        # Ensure contiguous memory and uint8 dtype
        if not roi.flags['C_CONTIGUOUS'] or roi.dtype != np.uint8:
            roi = np.ascontiguousarray(roi, dtype=np.uint8)
        
        # Verify ROI shape
        if roi.shape != (256, 256, 3):
            print(f"[FACE LANDMARK] Warning: Unexpected ROI shape: {roi.shape}")
            continue
        
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
                    y_norm = y_orig / transform.get('frame_height', 720)
                    
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
            import traceback
            traceback.print_exc()
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
    """Face worker process - GPU-ONLY VERSION"""
    print(f"[FACE WORKER] Starting GPU-ONLY worker with max_participants={max_participants}")
    
    # Verify GPU model path is provided
    if not retinaface_model_path:
        raise ValueError("[FACE WORKER] FATAL: retinaface_model_path is required for GPU detection")
    
    # Create internal queues with appropriate sizes
    detection_queue = MPQueue(maxsize=10)  # Reasonable size
    detection_result_queue = MPQueue(maxsize=5)  # Allow multiple results
    roi_queue = MPQueue(maxsize=5)
    landmark_result_queue = MPQueue(maxsize=5)
    debug_detection_queue = MPQueue(maxsize=1)
    
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
    
    # Initialize tracker
    tracker = LightweightTracker(
        max_age=30,  # Increase max age
        min_hits=1,
        iou_threshold=0.4,
        detection_interval=detection_interval,
        min_hits_in_window=1,  # Reduce requirement
        window_cycles=2,  # Shorter window
        max_tracks=max_participants
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
    detection_cache = {}  # frame_id -> (detections, timestamp)
    MAX_CACHE_SIZE = 30  # Keep last 30 frames of detections
    
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
        
        # Get frame - try to get the most recent one
        frame_data = None
        frames_dropped = 0
        
        # Drop old frames if queue has multiple items
        while True:
            try:
                new_frame = frame_queue.get_nowait()
                if frame_data is not None:
                    frames_dropped += 1
                frame_data = new_frame
            except:
                break
        
        # If we still don't have a frame, wait for one
        if frame_data is None:
            try:
                frame_data = frame_queue.get(timeout=0.1)
            except Exception as e:
                continue
        
        if frames_dropped > 0 and frame_count % 30 == 0:
            print(f"[FACE WORKER] Dropped {frames_dropped} old frames to catch up")
        
        rgb = frame_data['rgb']
        timestamp = frame_data['timestamp']
        frame_id = frame_data.get('frame_id', frame_count)
        frame_count += 1
        
        # Send frame to detector - drop old frames if queue is getting full
        try:
            # If queue is more than half full, drop oldest frames
            if detection_queue.qsize() > 5:  # Half of maxsize=10
                dropped = 0
                while detection_queue.qsize() > 3:
                    try:
                        detection_queue.get_nowait()
                        dropped += 1
                    except:
                        break
                if dropped > 0:
                    print(f"[FACE WORKER] Dropped {dropped} old frames from detection queue")
            
            detection_queue.put(frame_data, timeout=0.001)
            if frame_count % 30 == 0:
                print(f"[FACE WORKER] Sent frame {frame_id} to detector, queue size: {detection_queue.qsize()}")
        except queue.Full:
            # Queue is full even after dropping - skip this frame
            if frame_count % 30 == 0:
                print(f"[FACE WORKER] Detection queue full, skipping frame {frame_id}")
        
        # Check for detection results - get ALL available
        results_received = 0
        while True:
            try:
                det_result = detection_result_queue.get_nowait()
                det_frame_id = det_result['frame_id']
                
                # Cache the detection with its frame ID
                detection_cache[det_frame_id] = {
                    'detections': det_result['detections'],
                    'timestamp': det_result.get('timestamp', time.time())
                }
                results_received += 1
                
                print(f"[FACE WORKER] Cached {len(det_result['detections'])} detections for frame {det_frame_id}")
                
            except queue.Empty:
                break
        
        if results_received > 0:
            # Clean old entries by size and age
            current_time = time.time()
            expired_ids = []
            
            # Remove expired entries
            for cache_id, cached in detection_cache.items():
                if current_time - cached['timestamp'] > 1.0:  # Remove entries older than 1 second
                    expired_ids.append(cache_id)
            
            for expired_id in expired_ids:
                del detection_cache[expired_id]
            
            # Then clean by size if needed
            if len(detection_cache) > MAX_CACHE_SIZE:
                oldest_ids = sorted(detection_cache.keys())[:len(detection_cache) - MAX_CACHE_SIZE]
                for old_id in oldest_ids:
                    del detection_cache[old_id]
        
        # Use detections for the CURRENT frame if available
        detections = []
        detection_source = None
        current_time = time.time()
        max_cache_age = 0.5  # 500ms max age for cached detections

        # First try exact match
        if frame_id in detection_cache:
            cached = detection_cache[frame_id]
            # Check if cache entry is still fresh
            if current_time - cached['timestamp'] < max_cache_age:
                detections = cached['detections']
                detection_source = frame_id
        else:
            # Look for closest recent detection
            best_match = None
            min_distance = float('inf')
            
            for cached_id in sorted(detection_cache.keys(), reverse=True):
                if cached_id <= frame_id:  # Only use past detections
                    distance = frame_id - cached_id
                    if distance < min_distance and distance <= 5:  # Max 5 frames old
                        # Check timestamp before considering this cache entry
                        cached = detection_cache[cached_id]
                        if current_time - cached['timestamp'] < max_cache_age:
                            min_distance = distance
                            best_match = cached_id
            
            if best_match is not None:
                cached = detection_cache[best_match]
                detections = cached['detections']
                detection_source = best_match

        if detection_source is not None and len(detections) > 0:
            print(f"[FACE WORKER] Frame {frame_id}: Using {len(detections)} detections from frame {detection_source} (age: {frame_id - detection_source})")
        # Update tracker
        tracked_objects = tracker.update(rgb, detections)
        
        if frame_count % 60 == 0 or (tracked_objects and frame_count < 100):
            print(f"[FACE WORKER] Frame {frame_id}: {len(tracked_objects)} tracked objects, "
                  f"{len(detection_cache)} cached detections")
            
            for obj in tracked_objects:
                print(f"  Track {obj['track_id']}: age={obj.get('age', -1)}, confidence={obj.get('confidence', 0):.3f}")
        
        # Extract ROIs for tracked objects
        rois_extracted = 0
        rois_submitted = 0
        
        for track_dict in tracked_objects:
            # Get ROI
            try:
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
                        'quality_score': roi_result.get('quality_score', 0.5),
                        'frame_id': frame_id
                    }
                    
                    try:
                        roi_queue.put(roi_data, timeout=0.001)
                        rois_submitted += 1
                    except queue.Full:
                        # Try to clear old item
                        try:
                            roi_queue.get_nowait()
                            roi_queue.put_nowait(roi_data)
                            rois_submitted += 1
                        except:
                            pass
                            
            except Exception as e:
                print(f"[FACE WORKER] Exception extracting ROI: {e}")
                import traceback
                traceback.print_exc()
        
        if rois_extracted > 0:
            print(f"[FACE WORKER] Frame {frame_id}: Extracted {rois_extracted} ROIs, submitted {rois_submitted}")
        
        # Collect landmark results
        landmarks_collected = 0
        while True:
            try:
                landmark_result = landmark_result_queue.get_nowait()
                landmarks_collected += 1
                
                # Cache the landmark result
                track_id = landmark_result['track_id']
                landmark_results_cache[track_id] = landmark_result
                
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
            if len(face_data) > 0 and frame_count % 60 == 0:
                print(f"[FACE WORKER] Sent {len(face_data)} faces to fusion")
        except Exception as e:
            if frame_count % 60 == 0:
                print(f"[FACE WORKER] Failed to send result: {e}")
    
    # Cleanup
    print("[FACE WORKER] Stopping components...")
    detector_proc.join(timeout=2.0)
    landmark_proc.join(timeout=2.0)
    roi_processor.stop()
    if face_recognition:
        face_recognition.stop()
    
    print("[FACE WORKER] Stopping...")

def pose_worker_process(frame_queue, result_queue, model_path: str, control_pipe):
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
    """Fusion with proper frame synchronization"""
    print(f"[FUSION] Starting for camera {cam_idx}")
    
    # Initialize participant manager
    global_participant_manager = GlobalParticipantManager(max_participants=max_participants)
    
    # State
    latest_face_data = []
    latest_pose_data = []
    latest_frame_bgr = None
    latest_timestamp = 0
    frame_count = 0
    original_resolution = resolution
    
    # Track when we last sent preview
    last_preview_time = 0
    min_preview_interval = 0.033  # 30 FPS max
    
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
        
        # Get latest frame (don't drain queue completely)
        frame_data = None
        try:
            frame_data = frame_queue.get(timeout=0.01)
        except:
            pass
        
        if frame_data:
            latest_frame_bgr = frame_data.get('bgr')
            latest_timestamp = frame_data['timestamp']
            frame_count += 1
        
        # Get latest face result (don't drain queue)
        try:
            face_result = face_result_queue.get_nowait()
            if face_result and face_result['type'] == 'face':
                face_data = face_result['data']
                tracked_faces = []
                
                for face in face_data:
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
        except:
            pass
        
        # Get latest pose result if enabled
        if enable_pose:
            try:
                pose_result = pose_result_queue.get_nowait()
                if pose_result and pose_result['type'] == 'pose':
                    pose_data = pose_result['data']
                    latest_pose_data = []
                    
                    for i, pose in enumerate(pose_data):
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
            except:
                pass
        
        # Send preview at controlled rate
        if (current_time - last_preview_time > min_preview_interval and 
            latest_frame_bgr is not None):
            
            preview_data = {
                'cam_idx': cam_idx,
                'faces': latest_face_data,
                'all_poses': latest_pose_data if enable_pose else [],
                'timestamp': latest_timestamp,
                'frame_bgr': latest_frame_bgr,
                'original_resolution': original_resolution
            }
            
            # Send preview
            try:
                preview_queue.put(preview_data, timeout=0.001)
                last_preview_time = current_time
            except:
                # If queue is full, try to clear it
                try:
                    preview_queue.get_nowait()
                    preview_queue.put_nowait(preview_data)
                    last_preview_time = current_time
                except:
                    pass
            
            # Send to other outputs
            if latest_face_data:
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
                
                # Update score buffer
                for face in latest_face_data:
                    if 'global_id' in face and 'blend' in face:
                        try:
                            score_buffer.write(face['blend'])
                            break
                        except:
                            pass
        
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
    Parallel participant worker - GPU-ONLY VERSION
    """
    print(f"\n{'='*60}")
    print(f"[PARALLEL WORKER] Camera {cam_idx} starting - GPU ONLY")
    print(f"[PARALLEL WORKER] This version requires GPU acceleration")
    print(f"{'='*60}\n")
    
    # Check GPU model path
    if not retinaface_model_path:
        raise ValueError("[PARALLEL WORKER] FATAL: retinaface_model_path is required for GPU detection")
    
    # Initialize shared buffer
    score_buffer = NumpySharedBuffer(name=score_buffer_name)
    
    # Create frame distributor
    frame_distributor = FrameDistributor(
        camera_index=cam_idx,
        resolution=resolution,
        fps=fps
    )
    
    # Create queues
    face_frame_queue = MPQueue(maxsize=2)
    pose_frame_queue = MPQueue(maxsize=2) if enable_pose else None
    fusion_frame_queue = MPQueue(maxsize=2)
    face_result_queue = MPQueue(maxsize=1)
    pose_result_queue = MPQueue(maxsize=1) if enable_pose else None
    
    # Add subscribers to frame distributor
    frame_distributor.add_subscriber({
        'name': 'face',
        'queue': face_frame_queue,
        'full_res': True,
        'include_bgr': False
    })
    
    if enable_pose:
        frame_distributor.add_subscriber({
            'name': 'pose',
            'queue': pose_frame_queue,
            'full_res': False,
            'include_bgr': False,
            'downsample_to': (640, 480)  # Pose doesn't need full res
        })
    
    frame_distributor.add_subscriber({
        'name': 'fusion',
        'queue': fusion_frame_queue,
        'full_res': True,  # Fusion needs full res for preview
        'include_bgr': True  # Fusion needs BGR for display
    })
    
    # Start frame distribution
    try:
        frame_distributor.start()
        print(f"[PARALLEL WORKER] Frame distributor started for camera {cam_idx}")
    except Exception as e:
        print(f"[PARALLEL WORKER] Failed to start frame distributor: {e}")
        raise
    
    # Create control pipes
    face_control_parent, face_control_child = Pipe()
    pose_control_parent, pose_control_child = Pipe() if enable_pose else (None, None)
    
    # Start face worker process
    face_proc = Process(
        target=face_worker_process,
        args=(face_frame_queue, face_result_queue, face_model_path,
              enable_mesh, face_control_child,
              retinaface_model_path, arcface_model_path, enable_recognition,
              3, resolution, max_participants)
    )
    face_proc.start()
    
    # Start pose worker process if enabled
    if enable_pose:
        pose_proc = Process(
            target=pose_worker_process,
            args=(pose_frame_queue, pose_result_queue, pose_model_path,
                  pose_control_child)
        )
        pose_proc.start()
    
    # Start fusion process
    fusion_proc = Process(
        target=fusion_process,
        args=(face_result_queue, pose_result_queue, fusion_frame_queue,
              preview_queue, score_buffer,
              result_pipe, recording_queue, lsl_queue, participant_update_queue,
              worker_pipe, correlation_queue, cam_idx, enable_pose, resolution, max_participants)
    )
    fusion_proc.start()
    
    print(f"[PARALLEL WORKER] All components started for camera {cam_idx}")
    
    # Monitor loop
    last_health_check = time.time()
    health_check_interval = 5.0
    
    try:
        while True:
            if result_pipe.poll():
                msg = result_pipe.recv()
                if msg == 'stop':
                    break
                elif isinstance(msg, tuple) and msg[0] == 'set_mesh':
                    face_control_parent.send(msg)
                    enable_mesh = msg[1]
                    lsl_queue.put({
                        'type': 'config_update',
                        'camera_index': cam_idx,
                        'mesh_enabled': enable_mesh
                    })
            
            # Periodic health check
            current_time = time.time()
            if current_time - last_health_check > health_check_interval:
                if not frame_distributor.is_healthy():
                    print(f"[PARALLEL WORKER] Frame distributor unhealthy, attempting restart...")
                    frame_distributor.stop()
                    time.sleep(0.5)
                    frame_distributor.start()
                
                # Get stats
                stats = frame_distributor.get_stats()
                if stats['frame_count'] % 300 == 0:  # Every ~10 seconds at 30fps
                    print(f"[PARALLEL WORKER] Camera {cam_idx} stats: {stats['actual_fps']:.1f} FPS, "
                          f"{stats['stats']['queue_drops']} drops")
                
                last_health_check = current_time
            
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        print(f"[PARALLEL WORKER] Interrupted for camera {cam_idx}")
    except Exception as e:
        print(f"[PARALLEL WORKER] Error in monitor loop: {e}")
        import traceback
        traceback.print_exc()
    
    # Cleanup
    print(f"[PARALLEL WORKER] Stopping camera {cam_idx}...")
    
    # Stop frame distributor
    frame_distributor.stop()
    
    # Send stop signals
    face_control_parent.send('stop')
    if enable_pose:
        pose_control_parent.send('stop')
    
    # Wait for processes
    face_proc.join(timeout=2.0)
    if enable_pose:
        pose_proc.join(timeout=2.0)
    
    # Stop fusion
    if worker_pipe:
        try:
            worker_pipe.send('stop')
        except:
            pass
    fusion_proc.terminate()
    fusion_proc.join(timeout=2.0)
    
    print(f"[PARALLEL WORKER] Camera {cam_idx} stopped")
