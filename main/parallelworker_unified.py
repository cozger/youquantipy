# Parallel worker for face detection and processing - GPU ONLY VERSION

import cv2
import time
import threading
import numpy as np
from collections import deque
from multiprocessing import Process, Queue as MPQueue, Pipe, SimpleQueue
import queue
from sharedbuffer import NumpySharedBuffer
from tracker import UnifiedTracker
from frame_distributor import FrameDistributor

# TensorRT imports for GPU landmark processing
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
except ImportError:
    print("ERROR: TensorRT and PyCUDA are required for GPU landmark processing!")
    print("Install with: pip install tensorrt pycuda")
    import sys
    sys.exit(1)

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

# Fixed face_landmark_process that handles 478 landmarks

def face_landmark_process(roi_queue, landmark_result_queue, model_path: str,
                         enable_mesh: bool, control_pipe):
    """Face landmark extraction using TensorRT - handles 478 landmark models"""
    print("[FACE LANDMARK] Starting TensorRT landmark extraction process")
    
    # CRITICAL: Set CUDA device order before any CUDA operations
    import os
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    import time
    import numpy as np
    import queue
    from confighandler import ConfigHandler
    
    # Initialize variables
    cuda_context = None
    landmark_engine = None
    blendshape_engine = None
    landmark_context = None
    blendshape_context = None
    stream = None

    def extract_face_contour_points(landmarks_3d, num_landmarks):
        """
        Extract face contour (jaw line) points that are compatible with the drawing module.
        Returns normalized (0-1) coordinates for the jaw line.
        """
        # Standard MediaPipe jaw line indices (first 17 points form the jaw contour)
        simple_jaw_indices = list(range(17))
        
        face_contour = []
        
        # Extract jaw line points with normalization
        for idx in simple_jaw_indices:
            if idx < len(landmarks_3d) and idx < num_landmarks:
                lm = landmarks_3d[idx]
                # Ensure we have at least x, y coordinates
                if hasattr(lm, '__len__') and len(lm) >= 2:
                    # landmarks_3d contains values in [0, 1] range already
                    x_norm = float(lm[0])
                    y_norm = float(lm[1])
                    face_contour.append((x_norm, y_norm))
        
        return face_contour
    
    try:
        # Import CUDA/TensorRT modules
        import tensorrt as trt
        import pycuda.driver as cuda
        
        # Manual CUDA initialization
        cuda.init()
        
        # Check available devices
        device_count = cuda.Device.count()
        print(f"[FACE LANDMARK] CUDA devices available: {device_count}")
        
        if device_count == 0:
            raise RuntimeError("No CUDA devices available")
        
        # Create context
        cuda_device = cuda.Device(0)
        cuda_context = cuda_device.make_context(flags=cuda.ctx_flags.MAP_HOST)
        
        print(f"[FACE LANDMARK] Using CUDA device: {cuda_device.name()}")
        
        # Get TensorRT model paths
        config = ConfigHandler().config
        landmark_trt_path = config['advanced_detection']['landmark_trt_path']
        blendshape_trt_path = config['advanced_detection']['blendshape_trt_path']
        batch_size = config['advanced_detection']['gpu_settings'].get('landmark_batch_size', 4)
        
        # Load engines
        trt_logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(trt_logger)
        
        # Load landmark engine
        with open(landmark_trt_path, 'rb') as f:
            landmark_engine = runtime.deserialize_cuda_engine(f.read())
        
        # Try loading blendshape engine - make it optional
        blendshape_available = False
        try:
            if os.path.exists(blendshape_trt_path):
                with open(blendshape_trt_path, 'rb') as f:
                    blendshape_engine = runtime.deserialize_cuda_engine(f.read())
                blendshape_available = True
                print("[FACE LANDMARK] Blendshape engine loaded successfully")
            else:
                print("[FACE LANDMARK] Blendshape engine not found, running without blend scores")
        except Exception as e:
            print(f"[FACE LANDMARK] Warning: Could not load blendshape engine: {e}")
            blendshape_engine = None
        
        # Create contexts
        landmark_context = landmark_engine.create_execution_context()
        if blendshape_available and blendshape_engine:
            blendshape_context = blendshape_engine.create_execution_context()
        
        # Create CUDA stream
        stream = cuda.Stream()
        
        # Allocate buffers with better error handling
        def allocate_buffers(engine, context, batch_size_override=None):
            inputs = []
            outputs = []
            bindings = []
            
            # Try new API first
            try:
                num_io_tensors = engine.num_io_tensors
                use_new_api = True
            except AttributeError:
                num_io_tensors = engine.num_bindings
                use_new_api = False
            
            if use_new_api:
                for i in range(num_io_tensors):
                    tensor_name = engine.get_tensor_name(i)
                    tensor_shape = list(engine.get_tensor_shape(tensor_name))
                    tensor_dtype = engine.get_tensor_dtype(tensor_name)
                    is_input = engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT
                    
                    if tensor_shape[0] == -1 and batch_size_override:
                        tensor_shape[0] = batch_size_override
                        if is_input:
                            context.set_input_shape(tensor_name, tensor_shape)
                    
                    size = int(np.prod(tensor_shape))
                    dtype = trt.nptype(tensor_dtype)
                    
                    host_mem = cuda.pagelocked_empty(size, dtype)
                    device_mem = cuda.mem_alloc(host_mem.nbytes)
                    
                    bindings.append(int(device_mem))
                    
                    buffer_info = {
                        'host': host_mem,
                        'device': device_mem,
                        'shape': tensor_shape,
                        'name': tensor_name,
                        'dtype': dtype
                    }
                    
                    if is_input:
                        inputs.append(buffer_info)
                    else:
                        outputs.append(buffer_info)
                        print(f"[FACE LANDMARK] Output tensor '{tensor_name}': shape={tensor_shape}")
            
            return inputs, outputs, bindings, use_new_api
        
        # Allocate buffers
        print("[FACE LANDMARK] Allocating buffers...")
        landmark_inputs, landmark_outputs, landmark_bindings, landmark_use_new_api = \
            allocate_buffers(landmark_engine, landmark_context, batch_size)
        
        if blendshape_available and blendshape_context:
            blendshape_inputs, blendshape_outputs, blendshape_bindings, blendshape_use_new_api = \
                allocate_buffers(blendshape_engine, blendshape_context)
        
        # Detect number of landmarks from output shape - FIXED VERSION
        num_landmarks = 468  # Default
        landmark_dims = 3
        
        if landmark_outputs:
            # The model might output multiple tensors, find the landmarks one
            for output in landmark_outputs:
                output_shape = output['shape']
                total_elements = np.prod(output_shape)
                
                # Try different interpretations
                if len(output_shape) == 4:  # [batch, 1, 1, elements]
                    elements_per_sample = output_shape[-1]
                elif len(output_shape) == 3:  # [batch, landmarks, dims]
                    if output_shape[1] in [468, 478] and output_shape[2] == 3:
                        num_landmarks = output_shape[1]
                        landmark_dims = output_shape[2]
                        break
                    elements_per_sample = output_shape[1] * output_shape[2]
                elif len(output_shape) == 2:  # [batch, elements]
                    elements_per_sample = output_shape[1]
                else:
                    elements_per_sample = total_elements // batch_size
                
                # Check for known landmark counts
                if elements_per_sample == 1434:  # 478 * 3
                    num_landmarks = 478
                    landmark_dims = 3
                elif elements_per_sample == 1404:  # 468 * 3
                    num_landmarks = 468
                    landmark_dims = 3
                elif elements_per_sample % 3 == 0:
                    possible_landmarks = elements_per_sample // 3
                    if possible_landmarks in [468, 478]:
                        num_landmarks = possible_landmarks
                        landmark_dims = 3
            
            print(f"[FACE LANDMARK] Detected model configuration:")
            print(f"  - Number of landmarks: {num_landmarks}")
            print(f"  - Dimensions per landmark: {landmark_dims}")
        
        # Define blendshape indices for 468 landmarks (MediaPipe standard)
        blendshape_indices = [
            0, 1, 4, 5, 6, 10, 12, 13, 14, 17, 18, 21, 33, 37, 39, 40, 46, 52, 53, 54, 
            55, 58, 61, 63, 65, 66, 67, 70, 78, 80, 81, 82, 84, 87, 88, 91, 93, 95, 103, 105, 
            107, 109, 127, 132, 133, 136, 144, 145, 146, 148, 149, 150, 152, 153, 154, 155, 157, 158, 159, 160, 
            161, 162, 163, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 
            185, 191, 195, 197, 234, 246, 249, 251, 263, 267, 269, 270, 271, 272, 276, 282, 283, 284, 285, 288, 
            291, 293, 295, 296, 297, 300, 308, 310, 311, 312, 314, 317, 318, 321, 323, 324, 332, 334, 336, 338, 
            356, 361, 362, 365, 373, 374, 375, 377, 378, 379, 380, 381, 382, 384, 385, 386, 387, 388, 389, 390, 
            397, 398, 400, 402, 405, 409
        ]
        
        print(f"[FACE LANDMARK] TensorRT engines loaded successfully")
        
        # Processing variables
        processed_count = 0
        failed_count = 0
        last_debug_time = time.time()
        
        # Batch processing
        batch_rois = []
        batch_metadata = []
        batch_timeout = 0.02
        last_batch_time = time.time()
        
        # Main loop
        while True:
            try:
                # Check control messages
                if control_pipe.poll():
                    msg = control_pipe.recv()
                    if msg == 'stop':
                        break
                    elif isinstance(msg, tuple) and msg[0] == 'set_mesh':
                        enable_mesh = msg[1]
                
                # Get ROI
                roi_data = None
                try:
                    timeout = batch_timeout if not batch_rois else 0.001
                    roi_data = roi_queue.get(timeout=timeout)
                except queue.Empty:
                    pass
                
                # Add to batch
                if roi_data:
                    batch_rois.append(roi_data['roi'])
                    batch_metadata.append(roi_data)
                
                # Process batch
                current_time = time.time()
                should_process = (
                    len(batch_rois) >= batch_size or
                    (len(batch_rois) > 0 and current_time - last_batch_time > batch_timeout)
                )
                
                if should_process and batch_rois:
                    actual_batch_size = min(len(batch_rois), batch_size)
                    batch_input = np.stack(batch_rois[:actual_batch_size])
                    
                    # Pad if needed
                    if actual_batch_size < batch_size:
                        padding = np.zeros((batch_size - actual_batch_size, 256, 256, 3), dtype=np.float32)
                        batch_input = np.concatenate([batch_input, padding])
                    
                    # Transfer to GPU
                    np.copyto(landmark_inputs[0]['host'], batch_input.ravel())
                    cuda.memcpy_htod_async(landmark_inputs[0]['device'], 
                                         landmark_inputs[0]['host'], stream)
                    
                    # Execute landmark model
                    if landmark_use_new_api:
                        for i in range(landmark_engine.num_io_tensors):
                            tensor_name = landmark_engine.get_tensor_name(i)
                            landmark_context.set_tensor_address(tensor_name, landmark_bindings[i])
                        landmark_context.execute_async_v3(stream_handle=stream.handle)
                    else:
                        landmark_context.execute_async_v2(
                            bindings=landmark_bindings,
                            stream_handle=stream.handle
                        )
                    
                    # Transfer back
                    for output in landmark_outputs:
                        cuda.memcpy_dtoh_async(output['host'], output['device'], stream)
                    stream.synchronize()
                    
                    # Process outputs - FIXED RESHAPING
                    # Find the main landmark output
                    landmarks_batch = None
                    
                    # Debug all outputs
                    if processed_count % 30 == 0:
                        print(f"[FACE LANDMARK] Processing {len(landmark_outputs)} outputs:")
                        for idx, output in enumerate(landmark_outputs):
                            print(f"  Output {idx}: shape={output['shape']}, dtype={output['dtype']}")
                            print(f"    Data sample: {output['host'][:10]}")
                    
                    for output in landmark_outputs:
                        output_data = output['host']
                        output_shape = output['shape']
                        
                        # Try to reshape based on expected format
                        try:
                            if len(output_shape) == 4 and output_shape[-1] in [1404, 1434]:
                                # Shape like [batch, 1, 1, 1434]
                                flat_data = output_data[:actual_batch_size * output_shape[-1]]
                                landmarks_batch = flat_data.reshape((actual_batch_size, -1, 3))
                            elif len(output_shape) == 3:
                                # Already in correct shape [batch, landmarks, 3]
                                landmarks_batch = output_data.reshape((batch_size, output_shape[1], output_shape[2]))
                            elif len(output_shape) == 2:
                                # Shape like [batch, 1434]
                                landmarks_batch = output_data.reshape((batch_size, -1, 3))
                            
                            if landmarks_batch is not None:
                                actual_num_landmarks = landmarks_batch.shape[1]
                                print(f"[FACE LANDMARK] Reshaped output to {landmarks_batch.shape}")
                                break
                        except Exception as reshape_error:
                            print(f"[FACE LANDMARK] Reshape attempt failed: {reshape_error}")
                            continue
                    
                    if landmarks_batch is None:
                        print(f"[FACE LANDMARK] ERROR: Could not reshape landmark output")
                        # Clear batch and continue
                        batch_rois = batch_rois[actual_batch_size:]
                        batch_metadata = batch_metadata[actual_batch_size:]
                        continue
                    
                    # Process each result
                    for i in range(actual_batch_size):
                        try:
                            landmarks_3d = landmarks_batch[i]
                            metadata = batch_metadata[i]
                            
                            # Debug raw landmark values
                            if processed_count % 30 == 0:
                                print(f"[FACE LANDMARK] Raw landmark check:")
                                print(f"  - Shape: {landmarks_3d.shape}")
                                print(f"  - First 3 landmarks raw: {landmarks_3d[:3]}")
                                print(f"  - Min/max x: {landmarks_3d[:, 0].min():.4f}, {landmarks_3d[:, 0].max():.4f}")
                                print(f"  - Min/max y: {landmarks_3d[:, 1].min():.4f}, {landmarks_3d[:, 1].max():.4f}")
                            
                            # Create default blend scores if blendshape not available
                            blend_scores = [0.0] * 52
                            
                            # Try to run blendshape if available
                            if blendshape_available and blendshape_context and num_landmarks >= 468:
                                try:
                                    # For 478 landmark model, use first 468
                                    landmarks_to_use = landmarks_3d[:468] if num_landmarks > 468 else landmarks_3d
                                    
                                    # Extract the required landmark subset
                                    selected_points = []
                                    for idx in blendshape_indices:
                                        if idx < len(landmarks_to_use):
                                            selected_points.append(landmarks_to_use[idx, :2])  # Only x,y
                                    
                                    if len(selected_points) == 146:
                                        selected_points = np.array(selected_points, dtype=np.float32)
                                        
                                        # Run blendshape model
                                        np.copyto(blendshape_inputs[0]['host'], selected_points.ravel())
                                        cuda.memcpy_htod_async(blendshape_inputs[0]['device'],
                                                             blendshape_inputs[0]['host'], stream)
                                        
                                        if blendshape_use_new_api:
                                            for j in range(blendshape_engine.num_io_tensors):
                                                tensor_name = blendshape_engine.get_tensor_name(j)
                                                blendshape_context.set_tensor_address(tensor_name, blendshape_bindings[j])
                                            blendshape_context.execute_async_v3(stream_handle=stream.handle)
                                        else:
                                            blendshape_context.execute_async_v2(
                                                bindings=blendshape_bindings,
                                                stream_handle=stream.handle
                                            )
                                        
                                        cuda.memcpy_dtoh_async(blendshape_outputs[0]['host'],
                                                             blendshape_outputs[0]['device'], stream)
                                        stream.synchronize()
                                        
                                        blend_scores = blendshape_outputs[0]['host'].tolist()
                                except Exception as blend_error:
                                    print(f"[FACE LANDMARK] Blendshape processing error: {blend_error}")
                            
                            # Transform landmarks - use only first 468 for compatibility
                            transform = metadata['transform']
                            landmarks = []

                            # Get frame dimensions for proper normalization
                            frame_width = transform.get('frame_width', 1280)
                            frame_height = transform.get('frame_height', 720)

                            # Use min to ensure we don't exceed 468 landmarks for compatibility
                            landmarks_to_use = min(468, landmarks_3d.shape[0])

                            for j in range(landmarks_to_use):
                                lm = landmarks_3d[j]
                                # Debug individual landmark access
                                if j == 0 and processed_count % 30 == 0:
                                    print(f"[FACE LANDMARK] First landmark access debug:")
                                    print(f"  - lm type: {type(lm)}, shape: {lm.shape if hasattr(lm, 'shape') else 'no shape'}")
                                    print(f"  - lm value: {lm}")
                                    print(f"  - lm[0]: {lm[0]}, lm[1]: {lm[1]}, lm[2]: {lm[2] if len(lm) > 2 else 'N/A'}")
                                
                                # Model outputs are already in pixel coordinates (0-256)
                                x = float(lm[0])  # Already in ROI pixel space (256x256)
                                y = float(lm[1])
                                z = float(lm[2]) if landmark_dims > 2 else 0.0
                                
                                # Transform from ROI space to frame space
                                x_frame = x * transform['scale'] + transform['offset_x']
                                y_frame = y * transform['scale'] + transform['offset_y']
                                
                                # CRITICAL: Normalize to [0, 1] for GUI
                                x_norm = x_frame / frame_width
                                y_norm = y_frame / frame_height
                                
                                # Ensure values are in valid range [0, 1]
                                x_norm = max(0.0, min(1.0, x_norm))
                                y_norm = max(0.0, min(1.0, y_norm))
                                
                                landmarks.append((x_norm, y_norm, z))

                            # Calculate centroid from normalized coordinates
                            if landmarks:
                                x_coords = [l[0] for l in landmarks]
                                y_coords = [l[1] for l in landmarks]
                                centroid = (np.mean(x_coords), np.mean(y_coords))
                            else:
                                centroid = (0.5, 0.5)

                            # Face contour (jaw line - first 17 points) - properly normalized
                            face_contour = []
                            for idx in range(min(17, landmarks_3d.shape[0])):
                                lm = landmarks_3d[idx]
                                # Same transformation as landmarks - already in pixel coordinates
                                x = float(lm[0])  # Already in ROI pixel space (256x256)
                                y = float(lm[1])
                                
                                x_frame = x * transform['scale'] + transform['offset_x']
                                y_frame = y * transform['scale'] + transform['offset_y']
                                
                                x_norm = x_frame / frame_width
                                y_norm = y_frame / frame_height
                                
                                # Ensure valid range
                                x_norm = max(0.0, min(1.0, x_norm))
                                y_norm = max(0.0, min(1.0, y_norm))
                                
                                face_contour.append((x_norm, y_norm))

                            # Debug to verify normalization
                            if processed_count % 30 == 0:
                                print(f"[FACE LANDMARK] Coordinate check for track {metadata['track_id']}:")
                                print(f"  - Frame dimensions: {frame_width}x{frame_height}")
                                print(f"  - Transform scale: {transform['scale']}, offset: ({transform['offset_x']}, {transform['offset_y']})")
                                if landmarks:
                                    print(f"  - First landmark (normalized): {landmarks[0]}")
                                    print(f"  - Centroid (normalized): {centroid}")
                                if face_contour:
                                    print(f"  - First contour point (normalized): {face_contour[0]}")
                            
                            # Send result
                            result = {
                                'track_id': metadata['track_id'],
                                'landmarks': landmarks,
                                'blend': blend_scores,
                                'centroid': centroid,
                                'mesh': [coord for lm in landmarks for coord in lm] if enable_mesh else None,
                                'timestamp': metadata['timestamp'],
                                'quality_score': metadata.get('quality_score', 0.5),
                                'frame_id': metadata.get('frame_id', -1),
                                'face_contour': face_contour
                            }
                            
                            landmark_result_queue.put(result, timeout=0.01)
                            processed_count += 1
                            
                            if processed_count % 30 == 0:
                                print(f"[FACE LANDMARK] Processed {processed_count} ROIs, landmarks: {len(landmarks)}")
                            
                        except Exception as e:
                            print(f"[FACE LANDMARK] Error processing ROI: {e}")
                            import traceback
                            traceback.print_exc()
                            failed_count += 1
                    
                    # Clear processed
                    batch_rois = batch_rois[actual_batch_size:]
                    batch_metadata = batch_metadata[actual_batch_size:]
                    last_batch_time = current_time
                
                # Status update
                if current_time - last_debug_time > 5.0:
                    print(f"[FACE LANDMARK] Status: processed={processed_count}, failed={failed_count}")
                    last_debug_time = current_time
                    
            except Exception as e:
                print(f"[FACE LANDMARK] Error in main loop: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.1)
        
        print(f"[FACE LANDMARK] Exiting. Processed {processed_count} ROIs total")
        
    except Exception as e:
        print(f"[FACE LANDMARK] Fatal error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Cleanup
        if landmark_context:
            del landmark_context
        if blendshape_context:
            del blendshape_context
        if landmark_engine:
            del landmark_engine
        if blendshape_engine:
            del blendshape_engine
        if stream:
            del stream
        if cuda_context:
            try:
                cuda_context.pop()
                cuda_context.detach()
            except:
                pass
def validate_face_data_format(face_info):
    """
    Validates and normalizes face data to ensure compatibility with canvasdrawing.py
    """
    # Ensure all required fields exist
    required_fields = ['track_id', 'bbox', 'confidence', 'landmarks', 'blend', 'centroid', 'timestamp']
    
    for field in required_fields:
        if field not in face_info:
            if field == 'blend':
                face_info[field] = [0.0] * 52
            elif field == 'landmarks':
                face_info[field] = []
            elif field == 'centroid':
                if 'bbox' in face_info and face_info['bbox']:
                    bbox = face_info['bbox']
                    # Centroid should be normalized (0-1)
                    face_info[field] = (0.5, 0.5)  # Default to center
                else:
                    face_info[field] = (0.5, 0.5)
            elif field == 'confidence':
                face_info[field] = 0.5
            elif field == 'timestamp':
                face_info[field] = time.time()
    
    # Ensure face_contour is properly formatted
    if 'face_contour' not in face_info or not face_info['face_contour']:
        # Try to extract from landmarks if available
        if face_info['landmarks'] and len(face_info['landmarks']) >= 17:
            face_contour = []
            for i in range(17):  # First 17 landmarks form jaw line
                lm = face_info['landmarks'][i]
                if isinstance(lm, (list, tuple)) and len(lm) >= 2:
                    face_contour.append((float(lm[0]), float(lm[1])))
            face_info['face_contour'] = face_contour
    else:
        # Validate existing face contour
        valid_contour = []
        for point in face_info['face_contour']:
            if isinstance(point, (list, tuple)) and len(point) >= 2:
                valid_contour.append((float(point[0]), float(point[1])))
        face_info['face_contour'] = valid_contour
    
    # Ensure landmarks are tuples of (x, y, z)
    if face_info['landmarks']:
        valid_landmarks = []
        for lm in face_info['landmarks']:
            if isinstance(lm, (list, tuple)):
                if len(lm) >= 3:
                    valid_landmarks.append((float(lm[0]), float(lm[1]), float(lm[2])))
                elif len(lm) >= 2:
                    valid_landmarks.append((float(lm[0]), float(lm[1]), 0.0))
        face_info['landmarks'] = valid_landmarks
    
    return face_info

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
                if dropped > 3:
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
         # Collect landmark results
        landmarks_collected = 0
        while True:
            try:
                landmark_result = landmark_result_queue.get_nowait()
                landmarks_collected += 1
                
                # Cache the landmark result
                track_id = landmark_result['track_id']
                landmark_results_cache[track_id] = landmark_result
                
                # Debug landmark data
                if frame_count % 600 == 0:
                    print(f"[FACE WORKER] Collected landmark result for track {track_id}:")
                    print(f"  - Has landmarks: {bool(landmark_result.get('landmarks'))}")
                    if landmark_result.get('landmarks'):
                        print(f"  - Landmark count: {len(landmark_result['landmarks'])}")
                        print(f"  - First landmark: {landmark_result['landmarks'][0]}")
                    print(f"  - Has face_contour: {bool(landmark_result.get('face_contour'))}")
                    if landmark_result.get('face_contour'):
                        print(f"  - Face contour points: {len(landmark_result['face_contour'])}")
                
            except:
                break
        
        if landmarks_collected > 0 and frame_count % 30 == 0:
            print(f"[FACE WORKER] Collected {landmarks_collected} landmark results, cache size: {len(landmark_results_cache)}")
        
        # Build face data from tracked objects with cached landmarks
        face_data = []
        for track in tracked_objects:
            track_id = track['track_id']
            
            # Get cached landmark data if available
            landmark_data = landmark_results_cache.get(track_id, {})
            
            # Debug what we have
            if frame_count % 30 == 0 and track_id in landmark_results_cache:
                print(f"[FACE WORKER] Building face data for track {track_id}:")
                print(f"  - Has cached landmarks: {bool(landmark_data.get('landmarks'))}")
                print(f"  - Cached data keys: {list(landmark_data.keys())}")
            
            face_info = {
                'track_id': track_id,
                'bbox': track['bbox'],
                'confidence': track['confidence'],
                'landmarks': landmark_data.get('landmarks', []),
                'blend': landmark_data.get('blend', [0.0] * 52),
                'centroid': landmark_data.get('centroid', 
                           ((track['bbox'][0] + track['bbox'][2]) / 2 / rgb.shape[1],
                            (track['bbox'][1] + track['bbox'][3]) / 2 / rgb.shape[0])),  # Normalize to 0-1
                'mesh': landmark_data.get('mesh') if enable_mesh else None,
                'timestamp': timestamp,
                'face_contour': landmark_data.get('face_contour', [])  # This should now have proper values
            }
            
            # Validate face data format
            face_info = validate_face_data_format(face_info)
            face_data.append(face_info)
            
            # Debug final face data
            if frame_count % 60 == 0:
                print(f"[FACE WORKER] Final face data for track {track_id}:")
                print(f"  - Has landmarks: {bool(face_info['landmarks'])}")
                print(f"  - Landmark count: {len(face_info['landmarks'])}")
                print(f"  - Has face_contour: {bool(face_info['face_contour'])}")
                print(f"  - Face contour count: {len(face_info['face_contour'])}")
        
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
    """Dedicated process for pose detection with MediaPipe import and protobuf fixes"""
    print("[POSE WORKER] Starting")
    print(f"[POSE WORKER] frame_queue type: {type(frame_queue)}")
    print(f"[POSE WORKER] has get method: {hasattr(frame_queue, 'get')}")
    
    # Fix protobuf compatibility issue
    import os
    os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
    import sys

    # Remove any existing protobuf from modules
    modules_to_remove = []
    for module_name in sys.modules:
        if module_name.startswith('google.protobuf'):
            modules_to_remove.append(module_name)
    
    for module_name in modules_to_remove:
        del sys.modules[module_name]
    
    # Now import protobuf fresh
    try:
        import google.protobuf
        print(f"[POSE WORKER] Protobuf version: {google.protobuf.__version__}")
    except ImportError:
        print("[POSE WORKER] Protobuf not installed")

    
    # Try different MediaPipe import methods
    MEDIAPIPE_AVAILABLE = False
    pose_landmarker = None
    mp = None
    
    try:
        # Try the new import style first (MediaPipe >= 0.10.0)
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks import vision
        MEDIAPIPE_AVAILABLE = True
        print("[POSE WORKER] MediaPipe tasks API available")
    except ImportError as e:
        print(f"[POSE WORKER] MediaPipe tasks import failed: {e}")
        try:
            # Try basic MediaPipe import
            import mediapipe as mp
            # Check if we have the solutions API
            if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'pose'):
                MEDIAPIPE_AVAILABLE = True
                print("[POSE WORKER] Using MediaPipe solutions API (legacy mode)")
                python = None
                vision = None
            else:
                print("[POSE WORKER] MediaPipe solutions API not available")
        except ImportError as e2:
            print(f"[POSE WORKER] MediaPipe not installed: {e2}")
            MEDIAPIPE_AVAILABLE = False

    if not MEDIAPIPE_AVAILABLE:
        print("[POSE WORKER] ERROR: MediaPipe not properly installed")
        print("[POSE WORKER] Install with: pip install mediapipe==0.10.9 protobuf==3.20.3")
        print("[POSE WORKER] Running in fallback mode - no pose detection")
        
        # Fallback loop - just pass through empty results
        while True:
            if control_pipe.poll():
                msg = control_pipe.recv()
                if msg == 'stop':
                    break
            
            try:
                frame_data = frame_queue.get(timeout=0.1)
                result = {
                    'type': 'pose',
                    'data': [],
                    'timestamp': frame_data['timestamp']
                }
                result_queue.put_nowait(result)
            except:
                pass
        
        print("[POSE WORKER] Fallback worker stopped")
        return

    # Initialize pose detection based on available API
    if python and vision:
        # New API (MediaPipe >= 0.10.0)
        try:
            with open(model_path, 'rb') as f:
                model_buffer = f.read()
            base_opts = python.BaseOptions(model_asset_buffer=model_buffer)
            
            pose_options = vision.PoseLandmarkerOptions(
                base_options=base_opts,
                running_mode=vision.RunningMode.VIDEO,
                num_poses=5,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                output_segmentation_masks=False
            )
            pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)
            print("[POSE WORKER] Initialized with new MediaPipe API")
        except Exception as e:
            print(f"[POSE WORKER] Failed to initialize with new API: {e}")
            MEDIAPIPE_AVAILABLE = False
    else:
        # Legacy API (MediaPipe < 0.10.0)
        try:
            # Import with protobuf fix
            import mediapipe as mp
            mp_pose = mp.solutions.pose
            
            # Try to create pose instance with error handling
            try:
                pose_landmarker = mp_pose.Pose(
                    static_image_mode=False,
                    model_complexity=1,
                    enable_segmentation=False,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                print("[POSE WORKER] Initialized with legacy MediaPipe API")
            except Exception as init_error:
                print(f"[POSE WORKER] Pose initialization error: {init_error}")
                # Try simpler configuration
                pose_landmarker = mp_pose.Pose(
                    static_image_mode=False,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                print("[POSE WORKER] Initialized with minimal configuration")
                
        except Exception as e:
            print(f"[POSE WORKER] Failed to initialize with legacy API: {e}")
            import traceback
            traceback.print_exc()
            MEDIAPIPE_AVAILABLE = False

    if not MEDIAPIPE_AVAILABLE or pose_landmarker is None:
        print("[POSE WORKER] Failed to initialize pose detection, running in fallback mode")
        while True:
            if control_pipe.poll():
                msg = control_pipe.recv()
                if msg == 'stop':
                    break
            
            try:
                frame_data = frame_queue.get(timeout=0.1)
                result = {
                    'type': 'pose',
                    'data': [],
                    'timestamp': frame_data['timestamp']
                }
                result_queue.put_nowait(result)
            except:
                pass
        
        return

    # Monotonic timestamp generator
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
    
    # Main processing loop
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
        
        rgb = frame_data['rgb']
        timestamp = frame_data['timestamp']
        
        # Process based on API version
        pose_data = []
        
        try:
            if python and vision:
                # New API processing
                timestamp_ms = ts_gen.next()
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                pose_result = pose_landmarker.detect_for_video(mp_image, timestamp_ms)
                
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
                        
                        # Create pose values array
                        pose_vals = []
                        for lm in pose_landmarks:
                            pose_vals.extend([lm.x, lm.y, lm.z, lm.visibility])
                        
                        pose_data.append({
                            'landmarks': [(lm.x, lm.y, lm.z) for lm in pose_landmarks],
                            'centroid': pose_centroid,
                            'values': pose_vals,
                            'timestamp': timestamp
                        })
            else:
                # Legacy API processing with error handling
                # Ensure frame is uint8
                if rgb.dtype != np.uint8:
                    rgb_uint8 = (rgb * 255).astype(np.uint8) if rgb.max() <= 1.0 else rgb.astype(np.uint8)
                else:
                    rgb_uint8 = rgb
                    
                results = pose_landmarker.process(rgb_uint8)
                
                if results and results.pose_landmarks:
                    # Legacy API returns single pose
                    pose_landmarks = results.pose_landmarks
                    
                    # Calculate centroid
                    key_indices = [11, 12, 23, 24]  # shoulders and hips
                    key_points = [pose_landmarks.landmark[i] for i in key_indices if i < len(pose_landmarks.landmark)]
                    if key_points:
                        x_coords = [p.x for p in key_points]
                        y_coords = [p.y for p in key_points]
                        pose_centroid = (np.mean(x_coords), np.mean(y_coords))
                    else:
                        pose_centroid = (0.5, 0.5)
                    
                    # Create pose values array
                    pose_vals = []
                    for lm in pose_landmarks.landmark:
                        pose_vals.extend([lm.x, lm.y, lm.z, lm.visibility if hasattr(lm, 'visibility') else 1.0])
                    
                    pose_data.append({
                        'landmarks': [(lm.x, lm.y, lm.z) for lm in pose_landmarks.landmark],
                        'centroid': pose_centroid,
                        'values': pose_vals,
                        'timestamp': timestamp
                    })
                    
        except Exception as e:
            if frame_count % 30 == 0:
                print(f"[POSE WORKER] Processing error: {e}")
        
        if pose_data:
            detection_count += len(pose_data)
        
        # Send result
        result = {
            'type': 'pose',
            'data': pose_data,
            'timestamp': timestamp
        }
        
        try:
            result_queue.put_nowait(result)
        except:
            pass
        
        frame_count += 1
        
        # Report FPS periodically
        current_time = time.time()
        if current_time - last_fps_report > 30.0:  # Report every 30 seconds
            elapsed = current_time - last_fps_report
            fps = frame_count / elapsed
            print(f"[POSE WORKER] FPS: {fps:.1f}, Detection rate: {detection_count/frame_count:.2f}")
            frame_count = 0
            detection_count = 0
            last_fps_report = current_time
    
    # Cleanup
    if pose_landmarker:
        if hasattr(pose_landmarker, 'close'):
            pose_landmarker.close()
        elif hasattr(pose_landmarker, 'reset'):
            pose_landmarker.reset()
    
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

                if face_result and face_result['type'] == 'face':
                    face_data = face_result['data']
                    tracked_faces = []
                    
                    for face in face_data:
                        track_id = face.get('track_id', -1)
                        
                        # Debug incoming face data
                        if frame_count % 60 == 0:
                            print(f"[FUSION] Received face data for track {track_id}:")
                            print(f"  - Has landmarks: {bool(face.get('landmarks'))}")
                            if face.get('landmarks'):
                                print(f"  - Landmark count: {len(face['landmarks'])}")
                                print(f"  - First landmark: {face['landmarks'][0]}")
                            print(f"  - Has face_contour: {bool(face.get('face_contour'))}")
                            if face.get('face_contour'):
                                print(f"  - Face contour count: {len(face['face_contour'])}")
                        
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

            if frame_count % 60 == 0:
                print(f"\n[FUSION] Preparing preview data at frame {frame_count}:")
                print(f"  - Frame BGR shape: {latest_frame_bgr.shape if latest_frame_bgr is not None else 'None'}")
                print(f"  - Number of faces: {len(latest_face_data)}")
                
            for i, face in enumerate(latest_face_data[:1]):  # Just first face
                print(f"  Face {i}:")
                print(f"    - track_id: {face.get('track_id')}")
                print(f"    - global_id: {face.get('global_id')}")
                print(f"    - Has landmarks: {bool(face.get('landmarks'))}")
                if face.get('landmarks'):
                    print(f"    - Landmark count: {len(face['landmarks'])}")
                    print(f"    - Landmark sample: {face['landmarks'][:3]}")
                print(f"    - Has face_contour: {bool(face.get('face_contour'))}")
                if face.get('face_contour'):
                    print(f"    - Face contour count: {len(face['face_contour'])}")
                    print(f"    - Face contour sample: {face['face_contour'][:3]}")
                print(f"    - Has bbox: {bool(face.get('bbox'))}")
                if face.get('bbox'):
                    print(f"    - Bbox: {face['bbox']}")
            
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
