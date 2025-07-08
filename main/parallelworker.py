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

def robust_initialize_camera(camera_index, fps, resolution, settle_time=2.0, target_brightness_range=(30, 200), max_attempts=1):
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
    """Distributes frames to multiple processing pipelines"""
    def __init__(self, camera_index, resolution, fps):
        self.camera_index = camera_index
        self.resolution = resolution
        self.fps = fps
        self.subscribers = []
        self.running = False
        
    def add_subscriber(self, queue):
        """Add a queue that will receive frames"""
        self.subscribers.append(queue)
        
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

        while self.running:
            ret, frame_bgr = cap.read()
            if ret:
                # Convert once
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frame_data = {
                    'bgr': frame_bgr,
                    'rgb': rgb,
                    'timestamp': time.time()
                }
                
                # Send to all subscribers
                for q in self.subscribers:
                    if not q.full():
                        try:
                            q.put_nowait(frame_data)
                        except:
                            pass
            else:
                time.sleep(0.001)
                
        cap.release()

        # ─── Camera setup ──────────────────────────────────────────────
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        for b in backends:
            cap = cv2.VideoCapture(self.camera_index, b)
            if cap.isOpened():
                self.cap = cap
                break
        else:
            raise RuntimeError(f"Cannot open camera {self.camera_index!r}")

        # Use the new initialization method with auto-exposure settling
        camera_settings = self.initialize_camera(
            self.cap, 
            self.fps, 
            self.resolution,
        )

        # Store the settled values for potential later use
        self.camera_settings = camera_settings

        print("[Camera] Camera ready!")

        # Start capture thread
        self._start_capture_thread()
        
    def initialize_camera(cap, fps, resolution, 
                                                settle_time=3.0, 
                                                target_brightness_range=(0, 200),
                                                max_attempts=1):
        """
        Initialize camera with auto-exposure, validate brightness, and lock optimal settings.

        Args:
            cap: cv2.VideoCapture object
            fps: target FPS
            resolution: tuple of (width, height)
            settle_time: seconds to let auto-exposure settle
            target_brightness_range: tuple of (min, max) acceptable mean brightness
            max_attempts: maximum attempts to get good exposure

        Returns:
            dict with final camera settings
        """
        print(f"[Camera] Initializing for {resolution[0]}x{resolution[1]} @ {fps} FPS...")
        # Basic settings
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        cap.set(cv2.CAP_PROP_FPS, fps)

        if fps > 30:
            try:
                # Try to set high-speed mode if available
                cap.set(cv2.CAP_PROP_SETTINGS, 1)  # Some cameras use this
            except Exception as e:
                print(f"[Camera] Camera not on the fly setting capable: {e}")

            # Reduce exposure time for high FPS
            cap.set(cv2.CAP_PROP_EXPOSURE, -6)  # Lower exposure for faster capture
            


        for attempt in range(max_attempts):
            print(f"\n[Camera] Initialization attempt {attempt + 1}/{max_attempts}")
            
            # Enable auto features
            for val in [3, 1, 0.75, -1]:
                try:
                    if cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, val):
                        break
                except:
                    continue
            
            # Let camera settle
            print(f"[Camera] Settling for {settle_time} seconds...")
            
            exposure_readings = []
            brightness_readings = []
            gain_readings = []
            frame_brightnesses = []
            
            start_time = time.time()
            frame_count = 0
            
            while time.time() - start_time < settle_time:
                ret, frame = cap.read()
                if not ret:
                    continue
                    
                frame_count += 1
                
                # Calculate frame brightness
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                mean_brightness = np.mean(gray)
                frame_brightnesses.append(mean_brightness)
                
                # Sample camera values
                if frame_count % 10 == 0:
                    exposure = cap.get(cv2.CAP_PROP_EXPOSURE)
                    brightness = cap.get(cv2.CAP_PROP_BRIGHTNESS)
                    gain = cap.get(cv2.CAP_PROP_GAIN)
                    
                    if exposure != 0:
                        exposure_readings.append(exposure)
                    if brightness != 0:
                        brightness_readings.append(brightness)
                    if gain != 0:
                        gain_readings.append(gain)
            
            # Analyze brightness stability (use last 1/3 of frames)
            recent_brightnesses = frame_brightnesses[-len(frame_brightnesses)//3:]
            mean_brightness = np.mean(recent_brightnesses)
            brightness_std = np.std(recent_brightnesses)
            
            print(f"[Camera] Brightness analysis - Mean: {mean_brightness:.1f}, "
                    f"Std: {brightness_std:.1f}")
            
            # Check if brightness is in acceptable range
            if target_brightness_range[0] <= mean_brightness <= target_brightness_range[1]:
                print("[Camera] Brightness is in acceptable range!")
                
                # Calculate settled values
                settled_exposure = np.median(exposure_readings) if exposure_readings else None
                settled_brightness = np.median(brightness_readings) if brightness_readings else None
                settled_gain = np.median(gain_readings) if gain_readings else None
                
                # Switch to manual mode
                for val in [0.25, 1, 0]:
                    try:
                        if cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, val):
                            break
                    except:
                        continue
                
                # Apply settled values
                if settled_exposure is not None:
                    cap.set(cv2.CAP_PROP_EXPOSURE, settled_exposure)
                if settled_brightness is not None:
                    cap.set(cv2.CAP_PROP_BRIGHTNESS, settled_brightness)
                if settled_gain is not None:
                    cap.set(cv2.CAP_PROP_GAIN, settled_gain)
                
                # Success!
                break
                
            else:
                print(f"[Camera] Brightness {mean_brightness:.1f} is outside target range "
                        f"{target_brightness_range}")
                
                if attempt < max_attempts - 1:
                    # Try adjusting initial brightness for next attempt
                    if mean_brightness < target_brightness_range[0]:
                        print("[Camera] Attempting to increase initial brightness...")
                        cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.7)
                    else:
                        print("[Camera] Attempting to decrease initial brightness...")
                        cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.3)

        # Final settings
        final_settings = {
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'auto_exposure': cap.get(cv2.CAP_PROP_AUTO_EXPOSURE),
            'exposure': cap.get(cv2.CAP_PROP_EXPOSURE),
            'brightness': cap.get(cv2.CAP_PROP_BRIGHTNESS),
            'gain': cap.get(cv2.CAP_PROP_GAIN),
            'mean_frame_brightness': mean_brightness,
            'brightness_stability': brightness_std
        }

        print("\n[Camera] Final settings:")
        for key, value in final_settings.items():
            print(f"  {key}: {value}")

        return final_settings
    
def face_worker_process(frame_queue: MPQueue, 
                       result_queue: MPQueue,
                       model_path: str,
                       enable_mesh: bool,
                       control_pipe):
    """Dedicated process for face detection with performance monitoring"""
    print("[FACE WORKER] Starting")
    
    # Initialize face landmarker
    with open(model_path, 'rb') as f:
        model_buffer = f.read()
    base_opts = python.BaseOptions(model_asset_buffer=model_buffer)
    
    face_options = vision.FaceLandmarkerOptions(
        base_options=base_opts,
        output_face_blendshapes=True,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=2,
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
    
    # Performance monitoring
    frame_count = 0
    detection_count = 0
    last_fps_report = time.time()
    fps_report_interval = 10.0  # Report every 30 seconds
    
    while True:
        # Check for control messages
        if control_pipe.poll():
            msg = control_pipe.recv()
            if msg == 'stop':
                break
            elif isinstance(msg, tuple) and msg[0] == 'set_mesh':
                enable_mesh = msg[1]
                print(f"[FACE WORKER] Mesh data {'enabled' if enable_mesh else 'disabled'}")
            elif msg == 'get_stats':
                # Send performance stats
                elapsed = time.time() - last_fps_report
                if elapsed > 0:
                    stats = {
                        'type': 'face_stats',
                        'fps': frame_count / elapsed,
                        'detection_rate': detection_count / frame_count if frame_count > 0 else 0
                    }
                    control_pipe.send(stats)
                
        # Get frame
        try:
            frame_data = frame_queue.get(timeout=0.1)
        except:
            continue
            
        rgb = frame_data['rgb']
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
                    'timestamp': frame_data['timestamp']
                })

            detection_count += len(face_data)

            
        # Send result
        result = {
            'type': 'face',
            'data': face_data,
            'timestamp': frame_data['timestamp']
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
                print(f"[FACE WORKER] Performance: {fps:.1f} FPS, {detection_rate:.1f}% detection rate")
                
            # Reset counters
            frame_count = 0
            detection_count = 0
            last_fps_report = current_time
            
    # Cleanup
    face_landmarker.close()
    print("[FACE WORKER] Stopped")


def pose_worker_process(frame_queue: MPQueue,
                       result_queue: MPQueue,
                       pose_model_path: str,
                       control_pipe):
    """Dedicated process for pose detection with performance monitoring"""
    print("[POSE WORKER] Starting")
    
    # Initialize pose landmarker
    pose_base_opts = python.BaseOptions(
        model_asset_path=pose_model_path
    )
    pose_options = vision.PoseLandmarkerOptions(
        base_options=pose_base_opts,
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
    fps_report_interval = 30.0  # Report every 30 seconds
    enabled = True
    
    while True:
        # Check for control messages
        if control_pipe.poll():
            msg = control_pipe.recv()
            if msg == 'stop':
                break
            elif isinstance(msg, tuple) and msg[0] == 'enable_pose':
                enabled = msg[1]
                print(f"[POSE WORKER] {'Enabled' if enabled else 'Disabled'}")
            elif msg == 'get_stats':
                # Send performance stats
                elapsed = time.time() - last_fps_report
                if elapsed > 0:
                    stats = {
                        'type': 'pose_stats',
                        'fps': frame_count / elapsed,
                        'detection_rate': detection_count / frame_count if frame_count > 0 else 0,
                        'enabled': enabled
                    }
                    control_pipe.send(stats)
                
        if not enabled:
            time.sleep(0.01)
            continue
            
        # Get frame
        try:
            frame_data = frame_queue.get(timeout=0.1)
        except:
            continue
            
        rgb = frame_data['rgb']
        timestamp_ms = ts_gen.next()
        
        # Process
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        pose_result = pose_landmarker.detect_for_video(mp_image, timestamp_ms)
        
        # Extract data
        pose_data = []
        if pose_result and pose_result.pose_landmarks:
            for i, pose_landmarks in enumerate(pose_result.pose_landmarks):
                # Calculate centroid (hip center)
                left_hip = pose_landmarks[23]
                right_hip = pose_landmarks[24]
                pose_centroid = ((left_hip.x + right_hip.x) / 2, (left_hip.y + right_hip.y) / 2)
                
                # Pose values
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
            'data': pose_data,  # Now a list of poses
            'timestamp': frame_data['timestamp']
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
                print(f"[POSE WORKER] Performance: {fps:.1f} FPS, {detection_rate:.1f}% detection rate")
                
            # Reset counters
            frame_count = 0
            detection_count = 0
            last_fps_report = current_time
            
    # Cleanup
    pose_landmarker.close()
    print("[POSE WORKER] Stopped")

def data_fusion_process(
    face_queue: MPQueue,
    pose_queue: MPQueue,
    frame_queue: MPQueue,
    preview_queue: MPQueue,
    recording_queue: MPQueue,
    lsl_data_queue: MPQueue,
    score_buffer_name: str,
    camera_index: int,
    fps: int,
    participant_update_queue: MPQueue,
    participant_mapping_pipe,
    correlation_queue: MPQueue = None,
    max_participants: int = 2
):
    """
    Combines face and pose data with robust multi-person tracking and correct preview output.
    """ 
    print(f"[FUSION Camera {camera_index}] Process started")
    tracker = UnifiedTracker(max_participants=max_participants)
    streaming_active = False

    participant_names = {}  

    # Connect to score buffer
    score_buffer = None
    if score_buffer_name:
        try:
            score_buffer = NumpySharedBuffer(size=52, name=score_buffer_name)
        except Exception as e:
            print(f"[FUSION Camera {camera_index}] Failed to connect to score buffer: {e}")

    # MAINTAIN LAST KNOWN STATE
    latest_frame = None
    latest_face_list = None  # None means no data yet
    latest_pose_list = None  # None means no data yet
    last_output_time = 0
    output_interval = 1.0 / fps

    face_count = 0
    pose_count = 0
    lsl_count = 0
    frame_count = 0
    last_fps_report = time.time()
    fps_report_interval = 30.0  # Increased to 30 seconds for less spam
    
    # Track local to global ID mapping
    local_to_global = {}

    # Always running
    while True:
        # Check for pipe messages
        if participant_mapping_pipe.poll():
            try:
                message = participant_mapping_pipe.recv()
                if isinstance(message, dict):
                    # Handle participant ID mapping
                    if 'local_id' in message and 'global_id' in message:
                        local_id = message['local_id']
                        global_id = message['global_id']
                        if global_id is not None:
                            local_to_global[local_id] = global_id
                        else:
                            # Max participants reached
                            if max_participants == 1:
                                # In single participant mode, force map to 1
                                local_to_global[local_id] = 1
                                print(f"[FUSION Camera {camera_index}] Single participant mode: forcing local {local_id} -> global 1")
                            else:
                                # In multi-participant mode, mark as rejected
                                local_to_global[local_id] = None
                                print(f"[FUSION Camera {camera_index}] Max participants reached, rejecting local {local_id}")
                    
                    # Handle streaming state changes
                    elif message.get('type') == 'streaming_state':
                        streaming_active = message.get('active', False)
                        print(f"[FUSION Camera {camera_index}] Streaming state: {streaming_active}")
                    # Handle LSL data
                    elif message.get('type') == 'participant_names':
                        names_dict = message.get('names', {})
                        # Update local participant names - the dict is 0-based index to name
                        participant_names.clear()
                        for idx, name in names_dict.items():
                            participant_names[idx] = name
                        print(f"[FUSION Camera {camera_index}] Updated participant names: {participant_names}")


            except Exception as e:
                print(f"[FUSION Camera {camera_index}] Error processing pipe message: {e}")

        # Get latest frame data
        try:
            while not frame_queue.empty():
                latest_frame = frame_queue.get_nowait()
                frame_count += 1
        except:
            pass

        # Get latest face data (only update if new data arrives)
        try:
            while not face_queue.empty():
                face_result = face_queue.get_nowait()
                if face_result['type'] == 'face' and face_result['data'] is not None:
                    latest_face_list = face_result['data']
                    face_count += 1
        except:
            pass

        # Get latest pose data (only update if new data arrives)
        try:
            while not pose_queue.empty():
                pose_result = pose_queue.get_nowait()
                if pose_result['type'] == 'pose' and pose_result['data'] is not None:
                    latest_pose_list = pose_result['data']
                    pose_count += 1
        except:
            pass

        current_time = time.time()
        if current_time - last_output_time >= output_interval:
            # Always send preview if we have a frame, regardless of face detection
            if latest_frame and 'bgr' in latest_frame:
                # Default empty tracking data
                faces_to_draw = []
                primary_pose = None
                all_tracked_poses = []
                local_participant_ids = []
                
                # Only update tracking if we have face data
                if latest_face_list is not None:
                    # === TRACKER UPDATE ===
                    face_detections = []
                    for f in latest_face_list:
                        face_detections.append({
                            'centroid': f['centroid'],
                            'landmarks': f['landmarks'],
                            'blend': f['blend'],
                            'mesh': f.get('mesh', None)
                        })
                    
                    pose_detections = []
                    # Only include poses if we've received pose data
                    if latest_pose_list is not None:
                        for p in latest_pose_list:
                            pose_detections.append({
                                'centroid': p['centroid'],
                                'landmarks': p['landmarks'],
                                'values': p.get('values', None)
                            })

                    # Tracker returns ordered participant IDs for each detected face/pose
                    local_participant_ids = tracker.update(face_detections, pose_detections)

                    # Send participant updates to main process
                    for local_id in local_participant_ids:
                        participant = tracker.get_participant(local_id)
                        if participant and participant.get('face'):
                            centroid = participant['face']['centroid']
                            shape = participant.get('shape')
                            
                            # Send update to main process
                            update_msg = {
                                'camera_idx': camera_index,
                                'local_id': local_id,
                                'centroid': centroid,
                                'shape': shape
                            }
                            try:
                                participant_update_queue.put_nowait(update_msg)
                            except:
                                pass

                    # Build overlays for preview: use global IDs if available
                    for i, local_id in enumerate(local_participant_ids):
                        participant = tracker.get_participant(local_id)
                        if participant and participant.get('face'):
                            face_info = participant['face']
                            
                            # Check if we have a global mapping
                            if local_id in local_to_global:
                                global_id = local_to_global[local_id]
                                if global_id is None:
                                    # This face was rejected (max participants)
                                    if max_participants == 1:
                                        # In single participant mode, still show as participant 1
                                        display_id = 1
                                    else:
                                        # In multi-participant mode, skip this face entirely
                                        continue
                                else:
                                    display_id = global_id
                            else:
                                # No mapping yet - use a temporary local ID for preview
                                # This ensures faces are drawn immediately while waiting for global ID
                                display_id = f"local_{local_id}"
                                
                            faces_to_draw.append({
                                'landmarks': face_info['landmarks'],
                                'id': display_id,
                                'local_id': local_id,
                                'centroid': face_info['centroid'],
                                'blend': face_info['blend'],
                                'mesh': face_info.get('mesh'),  # Include mesh data
                                'signature': participant.get('signature')
                            })

                    # Get pose data for tracked participants
                    for local_id in local_participant_ids:
                        participant = tracker.get_participant(local_id)
                        if participant and participant.get('pose'):
                            pose_info = participant['pose']
                            
                            # Get global ID if available
                            if local_id in local_to_global:
                                global_id = local_to_global[local_id]
                                if global_id is None:
                                    # Rejected due to max participants
                                    if max_participants == 1:
                                        global_id = 1  # Force to 1 in single participant mode
                                    else:
                                        continue  # Skip in multi-participant mode
                                else:
                                    display_id = global_id
                            else:
                                # No mapping yet - use temporary ID for preview
                                display_id = f"local_{local_id}"
                                global_id = None  # Mark as temporary
                                
                            tracked_pose = {
                                'landmarks': pose_info['landmarks'],
                                'centroid': pose_info['centroid'],
                                'values': pose_info.get('values'),
                                'id': display_id if global_id is None else global_id,
                                'local_id': local_id
                            }
                            all_tracked_poses.append(tracked_pose)
                            if primary_pose is None:  # First pose becomes primary
                                primary_pose = tracked_pose

                frame_bgr = latest_frame['bgr']

                # === ALWAYS emit preview/record ===
                preview_data = {
                    'mode': 'unified',
                    'faces': faces_to_draw,         # List of tracked faces with IDs
                    'primary_face': faces_to_draw[0] if faces_to_draw else None,
                    'all_faces': latest_face_list if latest_face_list is not None else [],  # Raw detected faces
                    'pose': primary_pose,           # First tracked pose
                    'all_poses': all_tracked_poses,  # All tracked poses
                    'frame_bgr': frame_bgr,
                    'camera_index': camera_index,
                    'tracker': tracker
                }

                if preview_queue and not preview_queue.full():
                    try:
                        preview_queue.put_nowait(preview_data)
                    except:
                        pass

                if recording_queue and not recording_queue.full():
                    try:
                        recording_queue.put_nowait(preview_data.copy())
                    except:
                        pass

                # === Send face data for correlation (only when faces detected) ===
                if correlation_queue and faces_to_draw:
                        for face in faces_to_draw:
                            face_id = face['id']
                            
                            # Skip temporary IDs - they don't have global assignments yet
                            if isinstance(face_id, str) and face_id.startswith('local_'):
                                continue
                                
                            # Only process faces with valid global IDs
                            if isinstance(face_id, int):
                                # Get participant name - global_id is 1-based, names are 0-based
                                participant_name = participant_names.get(face_id - 1, f"P{face_id}")
                                
                                correlation_data = {
                                    'participant_id': participant_name,
                                    'blend_scores': face['blend'],
                                    'camera_index': camera_index
                                }
                                try:
                                    correlation_queue.put_nowait(correlation_data)
                                except:
                                    pass

                # === Export blendshapes/LSL (only when streaming and faces detected) ===
                if lsl_data_queue and faces_to_draw and streaming_active:
                    for face in faces_to_draw:
                        global_id = face.get('id')
                        
                        # Skip faces without proper global IDs
                        if global_id is None:
                            continue
                            
                        # get participant name - use 0-based index
                        participant_name = participant_names.get(global_id - 1, f"P{global_id}")
                        
                        if score_buffer and face == faces_to_draw[0]:
                            score_buffer.write(face['blend'])
                        
                        mesh_data = face.get('mesh')

                        lsl_data = {
                            'type': 'participant_data',
                            'participant_id': participant_name,
                            'global_id': global_id,
                            'blend_scores': face['blend'],
                            'mesh_data': mesh_data,  # This should contain the mesh data from face detection
                        }
                        try:
                            lsl_data_queue.put_nowait(lsl_data)
                            lsl_count += 1
                        except Exception as e:
                            print(f"[FUSION Camera {camera_index}] Error sending LSL data: {e}")

                    # === Export pose for each tracked participant ===
                    for pose in all_tracked_poses:
                        global_id = pose['id']
                        # get participant name
                        participant_name = participant_names.get(global_id-1, f"P{global_id}") #global_id is 1-based, names are 0-based
                        
                        pose_data_flat = pose.get('values')
                        if pose_data_flat is not None:
                            try:
                                lsl_data_queue.put_nowait({
                                    'type': 'pose_data',
                                    'participant_id': participant_name,
                                    'global_id': global_id,
                                    'pose_data': pose_data_flat
                                })
                            except queue.Full:
                                pass

                last_output_time = current_time

        # === Performance reporting ===
        if current_time - last_fps_report >= fps_report_interval:
            elapsed = fps_report_interval
            print(f"[FUSION Camera {camera_index}] Performance - "
                  f"Face: {face_count/elapsed:.1f} FPS, "
                  f"Pose: {pose_count/elapsed:.1f} FPS, "
                  f"LSL: {lsl_count/elapsed:.1f} FPS")

            face_count = pose_count = lsl_count = frame_count = 0
            last_fps_report = current_time

        time.sleep(0.001)

def parallel_participant_worker(camera_index: int,
                              model_path: str,
                              pose_model_path: str,
                              fps: int,
                              enable_raw_facemesh: bool,
                              enable_pose: bool,
                              preview_queue: MPQueue,
                              score_buffer_name: str,
                              control_conn,
                              recording_queue: MPQueue,
                              lsl_data_queue: MPQueue,
                              participant_update_queue,
                              worker_pipe,
                              correlation_queue,
                              max_participants=2,
                              resolution=(640, 480)):
    """
    Main coordinator that spawns separate face and pose processes.
    """
    print(f"[WORKER Camera {camera_index}] Starting")
    
    # Create queues for inter-process communication
    frame_queue_face = MPQueue(maxsize=2)
    frame_queue_pose = MPQueue(maxsize=2)
    face_result_queue = MPQueue(maxsize=2)
    pose_result_queue = MPQueue(maxsize=2)
    
    # Create frame distributor
    distributor = FrameDistributor(camera_index, resolution, fps)
    distributor.add_subscriber(frame_queue_face)
    distributor.add_subscriber(frame_queue_pose)
    
    # Create control pipes for sub-processes
    face_parent_conn, face_child_conn = Pipe()
    pose_parent_conn, pose_child_conn = Pipe()
    
    # Track current mesh state
    current_mesh_enabled = enable_raw_facemesh
    
    # Start processes
    face_proc = Process(
        target=face_worker_process,
        args=(frame_queue_face, face_result_queue, model_path, 
              enable_raw_facemesh, face_child_conn),
        daemon=True
    )
    face_proc.start()
    
    pose_proc = None
    if enable_pose:
        pose_model_path = 'D:/Projects/youquantipy/pose_landmarker_full.task'
        pose_proc = Process(
            target=pose_worker_process,
            args=(frame_queue_pose, pose_result_queue, 
                  pose_model_path, pose_child_conn),
            daemon=True
        )
        pose_proc.start()
    
    frame_queue_fusion = MPQueue(maxsize=2)
    distributor.add_subscriber(frame_queue_fusion)

    # Start data fusion
    fusion_proc = Process(
        target=data_fusion_process,
        args=(
            face_result_queue, pose_result_queue, frame_queue_fusion,
            preview_queue, recording_queue, lsl_data_queue, score_buffer_name,
            camera_index, fps, participant_update_queue, worker_pipe, correlation_queue,
            max_participants  
        ),
        daemon=True
    )
    fusion_proc.start()
    
    # Start frame distribution
    distributor.start()
    
    # Main control loop
    while True:
        # Check for control messages
        if control_conn and control_conn.poll():
            try:
                msg = control_conn.recv()
                
                if isinstance(msg, tuple):
                    command, *args = msg
                else:
                    command = msg
                    args = []
                
                if command == 'stop':
                    break
                elif command == 'enable_pose':
                    enabled = args[0] if args else True
                    if pose_parent_conn:
                        pose_parent_conn.send(('enable_pose', enabled))
                    # Start pose process if not running
                    if enabled and pose_proc is None:
                        pose_model_path = 'D:/Projects/youquantipy/pose_landmarker_heavy.task'
                        pose_proc = Process(
                            target=pose_worker_process,
                            args=(frame_queue_pose, pose_result_queue,
                                pose_model_path, pose_child_conn),
                            daemon=True
                        )
                        pose_proc.start()
                elif command == 'set_mesh':
                    current_mesh_enabled = args[0] if args else False
                    face_parent_conn.send(('set_mesh', current_mesh_enabled))
                    
                    # Notify LSL helper about mesh state change
                    if lsl_data_queue:
                        try:
                            lsl_data_queue.put_nowait({
                                'type': 'config_update',
                                'camera_index': camera_index,
                                'mesh_enabled': current_mesh_enabled
                            })
                        except:
                            pass
                            
                elif command == 'streaming_state':
                    # Just acknowledge receipt - GUI sends directly to fusion
                    streaming_active = args[0] if args else False
                    
                    # When streaming starts, notify LSL helper about mesh state
                    if streaming_active and lsl_data_queue:
                        try:
                            lsl_data_queue.put_nowait({
                                'type': 'config_update',
                                'camera_index': camera_index,
                                'mesh_enabled': current_mesh_enabled
                            })
                        except:
                            pass

            except Exception as e:
                print(f"[WORKER Camera {camera_index}] Error processing control message: {e}")
                
        time.sleep(0.01)
    
    # Cleanup
    print(f"[WORKER Camera {camera_index}] Stopping")
    distributor.stop()
    
    # Stop child processes
    face_parent_conn.send('stop')
    if pose_parent_conn:
        pose_parent_conn.send('stop')
    
    # Wait for processes to finish
    face_proc.join(timeout=2.0)
    if pose_proc:
        pose_proc.join(timeout=2.0)
    fusion_proc.terminate()
    fusion_proc.join(timeout=1.0)
    
    print(f"[WORKER Camera {camera_index}] Stopped")