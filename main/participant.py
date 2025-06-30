import cv2
import time
import threading
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pylsl import StreamInfo, StreamOutlet
from collections import OrderedDict, deque
from mediapipe.framework.formats import landmark_pb2
from facetracker import FaceTracker           # per-face centroid tracker
import queue     # for threading Queue
from multiprocessing import Queue as MPQueue 
from fastreader import NumpySharedBuffer
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings("ignore", category=UserWarning)



class DynamicFPSController:
    """Monitors actual frame rate and prevents duplicate processing"""
    def __init__(self, target_fps=30, window_size=30):
        self.target_fps = target_fps
        self.window_size = window_size
        self.frame_times = deque(maxlen=window_size)
        self.last_frame_hash = None
        self.duplicate_count = 0
        self.unique_count = 0
        self.actual_fps = 0
        self.effective_interval = 1.0 / target_fps
        
    def should_process_frame(self, frame):
        """Check if frame is unique and should be processed"""
        current_time = time.time()
        
        # Calculate frame hash to detect duplicates
        frame_hash = hash(frame.tobytes()[::1000])  # Sample every 1000th byte for speed
        
        is_duplicate = (frame_hash == self.last_frame_hash)
        self.last_frame_hash = frame_hash
        
        if is_duplicate:
            self.duplicate_count += 1
            return False
        
        self.unique_count += 1
        self.frame_times.append(current_time)
        
        # Calculate actual FPS from unique frames
        if len(self.frame_times) >= 2:
            time_span = self.frame_times[-1] - self.frame_times[0]
            self.actual_fps = len(self.frame_times) / time_span if time_span > 0 else 0
            
            # Adjust effective interval based on actual performance
            if self.actual_fps > 0:
                self.effective_interval = 1.0 / self.actual_fps
        
        # Log statistics periodically
        if (self.unique_count + self.duplicate_count) % 100 == 0:
            total = self.unique_count + self.duplicate_count
            dup_rate = (self.duplicate_count / total * 100) if total > 0 else 0
            print(f"[FPS] Actual: {self.actual_fps:.1f}, Target: {self.target_fps}, "
                  f"Duplicates: {dup_rate:.1f}%, Interval: {self.effective_interval*1000:.1f}ms")
        
        return True
    
    def get_sleep_time(self, processing_time):
        """Calculate optimal sleep time to maintain frame rate"""
        target_interval = 1.0 / self.target_fps
        sleep_time = max(0, target_interval - processing_time)
        
        # If we're getting duplicates, increase sleep time
        if self.duplicate_count > self.unique_count * 0.1:  # More than 10% duplicates
            sleep_time = max(sleep_time, self.effective_interval * 0.9)
            
        return sleep_time


def participant_worker(camera_index: int,
                    model_path: str,
                    participant_id: str,
                    stream_name: str,
                    fps: int,
                    enable_raw_facemesh: bool = False,
                    multi_face_mode: bool = False,
                    preview_queue: MPQueue = None,
                    score_buffer_name: str = None,
                    control_conn=None,
                    resolution= (640, 480)):
    """
    Runs in its own process.  Captures frames, runs Mediapipe models,
    builds payloads for the GUI (landmarks + IDs), streams to LSL,
    and pushes raw blendshape/pose vectors for the correlator.
    """
    # ─── 1) Initialize Participant and main LSL stream ────────────────
    fps_controller = DynamicFPSController(target_fps=fps)

    p = Participant(
        camera_index=camera_index,
        model_path=model_path,
        enable_raw_facemesh=enable_raw_facemesh,
        multi_face_mode=multi_face_mode,
        fps=fps,
        resolution=resolution
    )
    lsl_enabled = False
    score_buffer = None
    if score_buffer_name:
        try:
            score_buffer = NumpySharedBuffer(
                size=104 if multi_face_mode else 52, 
                name=score_buffer_name
            )
            print(f"[WORKER] Connected to shared buffer: {score_buffer_name}")
        except Exception as e:
            print(f"[WORKER] Failed to connect to shared buffer: {e}")

    if multi_face_mode:
        tracker = FaceTracker(track_threshold=200, max_missed=1500)

    last_sent_ts = 0
    cap = p.cap
    executor = ThreadPoolExecutor(max_workers=5, thread_name_prefix="participant_output")
    

    # ─── 4) Main loop ───────────────────────────────────────────────────
    while True:
        loop_start = time.time()
        
        # Check for control messages
        if control_conn is not None and control_conn.poll():
            msg = control_conn.recv()
            
            if isinstance(msg, tuple):
                command, *args = msg
            else:
                command = msg
                args = []
            
            if command == 'start_stream':
                if not multi_face_mode and args:
                    participant_id = args[0]
                    stream_name = f"{participant_id}_landmarks"
                
                if not multi_face_mode:
                    p.setup_stream(participant_id, stream_name)
                    print(f"[WORKER] outlet created: name={stream_name}, id={participant_id}")
                else:
                    if args and isinstance(args[0], dict):
                        p.set_face_names(args[0])
                    print(f"[WORKER] multiface mode - streams will be created per face")
                lsl_enabled = True
                
            elif command == 'stop_stream':
                p.stop_streaming()
                lsl_enabled = False
                
            elif command == 'set_mesh':
                new_val = args[0]
                p.enable_raw_facemesh = new_val
                enable_raw_facemesh = new_val
                if lsl_enabled:
                    p.stop_streaming()
                    if not p.multi_face_mode:
                        p.setup_stream(participant_id, stream_name)
                    lsl_enabled = True
        
        # Get next frame
        frame_data = p.get_next_frame()
        if frame_data is None:
            time.sleep(0.001)
            continue
        
        frame_bgr = frame_data['bgr']
        rgb = frame_data['rgb']
        
        # Check if this is a unique frame
        if not fps_controller.should_process_frame(frame_bgr):
            # Skip duplicate frame
            processing_time = time.time() - loop_start
            sleep_time = fps_controller.get_sleep_time(processing_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
            continue
        
        # Process unique frame
        if p.multi_face_mode:
            # ... (multiface processing code remains the same) ...
            loop_start_time = time.time()

            # Generate unique timestamp
            ts = int(time.monotonic() * 1000)
            if ts <= last_sent_ts:
                ts = last_sent_ts + 1
            last_sent_ts = ts
            
            # Send frame for async processing
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            p.multiface_landmarker.detect_async(mp_img, timestamp_ms=ts)

            # Try to get latest result
            latest_faces = p.get_latest_faces(frame_bgr.shape[1], frame_bgr.shape[0])

            # --- Centroid-track for stable IDs ---
            detections = [f['centroid'] for f in latest_faces]
            ids = tracker.update(detections)

            # --- Build per-face payload ---
            payload = []
            for fdata, fid in zip(latest_faces, ids):
                blends = fdata['blend']
                centroid = fdata['centroid']
                payload.append({
                    'id': fid,
                    'landmarks': fdata['landmarks'],
                    'blend': blends,
                    'centroid': centroid
                })

            # Parallel output operations
            futures = []
            
            # --- Send overlays to GUI ---
            if preview_queue is not None and not preview_queue.full():
                futures.append(executor.submit(
                    lambda: preview_queue.put_nowait({
                        'mode':  'multiface',
                        'faces': payload,
                        'frame_bgr': frame_bgr,
                    }) if not preview_queue.full() else None
                ))
                    
            if score_buffer is not None:
                def write_multiface_scores():
                    try:
                        if len(payload) >= 2:
                            face1_blend = payload[0]['blend']
                            face2_blend = payload[1]['blend']
                            combined_scores = face1_blend + face2_blend
                        elif len(payload) == 1:
                            face1_blend = payload[0]['blend']
                            combined_scores = face1_blend + [0.0] * 52
                        else:
                            combined_scores = [0.0] * 104
                        score_buffer.write(combined_scores)
                    except Exception as e:
                        print(f"[WORKER] Error sending multiface scores: {e}")
                
                futures.append(executor.submit(write_multiface_scores))

            if lsl_enabled:
                # --- Stream LSL ---
                futures.append(executor.submit(
                    p.push_multiface_samples,
                    faces=latest_faces,
                    face_ids=ids,
                    include_mesh=enable_raw_facemesh
                ))
                
            # Wait for all outputs to complete
            for future in futures:
                try:
                    future.result(timeout=0.01)  # 10ms timeout
                except:
                    pass
                    
            total_time = (time.time() - loop_start_time) * 1000
            frame_count = getattr(tracker, '_timing_frame_count', 0) + 1
            tracker._timing_frame_count = frame_count
            
            if frame_count % 30 == 0:
                effective_fps = 1000 / total_time if total_time > 0 else 0
                print(f"[MULTIFACE] Processing: {total_time:.1f}ms, Model FPS: {effective_fps:.1f}, "
                      f"Actual FPS: {fps_controller.actual_fps:.1f}")
        else:
            # Holistic mode processing
            loop_start_time = time.time()
            
            # Generate unique timestamp
            ts = int(time.monotonic() * 1000)
            if ts <= last_sent_ts:
                ts = last_sent_ts + 1
            last_sent_ts = ts
            
            # Start processing current frame's blendshapes
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            p.blendshape_landmarker.detect_async(mp_img, timestamp_ms=ts)
            
            # Check if we have a pending holistic result from previous frame
            holistic_res = None
            if p._pending_holistic:
                try:
                    result = p._pending_holistic.result(timeout=0.001)
                    if result is not None:
                        holistic_res = result
                        p._last_holistic_result = result
                except:
                    # Use last known good result on timeout
                    holistic_res = p._last_holistic_result
                finally:
                    # Always clear the pending future
                    p._pending_holistic = None
            
            # Start holistic processing for current frame (async)
            p._pending_holistic = p._holistic_executor.submit(p.process_holistic_frame, rgb)
            
            # For the very first frames, wait a bit longer for initial detection
            if holistic_res is None and p._last_holistic_result is None:
                try:
                    initial = p._pending_holistic.result(timeout=0.03)
                    if initial is not None:
                        holistic_res = initial
                        p._last_holistic_result = initial
                    p._pending_holistic = None
                except:
                    pass
            
            # Use the most recent valid result (could be from this frame or previous)
            face_lms = []
            pose_lms = []
            mesh_vals = []
            pose_vals = []
            
            if holistic_res is not None:
                face_lms = holistic_res.get('face_lms', [])
                pose_lms = holistic_res.get('pose_lms', [])
                mesh_vals = holistic_res.get('mesh_vals', []) if enable_raw_facemesh else []
                pose_vals = holistic_res.get('pose_vals', [])
            
            scores = p.get_latest_blend_scores()
            
            # Always send preview with current frame and latest landmarks
            if preview_queue is not None and not preview_queue.full():
                preview_queue.put_nowait({
                    'mode':      'holistic',
                    'face':      face_lms.copy() if face_lms else [],
                    'pose':      pose_lms.copy() if pose_lms else [],
                    'frame_bgr': frame_bgr.copy()
                })
            
            # Score buffer write
            if score_buffer is not None:
                def write_holistic_scores():
                    try:
                        scores_to_send = scores[:52] if len(scores) >= 52 else scores + [0.0] * (52 - len(scores))
                        score_buffer.write(scores_to_send)
                    except Exception as e:
                        print(f"[WORKER] Error sending holistic scores: {e}")
                
                executor.submit(write_holistic_scores)
            
            # LSL streaming
            if lsl_enabled:
                sample = scores + mesh_vals + pose_vals
                if len(sample) == p._lsl_info.channel_count():
                    if not hasattr(p, '_last_lsl_sample'):
                        p._last_lsl_sample = None
                        
                    if p._last_lsl_sample is not None and sample == p._last_lsl_sample:
                        pass  # Skip duplicate
                    else:
                        p._last_lsl_sample = sample.copy()
                        executor.submit(p.outlet.push_sample, sample)
                else:
                    print(f"[ERROR] Sample length {len(sample)} != LSL channels {p._lsl_info.channel_count()}")
            
            total_time = (time.time() - loop_start_time) * 1000
            
            # Print timing every 30 frames
            frame_count = getattr(p, '_timing_frame_count', 0) + 1
            p._timing_frame_count = frame_count
            
            if frame_count % 30 == 0:
                effective_fps = 1000 / total_time if total_time > 0 else 0
                print(f"[HOLISTIC] Processing: {total_time:.1f}ms, Model FPS: {effective_fps:.1f}, "
                      f"Actual FPS: {fps_controller.actual_fps:.1f}")
        
        # Dynamic frame rate control
        processing_time = time.time() - loop_start
        sleep_time = fps_controller.get_sleep_time(processing_time)
        if sleep_time > 0:
            time.sleep(sleep_time)
    
    # Cleanup
    executor.shutdown(wait=False)
    p.release()


class Participant:
    """
    Handles data acquisition (blendshapes, pose landmarks) and LSL streaming for a participant.
    Supports single-person holistic or multi-face modes with debug timing and robust camera capture.
    """
    def __init__(
        self,
        camera_index: int,
        model_path: str,
        enable_raw_facemesh: bool = False,
        multi_face_mode: bool = False,
        fps: int = 30,
        resolution= (640, 480) #default resolution is 640x480
    ):
        self.camera_index = camera_index
        self.resolution = resolution
        self.model_path   = model_path
        self.enable_raw_facemesh = enable_raw_facemesh
        self.multi_face_mode     = multi_face_mode
        self.fps = fps

        # Internal state and locking
        self._lock = threading.Lock()
        self.async_results = OrderedDict()
        self.result_queue = queue.Queue(maxsize=10)  # Increased size

        self.last_blend_scores = [0.0] * 52
        self.last_detected_faces = []
        self.blend_labels = None
        self._frame_idx = 0
        self._cached_result = None
        self.face_outlets = {}
        self.multiface_names = {}
        
        # Frame capture queue and thread
        self._frame_queue = deque(maxlen=2)  # Keep only latest frames
        self._capture_thread = None
        self._capture_running = False
        
        # Holistic processing pipeline
        self._holistic_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="holistic_model")
        self._pending_holistic = None
        self._last_holistic_result = None

        # Load model and setup Mediapipe base options
        with open(self.model_path, 'rb') as f:
            model_buffer = f.read()
        base_opts = python.BaseOptions(model_asset_buffer=model_buffer)

        # ─── Select and build the landmark processor ───────────────────
        if self.multi_face_mode:
            options = vision.FaceLandmarkerOptions(
                base_options=base_opts,
                output_face_blendshapes=True,
                running_mode=vision.RunningMode.LIVE_STREAM,
                num_faces=4,
                result_callback=self._on_multiface_result
            )
            self.multiface_landmarker = vision.FaceLandmarker.create_from_options(options)
        else:
            self.holistic = mp.solutions.holistic.Holistic(
                model_complexity=1,  # Reduced from 2 for better performance
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                refine_face_landmarks=False,
                smooth_landmarks=True, 
                enable_segmentation=False, 
                smooth_segmentation=False,
                static_image_mode=False  # Ensure tracking mode is enabled
            )
            blend_opts = vision.FaceLandmarkerOptions(
                base_options=base_opts,
                output_face_blendshapes=True,
                running_mode=vision.RunningMode.LIVE_STREAM,
                result_callback=self._on_blendshape_result
            )
            self.blendshape_landmarker = vision.FaceLandmarker.create_from_options(blend_opts)
            self.face_landmarker = self.blendshape_landmarker

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

    def initialize_camera(self,cap, fps, resolution, 
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
        print("[Camera] Starting advanced camera initialization...")
        # Basic settings
        cap.set(cv2.CAP_PROP_FPS, fps)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

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

    def _start_capture_thread(self):
        """Start the background thread for continuous frame capture"""
        self._capture_running = True
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()

    
    def get_camera_info(self):
        """Get camera capabilities and actual FPS"""
        if not hasattr(self, 'cap') or not self.cap.isOpened():
            return None
            
        info = {
            'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': self.cap.get(cv2.CAP_PROP_FPS),
            'backend': self.cap.get(cv2.CAP_PROP_BACKEND),
            'fourcc': self.cap.get(cv2.CAP_PROP_FOURCC)
        }
        
        # Test actual frame rate
        test_frames = 30
        start_time = time.time()
        unique_frames = 0
        last_frame = None
        
        for _ in range(test_frames):
            ret, frame = self.cap.read()
            if ret and frame is not None:
                # Simple check for frame uniqueness
                if last_frame is None or not np.array_equal(frame[::10, ::10], last_frame):
                    unique_frames += 1
                    last_frame = frame[::10, ::10].copy()
        
        elapsed = time.time() - start_time
        info['tested_fps'] = unique_frames / elapsed if elapsed > 0 else 0
        
        print(f"[Camera Info] Resolution: {info['width']}x{info['height']}, "
              f"Reported FPS: {info['fps']}, Tested FPS: {info['tested_fps']:.1f}")
        
        return info
        
    def _capture_loop(self):
        """Continuous capture loop running in separate thread"""
        while self._capture_running:
            ret, frame_bgr = self.cap.read()
            if ret:
                # Convert to RGB once
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                
                # Store in queue (automatically drops old frames due to maxlen)
                with self._lock:
                    self._frame_queue.append({
                        'bgr': frame_bgr,
                        'rgb': rgb,
                        'timestamp': time.time()
                    })
            else:
                time.sleep(0.001)
                
    def get_next_frame(self):
        """Get the latest captured frame"""
        with self._lock:
            if self._frame_queue:
                return self._frame_queue[-1]  # Return latest frame
        return None
        
    def process_blendshape_async(self, mp_img, timestamp_ms):
        """Wrapper to call async blendshape detection synchronously from thread"""
        self.blendshape_landmarker.detect_async(mp_img, timestamp_ms=timestamp_ms)
        # The callback _on_blendshape_result will update self.last_blend_scores
        
    def process_holistic_frame(self, rgb):
        """Process a frame through holistic model and return structured data"""
        res = self.holistic.process(rgb)
        
        if not res.face_landmarks and not res.pose_landmarks:
            return None
            
        # Extract face landmarks
        face_lms = []
        if res.face_landmarks:
            face_lms = [(lm.x, lm.y, lm.z) for lm in res.face_landmarks.landmark]
            
        # Extract pose landmarks
        pose_lms = []
        pose_vals = []
        if res.pose_landmarks:
            pose_lms = [(lm.x, lm.y, lm.z) for lm in res.pose_landmarks.landmark]
            for lm in res.pose_landmarks.landmark:
                pose_vals.extend([lm.x, lm.y, lm.z, lm.visibility])
        else:
            pose_vals = [0.0] * (33 * 4)
            
        # Build mesh values if enabled
        mesh_vals = []
        if self.enable_raw_facemesh and res.face_landmarks:
            for lm in res.face_landmarks.landmark:
                mesh_vals.extend([lm.x, lm.y, lm.z])
                
        return {
            'face_lms': face_lms,
            'pose_lms': pose_lms,
            'mesh_vals': mesh_vals,
            'pose_vals': pose_vals
        }

    # ─── Callback functions for results ─────────────────────────────

    def _on_multiface_result(self, res, image, timestamp_ms):
        """Callback for multiface detection results - runs in MediaPipe thread"""
        try:
            with self._lock:
                # Keep only recent results to prevent memory buildup
                if len(self.async_results) > 5:
                    # Remove oldest result
                    self.async_results.popitem(last=False)
                
                self.async_results[timestamp_ms] = res
                
                # Update blend labels if we haven't set them yet
                if self.blend_labels is None and res.face_blendshapes:
                    self.blend_labels = [cat.category_name for cat in res.face_blendshapes[0]]
                
                # Notify that new result is available
                if not self.result_queue.full():
                    self.result_queue.put_nowait(timestamp_ms)
        except Exception as e:
            print(f"[ERROR] _on_multiface_result: {e}")

    def _on_blendshape_result(self, res, image, timestamp_ms):
        """Callback for blendshape detection results - runs in MediaPipe thread"""
        try:
            shapes = res.face_blendshapes[0] if res.face_blendshapes else []
            scores = [cat.score for cat in shapes]
            scores = scores[:52] + [0.0] * (52 - len(scores))
            
            with self._lock:
                self.last_blend_scores = scores
                
                if self.blend_labels is None and res.face_blendshapes:
                    self.blend_labels = [cat.category_name for cat in res.face_blendshapes[0]]
        except Exception as e:
            print(f"[ERROR] _on_blendshape_result: {e}")

    def get_latest_faces(self, frame_w, frame_h):
        """Get the latest detected faces, processing any new results"""
        # Process any new results that have arrived
        processed_any = False
        try:
            while not self.result_queue.empty():
                try:
                    timestamp_ms = self.result_queue.get_nowait()
                    with self._lock:
                        if timestamp_ms in self.async_results:
                            res = self.async_results.pop(timestamp_ms)
                            self.last_detected_faces = self._parse_multiface_result(res, frame_w, frame_h)
                            processed_any = True
                except queue.Empty:
                    break
        except Exception as e:
            print(f"[ERROR] get_latest_faces: {e}")
        
        return self.last_detected_faces
    
    def set_face_names(self, names_dict):
        """Update face names for multiface streaming"""
        self.multiface_names = names_dict

    def get_latest_blend_scores(self):
        """Thread-safe access to latest blend scores"""
        with self._lock:
            return self.last_blend_scores.copy()

    def _parse_multiface_result(self, res, width, height):
        """
        Convert FaceLandmarkerResult to a list of dictionaries usable by GUI/LSL.
        """
        faces = []
        if not res.face_landmarks:
            return faces
            
        for idx, (landmarks, blendshapes) in enumerate(zip(res.face_landmarks, res.face_blendshapes)):
            mesh_vals = [(pt.x, pt.y, pt.z) for pt in landmarks]
            blend_vals = [b.score for b in blendshapes][:52] + [0.0]*(52 - len(blendshapes))
            cx = np.mean([pt.x for pt in landmarks])
            cy = np.mean([pt.y for pt in landmarks])
            faces.append({
                'id': idx,
                'landmarks': mesh_vals,
                'mesh': mesh_vals,
                'blend': blend_vals,
                'centroid': (cx, cy)
            })
        return faces

    def setup_stream(self, participant_id: str, stream_name: str):
        """
        Initialize LSL stream outlet.
        """
        self.participant_id = participant_id
        self.stream_name    = stream_name
        self.source_id      = f"{participant_id}_uid"
        
        if self.multi_face_mode:
            mesh_ch = (478*3) if self.enable_raw_facemesh else 0
        else:
            mesh_ch = (468*3) if self.enable_raw_facemesh else 0
        total_ch = 52 + mesh_ch + (33*4)
        info = StreamInfo(
            name          = self.stream_name,
            type          = "Landmark",
            channel_count = total_ch,
            nominal_srate = self.fps,
            channel_format="float32",
            source_id     = self.source_id
        )
        self._lsl_info = info
        self.outlet = StreamOutlet(info)
        print(f"[WORKER] LSL outlet up → {stream_name} @ {self.source_id} ({info.channel_count} ch)", flush=True)

    def push_multiface_samples(self, faces: list, face_ids: list, include_mesh: bool):
        """
        Stream each face's blendshapes and optional mesh to LSL.
        """
        # Check if face names have changed and recreate outlets if needed
        for fid in face_ids:
            current_name = getattr(self, 'multiface_names', {}).get(fid, f"ID{fid}")
            if fid in self.face_outlets:
                # Check if name changed by comparing the stream name
                outlet = self.face_outlets[fid]
                # Store the expected name for this outlet
                expected_name = f"{current_name}_face{fid}"
                existing_name = getattr(outlet, '_stream_name', None)
                
                if existing_name != expected_name:
                    # Name changed, close old outlet
                    try:
                        outlet.close()
                    except:
                        pass
                    del self.face_outlets[fid]
        
        # Now create streams for each face
        for fdata, fid in zip(faces, face_ids):
            blend = fdata['blend']
            if include_mesh:
                mesh_tuples = fdata['mesh']
                mesh = [coord for pt in mesh_tuples for coord in (pt[0], pt[1], pt[2])]
            else:
                mesh = []
                
            sample = blend + mesh

            base = getattr(self, 'multiface_names', {}).get(fid, f"ID{fid}")
            stream_name = f"{base}_face{fid}"

            if fid not in self.face_outlets:
                ch_count = len(blend) + len(mesh)
                info = StreamInfo(
                    name=stream_name,
                    type="Landmark",
                    channel_count=ch_count,
                    nominal_srate=self.fps,
                    channel_format="float32",
                    source_id=stream_name
                )
                self.face_outlets[fid] = StreamOutlet(info)
                # Store the stream name for future comparison
                self.face_outlets[fid]._stream_name = stream_name
                print(f"[WORKER] Created LSL outlet for face {fid}: {stream_name}")

            # Create a per-face sample cache if it doesn't exist
            if not hasattr(self, '_last_face_samples'):
                self._last_face_samples = {}
                
            # Check for duplicate samples per face
            if fid in self._last_face_samples and sample == self._last_face_samples[fid]:
                continue  # Skip duplicate for this face
                
            self._last_face_samples[fid] = sample.copy()
            self.face_outlets[fid].push_sample(sample)

    def release(self):
        """
        Release all camera and Mediapipe resources.
        """
        # Stop capture thread
        self._capture_running = False
        if self._capture_thread:
            self._capture_thread.join(timeout=1.0)
            
        # Shutdown executor
        if hasattr(self, '_holistic_executor'):
            self._holistic_executor.shutdown(wait=False)
            
        if hasattr(self, 'holistic'):
            self.holistic.close()
        if hasattr(self, 'blendshape_landmarker'):
            self.blendshape_landmarker.close()
        if hasattr(self, 'multiface_landmarker'):
            self.multiface_landmarker.close()
        if getattr(self, 'outlet', None):
            try: 
                self.outlet.close()
            except: 
                pass
        if hasattr(self, 'face_outlets'):
            for ol in self.face_outlets.values():
                try: 
                    ol.close()
                except: 
                    pass
            self.face_outlets.clear()
        if hasattr(self, 'cap'):
            self.cap.release()

    def stop_streaming(self):
        """
        Close all LSL outlets but retain camera.
        """
        if getattr(self, 'outlet', None):
            try: 
                self.outlet.close()
            except: 
                pass
            self.outlet = None
        for fid, ol in list(self.face_outlets.items()):
            try: 
                ol.close()
            except: 
                pass
        self.face_outlets.clear()