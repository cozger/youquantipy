import cv2
import time
import threading
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pylsl import StreamInfo, StreamOutlet
from collections import OrderedDict 
from mediapipe.framework.formats import landmark_pb2
from facetracker import FaceTracker           # per-face centroid tracker
import queue     # for threading Queue
from multiprocessing import Queue as MPQueue 
from fastreader import NumpySharedBuffer


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

    # ─── 2) If multi-face, prepare a local FaceTracker ─
    if multi_face_mode:
        tracker = FaceTracker(track_threshold=200, max_missed=1500)

    last_sent_ts = 0
    cap = p.cap
    frame_delay = 1.0 / fps

    # ─── 4) Main loop ───────────────────────────────────────────────────
    while True:
        loop_start = time.time()
        
        # Check for control messages
        if control_conn is not None and control_conn.poll():
            msg = control_conn.recv()
            
            # Handle both old string format and new tuple format
            if isinstance(msg, tuple):
                command, *args = msg
            else:
                command = msg
                args = []
            
            print(f"[WORKER] received command → {command}", flush=True)
            
            if command == 'start_stream':
                if not multi_face_mode and args:
                    # Update participant_id with the provided name
                    participant_id = args[0]
                    stream_name = f"{participant_id}_landmarks"
                
                if not multi_face_mode:
                    # Holistic mode - single stream for this participant
                    p.setup_stream(participant_id, stream_name)
                    print(f"[WORKER] outlet created: name={stream_name}, id={participant_id}", flush=True)
                else:
                    if args and isinstance(args[0], dict):
                        p.set_face_names(args[0])
                        print(f"[WORKER] multiface mode - face names received: {args[0]}", flush=True)
                    print(f"[WORKER] multiface mode - streams will be created per face", flush=True)
                lsl_enabled = True
            elif command == 'stop_stream':
                p.stop_streaming()
                print(f"[WORKER] streaming stopped", flush=True)
                lsl_enabled = False
        
        ret, frame_bgr = cap.read()
        if not ret:
            time.sleep(frame_delay)
            continue

        # Convert once to RGB
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        if p.multi_face_mode:
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

            # --- Send overlays to GUI ---
            if preview_queue is not None and not preview_queue.full():
                try:
                    preview_queue.put_nowait({
                        'mode':  'multiface',
                        'faces': payload,
                        'frame_bgr': frame_bgr,
                    })
                except:
                    pass  # Queue full, skip this frame
                    
            if score_buffer is not None:
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

            if lsl_enabled:
                # --- Stream LSL ---
                p.push_multiface_samples(faces=latest_faces,
                                        face_ids=ids,
                                        include_mesh=enable_raw_facemesh)
            total_time = (time.time() - loop_start_time) * 1000
            frame_count = getattr(tracker, '_timing_frame_count', 0) + 1
            tracker._timing_frame_count = frame_count
            
            if frame_count % 30 == 0:
                effective_fps = 1000 / total_time if total_time > 0 else 0
                print(f"[MULTIFACE] Loop time: {total_time:.1f}ms, Effective FPS: {effective_fps:.1f}")

        else:
            # Holistic mode with detailed timing
            loop_start_time = time.time()
            
            # Generate unique timestamp for blendshape detection
            ts = int(time.monotonic() * 1000)
            if ts <= last_sent_ts:
                ts = last_sent_ts + 1
            last_sent_ts = ts
            
            # Time the async blendshape call
            t1 = time.time()
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            p.blendshape_landmarker.detect_async(mp_img, timestamp_ms=ts)
            async_time = (time.time() - t1) * 1000
            
            # Time the holistic processing (the suspected bottleneck)
            t2 = time.time()
            res = p.holistic.process(rgb)
            holistic_time = (time.time() - t2) * 1000
            
            # Time landmark extraction
            t3 = time.time()
            if res.face_landmarks:
                face_lms = [(lm.x, lm.y, lm.z) for lm in (res.face_landmarks.landmark or [])]
            else:
                face_lms = []
            if res.pose_landmarks:
                pose_lms = [(lm.x, lm.y, lm.z) for lm in (res.pose_landmarks.landmark or [])]
            else:
                pose_lms = []
            extract_time = (time.time() - t3) * 1000

            # Time mesh/pose data building
            t4 = time.time()
            mesh_vals = []
            if enable_raw_facemesh and getattr(res, 'face_world_mesh', None):
                for lm in res.face_landmarks.landmark:
                    mesh_vals.extend([lm.x, lm.y, lm.z])

            pose_vals = []
            if res.pose_landmarks:
                for lm in res.pose_landmarks.landmark:
                    pose_vals.extend([lm.x, lm.y, lm.z, lm.visibility])
            else:
                pose_vals = [0.0] * (33 * 4)
            build_time = (time.time() - t4) * 1000

            # Time getting blend scores
            t5 = time.time()
            scores = p.get_latest_blend_scores()
            frame_count = getattr(p, '_scores_debug_count', 0) + 1
            p._scores_debug_count = frame_count
            if frame_count % 30 == 0:
                print(f"[WORKER] Got {len(scores)} blend scores, first few: {scores[:5] if scores else 'None'}")
            scores_time = (time.time() - t5) * 1000

            # Time GUI update
            t6 = time.time()
            if preview_queue is not None and not preview_queue.full():
                try:
                    preview_queue.put_nowait({
                        'mode': 'holistic',
                        'face': face_lms,
                        'pose': pose_lms,
                        'frame_bgr': frame_bgr
                    })
                except:
                    pass
            gui_time = (time.time() - t6) * 1000
                    
            # Time score buffer write
            t7 = time.time()
            if score_buffer is not None:
                    try:
                        scores_to_send = scores[:52] if len(scores) >= 52 else scores + [0.0] * (52 - len(scores))
                        score_buffer.write(scores_to_send)
                        
                        # Debug output every 30 frames
                        frame_count = getattr(p, '_buffer_write_count', 0) + 1
                        p._buffer_write_count = frame_count
                        if frame_count % 30 == 0:
                            print(f"[WORKER] Writing {len(scores_to_send)} scores to buffer, first few: {scores_to_send[:5]}")
                            
                    except Exception as e:
                        print(f"[WORKER] Error sending holistic scores: {e}")
            # Time LSL streaming
            t8 = time.time()
            if lsl_enabled:
                sample = scores + mesh_vals + pose_vals
                if len(sample) == p._lsl_info.channel_count():
                    p.outlet.push_sample(sample)
                else:
                    print(f"[ERROR] Sample length {len(sample)} != LSL channels {p._lsl_info.channel_count()}")
            lsl_time = (time.time() - t8) * 1000
            
            total_time = (time.time() - loop_start_time) * 1000
            
            # Print timing every 10 frames (more frequent for debugging)
            frame_count = getattr(p, '_timing_frame_count', 0) + 1
            p._timing_frame_count = frame_count
            
            if frame_count % 10 == 0:
                print(f"[HOLISTIC TIMING] Async: {async_time:.1f}ms, Holistic: {holistic_time:.1f}ms, "
                      f"Extract: {extract_time:.1f}ms, Build: {build_time:.1f}ms, Scores: {scores_time:.1f}ms, "
                      f"GUI: {gui_time:.1f}ms, LSL: {lsl_time:.1f}ms, "
                      f"TOTAL: {total_time:.1f}ms")
                
                # Calculate effective FPS
                effective_fps = 1000 / total_time if total_time > 0 else 0
                print(f"[HOLISTIC] Effective FPS: {effective_fps:.1f}")

        # Frame rate control
        elapsed = time.time() - loop_start
        sleep_time = max(0, frame_delay - elapsed)
        if sleep_time > 0:
            time.sleep(sleep_time)


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
                model_complexity=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                refine_face_landmarks=False,
                smooth_landmarks=True, 
                enable_segmentation=False, 
                smooth_segmentation=False
            )
            blend_opts = vision.FaceLandmarkerOptions(
                base_options=base_opts,
                output_face_blendshapes=True,
                running_mode=vision.RunningMode.LIVE_STREAM,
                result_callback=self._on_blendshape_result
            )
            self.blendshape_landmarker = vision.FaceLandmarker.create_from_options(blend_opts)
            self.face_landmarker = self.blendshape_landmarker

        # ─── Setup camera capture ──────────────────────────────────────
        # Replace the camera setup section in Participant.__init__ with this:

        # ─── Setup camera capture ──────────────────────────────────────
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        for b in backends:
            cap = cv2.VideoCapture(self.camera_index, b)
            if cap.isOpened():
                self.cap = cap
                break
        else:
            raise RuntimeError(f"Cannot open camera {self.camera_index!r}")

        # Basic settings
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer lag
        
        # Try multiple approaches for auto-exposure
        print("[Camera] Attempting to enable auto-exposure...")
        auto_exposure_values = [3, 1, 0.75, 0.25, -1]  # Different values cameras might accept
        for val in auto_exposure_values:
            try:
                result = self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, val)
                current_val = self.cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
                print(f"[Camera] Auto-exposure set to {val}, result: {result}, current: {current_val}")
                if result:
                    break
            except:
                continue
        
        # Try autofocus
        print("[Camera] Attempting to enable auto-focus...")
        autofocus_values = [1, -1, True]
        for val in autofocus_values:
            try:
                result = self.cap.set(cv2.CAP_PROP_AUTOFOCUS, val)
                current_val = self.cap.get(cv2.CAP_PROP_AUTOFOCUS)
                print(f"[Camera] Auto-focus set to {val}, result: {result}, current: {current_val}")
                if result:
                    break
            except:
                continue
        
        # Try auto white balance
        print("[Camera] Attempting to enable auto white balance...")
        auto_wb_values = [1, -1, True]
        for val in auto_wb_values:
            try:
                result = self.cap.set(cv2.CAP_PROP_AUTO_WB, val)
                current_val = self.cap.get(cv2.CAP_PROP_AUTO_WB)
                print(f"[Camera] Auto WB set to {val}, result: {result}, current: {current_val}")
                if result:
                    break
            except:
                continue
        
        # Manual brightness/exposure adjustments if auto doesn't work
        print("[Camera] Checking current exposure/brightness...")
        current_exposure = self.cap.get(cv2.CAP_PROP_EXPOSURE)
        current_brightness = self.cap.get(cv2.CAP_PROP_BRIGHTNESS)
        current_gain = self.cap.get(cv2.CAP_PROP_GAIN)
        
        print(f"[Camera] Current - Exposure: {current_exposure}, Brightness: {current_brightness}, Gain: {current_gain}")
        
        # If image is too dark, try manual adjustments
        if current_exposure < -5:  # Very low exposure
            print("[Camera] Exposure seems low, attempting to increase...")
            self.cap.set(cv2.CAP_PROP_EXPOSURE, -3)  # Increase exposure
            
        if current_brightness < 0.3:  # Low brightness
            print("[Camera] Brightness seems low, attempting to increase...")
            self.cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.5)  # Increase brightness
            
        if current_gain < 0.3:  # Low gain
            print("[Camera] Gain seems low, attempting to increase...")
            self.cap.set(cv2.CAP_PROP_GAIN, 0.5)  # Increase gain
        
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        # Final status check
        final_exposure = self.cap.get(cv2.CAP_PROP_EXPOSURE)
        final_brightness = self.cap.get(cv2.CAP_PROP_BRIGHTNESS)
        final_auto_exposure = self.cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
        final_autofocus = self.cap.get(cv2.CAP_PROP_AUTOFOCUS)
        final_auto_wb = self.cap.get(cv2.CAP_PROP_AUTO_WB)
        
        print(f"[Camera] Final settings:")
        print(f"  Resolution: {w}x{h}@{actual_fps}fps")
        print(f"  Auto-exposure: {final_auto_exposure}")
        print(f"  Auto-focus: {final_autofocus}")
        print(f"  Auto-WB: {final_auto_wb}")
        print(f"  Exposure: {final_exposure}")
        print(f"  Brightness: {final_brightness}")
        
        # Warm up camera (let it adjust)
        print("[Camera] Warming up for 2 seconds...")
        start_time = time.time()
        while time.time() - start_time < 2.0:
            ret, frame = self.cap.read()
            if not ret:
                break
            time.sleep(0.1)
        print("[Camera] Camera ready!")
        
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        print(f"[Participant] camera {self.camera_index}: {w}x{h}@{actual_fps}fps")
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

            self.face_outlets[fid].push_sample(sample)

    def release(self):
        """
        Release all camera and Mediapipe resources.
        """
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