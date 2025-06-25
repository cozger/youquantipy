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

def participant_worker(camera_index: int,
                    model_path: str,
                    participant_id: str,
                    stream_name: str,
                    fps: int,
                    enable_raw_facemesh: bool = False,
                    multi_face_mode: bool = False,
                    preview_queue: MPQueue = None,
                    score_queue: MPQueue = None,
                    control_conn=None):
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
        fps=fps
    )
    lsl_enabled = False

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
            print(f"[WORKER] received command → {msg}", flush=True)
            if msg == 'start_stream':
                p.setup_stream(participant_id, stream_name)
                print(f"[WORKER] outlet created: name={stream_name}, id={participant_id}", flush=True)
                lsl_enabled = True
            elif msg == 'stop_stream':
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
                    
            if score_queue is not None and not score_queue.full():
                try:
                    score_queue.put_nowait(payload[0]['blend'] if payload else [0.0]*52)
                except:
                    pass  # Queue full, skip

            if lsl_enabled:
                # --- Stream LSL ---
                p.push_multiface_samples(faces=latest_faces,
                                        face_ids=ids,
                                        include_mesh=enable_raw_facemesh)

        else:
            # Holistic mode
            # Generate unique timestamp for blendshape detection
            ts = int(time.monotonic() * 1000)
            if ts <= last_sent_ts:
                ts = last_sent_ts + 1
            last_sent_ts = ts
            
            # Start async blendshape detection
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            p.blendshape_landmarker.detect_async(mp_img, timestamp_ms=ts)

            # Run holistic processing (synchronous)
            res = p.holistic.process(rgb)

            # Extract landmarks for GUI
            if res.face_landmarks:
                face_lms = [(lm.x, lm.y, lm.z) for lm in (res.face_landmarks.landmark or [])]
            else:
                face_lms = []
            if res.pose_landmarks:
                pose_lms = [(lm.x, lm.y, lm.z) for lm in (res.pose_landmarks.landmark or [])]
            else:
                pose_lms = []

            # Build mesh and pose data for LSL
            mesh_vals = []
            if enable_raw_facemesh and getattr(res, 'face_world_mesh', None):
                for v in res.face_world_mesh.vertices:
                    mesh_vals.extend([v.x, v.y, v.z])

            pose_vals = []
            if res.pose_landmarks:
                for lm in res.pose_landmarks.landmark:
                    pose_vals.extend([lm.x, lm.y, lm.z, lm.visibility])
            else:
                pose_vals = [0.0] * (33 * 4)

            # Get latest blendshape scores
            scores = p.get_latest_blend_scores()

            # Send to GUI
            if preview_queue is not None and not preview_queue.full():
                try:
                    preview_queue.put_nowait({
                        'mode': 'holistic',
                        'face': face_lms,
                        'pose': pose_lms,
                        'frame_bgr': frame_bgr
                    })
                except:
                    pass  # Queue full, skip
                    
            if score_queue is not None and not score_queue.full():
                try:
                    score_queue.put_nowait(scores)
                except:
                    pass  # Queue full, skip

            # Build and stream full sample to LSL
            if lsl_enabled:
                sample = scores + mesh_vals + pose_vals
                if len(sample) == p._lsl_info.channel_count():
                    p.outlet.push_sample(sample)
                else:
                    print(f"[ERROR] Sample length {len(sample)} != LSL channels {p._lsl_info.channel_count()}")

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
                model_complexity=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                refine_face_landmarks=True
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
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        for b in backends:
            cap = cv2.VideoCapture(self.camera_index, b)
            if cap.isOpened():
                self.cap = cap
                break
        else:
            raise RuntimeError(f"Cannot open camera {self.camera_index!r}")

        # Set properties explicitly
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])

        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[Participant] camera {self.camera_index} resolution = {w}x{h}")

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
        
        mesh_ch = (478*3) if self.enable_raw_facemesh else 0
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
        for fdata, fid in zip(faces, face_ids):
            blend = fdata['blend']
            mesh  = fdata['mesh'] if include_mesh else []
            sample = blend + mesh

            base = getattr(self, 'multiface_names', {}).get(fid, f"ID{fid}")

            if fid not in self.face_outlets:
                ch_count = len(blend) + len(mesh)
                info = StreamInfo(
                    name          = f"{base}_face{fid}",
                    type          = "Landmark",
                    channel_count = ch_count,
                    nominal_srate = self.fps,
                    channel_format= "float32",
                    source_id     = f"{base}_face{fid}"
                )
                self.face_outlets[fid] = StreamOutlet(info)

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