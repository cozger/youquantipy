import cv2
import time
import threading
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pylsl import StreamInfo, StreamOutlet
from collections import OrderedDict 


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
        multi_face_mode: bool = False,      # ← new arg
        fps: int = 30
    ):
        self.camera_index = camera_index
        self.model_path   = model_path
        self.enable_raw_facemesh = enable_raw_facemesh
        self.multi_face_mode     = multi_face_mode
        self.fps = fps

        # threading lock & async state
        self._lock = threading.Lock()
        self._detect_busy = False
        self.async_results = OrderedDict()
        self.last_blend_scores = [0.0] * 52
        self.last_detected_faces = []
        self.blend_labels = None


        # Load model buffer once
        with open(self.model_path, 'rb') as f:
            model_buffer = f.read()
        base_opts = python.BaseOptions(model_asset_buffer=model_buffer)

        # Callback for multi-face mode
        def _on_multiface_result(res, image, timestamp_ms):
            with self._lock:
                self.async_results[timestamp_ms] = res
                self._detect_busy = False
                if self.blend_labels is None and res.face_blendshapes:
                    self.blend_labels = [
                        cat.category_name for cat in res.face_blendshapes[0]
                    ]

        # Callback for single-face blendshapes
        def _on_blendshape_result(res, image, timestamp_ms):
            shapes = res.face_blendshapes[0] if res.face_blendshapes else []
            scores = [cat.score for cat in shapes]
            scores = scores[:52] + [0.0] * (52 - len(scores))
            with self._lock:
                self.last_blend_scores = scores
                self._detect_busy = False
                if self.blend_labels is None and res.face_blendshapes:
                  self.blend_labels = [
                    cat.category_name for cat in res.face_blendshapes[0]
                ]

        # ─── select and build exactly one landmarker ─────────────────────────────
        if self.multi_face_mode:
            # multi-face: return up to 4 sets of meshes + blendshapes
            options = vision.FaceLandmarkerOptions(
                base_options=base_opts,
                output_face_blendshapes=True,
                running_mode=vision.RunningMode.LIVE_STREAM,
                num_faces=4,
                result_callback=_on_multiface_result
            )
            self.multiface_landmarker = vision.FaceLandmarker.create_from_options(options)
        else:
            # single-face: holistic for mesh+pose, plus blendshape-only landmarker
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
                result_callback=_on_blendshape_result
            )
            self.blendshape_landmarker = vision.FaceLandmarker.create_from_options(blend_opts)
            # alias for compatibility with read_frame
            self.face_landmarker = self.blendshape_landmarker

        # Placeholders for LSL outlets
        self.outlet = None
        self.participant_id = None
        self.stream_name = None
        self.source_id = None
        self.face_outlets = {}

        # Robust VideoCapture: try DSHOW, MSMF, then default
        backends = []
        if hasattr(cv2, 'CAP_DSHOW'): backends.append(cv2.CAP_DSHOW)
        if hasattr(cv2, 'CAP_MSMF'):  backends.append(cv2.CAP_MSMF)
        backends.append(0)

        cap = None
        for b in backends:
            try:
                c = cv2.VideoCapture(self.camera_index, b) if b != 0 else cv2.VideoCapture(self.camera_index)
                if c.isOpened():
                    cap = c
                    break
                else:
                    c.release()
            except Exception:
                continue
        if cap is None or not cap.isOpened():
            raise RuntimeError(f"Could not open camera #{self.camera_index}")
        self.cap = cap
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[Participant] camera {self.camera_index} resolution = {w}x{h}")

    def detect_faces_multiface(self, mp_img):
        """
        Runs async multi-face detection and returns parsed faces.
        """
        frame_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ts = int(time.monotonic() * 1e3)

        with self._lock:
            busy = self._detect_busy
        if not busy:
            with self._lock:
                self._detect_busy = True
            self.multiface_landmarker.detect_async(mp_img, timestamp_ms=ts)

        faces = []
        with self._lock:
            res = self.async_results.pop(ts, None)
        if res:
            faces = self._parse_multiface_result(res, frame_w, frame_h)
        return faces

    def _parse_multiface_result(self, res, width, height):
        """Convert FaceLandmarkerResult -> list[dict] usable by GUI/LSL."""
        faces = []
        for landmarks, blendshapes in zip(
            res.face_landmarks, res.face_blendshapes
        ):
            mesh_vals = [coord for pt in landmarks
                                for coord in (pt.x, pt.y, pt.z)]

            blend_vals = [b.score for b in blendshapes]
            blend_vals = blend_vals[:52] + [0.0]*(52 - len(blend_vals))

            # centroid ~ mean landmark location
            xs = [pt.x for pt in landmarks]
            ys = [pt.y for pt in landmarks]
            cx = int(np.mean(xs) * width)
            cy = int(np.mean(ys) * height)

            faces.append({
                'landmarks': landmarks,
                'mesh': mesh_vals,
                'blend': blend_vals,
                'centroid': (cx, cy)
            })
        return faces

    def setup_stream(self, participant_id: str, stream_name: str):
        """
        Initialize LSL stream for single or multi-face mode.
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
        self.outlet = StreamOutlet(info)

    def read_frame(self):
        """
        Capture one frame, run detection(s), and prepare data for streaming or display.
        """
        success, frame = self.cap.read()
        if not success:
            return None
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        if self.multi_face_mode:
            # multi-face path
            t0 = time.time()
            faces = self.detect_faces_multiface(mp_img)
            print(f"[Participant] multi detect {(time.time()-t0)*1000:.1f}ms")
            self.last_detected_faces = faces
            return faces

        else:
            # single-face path
            ts = int(time.monotonic() * 1e3)
            self.face_landmarker.detect_async(mp_img, timestamp_ms=ts)
            t1 = time.time()
            res = self.holistic.process(rgb)
            print(f"[Participant] holistic.process {(time.time()-t1)*1000:.1f}ms")

            mesh_vals = []
            if res.face_landmarks:
                mesh_vals = [coord for lm in res.face_landmarks.landmark
                             for coord in (lm.x, lm.y, lm.z)]
            else:
                mesh_vals = [0.0] * (468*3)

            pose_vals = []
            if res.pose_landmarks:
                pose_vals = [coord for lm in res.pose_landmarks.landmark
                             for coord in (lm.x, lm.y, lm.z, lm.visibility)]
            else:
                pose_vals = [0.0] * (33*4)

            return {
                'mesh': mesh_vals,
                'pose': pose_vals,
                'blend': self.last_blend_scores
            }

    def push_multiface_samples(self, faces: list, face_ids: list, include_mesh: bool):
        """
        Streams each face’s blendshapes and optional mesh to separate LSL outlets,
        using per-face custom names stored in self.multiface_names.
        """
        for fdata, fid in zip(faces, face_ids):
            # gather data
            blend = fdata['blend']
            mesh  = fdata['mesh'] if include_mesh else []
            sample = blend + mesh

            # determine the base label for this face (Alice, Bob, or ID#)
            base = getattr(self, 'multiface_names', {}).get(fid, f"ID{fid}")

            # create outlet on first sample
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

            # push the sample
            self.face_outlets[fid].push_sample(sample)

    def release(self):
        """Release camera and Mediapipe resources."""
        if self.cap:
            self.cap.release()
        if hasattr(self, 'holistic'):
            self.holistic.close()
        if hasattr(self, 'blendshape_landmarker'):
            self.blendshape_landmarker.close()
        if hasattr(self, 'multiface_landmarker'):
            self.multiface_landmarker.close()
        if hasattr(self, 'outlet') and self.outlet:
            try: self.outlet.close()
            except: pass
        for ol in self.face_outlets.values():
            try: ol.close()
            except: pass
            
    def stop_streaming(self):
        """Close all LSL outlets so we can restart later without re-opening camera."""
        # main outlet (holistic mode)
        if getattr(self, 'outlet', None):
            try: self.outlet.close()
            except: pass
            self.outlet = None
        # per-face outlets (multi-face mode)
        for fid, ol in list(self.face_outlets.items()):
            try: ol.close()
            except: pass
        self.face_outlets.clear()