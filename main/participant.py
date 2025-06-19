import cv2
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pylsl import StreamInfo, StreamOutlet

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
        self.multi_face_mode     = multi_face_mode  # ← store it
        self.fps = fps
    
        # existing: load Holistic, FaceLandmarker, VideoCapture…

        
        # Load per-mode landmark models
        if not self.multi_face_mode:
            self.holistic = mp.solutions.holistic.Holistic(
                model_complexity=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                refine_face_landmarks=True
            )
        else:
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=4,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )


        # Placeholders for LSL
        self.outlet = None
        self.participant_id = None
        self.stream_name = None
        self.source_id = None

        # Robust VideoCapture: try DSHOW, MSMF, then default
        backends = []
        if hasattr(cv2, 'CAP_DSHOW'): backends.append(cv2.CAP_DSHOW)
        if hasattr(cv2, 'CAP_MSMF'):  backends.append(cv2.CAP_MSMF)
        backends.append(0)

        cap = None
        chosen_backend = None
        for b in backends:
            try:
                # Open with backend b (0 means default API)
                c = cv2.VideoCapture(self.camera_index, b) if b != 0 else cv2.VideoCapture(self.camera_index)
                if c.isOpened():
                    cap = c
                    chosen_backend = b
                    break
                else:
                    c.release()
            except Exception:
                continue
        if cap is None or not cap.isOpened():
            raise RuntimeError(f"Could not open camera #{self.camera_index} on any backend")

        self.cap = cap
        # Debug: report backend and negotiated resolution
        print(f"[Participant] camera {self.camera_index} opened with backend={chosen_backend}")
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[Participant] actual resolution = {w}x{h}")

        # Load model buffer once
        with open(self.model_path, 'rb') as f:
            model_buffer = f.read()
        base_opts = python.BaseOptions(model_asset_buffer=model_buffer)

        # Blendshape detector (LIVE_STREAM)
        self.last_blend_scores = [0.0] * 52
        def _on_result(res, img, ts):
            shapes = res.face_blendshapes[0] if res.face_blendshapes else []
            scores = [cat.score for cat in shapes]
            # pad/truncate to exactly 52
            scores = scores[:52] + [0.0] * max(0, 52 - len(scores))
            self.last_blend_scores = scores

        blend_opts = vision.FaceLandmarkerOptions(
            base_options=base_opts,
            output_face_blendshapes=True,
            running_mode=vision.RunningMode.LIVE_STREAM,
            result_callback=_on_result
        )
        self.face_landmarker = vision.FaceLandmarker.create_from_options(blend_opts)

    def setup_stream(self, participant_id: str, stream_name: str):
        """Initializes the LSL outlet."""
        self.participant_id = participant_id
        self.stream_name = stream_name
        self.source_id = f"{participant_id}_uid"

        mesh_ch = (468 * 3) if self.enable_raw_facemesh else 0
        total_ch = 52 + mesh_ch + (33 * 4)

        info = StreamInfo(
            name=self.stream_name,
            type='Landmark',
            channel_count=total_ch,
            nominal_srate=self.fps,
            channel_format='float32',
            source_id=self.source_id
        )
        self.outlet = StreamOutlet(info)

    def read_frame(self):
        """Reads a frame, processes landmarks/blendshapes, returns the LSL sample with timing logs."""
        t0 = time.time()
        success, frame = self.cap.read()
        t1 = time.time()
        print(f"[Participant] cap.read() {(t1-t0)*1000:.1f}ms")
        if not success:
            return None

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mesh_vals, pose_vals = [], []

        # Landmark processing timing
        t2 = time.time()
        if not self.multi_face_mode:
            res = self.holistic.process(rgb)
            t3 = time.time()
            print(f"[Participant] holistic.process() {(t3-t2)*1000:.1f}ms")
            if res.face_landmarks:
                mesh_vals = [coord for lm in res.face_landmarks.landmark for coord in (lm.x, lm.y, lm.z)]
            else:
                mesh_vals = [0.0] * (468*3)
            if res.pose_landmarks:
                pose_vals = [coord for lm in res.pose_landmarks.landmark for coord in (lm.x, lm.y, lm.z, lm.visibility)]
            else:
                pose_vals = [0.0] * (33*4)
        else:
            multi = self.face_mesh.process(rgb)
            t3 = time.time()
            print(f"[Participant] face_mesh.process() {(t3-t2)*1000:.1f}ms")
            faces = multi.multi_face_landmarks or []
            if faces:
                mesh_vals = [coord for lm in faces[0].landmark for coord in (lm.x, lm.y, lm.z)]
            else:
                mesh_vals = [0.0] * (468*3)
            pose_vals = [0.0] * (33*4)

        # Blendshape async scheduling timing
        t4 = time.time()
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        self.face_landmarker.detect_async(mp_img, timestamp_ms=int(time.time()*1e3))
        t5 = time.time()
        print(f"[Participant] detect_async() scheduling {(t5-t4)*1000:.1f}ms")

        blend_scores = self.last_blend_scores
        if self.enable_raw_facemesh:
            sample = blend_scores + mesh_vals + pose_vals
        else:
            sample = blend_scores + pose_vals
        self.last_full_mesh = mesh_vals
        return sample

    def release(self):
        """Releases camera and Mediapipe resources."""
        if getattr(self, 'outlet', None):
            try:
                self.outlet.close()
            except:
                pass
        if self.cap:
            self.cap.release()
        if hasattr(self, 'holistic') and self.holistic:
            try:
                self.holistic.close()
            except:
                pass
        if hasattr(self, 'face_mesh') and self.face_mesh:
            try:
                self.face_mesh.close()
            except:
                pass
        if self.face_landmarker:
            try:
                self.face_landmarker.close()
            except:
                pass
