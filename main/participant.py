import cv2
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pylsl import StreamInfo, StreamOutlet

class Participant:
    """
    Handles data acquisition (blendshapes, pose landmarks) and LSL streaming for a participant.
    LSL outlet setup is deferred until setup_stream() is called.
    """
    def __init__(
        self,
        camera_index: int,
        model_path: str,
        enable_raw_facemesh: bool = False,
        fps: int = 30
    ):
        self.camera_index = camera_index
        self.model_path = model_path
        self.enable_raw_facemesh = enable_raw_facemesh
        self.fps = fps

        # Placeholders
        self.outlet = None
        self.participant_id = None
        self.stream_name = None
        self.source_id = None

        # Load Mediapipe Holistic
        self.holistic = mp.solutions.holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            refine_face_landmarks=True
        )

        # Prepare a buffer for the latest blendshape scores
        self.last_blend_scores = [0.0] * 52

        # Result callback for LIVE_STREAM mode
        def _on_result(result, output_image, timestamp_ms):
            # extract blendshape scores (pad/truncate to 52)
            shapes = result.face_blendshapes[0] if result.face_blendshapes else []
            scores = [cat.score for cat in shapes]
            if len(scores) < 52:
                scores += [0.0] * (52 - len(scores))
            else:
                scores = scores[:52]
            self.last_blend_scores = scores

        # Load FaceLandmarker in LIVE_STREAM mode
        with open(self.model_path, 'rb') as f:
            base_options = python.BaseOptions(model_asset_buffer=f.read())
        blend_opts = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            running_mode=vision.RunningMode.LIVE_STREAM,
            result_callback=_on_result
        )
        self.face_landmarker = vision.FaceLandmarker.create_from_options(blend_opts)

        # Video capture
        self.cap = cv2.VideoCapture(self.camera_index)

    def setup_stream(self, participant_id: str, stream_name: str):
        """
        Initializes the LSL outlet with finalized participant ID and stream name.
        """
        self.participant_id = participant_id
        self.stream_name = stream_name
        self.source_id = f"{participant_id}_uid"

        # Calculate channel count
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
        """
        Reads a frame, enqueues it for async blendshape detection,
        and returns [blend(52), mesh(468*3), pose(132)].
        """
        success, frame = self.cap.read()
        if not success:
            return None

        # RGB for holistic
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hol_res = self.holistic.process(rgb)

        # Enqueue for async blendshape detection
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        self.face_landmarker.detect_async(
            mp_img,
            timestamp_ms=int(time.time() * 1e3)
        )
        # Grab latest from callback
        blend_scores = self.last_blend_scores

        # extract full 468×3 mesh for tracking/visualization
        if hol_res.face_landmarks:
            mesh_vals = []
            for lm in hol_res.face_landmarks.landmark:
                mesh_vals.extend([lm.x, lm.y, lm.z])
        else:
            mesh_vals = [0.0] * (468 * 3)

        # Pose landmarks
        pose_vals = []
        if hol_res.pose_landmarks:
            for lm in hol_res.pose_landmarks.landmark:
                pose_vals.extend([lm.x, lm.y, lm.z, lm.visibility])
        else:
            pose_vals = [0.0] * (33 * 4)

        # build the full sample for LSL only if requested
        if self.enable_raw_facemesh:
            sample = blend_scores + mesh_vals + pose_vals
        else:
            sample = blend_scores + pose_vals

        # stash mesh_vals for GUI or other processing
        self.last_full_mesh = mesh_vals

        return sample
    
    def stream_loop(self):
        """
        Loop: read frames and push to LSL outlet. Requires setup_stream call.
        """
        if not self.outlet:
            raise RuntimeError("LSL outlet not initialized. Call setup_stream() first.")
        while True:
            sample = self.read_frame()
            if sample is None:
                continue
            self.outlet.push_sample(sample)

    def release(self):
        """
        Releases camera and Mediapipe resources. Idempotent.
        """
        # Close LSL outlet if it exists
        if getattr(self, 'outlet', None):
            try:
                self.outlet.close()
            except Exception:
                pass
            self.outlet = None

        # Release camera
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()
            self.cap = None
        # Close holistic
        if hasattr(self, 'holistic') and self.holistic:
            try:
                self.holistic.close()
            except ValueError:
                pass
            self.holistic = None
        # Close face_landmarker
        if hasattr(self, 'face_landmarker') and self.face_landmarker:
            try:
                self.face_landmarker.close()
            except Exception:
                pass
            self.face_landmarker = None
