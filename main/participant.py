import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pylsl import StreamInfo, StreamOutlet

class Participant:
    """
    Handles data acquisition (blendshapes, pose landmarks) and LSL streaming for a participant.
    LSL outlet setup is deferred until setup_stream() is called, so that participant ID
    and stream name can be finalized.
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

        # Placeholders for LSL outlet and identifiers
        self.outlet = None
        self.participant_id = None
        self.stream_name = None
        self.source_id = None

        # Initialize Mediapipe Holistic (pose + facemesh)
        self.holistic = mp.solutions.holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            refine_face_landmarks=True
        )

        # Initialize FaceLandmarker (blendshapes)
        with open(self.model_path, 'rb') as f:
            base_options = python.BaseOptions(model_asset_buffer=f.read())
        blend_opts = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            running_mode=vision.RunningMode.IMAGE
        )
        self.face_landmarker = vision.FaceLandmarker.create_from_options(blend_opts)

        # Video capture
        self.cap = cv2.VideoCapture(self.camera_index)

    def setup_stream(self, participant_id: str, stream_name: str):
        """
        Sets up the LSL outlet with the given participant ID and stream name.
        Must be called before starting stream_loop.
        """
        self.participant_id = participant_id
        self.stream_name = stream_name
        self.source_id = f"{participant_id}_uid"

        # Determine channel count:
        # Blendshapes: 52
        # Raw facemesh: 468 landmarks × 3 coords = 1434 (if enabled)
        # Pose: 33 landmarks × 4 values = 132
        mesh_ch = 478 * 3 if self.enable_raw_facemesh else 0
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
        Reads a frame from the camera, processes with Holistic and FaceLandmarker,
        and returns a combined feature vector:
        - blendshapes (52 values)
        - optional facemesh coords (478*3 values)
        - pose landmarks (33*4 values)

        If a detection model returns fewer values (e.g. no face detected),
        we "zero pad" the list to ensure a constant-length output for streaming.
        """
        success, frame = self.cap.read()
        if not success:
            return None

        # Convert to RGB for Mediapipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hol_res = self.holistic.process(rgb)

        # Blendshape detection
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        blend_res = self.face_landmarker.detect(mp_img)

        # Extract blendshape scores
        blend_scores = []
        if blend_res.face_blendshapes:
            for cat in blend_res.face_blendshapes[0]:
                blend_scores.append(cat.score)
        # If fewer than 52 categories detected, pad with zeros
        if len(blend_scores) < 52:
            blend_scores.extend([0.0] * (52 - len(blend_scores)))

        # Extract raw facemesh coords (if enabled)
        mesh_vals = []
        if self.enable_raw_facemesh:
            if hol_res.face_landmarks:
                for lm in hol_res.face_landmarks.landmark:
                    mesh_vals.extend([lm.x, lm.y, lm.z])
            else:
                # No face detected: pad all mesh coords with zeros
                mesh_vals = [0.0] * (478 * 3)

        # Extract pose landmarks
        pose_vals = []
        if hol_res.pose_landmarks:
            for lm in hol_res.pose_landmarks.landmark:
                pose_vals.extend([lm.x, lm.y, lm.z, lm.visibility])
        else:
            # No pose detected: pad with zeros
            pose_vals = [0.0] * (33 * 4)

        # Combined sample has fixed length of 52 + (478*3 if enabled) + 132
        return blend_scores + mesh_vals + pose_vals

    def stream_loop(self):
        """
        Continuously reads frames and pushes samples to the LSL outlet.
        Requires setup_stream() to be called first.
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
        Releases camera and Mediapipe resources.
        """
        self.cap.release()
        self.holistic.close()
        self.face_landmarker.close()
