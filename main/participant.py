import cv2
import mediapipe as mp
from pylsl import StreamInfo, StreamOutlet
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class Participant:
    """
    Represents a single participant's multimodal data streamer.

    Attributes:
        id (str): Identifier for the participant.
        camera_index (int): OpenCV camera index for video input.
        model_path (str): Path to the Mediapipe face_landmarker.task file.
        stream_name (str): Name for the LSL outlet stream.
        source_id (str): Unique source ID for the LSL stream.
        fps (int): Nominal frame rate for the LSL stream.
    """
    def __init__(
        self,
        participant_id: str,
        camera_index: int,
        model_path: str,
        stream_name: str,
        source_id: str,
        fps: int = 30
    ):
        self.id = participant_id
        self.camera_index = camera_index
        self.model_path = model_path
        self.stream_name = stream_name
        self.source_id = source_id
        self.fps = fps
        # 52 blendshape channels + 33 pose landmarks × 4 values
        self.landmark_channel_count = 52 + (33 * 4)

        self._setup_lsl_outlet()
        self._setup_models()

    def _setup_lsl_outlet(self):
        """
        Initializes the LSL StreamInfo and StreamOutlet for landmark data.
        """
        info = StreamInfo(
            name=self.stream_name,
            type='Landmark',
            channel_count=self.landmark_channel_count,
            nominal_srate=self.fps,
            channel_format='float32',
            source_id=self.source_id
        )
        self.outlet = StreamOutlet(info)

    def _setup_models(self):
        """
        Loads Mediapipe models: Holistic for pose and FaceLandmarker for blendshapes.
        """
        # Holistic (pose + facemesh)
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.holistic = mp.solutions.holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            refine_face_landmarks=True
        )

        # FaceLandmarker (blendshapes)
        with open(self.model_path, "rb") as f:
            model_bytes = f.read()
        base_options = python.BaseOptions(model_asset_buffer=model_bytes)
        blendshape_opts = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            running_mode=vision.RunningMode.IMAGE
        )
        self.face_landmarker = vision.FaceLandmarker.create_from_options(blendshape_opts)

        # Video capture
        self.cap = cv2.VideoCapture(self.camera_index)

    def run(self):
        """
        Main loop: captures video frames, processes with Mediapipe, and streams via LSL.
        """
        while self.cap.isOpened():
            success, frame = self.cap.read()
            if not success:
                continue

            # Prepare frame for Mediapipe
            frame.flags.writeable = False
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Holistic processing
            hol_res = self.holistic.process(rgb)

            # Blendshape processing
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            blend_res = self.face_landmarker.detect(mp_image)

            # Extract blendshape scores
            blend_scores = []
            if blend_res.face_blendshapes:
                for cat in blend_res.face_blendshapes[0]:
                    blend_scores.append(cat.score)
            # pad if needed
            blend_scores += [0.0] * (52 - len(blend_scores))

            # Extract pose landmarks
            pose_values = []
            if hol_res.pose_landmarks:
                for lm in hol_res.pose_landmarks.landmark:
                    pose_values.extend([lm.x, lm.y, lm.z, lm.visibility])
            else:
                pose_values = [0.0] * (33 * 4)

            # Combine samples and push to LSL
            sample = blend_scores + pose_values
            self.outlet.push_sample(sample)

            # Visualization (optional)
            disp = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            # draw face mesh
            self.mp_drawing.draw_landmarks(
                disp,
                hol_res.face_landmarks,
                mp.solutions.holistic.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style()
            )
            # draw pose
            self.mp_drawing.draw_landmarks(
                disp,
                hol_res.pose_landmarks,
                mp.solutions.holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )

            # Show window
            cv2.imshow(self.id, disp)
            if cv2.waitKey(5) & 0xFF == 27:
                break

        # Cleanup
        self.cap.release()
        cv2.destroyWindow(self.id)
