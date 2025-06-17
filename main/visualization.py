import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import threading

class ParticipantVisualizer:
    """
    A visualization-only module that processes video from a camera index,
    overlays Mediapipe Holistic landmarks and blendshape barplots, and displays live.
    Decoupled from LSL streaming and data export.
    """
    def __init__(
        self,
        camera_index: int,
        model_path: str,
        window_name: str,
        enable_raw_facemesh: bool = False
    ):
        self.camera_index = camera_index
        self.window_name = window_name
        self.enable_raw_facemesh = enable_raw_facemesh

        # Initialize video capture
        self.cap = cv2.VideoCapture(self.camera_index)

        # Initialize Mediapipe holistic for mesh + pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_styles  = mp.solutions.drawing_styles
        self.holistic   = mp.solutions.holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            refine_face_landmarks=True
        )

        # Initialize FaceLandmarker for blendshape categories
        with open(model_path, 'rb') as f:
            buf = f.read()
        base_options = python.BaseOptions(model_asset_buffer=buf)
        blend_opts = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            running_mode=vision.RunningMode.IMAGE
        )
        self.face_landmarker = vision.FaceLandmarker.create_from_options(blend_opts)

        self._stop = threading.Event()
        self._thread = None

    def start(self):
        """Begin the visualization in a background thread."""
        if self._thread is None:
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()

    def stop(self):
        """Signal the visualization loop to stop and close resources."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self.cap.release()
        self.holistic.close()
        self.face_landmarker.close()
        cv2.destroyWindow(self.window_name)

    def _run_loop(self):
        """Internal loop: capture, process, overlay, and display."""
        while not self._stop.is_set():
            ret, frame = self.cap.read()
            if not ret:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Holistic landmarks
            hol_res = self.holistic.process(rgb)

            # Blendshapes
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            blend_res = self.face_landmarker.detect(mp_img)

            # Overlay mesh and pose
            disp = frame.copy()
            self.mp_drawing.draw_landmarks(
                disp,
                hol_res.face_landmarks,
                mp.solutions.holistic.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_styles.get_default_face_mesh_contours_style()
            )
            self.mp_drawing.draw_landmarks(
                disp,
                hol_res.pose_landmarks,
                mp.solutions.holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_styles.get_default_pose_landmarks_style()
            )

            # Compute blendshape scores list and names
            scores = []
            names  = []
            if blend_res.face_blendshapes:
                for c in blend_res.face_blendshapes[0]:
                    scores.append(c.score)
                    names.append(c.category_name)
            # pad
            while len(scores) < 52:
                scores.append(0.0)
                names.append("")

            # Overlay barplot on left side
            h, w, _ = disp.shape
            plot_w = 300
            bar_h = int(h / 52)
            for i, (n, s) in enumerate(zip(names, scores)):
                y0 = i * bar_h
                y1 = y0 + bar_h
                length = int(s * (plot_w - 100))
                cv2.rectangle(disp, (0, y0), (length, y1), (0, 255, 0), -1)
                cv2.putText(
                    disp,
                    n,
                    (5, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA
                )

            cv2.imshow(self.window_name, disp)
            if cv2.waitKey(1) & 0xFF == 27:
                self._stop.set()
                break
