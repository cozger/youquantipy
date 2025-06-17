import tkinter as tk
from tkinter import ttk
from pygrabber.dshow_graph import FilterGraph
from PIL import Image, ImageTk
import threading
import queue

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pylsl import StreamInfo, StreamOutlet

from participant import Participant  # data-stream class

# Path to your Mediapipe face_landmarker.task
MODEL_PATH = r"D:\Projects\MovieSynchrony\face_landmarker.task"

# Utility to list video devices with names and resolution
def list_video_devices(max_devices=10):
    graph = FilterGraph()
    names = graph.get_input_devices()
    devices = []
    for i in range(min(len(names), max_devices)):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if not cap.isOpened():
            continue
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        devices.append((i, names[i], width, height))
        cap.release()
    return devices

class YouQuantiPyGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("YouQuantiPy Multi-Participant Stream")
        self.geometry("1200x900")

        # Shared Mediapipe models for preview
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_styles  = mp.solutions.drawing_styles
        self.holistic   = mp.solutions.holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            refine_face_landmarks=True
        )
        with open(MODEL_PATH, 'rb') as f:
            buf = f.read()
        base_opts = python.BaseOptions(model_asset_buffer=buf)
        self.blend_opts = vision.FaceLandmarkerOptions(
            base_options=base_opts,
            output_face_blendshapes=True,
            running_mode=vision.RunningMode.IMAGE
        )

        # Streaming control
        self.streaming = False

        # UI controls
        tk.Label(self, text="Number of participants:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.participant_count = tk.IntVar(value=2)
        tk.Spinbox(
            self, from_=1, to=6, textvariable=self.participant_count,
            command=self.build_frames, width=5
        ).grid(row=0, column=1, sticky="w", padx=5)
        self.enable_mesh = tk.BooleanVar(value=False)
        tk.Checkbutton(
            self, text="Enable full FaceMesh", variable=self.enable_mesh
        ).grid(row=0, column=2, padx=5)
        tk.Button(
            self, text="Start Data Stream", command=self.start_stream
        ).grid(row=0, column=3, padx=10)
        tk.Button(
            self, text="Stop Data Stream", command=self.stop_stream
        ).grid(row=0, column=4, padx=10)

        # Container for participant canvases
        self.container = ttk.Frame(self)
        self.container.grid(row=1, column=0, columnspan=5, pady=10)

        # frames info: list of dicts containing UI, preview & data threads
        self.frames = []
        self.build_frames()

    def build_frames(self):
        # Cleanup existing frames and threads
        for info in self.frames:
            info['stop_evt'].set()
            if info.get('thread'):
                info['thread'].join()
            if info.get('data_stop_evt'):
                info['data_stop_evt'].set()
                info['data_thread'].join()
            if info.get('participant'):
                info['participant'].release()
            info['frame'].destroy()
        self.frames.clear()

        # List cameras
        cams = list_video_devices()
        cam_vals = [f"{i}: {name} ({w}x{h})" for i,name,w,h in cams]

        # Build UI per participant
        for i in range(self.participant_count.get()):
            part_frame = ttk.LabelFrame(self.container, text=f"Participant {i+1}")
            part_frame.grid(row=0, column=i, padx=5)
            tk.Label(part_frame, text="ID:").pack(anchor='w')
            ent = tk.Entry(part_frame, width=15)
            ent.pack()
            canvas = tk.Canvas(part_frame, width=400, height=300)
            canvas.pack(pady=5)
            meta_lbl = tk.Label(part_frame, text="ID: -, Stream: -")
            meta_lbl.pack(pady=(0,5))
            tk.Label(part_frame, text="Camera:").pack()
            cmb = ttk.Combobox(part_frame, values=cam_vals, state="readonly", width=30)
            if cam_vals:
                cmb.current(0)
            cmb.pack(pady=(0,5))

            stop_evt = threading.Event()
            frame_queue = queue.Queue(maxsize=1)
            thread = None

            def on_select(event, idx=i):
                info = self.frames[idx]
                sel = info['combo'].get()
                cam_idx = int(sel.split(':')[0])
                # initialize new participant
                part = Participant(
                    camera_index=cam_idx,
                    model_path=MODEL_PATH,
                    enable_raw_facemesh=self.enable_mesh.get()
                )
                # stop old preview thread
                if info.get('stop_evt'):
                    info['stop_evt'].set()
                if info.get('thread'):
                    info['thread'].join()
                # assign new queues and thread
                info['participant'] = part
                info['stop_evt'] = threading.Event()
                info['frame_queue'] = queue.Queue(maxsize=1)
                t = threading.Thread(target=self.preview_loop, args=(idx,), daemon=True)
                info['thread'] = t
                t.start()
                self.schedule_preview(idx)

            cmb.bind("<<ComboboxSelected>>", on_select)

            info = {
                'frame': part_frame,
                'canvas': canvas,
                'entry': ent,
                'combo': cmb,
                'meta_label': meta_lbl,
                'participant': None,
                'stop_evt': stop_evt,
                'thread': thread,
                'data_stop_evt': None,
                'data_thread': None,
                'frame_queue': frame_queue
            }
            self.frames.append(info)

            # trigger initial preview start
            if cam_vals:
                self.after(100, lambda idx=i: on_select(None, idx))

    def preview_loop(self, idx):
        info = self.frames[idx]
        p = info['participant']
        stop_evt = info['stop_evt']
        q = info['frame_queue']
        while not stop_evt.is_set():
            if getattr(p, 'cap', None) is None:
                break
            success, frame = p.cap.read()
            if not success:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hol = p.holistic.process(rgb)
            blend_res = p.face_landmarker.detect(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            )
            disp = frame.copy()
            self.mp_drawing.draw_landmarks(
                disp, hol.face_landmarks,
                mp.solutions.holistic.FACEMESH_CONTOURS, None,
                self.mp_styles.get_default_face_mesh_contours_style()
            )
            self.mp_drawing.draw_landmarks(
                disp, hol.pose_landmarks,
                mp.solutions.holistic.POSE_CONNECTIONS,
                self.mp_styles.get_default_pose_landmarks_style()
            )
            scores = [c.score for c in (blend_res.face_blendshapes[0] if blend_res.face_blendshapes else [])]
            scores += [0.0] * (52 - len(scores))
            h, w, _ = disp.shape
            bar_h = int(h/52)
            for j, s in enumerate(scores):
                y0, y1 = j*bar_h, (j+1)*bar_h
                cv2.rectangle(disp, (0,y0), (int(s*150),y1), (0,255,0), -1)
            if not q.full():
                q.put(disp)
            stop_evt.wait(0.03)

    def schedule_preview(self, idx):
        info = self.frames[idx]
        canvas = info['canvas']
        q = info['frame_queue']
        try:
            disp = q.get_nowait()
            img = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
            imgtk = ImageTk.PhotoImage(Image.fromarray(img).resize((400,300)))
            canvas.create_image(0,0,anchor='nw',image=imgtk)
            canvas.image = imgtk
        except queue.Empty:
            pass
        if not info['stop_evt'].is_set():
            self.after(30, self.schedule_preview, idx)

    def data_loop(self, idx):
        info = self.frames[idx]
        p = info['participant']
        stop_evt = info['data_stop_evt']
        while not stop_evt.is_set():
            sample = p.read_frame()
            if sample is not None and p.outlet:
                p.outlet.push_sample(sample)
            stop_evt.wait(1.0/p.fps)

    def start_stream(self):
        self.streaming = True
        for i, info in enumerate(self.frames):
            pid = info['entry'].get().strip() or f"P{i+1}"
            stream_name = f"{pid}_landmarks"
            part = info['participant']
            part.setup_stream(pid, stream_name)
            stop_evt = threading.Event()
            t = threading.Thread(target=self.data_loop, args=(i,), daemon=True)
            info['data_stop_evt'] = stop_evt
            info['data_thread'] = t
            t.start()
            info['meta_label'].config(text=f"ID: {pid}, Stream: {stream_name}")

    def stop_stream(self):
        self.streaming = False
        for info in self.frames:
            info['stop_evt'].set()
            if info.get('data_stop_evt'):
                info['data_stop_evt'].set()
        self.after(50, self._cleanup_threads)

    def _cleanup_threads(self):
        alive = False
        for info in self.frames:
            t = info.get('thread')
            dt = info.get('data_thread')
            if (t and t.is_alive()) or (dt and dt.is_alive()):
                alive = True
                break
        if alive:
            self.after(50, self._cleanup_threads)
        else:
            for info in self.frames:
                if info.get('participant'):
                    info['participant'].release()
                info['meta_label'].config(text="ID: -, Stream: -")

if __name__ == '__main__':
    app = YouQuantiPyGUI()
    app.mainloop()
