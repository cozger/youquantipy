import tkinter as tk
from tkinter import ttk
from pygrabber.dshow_graph import FilterGraph
from PIL import Image, ImageTk
import threading
import queue
import time

import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from participant import Participant  # data-stream class :contentReference[oaicite:5]{index=5}
from correlator import ChannelCorrelator

# Path to your Mediapipe face_landmarker.task
MODEL_PATH = r"D:\Projects\MovieSynchrony\face_landmarker.task"

def list_video_devices(max_devices=10):
    graph = FilterGraph()
    names = graph.get_input_devices()
    devices = []
    for i in range(min(len(names), max_devices)):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        if not cap.isOpened():
            continue
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        devices.append((i, names[i], w, h))
        cap.release()
    return devices

class YouQuantiPyGUI(tk.Tk):
    BAR_HEIGHT = 150

    def __init__(self):
        super().__init__()

        # ─── HIDDEN TESTING MODE ──────────────────────────────────────────────
        # When True, participants share the same cv2.VideoCapture. Flip to False for real multi-camera.
        self.testing_mode = True
        # ──────────────────────────────────────────────────────────────────────

        print("[DEBUG] GUI __init__")
        self.title("YouQuantiPy Multi-Participant Stream")
        self.geometry("1200x1050")
        for col in range(5):
            self.grid_columnconfigure(col, weight=1)
        self.grid_rowconfigure(2, weight=0)

        # Shared Holistic for drawing preview
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_styles  = mp.solutions.drawing_styles
        self.holistic   = mp.solutions.holistic.Holistic(
            model_complexity=2, 
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            refine_face_landmarks=True
        )

        # Correlator for two-participant synchrony plot
        self.correlator = ChannelCorrelator(window_size=60, fps=30)
        self.data_stop_evt = threading.Event()
        self.data_thread = None
        self.streaming = False

        # Auto-start params
        self.blend_labels = None
        self.auto_start_threshold = 30

        # UI controls
        tk.Label(self, text="Number of participants:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.participant_count = tk.IntVar(value=2)
        tk.Spinbox(self, from_=1, to=6, textvariable=self.participant_count,
                   command=self.build_frames, width=5).grid(row=0, column=1, sticky="w", padx=5)
        self.enable_mesh = tk.BooleanVar(value=False)
        tk.Checkbutton(self, text="Enable full FaceMesh", variable=self.enable_mesh).grid(row=0, column=2, padx=5)
        tk.Button(self, text="Start Data Stream", command=self.start_stream).grid(row=0, column=3, padx=10)
        tk.Button(self, text="Stop Data Stream",  command=self.stop_stream).grid(row=0, column=4, padx=10)

        # Preview + plot areas
        # Preview area (participants)
        self.container = ttk.Frame(self)
        self.container.grid(row=1, column=0, columnspan=4, pady=10, sticky='ew')
        self.container.grid_columnconfigure(tuple(range(4)), weight=1)

        # Comodulation bar plots on the right
        self.bar_canvas = tk.Canvas(self, bg='white', width=200)
        self.bar_canvas.grid(row=1, column=4, rowspan=2, sticky='ns', padx=(10,0))
        # keep column 4 from expanding
        self.grid_columnconfigure(4, weight=0)




        self.bar_canvas.create_rectangle(10,10,110,60, fill='red', tags='debug')

        # Build per-participant frames
        self.frames = []
        self.build_frames()

    def on_landmarker_result(self, idx, result, output_image, timestamp_ms):
        """
        Live-stream callback for participant idx.
        """
        info = self.frames[idx]
        if result.face_blendshapes:
            scores = [cat.score for cat in result.face_blendshapes[0]]
        else:
            scores = []
        scores += [0.0] * (52 - len(scores))
        info['last_scores'] = scores

        if self.blend_labels is None and result.face_blendshapes:
            self.blend_labels = [cat.category_name for cat in result.face_blendshapes[0]]

        info['detect_count'] = info.get('detect_count', 0) + (1 if any(scores) else 0)

    def build_frames(self):
        print("[DEBUG] build_frames called")
        # Clean up existing
        for i, info in enumerate(self.frames):
            info['stop_evt'].set()
            if info.get('thread'):      info['thread'].join()
            if info.get('data_thread'): info['data_thread'].join()
            if info.get('participant') and not (self.testing_mode and i>0):
                info['participant'].release()
            info['frame'].destroy()
        self.frames.clear()

        cams = list_video_devices()
        cam_vals = [f"{i}: {name} ({w}x{h})" for i,name,w,h in cams]

        for i in range(self.participant_count.get()):
            frame = ttk.LabelFrame(self.container, text=f"Participant {i+1}")
            frame.grid(row=0, column=i, padx=5, sticky='n')
            frame.grid_columnconfigure(0, weight=1)

            tk.Label(frame, text="ID:").pack(anchor='w')
            ent    = tk.Entry(frame, width=15); ent.pack()
            canvas = tk.Canvas(frame, width=400, height=300); canvas.pack(pady=5)
            meta   = tk.Label(frame, text="ID: -, Stream: -"); meta.pack(pady=(0,5))
            tk.Label(frame, text="Camera:").pack()
            cmb = ttk.Combobox(frame, values=cam_vals, state='readonly', width=30)
            if cam_vals: cmb.current(0)
            cmb.pack(pady=(0,5))

            stop_evt    = threading.Event()
            frame_queue = queue.Queue(maxsize=1)

            def on_select(event=None, idx=i):
                print(f"[DEBUG] on_select(idx={idx})")
                try:
                    info = self.frames[idx]
                    cam_idx = int(info['combo'].get().split(':')[0])
                    part = Participant(cam_idx, MODEL_PATH, self.enable_mesh.get())
                    # share cap in testing mode
                    if self.testing_mode and idx>0 and self.frames[0]['participant']:
                        part.cap = self.frames[0]['participant'].cap

                    # set up a dedicated FaceLandmarker for this participant
                    with open(MODEL_PATH, 'rb') as f:
                        buf = f.read()
                    base_opts = python.BaseOptions(model_asset_buffer=buf)
                    opts = vision.FaceLandmarkerOptions(
                        base_options=base_opts,
                        output_face_blendshapes=True,
                        running_mode=vision.RunningMode.LIVE_STREAM,
                        # capture idx in the callback
                        result_callback=lambda rst, img, ts, idx=idx: self.on_landmarker_result(idx, rst, img, ts)
                    )
                    part.face_landmarker = vision.FaceLandmarker.create_from_options(opts)

                    # cleanup old preview
                    info['stop_evt'].set()
                    if info.get('thread'): info['thread'].join()

                    info.update(
                        participant=part,
                        face_landmarker=part.face_landmarker,
                        stop_evt=threading.Event(),
                        frame_queue=queue.Queue(maxsize=1),
                        detect_count=0,
                        last_scores=[0.0]*52
                    )
                    # start preview
                    t = threading.Thread(target=self.preview_loop, args=(idx,), daemon=True)
                    info['thread'] = t; t.start()
                    # schedule the Tkinter display loop
                    self.after(30, self.schedule_preview, idx)
                except Exception as e:
                    print(f"[ERROR] on_select idx={idx}: {e}")

            cmb.bind('<<ComboboxSelected>>', on_select)

            self.frames.append({
                'frame': frame, 'canvas': canvas, 'entry': ent,
                'combo': cmb, 'meta_label': meta,
                'participant': None, 'face_landmarker': None,
                'stop_evt': stop_evt, 'thread': None, 'frame_queue': frame_queue,
                'data_thread': None, 'detect_count': 0, 'last_scores': [0.0]*52
            })
            if cam_vals:
                self.after(100, on_select)

    def preview_loop(self, idx):
        print(f"[DEBUG] preview_loop started for participant {idx}")
        info = self.frames[idx]
        p    = info['participant']
        fl   = info['face_landmarker']
        stop = info['stop_evt']
        q    = info['frame_queue']

        while not stop.is_set():
            success, frame = p.cap.read()
            if not success:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # draw Holistic landmarks
            hol = p.holistic.process(rgb)
            disp = frame.copy()
                        # draw full 468-point mesh
            self.mp_drawing.draw_landmarks(
                disp, hol.face_landmarks,
                mp.solutions.holistic.FACEMESH_TESSELATION,  # full mesh triangles
                None,
                self.mp_styles.get_default_face_mesh_tesselation_style()
            )

            self.mp_drawing.draw_landmarks(
                disp, hol.pose_landmarks, mp.solutions.holistic.POSE_CONNECTIONS,
                self.mp_styles.get_default_pose_landmarks_style()
            )


              # ---- Live-stream FaceLandmarker API ----
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            # use a monotonic clock so timestamps always strictly increase
            ts = int(time.monotonic() * 1e3)
            fl.detect_async(mp_img, timestamp_ms=ts)

          # ---- Extract and cache mesh & pose for combined_loop ----
            # mesh: always extract the full 468-point mesh
            if hol.face_landmarks:
                mesh_vals = [
                    coord
                    for lm in hol.face_landmarks.landmark
                    for coord in (lm.x, lm.y, lm.z)
                ]
            else:
                mesh_vals = [0.0] * (468 * 3)
            info['last_mesh_vals'] = mesh_vals

            if hol.pose_landmarks:
                pose_vals = [
                    coord
                    for lm in hol.pose_landmarks.landmark
                    for coord in (lm.x, lm.y, lm.z, lm.visibility)
                ]
            else:
                pose_vals = [0.0] * (33 * 4)
            info['last_pose_vals'] = pose_vals


            # draw blendshape bars
            scores = info.get('last_scores', [0.0]*52)
            h, w, _ = disp.shape
            bar_h = h // 52
            for j, s in enumerate(scores):
                y0, y1 = j * bar_h, (j + 1) * bar_h
                cv2.rectangle(disp, (0, y0), (int(s * 150), y1), (0, 255, 0), -1)

            if not q.full():
                q.put(disp)
            stop.wait(0.03)

    def schedule_preview(self, idx):
        info   = self.frames[idx]
        canvas = info['canvas']
        q      = info['frame_queue']
        try:
            disp = q.get_nowait()
            imgtk = ImageTk.PhotoImage(
                Image.fromarray(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)).resize((400,300))
            )
            canvas.create_image(0,0,anchor='nw',image=imgtk)
            canvas.image = imgtk
        except queue.Empty:
            pass
        self.check_auto_start()
        if not info['stop_evt'].is_set():
            self.after(30, self.schedule_preview, idx)

    def check_auto_start(self):
        if self.streaming or len(self.frames) < 2:
            return
        if all(info['detect_count'] >= self.auto_start_threshold for info in self.frames[:2]):
            print("[DEBUG] Auto-start condition met")
            self.start_stream()

    def update_plot(self, corr):
        c = self.bar_canvas
        c.delete('all')
        W, H = c.winfo_width(), c.winfo_height()

        if corr is None:
            c.create_text(W//2, H//2, text="No data", fill='gray')
            return

        n = len(corr)
        bar_h = max(4, H / n)
        label_w = 100      # width reserved for labels on left
        margin  = 5        # space between label and bar
        bg = (255,255,255)

        for i, val in enumerate(corr):
            y0 = i * bar_h
            y1 = y0 + bar_h - 1
            x0 = label_w + margin
            length = val * (W - label_w - 2*margin)
            x1 = x0 + length

            # blend green/red toward white
            base = (0,255,0) if val >= 0 else (255,0,0)
            a    = min(max(abs(val), 0.0), 1.0)
            r = int(bg[0] + (base[0]-bg[0]) * a)
            g = int(bg[1] + (base[1]-bg[1]) * a)
            b = int(bg[2] + (base[2]-bg[2]) * a)
            color = f'#{r:02x}{g:02x}{b:02x}'

            # draw the horizontal bar
            c.create_rectangle(x0, y0, x1, y1, fill=color, outline='')

            # blendshape label to the left
            if self.blend_labels:
                c.create_text(
                    label_w - margin, y0 + bar_h/2,
                    text=self.blend_labels[i],
                    anchor='e',
                    font=('Arial', 8)
                )

    def combined_loop(self):
        p1, p2 = self.frames[0]['participant'], self.frames[1]['participant']
        stop = self.data_stop_evt
        fps  = getattr(self.frames[0]['participant'], 'fps', 30)
        while not stop.is_set():
            info1, info2 = self.frames[0], self.frames[1]

            # just pull the last-seen scores, mesh & pose
            s1 = info1.get('last_scores', [0.0]*52)
            s2 = info2.get('last_scores', [0.0]*52)
            m1 = info1.get('last_mesh_vals', [])
            m2 = info2.get('last_mesh_vals', [])
            p1 = info1.get('last_pose_vals', [])
            p2 = info2.get('last_pose_vals', [])

              # rebuild sample vectors, gating mesh on the toggle
            if self.enable_mesh.get():
                sample1 = s1 + m1 + p1
                sample2 = s2 + m2 + p2
            else:
                sample1 = s1 + p1
                sample2 = s2 + p2

            # stream to LSL
            if getattr(info1['participant'], 'outlet', None):
                info1['participant'].outlet.push_sample(sample1)
            if getattr(info2['participant'], 'outlet', None):
                info2['participant'].outlet.push_sample(sample2)

            # update correlation plot
            v1, v2 = np.array(s1), np.array(s2)
            corr = self.correlator.update(v1, v2)
            if corr is not None:
                self.after_idle(lambda c=corr.copy(): self.update_plot(c))

            time.sleep(1.0 / fps)

    def start_stream(self):
        if self.streaming:
            return
        valid = [info for info in self.frames if info['participant'] and hasattr(info['participant'], 'setup_stream')]
        if len(valid) < 2:
            return
        self.streaming = True
        for i, info in enumerate(valid[:2]):
            pid = info['entry'].get().strip() or f"P{i+1}"
            nm  = f"{pid}_landmarks"
            info['participant'].setup_stream(pid, nm)
            info['meta_label'].config(text=f"ID: {pid}, Stream: {nm}")
        self.data_stop_evt.clear()
        t = threading.Thread(target=self.combined_loop, daemon=True)
        self.data_thread = t
        t.start()

    def stop_stream(self):
        if not self.streaming:
            return
        self.streaming = False
        for info in self.frames:
            info['stop_evt'].set()
        self.data_stop_evt.set()

        def _cleanup():
            if any((info['thread'] and info['thread'].is_alive()) or
                   (self.data_thread and self.data_thread.is_alive())
                   for info in self.frames):
                self.after(100, _cleanup)
                return
            for i, info in enumerate(self.frames):
                if info['participant'] and not (self.testing_mode and i > 0):
                    info['participant'].release()
                info['meta_label'].config(text="ID: -, Stream: -")
            self.bar_canvas.delete('all')
            self.bar_canvas.create_rectangle(10,10,110,60, fill='red', tags='debug')
            self.correlator.close()

        self.after(50, _cleanup)

if __name__ == '__main__':
    YouQuantiPyGUI().mainloop()
