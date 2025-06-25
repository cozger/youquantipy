import tkinter as tk
from tkinter import ttk
from pygrabber.dshow_graph import FilterGraph
from PIL import Image
from multiprocessing import Queue as MPQueue
from multiprocessing import Process
from multiprocessing import Pipe
from queue import Queue as ThreadQueue
from concurrent.futures import ThreadPoolExecutor

import threading
import time
import base64

import numpy as np
import cv2
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2

from participant import participant_worker
from facetracker import FaceTracker 
from correlator import ChannelCorrelator


# TO DO: 
# Readd comdoulation plot for both modes
# Add back FPS counter- is the separate processes actually faster?
# Fix LSL stream set up/channel count errors
# 

# Path to your Mediapipe face_landmarker.task
MODEL_PATH = r"D:\Projects\MovieSynchrony\face_landmarker.task"
try:
    RESAMPLE = Image.Resampling.LANCZOS
except AttributeError:       # Pillow <10
    RESAMPLE = Image.LANCZOS
DESIRED_FPS = 30
CAM_RESOLUTION = (640, 480)
THROTTLE_N = 5    # Only update GUI every 5th frame (~6-7Hz at 30Hz input)

def list_video_devices(max_devices=10):
    graph = FilterGraph()
    names = graph.get_input_devices()
    devices = []

    for i in range(min(len(names), max_devices)):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_RESOLUTION[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_RESOLUTION[1])
        if not cap.isOpened():
            continue
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if (actual_w, actual_h) != (640, 480):
            print(f"[GUI] Warning: Device {i} locked to {actual_w}×{actual_h}")
        devices.append((i, names[i], actual_w, actual_h))
        cap.release()
    return devices


def frame_to_tk(bgr, target_wh, mode="contain"):
    """
    Fast BGR → Tk PhotoImage.

    • mode="contain"   keep the whole frame, add letter-boxing (old ImageOps.contain)
    • mode="cover"     fill & crop centre (old ImageOps.fit)

    Args
    ----
    bgr : np.ndarray   original frame
    target_wh : tuple  (canvas_w, canvas_h)
    mode : str         "contain" | "cover"
    """
    cw, ch = target_wh
    h, w   = bgr.shape[:2]

    # --- compute scale ----------------------------------------------------
    if mode == "cover":           # fill, then crop centre
        scale = max(cw / w, ch / h)
    else:                          # default = contain / letter-box
        scale = min(cw / w, ch / h)

    if scale != 1.0:
        bgr = cv2.resize(
            bgr,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA,
        )

    # --- centre-crop for "cover" ------------------------------------------
    if mode == "cover":
        h, w = bgr.shape[:2]
        x0 = (w - cw) // 2
        y0 = (h - ch) // 2
        bgr = bgr[y0 : y0 + ch, x0 : x0 + cw]
    # for "contain" we keep full frame; Tk centres it later

    # --- encode PNG + base64 for Tk ---------------------------------------
    ok, png = cv2.imencode(".png", bgr, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    if not ok:
        return None

    return tk.PhotoImage(data=base64.b64encode(png).decode("ascii"))

class YouQuantiPyGUI(tk.Tk):
    BAR_HEIGHT = 150

    def __init__(self):
        super().__init__()
        # Allows for mirroring in holistic mode. Will throw an error if used in multiface
        self.testing_mode = False 
        # Load face_landmarker model buffer
        with open(MODEL_PATH, 'rb') as f:
            self.model_buf = f.read()

        print("[DEBUG] GUI __init__")
        self.title("YouQuantiPy")
        self.geometry("1200x1050")
        self.grid_columnconfigure(0, weight=0)   # fixed width  (left)
        self.grid_columnconfigure(1, weight=1)   # expands      (middle  ♦)
        self.grid_columnconfigure(2, weight=0)   # fixed width  (right)
        # give the first row weight so everything grows vertically
        self.grid_rowconfigure(0, weight=1)

        # Drawing utilities
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles

        # Synchrony plot correlator & face tracker
        self.correlator = ChannelCorrelator(window_size=60, fps=30)
        self.face_tracker = FaceTracker(track_threshold=200, max_missed=1500)
        self.enable_mesh = tk.BooleanVar(value=False)
        self.multi_face_mode = tk.BooleanVar(value=False)

        self.data_thread = None
        self.streaming = False
        self.blend_labels = None 
        self.worker_procs = []
        self.preview_queues = []
        self.score_queues = []


        # UI controls
        self.participant_count = tk.IntVar(value=2)
        self.camera_count      = tk.IntVar(value=2)
        def _sync_counts(var_name, index, mode):
            """Mirror participants⇆cameras in holistic mode."""
            if not self.multi_face_mode.get():
                # participant changed → update cameras
                if var_name == self.participant_count._name:
                    self.camera_count.set(max(1, self.participant_count.get()))
                # camera changed → update participants
                elif var_name == self.camera_count._name:
                    self.participant_count.set(max(1, self.camera_count.get()))
        # attach trace callbacks (write-only)
        self._sync_counts = _sync_counts
        self.participant_count.trace_add('write', self._sync_counts)
        self.camera_count.trace_add('write',      self._sync_counts)
        
        # Left-side control panel
        self.control_panel = ttk.Frame(self)
        self.control_panel.grid(row=0, column=0, rowspan=3, sticky='ns', padx=10, pady=10)
        self.grid_columnconfigure(0, weight=0)
        # Participant count
        ttk.Label(self.control_panel, text="Participants:").pack(anchor='w')
        self.part_spin = tk.Spinbox(self.control_panel, from_=1, to=6, textvariable=self.participant_count,
                                    command=self.build_frames, width=5)
        self.part_spin.pack(anchor='w', pady=(0,20))

        # Camera count
        ttk.Label(self.control_panel, text="Cameras:").pack(anchor='w')
        self.cam_spin = tk.Spinbox(self.control_panel, from_=1, to=self.participant_count.get(),
                                textvariable=self.camera_count,
                                command=self.build_frames, width=5)
        self.cam_spin.pack(anchor='w', pady=(0,15))

        # Toggles
        ttk.Checkbutton(self.control_panel, text="Enable full FaceMesh", variable=self.enable_mesh).pack(anchor='w')
        ttk.Checkbutton(self.control_panel, text="Enable Multi-Face Mode",
                        variable=self.multi_face_mode, command=self.on_mode_toggle).pack(anchor='w', pady=(0,50))

        # Action Buttons
        ttk.Button(self.control_panel, text="Start Data Stream", command=self.start_stream).pack(fill='x', pady=(2))
        ttk.Button(self.control_panel, text="Stop Data Stream", command=self.stop_stream).pack(fill='x', pady=(2))
        ttk.Button(self.control_panel, text="Reset", command=self.reset).pack(fill='x', pady=(2))
       
        # Layout for previews
        self.container = ttk.Frame(self)
        self.container.grid(row=0, column=1, columnspan=1, rowspan=1, pady=10, sticky='nsew')
        self.container.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        #Comod plot frame
        self.comod_frame = ttk.Frame(self)
        self.comod_frame.grid(row=0, column=2, sticky='nsew', padx=(10, 0))
        self.grid_columnconfigure(2, weight=0)

        self.bar_label  = tk.Label(self.comod_frame, text="Co-Modulation",
                            font=("Arial", 10, "bold"))
        self.bar_label.pack(anchor='nw')

        self.bar_canvas = tk.Canvas(self.comod_frame, bg='white',
                                    width=250, height=700)
        self.bar_canvas.pack(fill='both', expand=True)
        self.frames = []
        self.build_frames()
        self.after(30, self.schedule_preview)
        self.after(30, self.preview_comodulation)
    
    def on_mode_toggle(self):
        if self.multi_face_mode.get():          # ── Multi-face
            # lock both boxes
            # single camera, but dynamic participants
            self.cam_spin .config(state='disabled')
            self.camera_count.set(1)
            self.part_spin.config(state='normal')
 
        else:                                   # ── Holistic
            # enable and let the trace keep them equal
            # both spinboxes enabled and mirrored
            self.part_spin.config(state='normal')
            self.cam_spin .config(state='normal')
        # re-sync counts by passing in any of the traced var names
        self._sync_counts(self.participant_count._name, None, None)
        # rebuild frames immediately
        self.build_frames()


    def on_landmarker_result(self, idx, result):
        info = self.frames[idx]
        if result.face_blendshapes:
            scores = [cat.score for cat in result.face_blendshapes[0]]
        else:
            scores = []
        scores += [0.0] * (52 - len(scores))
        if hasattr(info['participant'], 'last_blend_scores'):
            self.last_scores = list(info['participant'].last_blend_scores)


        if self.blend_labels is None and result.face_blendshapes:
            self.blend_labels = [cat.category_name for cat in result.face_blendshapes[0]]
        info['detect_count'] = info.get('detect_count', 0) + (1 if any(scores) else 0)

    def build_frames(self):
        print("[DEBUG] build_frames called")
        print(f"[DEBUG] build_frames: about to tear down {len(self.frames)} frames")

        # --- cancel any lingering preview callbacks before teardown ---
        # 1) cancel pending callbacks
        for info in self.frames:
            if aid := info.get('after_id'):
                self.after_cancel(aid)

        # 2) stop & join threads, release cameras
        for info in self.frames:
            if info.get('proc'):
                info['proc'].terminate()
                info['proc'].join(timeout=1.0)


        # 3) destroy all canvas frames
        for w in self.container.winfo_children():
            w.destroy()

        # 4) now clear the list
        self.frames.clear()
        self.cams = list_video_devices()
        cams = len(self.cams)

        # 5) clear IPC state
        self.worker_procs   = [None] * cams
        self.preview_queues = [None] * cams
        self.score_queues   = [None] * cams

        # holistically mirror cameras to participants; multiface always 1
        if self.multi_face_mode.get():
            cams = 1
        else:
            # clamp camera_count to [1, participant_count]
            cams = max(1, min(self.camera_count.get(), self.participant_count.get()))
        # update the IntVar and the spinbox bounds
        self.camera_count.set(cams)
        if self.multi_face_mode.get():
            self.cam_spin.config(to=1)
        else:
            self.cam_spin.config(to=self.participant_count.get())
        # Build camera selection values
        cam_vals = [f"{i}: {name} ({w}x{h})" for i,name,w,h in self.cams]

        def on_select(event, idx):
            print(f"[DEBUG] on_select(idx={idx})")

            if idx < 0 or idx >= len(self.frames):
                return

            info = self.frames[idx]
            parent_conn, child_conn = Pipe()
            info['control_conn'] = parent_conn 

            # Clean up any previous process
            if info.get('proc'):
                info['proc'].terminate()
                info['proc'].join(timeout=2.0)

            # Parse camera index
            sel = info['combo'].get()
            cam_idx = int(sel.split(":", 1)[0])

            # Prepare queues
            pv_q = MPQueue(maxsize=2)
            score_q = MPQueue(maxsize=2)
            if idx >= len(self.preview_queues):
                self.preview_queues.append(pv_q)
            else:
                self.preview_queues[idx] = pv_q

            info['preview_queue'] = pv_q
            info['score_queue'] = score_q

            enable_mesh = self.enable_mesh.get()
            multi_face = self.multi_face_mode.get()
            fps = DESIRED_FPS
            resolution = CAM_RESOLUTION

            if not multi_face:
                pid = info['entry'].get().strip() or f"P{idx+1}"
                stream_name = f"{pid}_landmarks"
                participant_id = pid
            else:
                stream_name = "multiface"
                participant_id = "MF"

            proc = Process(
                target=participant_worker,
                args=(cam_idx, MODEL_PATH, participant_id, stream_name, fps,
                    enable_mesh, multi_face, pv_q, score_q, child_conn),
                daemon=True
            )
            proc.start()
            info['proc'] = proc
            info['meta_label'].config(text=f"Camera → {cam_idx}")

        for i in range(self.camera_count.get()):
            frame = ttk.LabelFrame(self.container, text=f"Camera {i}")  
            frame.grid(row=0, column=i, padx=5, sticky='nsew')
            self.container.grid_columnconfigure(i, weight=1)
            self.container.grid_rowconfigure(0, weight=1) 
            frame.grid_columnconfigure(0, weight=1)
            frame.grid_rowconfigure(1, weight=1)
            row = 0
            ent    = ttk.Entry(frame, width=15) if not self.multi_face_mode.get() else None
            if ent: ent.grid(row=0, column=0, sticky='w', padx=4, pady=2)

            #Stretchy canvas
            canvas = tk.Canvas(frame, bg='black')
            canvas.grid(row=1, column=0, sticky='nsew', padx=2, pady=2)

            #Meta label + combobox sit under the canvas
            meta = ttk.Label(frame, text="ID: -, Stream: -")
            meta.grid(row=2, column=0, sticky='w', padx=2, pady=(4,0))

            cmb = ttk.Combobox(frame, values=cam_vals, state='readonly', width=30)
            if cam_vals: cmb.current(0)
            cmb.grid(row=4, column=0, sticky='we', padx=2, pady=(0,4))
            frame.grid_columnconfigure(0, weight=1)   

            # Bind the combobox selection event to on_select with the current index
            cmb.bind("<<ComboboxSelected>>", lambda event, idx=i: on_select(idx))

            # Store references for later use
            info = {
                'frame': frame,
                'entry': ent,
                'canvas': canvas,
                'meta_label': meta,
                'combo': cmb,
                'proc': None,
                'preview_queue': None,
                'score_queue': None,
            }
            self.frames.append(info)

            # Bind the combobox selection event to on_select
            cmb.bind("<<ComboboxSelected>>", lambda event, idx=i: on_select(event, idx))

    def schedule_preview(self):
        self.after(30, self.schedule_preview)

        for idx, q in enumerate(self.preview_queues):
            if not q or idx >= len(self.frames):
                continue
                
            # Drain the queue to get the most recent frame
            msg = None
            try:
                while not q.empty():
                    msg = q.get_nowait()
            except:
                continue
                
            if msg is None:
                continue

            canvas = self.frames[idx]['canvas']
            canvas.delete('all')
            W, H = canvas.winfo_width(), canvas.winfo_height()
            
            # Ensure canvas has been sized
            if W <= 1 or H <= 1:
                continue

            # ----- Handle missing frame_bgr gracefully -----
            frame_bgr = msg.get('frame_bgr', None)
            if frame_bgr is None:
                continue

            try:
                frame_h, frame_w = frame_bgr.shape[:2]
                scale = min(W / frame_w, H / frame_h)
                scaled_w, scaled_h = frame_w * scale, frame_h * scale
                offset_x, offset_y = (W - scaled_w) // 2, (H - scaled_h) // 2

                def transform_coords(lms):
                    return [
                        ((x * scaled_w) + offset_x, (y * scaled_h) + offset_y, z)
                        for x, y, z in lms
                    ]

                img = frame_to_tk(frame_bgr, (W, H), mode='contain')
                if img is not None:
                    canvas.create_image(W // 2, H // 2, image=img, anchor='center')
                    canvas.image = img  # Keep reference!

                    if msg['mode'] == 'holistic':
                        face_lms = transform_coords(msg.get('face', []))
                        pose_lms = transform_coords(msg.get('pose', []))
                        self._draw_holistic(canvas, face_lms, pose_lms)
                    else:  # multiface mode
                        faces = msg.get('faces', [])
                        for face in faces:
                            if 'landmarks' in face and face['landmarks']:
                                face['landmarks'] = transform_coords(face['landmarks'])
                        self._draw_multiface(canvas, faces)
            except Exception as e:
                print(f"[GUI] Error in preview for canvas {idx}: {e}")
                # Draw error indicator
                canvas.create_text(W//2, H//2, text="Preview Error", fill='red', font=('Arial', 12))


    def _draw_holistic(self, canvas, face_lms, pose_lms):
        """
        Draw holistic landmarks using pixel coordinates directly.
        """
        # Draw face landmarks
        for (cx, cy, _) in face_lms:
            r = 2  # point radius
            canvas.create_oval(cx - r, cy - r, cx + r, cy + r,
                            outline='lime', fill='lime')

        # Draw pose connections
        connections = mp.solutions.holistic.POSE_CONNECTIONS
        for start_idx, end_idx in connections:
            if start_idx < len(pose_lms) and end_idx < len(pose_lms):
                x1, y1, _ = pose_lms[start_idx]
                x2, y2, _ = pose_lms[end_idx]
                canvas.create_line(x1, y1, x2, y2, fill='lime', width=2)

    def _draw_multiface(self, canvas, faces):
        """
        Draw multi-face landmarks using pixel coordinates directly.
        """
        for face in faces:
            fid = face['id']
            lm_xyz = face['landmarks']
            cx_norm, cy_norm = face['centroid']

            # Draw landmarks
            for (cx, cy, _) in lm_xyz:
                r = 2
                canvas.create_oval(cx - r, cy - r, cx + r, cy + r,
                                outline='lime', fill='lime')

            # Label ID at centroid
            label_x = int(cx_norm * canvas.winfo_width())
            label_y = max(10, int(cy_norm * canvas.winfo_height()) - 20)
            canvas.create_text(label_x, label_y, text=f"ID {fid}",
                            fill='yellow', font=('Arial', 12, 'bold'))


        
    def preview_comodulation(self):
        # ─── reschedule first ─────────────────────────────────────────────
        self._comod_after_id = self.after(30, self.preview_comodulation)

        # ─── guard: must be streaming and have score queues ──────────────
        if not self.streaming:
            return
        if len(self.score_queues) < 2:
            return

        # ─── try to pull one new vector from each (non-blocking) ──────────
        scores = []
        for i, q in enumerate(self.score_queues[:2]):  # Only use first 2
            if q is None:
                return
            try:
                # Drain queue to get latest
                latest = None
                while not q.empty():
                    latest = q.get_nowait()
                if latest is not None:
                    scores.append(latest)
            except:
                pass

        if len(scores) < 2:
            return

        # ─── compute and draw ─────────────────────────────────────────────
        try:
            corr = self.correlator.update(np.array(scores[0]), np.array(scores[1]))
            if corr is not None:
                self.update_plot(corr)
        except Exception as e:
            print(f"[GUI] Error in comodulation: {e}")

    def update_plot(self, corr):
        c = self.bar_canvas
        now = time.time()
        if hasattr(self, '_last_plot') and now - self._last_plot < 0.1:
            return
        self._last_plot = now

        # Lazy load labels
        if getattr(self, 'blend_labels', None) is None:
            for info in self.frames:
                p = info.get('participant')
                labels = getattr(p, 'blend_labels', None)
                if labels:
                    self.blend_labels = labels
                    break

        c.delete('all')
        W, H = c.winfo_width(), c.winfo_height()
        n = len(corr)
        if n == 0:
            c.create_text(W//2, H//2, text="No data", fill='gray')
            return

        bar_h = max(10, H / n)  # Increased min height
        font_size = int(bar_h * 0.5)  # Dynamic font size
        font = ('Arial', max(6, font_size))
        label_w = 100
        margin = 5
        bg = (255, 255, 255)

        for i, val in enumerate(corr):
            y0 = i * bar_h
            y1 = y0 + bar_h - 1
            x0 = label_w + margin
            length = val * (W - label_w - 2 * margin)
            x1 = x0 + length

            # Label
            lbl = self.blend_labels[i] if self.blend_labels and i < len(self.blend_labels) else str(i)
            cy = int(y0 + bar_h / 2)

            # Color blend
            base = (0, 255, 0) if val >= 0 else (255, 0, 0)
            a = min(max(abs(val), 0.0), 1.0)
            r = int(bg[0] + (base[0] - bg[0]) * a)
            g = int(bg[1] + (base[1] - bg[1]) * a)
            b = int(bg[2] + (base[2] - bg[2]) * a)
            color = f'#{r:02x}{g:02x}{b:02x}'

            # Draw bar and text
            c.create_rectangle(x0, y0, x1, y1, fill=color, outline='')
            c.create_text(label_w - margin, cy, text=lbl, anchor='e', font=font, fill='black')

    def start_stream(self):
        if self.streaming: 
            return
        # find all camera slots where we've spawned a worker
        procs = [info for info in self.frames if info.get('control_conn')]
        if not procs:
            return

        self.streaming = True
        self.score_queues.clear()

        for info in procs:
            print("[GUI] sending start_stream to", len(procs), "worker(s)", flush=True)
        for info in procs:
            # ← send the magic message into the worker
            info['control_conn'].send('start_stream')
            # register its score queue for the comod plot
            self.score_queues.append(info['score_queue'])
            info['meta_label'].config(text="LSL → ON")
        
        # if you have 2+ participants, kick off combined_loop or rely on preview_comodulation
        if len(self.score_queues) >= 2:
            nominal_fps = DESIRED_FPS
            self.correlator.setup_stream(fps=nominal_fps)
            self.stop_evt = threading.Event()
            threading.Thread(target=self.combined_loop, daemon=True).start()

        # start drawing the comod bars
        self.preview_comodulation()


    def stop_stream(self):
        if not self.streaming:
            return
        self.streaming = False

        for info in self.frames:
            if info.get('control_conn'):
                info['control_conn'].send('stop_stream') 
            aid = info.get('after_id')
            if aid:
                try:
                    self.after_cancel(aid)
                except (tk.TclError, ValueError):
                    pass
                info['after_id'] = None

            stop_evt = info.get('stop_evt')
            if stop_evt:
                stop_evt.set()

            thread = info.get('thread')
            if thread and thread.is_alive():
                thread.join(timeout=1)

        if hasattr(self, '_comod_after_id') and self._comod_after_id is not None:
            try:
                self.after_cancel(self._comod_after_id)
            except (tk.TclError, ValueError):
                pass
            self._comod_after_id = None

        if hasattr(self, 'stop_evt'):
            self.stop_evt.set()

        for info in self.frames:
            part = info.get('participant')
            if part:
                part.stop_streaming()

        self.worker_procs.clear()
        self.preview_queues.clear()
        self.score_queues.clear()
        self.bar_canvas.delete('all')


        def _cleanup():
            if any((info.get('thread') and info.get('thread').is_alive()) or
                   (self.data_thread and self.data_thread.is_alive())
                   for info in self.frames):
                self.after(100, _cleanup)
                return
            for i, info in enumerate(self.frames):
                # just reset the label; camera & ML remain ready
                info['meta_label'].config(text="ID: -, Stream: -")

            self.bar_canvas.delete('all')
            self.correlator.close()

        self.after(50, _cleanup)

    def reset(self):
        """
        Stop all streams and visualizations, 
        then rebuild the GUI to its initial state.
        """
        # 1) Stop data stream (participants + comodulator)
        self.stop_stream()

        # 2) Fully close correlator outlet and reset state
        self.correlator.close()
        # Re-instantiate to clear any holding state
        self.correlator = ChannelCorrelator(window_size=60, fps=30)

        #Re-run build_frames, resetting threads & callbacks
        self.build_frames()

        # 4) Clear the bar plot back to its debug rectangle
        self.bar_canvas.delete('all')
        self.bar_canvas.create_rectangle(10,10,110,60, fill='red', tags='debug')

        # 5) Reset streaming flag
        self.streaming = False
        if hasattr(self, '_comod_after_id'):
            try:
                self.after_cancel(self._comod_after_id)
            except tk.TclError:
                pass
        #Reset executor
        if hasattr(self, 'exec') and self.exec:
            self.exec.shutdown(wait=False, cancel_futures=True)
        self.exec = ThreadPoolExecutor(max_workers=12)
        if hasattr(self, 'worker_procs'):
            for proc in self.worker_procs:
                if proc is not None:
                    proc.terminate()
                    proc.join()
            self.worker_procs = []
            self.preview_queues = []

if __name__ == '__main__':
    YouQuantiPyGUI().mainloop()
