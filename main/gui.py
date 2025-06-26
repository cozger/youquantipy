import tkinter as tk
from tkinter import ttk
from pygrabber.dshow_graph import FilterGraph
from PIL import Image, ImageTk
from multiprocessing import Queue as MPQueue
from multiprocessing import Process
from multiprocessing import Pipe
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
from fastreader import FastScoreReader, NumpySharedBuffer


# Path to your Mediapipe face_landmarker.task
MODEL_PATH = r"D:\Projects\MovieSynchrony\face_landmarker.task"
try:
    RESAMPLE = Image.Resampling.LANCZOS
except AttributeError:       # Pillow <10
    RESAMPLE = Image.LANCZOS
DESIRED_FPS = 30
CAM_RESOLUTION = (1920, 1080)
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
    Fast BGR → Tk PhotoImage with performance optimizations.
    """
    cw, ch = target_wh
    h, w = bgr.shape[:2]
    
    # Skip processing if target is too small
    if cw <= 1 or ch <= 1:
        return None

    # Aggressive downscaling for very large targets
    max_dimension = 1280  # Maximum dimension for preview
    if cw > max_dimension or ch > max_dimension:
        scale_factor = min(max_dimension / cw, max_dimension / ch)
        cw = int(cw * scale_factor)
        ch = int(ch * scale_factor)

    # Compute scale
    if mode == "cover":
        scale = max(cw / w, ch / h)
    else:
        scale = min(cw / w, ch / h)

    # Limit upscaling to prevent quality loss
    scale = min(scale, 2.0)  # Never upscale more than 2x

    if scale != 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        # Use INTER_LINEAR for better performance
        bgr = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Centre-crop for "cover"
    if mode == "cover":
        h, w = bgr.shape[:2]
        x0 = max(0, (w - cw) // 2)
        y0 = max(0, (h - ch) // 2)
        bgr = bgr[y0:y0 + ch, x0:x0 + cw]

    # Convert to RGB for PIL
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    
    # Use PIL's built-in PhotoImage which is more efficient
    try:
        return ImageTk.PhotoImage(pil_img)
    except:
        # Fallback to the original method if ImageTk not available
        ok, png = cv2.imencode(".png", bgr, [cv2.IMWRITE_PNG_COMPRESSION, 9])
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
        self.participant_names = {}
        self.score_reader = None
        self.gui_update_thread = None
        self.stop_evt = None


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
        self.after(32, self.continuous_correlation_monitor)
        self.after(1000, self.update_participant_names)
    
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

    def update_participant_names(self):
        """Periodically update participant names from entry fields"""
        # Schedule next update
        self.after(1000, self.update_participant_names)  # Update every second
        
        if self.multi_face_mode.get():
            # Update multiface participant names
            if hasattr(self, 'multi_entries'):
                self.participant_names = {}
                for idx, entry in enumerate(self.multi_entries):
                    name = entry.get().strip()
                    if name:
                        self.participant_names[idx + 1] = name  # Face IDs start at 1
                    else:
                        self.participant_names[idx + 1] = f"Face {idx + 1}"
        else:
            # Update holistic participant names
            self.participant_names = {}
            for idx, info in enumerate(self.frames):
                if info.get('entry'):
                    name = info['entry'].get().strip()
                    if name:
                        self.participant_names[idx] = name
                    else:
                        self.participant_names[idx] = f"P{idx + 1}"

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
            # --- Create Participant name input for multiface ---
        # Remove any old panel
        if hasattr(self, 'multi_panel'):
            self.multi_panel.destroy()

        # Create participant name entry panel for multiface mode
        if self.multi_face_mode.get():
            self.multi_panel = ttk.LabelFrame(self, text="Participant Names")
            self.multi_panel.grid(row=2, column=0, columnspan=3, padx=10, pady=10, sticky='ew')
            
            # Clear any old list
            self.multi_entries = []
            
            for i in range(self.participant_count.get()):
                row = ttk.Frame(self.multi_panel)
                row.pack(fill='x', padx=5, pady=2)
                ttk.Label(row, text=f"Face {i+1}:").pack(side='left', padx=(0,5))
                ent = ttk.Entry(row, width=20)
                ent.insert(0, f"Participant {i+1}")  # Default name
                ent.pack(side='left', padx=5)
                self.multi_entries.append(ent)
                self.participant_names[i + 1] = f"Participant {i+1}"
        else:
            # Clean up entries if switching away from multiface
            if hasattr(self, 'multi_entries'):
                self.multi_entries = []

        # Build camera selection values
        cam_vals = [f"{i}: {name} ({w}x{h})" for i,name,w,h in self.cams]

        def on_select(idx, event=None):
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

            pv_q = MPQueue(maxsize=2) #preview queue
            enable_mesh = self.enable_mesh.get()
            multi_face = self.multi_face_mode.get()

            if idx >= len(self.preview_queues):
                self.preview_queues.append(pv_q)
            else:
                self.preview_queues[idx] = pv_q

            # Create numpy shared buffer instead of regular SharedScoreBuffer
            score_buffer = NumpySharedBuffer(104 if multi_face else 52)
            info['score_buffer'] = score_buffer
            score_buffer_name = score_buffer.name  # Get the shared memory name

            fps = DESIRED_FPS
            resolution = CAM_RESOLUTION

            if not multi_face:
                pid = info['entry'].get().strip() or f"P{idx+1}"
                stream_name = f"{pid}_landmarks"
                participant_id = pid
                self.participant_names[idx] = participant_id
            else:
                stream_name = None
                participant_id = None

            # Pass the buffer name instead of the buffer object
            proc = Process(
                target=participant_worker,
                args=(cam_idx, MODEL_PATH, participant_id, stream_name, fps,
                    enable_mesh, multi_face, pv_q, score_buffer_name, child_conn, resolution),  # Changed score_buffer to score_buffer_name
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

            # Store references for later use
            info = {
                'frame': frame,
                'entry': ent,
                'canvas': canvas,
                'meta_label': meta,
                'combo': cmb,
                'proc': None,
                'preview_queue': None,
                'score_buffer': None,
            }
            self.frames.append(info)

            # Bind the combobox selection event to on_select
            cmb.bind("<<ComboboxSelected>>", lambda event, idx=i: on_select(idx, event))
    
    
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
        # Draw face mesh with tessellation
        if face_lms:
            # Draw tessellation connections
            connections = mp.solutions.face_mesh.FACEMESH_TESSELATION
            for connection in connections:
                start_idx, end_idx = connection
                if start_idx < len(face_lms) and end_idx < len(face_lms):
                    x1, y1, _ = face_lms[start_idx]
                    x2, y2, _ = face_lms[end_idx]
                    canvas.create_line(x1, y1, x2, y2, fill='#00FF00', width=1)

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

            # Draw tessellation instead of dots
            connections = mp.solutions.face_mesh.FACEMESH_TESSELATION
            for connection in connections:
                start_idx, end_idx = connection
                if start_idx < len(lm_xyz) and end_idx < len(lm_xyz):
                    x1, y1, _ = lm_xyz[start_idx]
                    x2, y2, _ = lm_xyz[end_idx]
                    canvas.create_line(x1, y1, x2, y2, fill='#00FF00', width=1)

            # Label ID above the face (hovering effect)
            label_x = int(cx_norm * canvas.winfo_width())
            label_y = max(20, int(cy_norm * canvas.winfo_height()) - 240)  # 240 pixels above centroid
            
            # Create background rectangle for better visibility
            label_text =getattr(self, 'participant_names', {}).get(fid, f"ID {fid}")
            text_id = canvas.create_text(label_x, label_y, text= label_text ,
                            fill='yellow', font=('Arial', 20, 'bold'))
            
            # Get text bounding box and create background
            bbox = canvas.bbox(text_id)
            if bbox:
                canvas.create_rectangle(bbox[0]-5, bbox[1]-2, bbox[2]+5, bbox[3]+2, 
                                    fill='black', outline='black')
                # Recreate text on top
                canvas.create_text(label_x, label_y, text=label_text,
                                fill='yellow', font=('Arial', 20, 'bold'))

    def continuous_correlation_monitor(self):
        """Continuously monitor for correlation data regardless of streaming state"""
        self.after(33, self.continuous_correlation_monitor)
        
        if self.streaming:
            if hasattr(self, 'latest_correlation') and self.latest_correlation is not None:
                self.update_plot(self.latest_correlation)
            return
        
        # Fallback to direct buffer reading when not streaming (for testing)
        scores = []
        
        if self.multi_face_mode.get():
            active_workers = [info for info in self.frames if info.get('proc') and info.get('score_buffer')]
            if len(active_workers) >= 1:
                buffer = active_workers[0]['score_buffer']
                if buffer is not None:
                    try:
                        data, ts = buffer.read_latest()
                        if data is not None and len(data) >= 104:
                            scores.append(data[:52])
                            scores.append(data[52:104])
                    except:
                        pass
        else:
            active_workers = [info for info in self.frames if info.get('proc') and info.get('score_buffer')]
            if len(active_workers) < 2:
                return
                
            for i, info in enumerate(active_workers[:2]):
                buffer = info['score_buffer']
                if buffer is None:
                    continue
                    
                try:
                    data, ts = buffer.read_latest()
                    if data is not None and len(data) >= 52:
                        scores.append(data[:52])
                except:
                    continue
        
        if len(scores) >= 2:
            try:
                corr = self.correlator.update(np.array(scores[0]), np.array(scores[1]))
                if corr is not None and len(corr) > 0:
                    self.update_plot(corr)
            except Exception as e:
                print(f"[GUI] Error in correlation monitor: {e}")
                
    def update_plot(self, corr):
        """
        Updated plot method that handles blend labels properly in multiprocessing setup
        """
        c = self.bar_canvas
        
        # Add this safety check
        W, H = c.winfo_width(), c.winfo_height()
        if W <= 1 or H <= 1:
            return  # Skip plotting if canvas not properly sized
        
        now = time.time()
        if hasattr(self, '_last_plot') and now - self._last_plot < 0.033:
            return
        self._last_plot = now
        # Use hardcoded blend labels since we can't access them from worker processes
        if getattr(self, 'blend_labels', None) is None:
            # Standard MediaPipe blendshape names (first 52)
            self.blend_labels = [
                "_neutral", "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft",
                "browOuterUpRight", "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
                "eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft", "eyeLookDownRight",
                "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft", "eyeLookOutRight",
                "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft", "eyeSquintRight",
                "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft", "jawOpen", "jawRight",
                "mouthClose", "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft",
                "mouthFrownRight", "mouthFunnel", "mouthLeft", "mouthLowerDownLeft",
                "mouthLowerDownRight", "mouthPressLeft", "mouthPressRight", "mouthPucker",
                "mouthRight", "mouthRollLower", "mouthRollUpper", "mouthShrugLower",
                "mouthShrugUpper", "mouthSmileLeft", "mouthSmileRight", "mouthStretchLeft",
                "mouthStretchRight", "mouthUpperUpLeft", "mouthUpperUpRight", "noseSneerLeft",
                "noseSneerRight", "tongueOut"
            ]

        c.delete('all')
        c.delete('debug')

        W, H = c.winfo_width(), c.winfo_height()
        n = len(corr)
        if n == 0:
            c.create_text(W//2, H//2, text="No correlation data", fill='gray', font=('Arial', 12))
            return

        bar_h = max(8, H / n)  # Minimum bar height
        font_size = max(6, min(10, int(bar_h * 0.6)))  # Dynamic but readable font size
        font = ('Arial', font_size)
        label_w = 120  # Slightly wider for blend shape names
        margin = 5
        bg = (255, 255, 255)

        for i, val in enumerate(corr):
            if i >= len(self.blend_labels):
                break
                
            y0 = i * bar_h
            y1 = y0 + bar_h - 1
            x0 = label_w + margin
            
            # Clamp correlation value and scale bar length
            val_clamped = max(-1.0, min(1.0, val))
            bar_width = W - label_w - 2 * margin
            
            if val_clamped >= 0:
                # Positive correlation - green bar extending right
                length = val_clamped * bar_width * 0.5
                x1 = x0 + length
                base_color = (0, 255, 0)
            else:
                # Negative correlation - red bar extending left
                length = abs(val_clamped) * bar_width * 0.5
                x1 = x0
                x0 = x0 - length
                base_color = (255, 0, 0)

            # Calculate color with alpha blending
            alpha = abs(val_clamped)
            r = int(bg[0] + (base_color[0] - bg[0]) * alpha)
            g = int(bg[1] + (base_color[1] - bg[1]) * alpha)
            b = int(bg[2] + (base_color[2] - bg[2]) * alpha)
            color = f'#{r:02x}{g:02x}{b:02x}'

            # Draw bar
            if abs(val_clamped) > 0.01:  # Only draw visible bars
                c.create_rectangle(min(x0, x1), y0, max(x0, x1), y1, fill=color, outline='')
            
            # Draw center line for reference
            center_x = label_w + margin
            if i == 0:  # Only draw once
                c.create_line(center_x, 0, center_x, H, fill='lightgray', width=1)

            # Label
            lbl = self.blend_labels[i] if i < len(self.blend_labels) else f"BS_{i}"
            cy = int(y0 + bar_h / 2)
            c.create_text(label_w - margin, cy, text=lbl, anchor='e', font=font, fill='black')
            
            # Value text
            if abs(val_clamped) > 0.05:  # Only show significant correlations
                val_text = f"{val_clamped:.2f}"
                text_x = center_x + (length / 2 if val_clamped >= 0 else -length / 2)
                c.create_text(text_x, cy, text=val_text, anchor='center', 
                            font=('Arial', max(6, font_size-1)), fill='white')
    def _gui_update_loop(self):
        """Separate thread just for updating the GUI plot"""
        print("[GUI] GUI update loop started")
        while self.streaming and not self.stop_evt.is_set():
            if hasattr(self, 'score_reader') and self.score_reader:
                corr = self.score_reader.get_latest_correlation()
                if corr is not None:
                    self.latest_correlation = corr
            time.sleep(0.033)  # 30 FPS for GUI updates
        print("[GUI] GUI update loop stopped")

    def start_stream(self):
        """
        Updated start_stream method with FastScoreReader and debugging
        """
        if self.streaming: 
            print("[GUI] Already streaming, ignoring start request")
            return
            
        # Find all camera slots where we've spawned a worker
        active_workers = [info for info in self.frames if info.get('control_conn')]
        print(f"[GUI] Found {len(active_workers)} active workers")
        if not active_workers:
            print("[GUI] No active workers found")
            return

        self.streaming = True
        self.score_queues.clear()
        
        # Initialize correlation storage
        self.latest_correlation = None

        print(f"[GUI] Starting stream for {len(active_workers)} worker(s)")
        print(f"[GUI] Multi-face mode: {self.multi_face_mode.get()}")
        
        # Send start commands to workers
        for idx, info in enumerate(active_workers):
            print(f"[GUI] Starting worker {idx}")
            
            # Get the appropriate name based on mode
            if self.multi_face_mode.get():
                face_names = {}
                if hasattr(self, 'multi_entries'):
                    for face_idx, entry in enumerate(self.multi_entries):
                        name = entry.get().strip() or f"Participant {face_idx+1}"
                        face_names[face_idx + 1] = name
                info['control_conn'].send(('start_stream', face_names))
                print(f"[GUI] Sent multiface start command with names: {face_names}")
            else:
                if info.get('entry'):
                    participant_name = info['entry'].get().strip() or f"P{idx+1}"
                    info['control_conn'].send(('start_stream', participant_name))
                    print(f"[GUI] Sent holistic start command for: {participant_name}")
                else:
                    info['control_conn'].send('start_stream')
                    print(f"[GUI] Sent basic start command")
            
            # Register score buffer for correlation analysis
            if info.get('score_buffer'):
                self.score_queues.append(info['score_buffer'])
                print(f"[GUI] Added score buffer {idx} to queues")
            else:
                print(f"[GUI] WARNING: No score buffer for worker {idx}")
                
            info['meta_label'].config(text="LSL → ON")

        print(f"[GUI] Total score queues: {len(self.score_queues)}")
        print(f"[GUI] Required for correlation: {1 if self.multi_face_mode.get() else 2}")

        # Setup correlator with FastScoreReader if we have multiple participants OR in multiface mode
        if (self.multi_face_mode.get() and len(self.score_queues) >= 1) or len(self.score_queues) >= 2:
            try:
                print("[GUI] Setting up correlator...")
                self.correlator.setup_stream(fps=DESIRED_FPS)
                print("[GUI] Correlator stream setup complete")
                
                # Use the FastScoreReader instead of combined_loop
                print("[GUI] Creating FastScoreReader...")
                self.score_reader = FastScoreReader(
                    self.score_queues, 
                    self.correlator, 
                    self.multi_face_mode.get(),
                    target_fps=DESIRED_FPS 
                )
                self.score_reader.start()
                print("[GUI] Fast score reader started")
                
                # Start a separate thread just for GUI updates
                self.stop_evt = threading.Event()
                self.gui_update_thread = threading.Thread(
                    target=self._gui_update_loop, 
                    daemon=True
                )
                self.gui_update_thread.start()
                print("[GUI] GUI update thread started")
                
            except Exception as e:
                print(f"[GUI] Error setting up correlator: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"[GUI] Not enough buffers for correlation: need {1 if self.multi_face_mode.get() else 2}, have {len(self.score_queues)}")
                
    def stop_stream(self):
        """
        Updated stop_stream with proper cleanup for FastScoreReader
        """
        if not self.streaming:
            return
            
        print("[GUI] Stopping stream...")
        self.streaming = False

        # Stop all workers
        for info in self.frames:
            if info.get('control_conn'):
                try:
                    info['control_conn'].send('stop_stream')
                except:
                    pass
            info['meta_label'].config(text="LSL → OFF")

        # Stop the fast score reader
        if hasattr(self, 'score_reader') and self.score_reader:
            self.score_reader.stop()
            self.score_reader = None

        # Stop the GUI update thread
        if hasattr(self, 'stop_evt'):
            self.stop_evt.set()
        
        if hasattr(self, 'gui_update_thread') and self.gui_update_thread and self.gui_update_thread.is_alive():
            self.gui_update_thread.join(timeout=2.0)

        # Clear data structures
        self.score_queues.clear()
        self.latest_correlation = None
        
        # Clear the plot
        self.bar_canvas.delete('all')
        self.bar_canvas.create_text(
            self.bar_canvas.winfo_width()//2, 
            self.bar_canvas.winfo_height()//2, 
            text="Stopped", fill='gray', font=('Arial', 14)
        )
        
        # Close correlator
        try:
            self.correlator.close()
        except:
            pass
            
        print("[GUI] Stream stopped and cleaned up")

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

        # 2) Clean up shared memory buffers
        for info in self.frames:
            if info.get('score_buffer'):
                try:
                    info['score_buffer'].cleanup()
                except:
                    pass

        # 3) Fully close correlator outlet and reset state
        self.correlator.close()
        # Re-instantiate to clear any holding state
        self.correlator = ChannelCorrelator(window_size=60, fps=30)

        # 4) Re-run build_frames, resetting threads & callbacks
        self.build_frames()

        # 5) Clear the bar plot back to its debug rectangle
        self.bar_canvas.delete('all')
        self.bar_canvas.create_rectangle(10,10,110,60, fill='red', tags='debug')

        # 6) Reset streaming flag and cleanup FastScoreReader
        self.streaming = False
        if hasattr(self, 'score_reader') and self.score_reader:
            self.score_reader.stop()
            self.score_reader = None
            
        if hasattr(self, '_comod_after_id'):
            try:
                self.after_cancel(self._comod_after_id)
            except tk.TclError:
                pass
                
        # Reset executor
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
