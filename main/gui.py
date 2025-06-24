import tkinter as tk
from tkinter import ttk
from pygrabber.dshow_graph import FilterGraph
from PIL import Image
import threading
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures

import queue
import time
import base64

import numpy as np
import cv2
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2

from participant import Participant
from facetracker import FaceTracker 
from correlator import ChannelCorrelator

# Path to your Mediapipe face_landmarker.task
MODEL_PATH = r"D:\Projects\MovieSynchrony\face_landmarker.task"
try:
    RESAMPLE = Image.Resampling.LANCZOS
except AttributeError:       # Pillow <10
    RESAMPLE = Image.LANCZOS

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
    ok, png = cv2.imencode(".png", bgr, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        return None

    return tk.PhotoImage(data=base64.b64encode(png).decode("ascii"))

class YouQuantiPyGUI(tk.Tk):
    BAR_HEIGHT = 150

    def __init__(self):
        super().__init__()
        # Allows for mirroring in holistic mode. Will throw an error if used in multiface
        self.testing_mode = False 

        self.exec = ThreadPoolExecutor(max_workers=24)
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


        self.data_stop_evt = threading.Event()
        self.data_thread = None
        self.streaming = False
        self.blend_labels = None 

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
        self._comod_after_id = self.after(30, self.preview_comodulation)


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
        for info in self.frames:
            aid = info.get('after_id')
            if aid:
                try:
                    self.after_cancel(aid)
                except tk.TclError:
                    pass
        #destroy all frames
        for widget in self.container.winfo_children():
            widget.destroy()
        self.frames.clear()

        # Teardown existing frames
        for i, info in enumerate(self.frames):
            print(f"tearing down frame {i}")
            # 1) Release the camera first so cap.read() unblocks:
            if info.get('participant') and not (self.testing_mode and i>0):
                info['participant'].release()
            # 2) Signal the thread to stop
            info['stop_evt'].set()
            # 3) Join (optionally with a short timeout)
            if info.get('thread'):
                info['thread'].join(timeout=1.0)
            # info['frame'].destroy()
            # self.frames.clear()

        self.cams = list_video_devices()
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

        for i in range(self.camera_count.get()):
            frame = ttk.LabelFrame(self.container, text=f"Camera {i}")  
            frame.grid(row=0, column=i, padx=5, sticky='nsew')
            self.container.grid_columnconfigure(i, weight=1)
            self.container.grid_rowconfigure(0, weight=1) 
            frame.grid_columnconfigure(0, weight=1)
            frame.grid_rowconfigure(1, weight=1)
            row = 0
            if not self.multi_face_mode.get():
                ent = ttk.Entry(frame, width=15)
                ent.grid(row=row, column=0, sticky='w', padx=4, pady=2)
                row += 1
            else:
                ent = None

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

            stop_evt = threading.Event()
            frame_queue = queue.Queue(maxsize=1)

            def on_select(event=None, idx=i):
                print(f"[DEBUG] on_select(idx={idx})")
                try:
                    info = self.frames[idx]
                    # Release old resources
                    if info.get('participant'):
                        info['participant'].release()

                    # Parse selected camera
                    sel = info['combo'].get()
                    cam_idx = int(sel.split(':')[0])
                    cam_info = next((c for c in self.cams if c[0]==cam_idx), None)
                    cam_w = cam_h = None
                    if cam_info:
                        _,_,cam_w,cam_h = cam_info

                    # Create new participant
                    part = Participant(cam_idx, MODEL_PATH, self.enable_mesh.get(),
                                       multi_face_mode=self.multi_face_mode.get(), fps=30)
                    if cam_w and cam_h:
                        part.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_w)
                        part.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_h)
                    if self.testing_mode and idx>0 and self.frames[0]['participant']:
                        part.cap = self.frames[0]['participant'].cap

                    # Stop old preview thread
                    info['stop_evt'].set()
                    if info.get('thread'): info['thread'].join()

                    # Update info dict
                    info.update(
                        participant=part,
                        stop_evt=threading.Event(),
                        frame_queue=queue.Queue(maxsize=1),
                        detect_count=0,
                        last_scores=[0.0]*52
                    )

                    # Start preview thread
                    t = threading.Thread(target=self.preview_loop, args=(idx,), daemon=True)
                    info['thread'] = t
                    t.start()
                    # Schedule preview updates
                    info['after_id'] = self.after(30, self.schedule_preview, idx)

                except Exception as e:
                    print(f"[ERROR] on_select idx={idx}: {e}")

            cmb.bind('<<ComboboxSelected>>', on_select)
            self.frames.append({
                'frame': frame, 'canvas': canvas, 'entry': ent,
                'combo': cmb, 'meta_label': meta,
                'participant': None, 'face_landmarker': None,
                'stop_evt': stop_evt, 'thread': None,
                'frame_queue': frame_queue,
                'detect_count': 0, 'last_scores': [0.0]*52
            })
            self.frames[-1]['face_tracker'] = FaceTracker(track_threshold=200, max_missed=1500)

            
        # first, remove any old panel
        if hasattr(self, 'multi_panel'):
            self.multi_panel.destroy()

        # then, if multi-face mode is active, build a new panel
        if self.multi_face_mode.get():
            self.multi_panel = ttk.LabelFrame(self, text="Participants")    
            self.multi_panel.grid(row=3, column=0, columnspan=self.camera_count.get(),
                                    pady=10, sticky='ew')                   
            # clear any old list
            self.multi_entries = []  

            for i in range(self.participant_count.get()):
                row = ttk.Frame(self.multi_panel)
                row.pack(fill='x', padx=5, pady=2)
                ttk.Label(row, text=f"Participant {i+1}:").pack(side='left')
                ent = ttk.Entry(row, width=20)
                ent.pack(side='left', padx=5)
                self.multi_entries.append(ent)
    # --------------------------------------------------------------------
    #                BACKGROUND WORKER – runs inside EXEC
    # --------------------------------------------------------------------
    def process_frame(self, idx, frame_bgr, multi_face_enabled):
        """
        Heavy work: MediaPipe detection, landmark drawing, labels & LSL push.
        Runs in a worker thread, returns a fully rendered BGR image.
        """
        info = self.frames[idx]
        p    = info['participant']

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        disp = frame_bgr.copy()
        mesh_vals = []
        pose_vals = []

        # Multi face mode (FaceLandmarker) ---------------------------------
        if p.multi_face_mode:
            # 1) pull oldest result
            with p._lock:
                if p.async_results:
                    _, res = p.async_results.popitem(last=False)
                else:
                    res = None

            if res:
                h, w = frame_bgr.shape[:2]
                faces = p._parse_multiface_result(res, w, h)
                p.last_detected_faces = faces
            else:
                faces = p.last_detected_faces

            # 2) schedule next detection
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_send = int(time.monotonic() * 1000)
            prev_ts = getattr(p, 'last_sent_ts', 0)
            if ts_send <= prev_ts:
                ts_send = prev_ts + 1
            p.last_sent_ts = ts_send
            p.multiface_landmarker.detect_async(mp_img, timestamp_ms=ts_send)

            # 3) draw + stream
            detections = [f['centroid'] for f in faces]
            tracker = info['face_tracker']
            ids = tracker.update(detections)
            for fdata in faces:
                converted = [landmark_pb2.NormalizedLandmark(x=pt.x, y=pt.y, z=pt.z)
                            for pt in fdata['landmarks']]
                landmark_list = landmark_pb2.NormalizedLandmarkList(landmark=converted)
                self.mp_drawing.draw_landmarks(
                    disp,
                    landmark_list,
                    mp.solutions.face_mesh.FACEMESH_TESSELATION,
                    None,
                    self.mp_styles.get_default_face_mesh_tesselation_style()
                )
            for (cx, cy), fid in zip(detections, ids):
                # default label
                label = f"ID {fid}"
                # only try to read a name if we have an entry for this face‐ID
                if multi_face_enabled and hasattr(self, 'multi_entries'):
                    entry_idx = fid - 1
                    if 0 <= entry_idx < len(self.multi_entries):
                        name = self.multi_entries[entry_idx].get().strip()
                        if name:
                            label = name

                cv2.putText(
                    disp,
                    label,
                    (cx - 30, cy - 240),  # “above head” coords
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2.2,                   # your chosen fontScale
                    (255, 255, 0),
                    3
                )
            if self.streaming:
                p.push_multiface_samples(
                    faces=faces,
                    face_ids=ids,
                    include_mesh=self.enable_mesh.get()
                )


            # ——— store ALL face blends for downstream plotting/correlator ———
            info['face_blends'] = {
                fid: fdata['blend']
                for fdata, fid in zip(faces, ids)
            }
                            # update fallback arrays and scores
            if faces:
                self.last_mesh_vals = faces[0]['mesh']
                self.last_scores    = faces[0]['blend']
            else:
                self.last_mesh_vals = [0.0] * (478*3)
                self.last_scores    = [0.0] * 52

            scores = self.last_scores

        # Holistic mode (holistic model) -----------------------------------
        else:
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts = int(time.monotonic() * 1000)
            prev_ts = getattr(p, 'last_sent_ts', 0)
            if ts <= prev_ts:
                ts = prev_ts + 1
            p.last_sent_ts = ts
            p.face_landmarker.detect_async(mp_img, timestamp_ms=ts)
            p.last_requested_ts = ts

            res = p.holistic.process(rgb)
            # draw face + pose
            self.mp_drawing.draw_landmarks(
                disp, res.face_landmarks,
                mp.solutions.holistic.FACEMESH_TESSELATION,
                None, self.mp_styles.get_default_face_mesh_tesselation_style()
            )
            self.mp_drawing.draw_landmarks(
                disp, res.pose_landmarks,
                mp.solutions.holistic.POSE_CONNECTIONS,
                self.mp_styles.get_default_pose_landmarks_style()
            )

            # extract mesh & pose values
            if res.face_landmarks:
                mesh_vals = [coord for lm in res.face_landmarks.landmark
                            for coord in (lm.x, lm.y, lm.z)]
            else:
                mesh_vals = [0.0] * (468*3)

            if res.pose_landmarks:
                pose_vals = [coord for lm in res.pose_landmarks.landmark
                            for coord in (lm.x, lm.y, lm.z, lm.visibility)]
            else:
                pose_vals = [0.0] * (33*4)

            # —— NEW: pull blendshape scores from participant ——
            scores = getattr(p, 'last_blend_scores', [0.0] * 52)

        # store for correlator & GUI
        info['last_mesh_vals'] = mesh_vals
        info['last_pose_vals'] = pose_vals
        info['last_scores']    = scores

        # draw green bar plot
        h, w, _ = disp.shape
        bar_h = h // 52
        for j, s in enumerate(scores):
            y0, y1 = j * bar_h, (j + 1) * bar_h
            cv2.rectangle(disp, (0, y0), (int(s * 150), y1), (0, 255, 0), -1)
            
        return disp      # BGR image ready for the GUI

    def preview_loop(self, idx):
        info   = self.frames[idx]
        p      = info['participant']
        q      = info['frame_queue']
        stop   = info['stop_evt']

        t0 = time.perf_counter()
        frame_idx = 0

        while not stop.is_set():
            ok, frame = p.cap.read()
            if not ok:
                continue

            frame_idx += 1

            # ---------- MULTI-FACE asynchronously ----------
            if self.multi_face_mode.get():
                # queue has size 1 ⇒ if full, replace the old Future
                if q.full():
                    try:
                        q.get_nowait()        # drop oldest future
                    except queue.Empty:
                        pass

                fut = self.exec.submit(self.process_frame, idx, frame, True)
                q.put_nowait(fut)

            # ---------- HOLISTIC synchronously --------------
            else:
                disp = self.process_frame(idx, frame, False)
                fut  = concurrent.futures.Future()
                fut.set_result(disp)

                if q.full():
                    try:
                        q.get_nowait()
                    except queue.Empty:
                        pass
                q.put_nowait(fut)

            # ----- optional live FPS monitor ----------------
            if frame_idx % 30 == 0:
                dt  = time.perf_counter() - t0
                fps = frame_idx / dt if dt else 0
                print(f"[Cam {idx}] q={q.qsize()}  fps≈{fps:4.1f}")

            stop.wait(0.001)


    def schedule_preview(self, idx):
        info   = self.frames[idx]
        canvas = info['canvas']
        q      = info['frame_queue']

        # arm next call immediately
        if not info['stop_evt'].is_set():
            info['after_id'] = self.after(30, self.schedule_preview, idx)

        # get the *latest* completed future in the queue
        try:
            fut = q.get_nowait()
            while not q.empty():
                fut = q.get_nowait()          # discard older ones
        except queue.Empty:
            return                            # nothing ready

        if not fut.done():
            return                            # still processing

        disp = fut.result()                   # BGR frame

        # ----------- draw on canvas -----------
        cw, ch = canvas.winfo_width(), canvas.winfo_height()
        if cw < 2 or ch < 2:
            return

        img = frame_to_tk(disp, (cw, ch), mode="contain")
        if img is None:
            return

        canvas.delete('all')
        canvas.create_image(cw//2, ch//2, image=img, anchor='center')
        canvas.image = img


    def preview_comodulation(self):
        # ←–– early-out during rebuild -----------------------------------------
        if not self.frames:        # nothing valid yet
            self._comod_after_id = self.after(30, self.preview_comodulation)
            return
        # ----------------------------------------------------------------------

        if self.multi_face_mode.get():
            info = self.frames[0]
            blends = info.get('face_blends', {})
            if len(blends) >= 2:
                fid1, fid2 = sorted(blends)[:2]
                s1, s2 = blends[fid1], blends[fid2]
                corr = self.correlator.update(np.array(s1), np.array(s2))
                if corr is not None:
                    self.update_plot(corr)
        else:
            if len(self.frames) >= 2:
                s1 = self.frames[0].get('last_scores')
                s2 = self.frames[1].get('last_scores')
                corr = self.correlator.update(np.array(s1), np.array(s2))
                if corr is not None:
                    self.update_plot(corr)

        # reschedule and remember the ID
        self._comod_after_id = self.after(30, self.preview_comodulation)


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

            stop.wait(0.001) 
    def start_stream(self):
        if self.streaming:
            return

        valid = [
            info for info in self.frames
            if info['participant'] and hasattr(info['participant'], 'setup_stream')
        ]

        # require at least one feed (camera or face)
        if len(valid) < 1:
            return

        self.streaming = True
        if self.multi_face_mode.get():
            # ── MULTI-FACE MODE ──
            p = valid[0]['participant']

            # (Re)build the per-face name→stream map
            p.face_outlets.clear()
            p.multiface_names = {}
            for fid, ent in enumerate(self.multi_entries, start=1):
                p.multiface_names[fid] = ent.get().strip() or f"ID{fid}"

           # Show which face streams will open
            valid[0]['meta_label'].config(
                text="Multi-Face → " +
                    ", ".join(p.multiface_names[fid]
                              for fid in sorted(p.multiface_names))
            )

            # Always clear the co-mod stop event
            self.data_stop_evt.clear()

            # If two or more faces are active, open the co-modulator outlet
            if self.participant_count.get() >= 2:
                # use camera's FPS as nominal rate
                self.correlator.setup_stream(fps=p.fps)


        else:
            # ── HOLISTIC MODE ──
            fps_list = []
            # 1) start each camera’s landmark stream
            for i, info in enumerate(valid):
                part = info['participant']
                pid  = info['entry'].get().strip() or f"P{i+1}"
                nm   = f"{pid}_landmarks"
                part.setup_stream(pid, nm)
                info['meta_label'].config(text=f"ID: {pid}, Stream: {nm}")
                fps_list.append(part.fps)

            # 2) always clear the co-mod stop event
            self.data_stop_evt.clear()

            # 3) if you have ≥ 2 cameras, open co-modulator and spawn the combiner
            if len(valid) >= 2:
                nominal_fps = int(round(sum(fps_list) / len(fps_list)))
                self.correlator.setup_stream(fps=nominal_fps)
                t = threading.Thread(target=self.combined_loop, daemon=True)
                self.data_thread = t
                t.start()


    def stop_stream(self):
        if not self.streaming:
            return
        self.streaming = False

        # --- 0) cancel all per-frame preview callbacks ---
        for info in self.frames:
            aid = info.get('after_id')
            if aid:
                try:
                    self.after_cancel(aid)
                except tk.TclError:
                    pass
           # 1) Signal all preview threads to stop
        for info in self.frames:
            info['stop_evt'].set()
            # 2) Cancel any pending preview callbacks
            if 'after_id' in info:
                self.after_cancel(info['after_id'])
            # 3) Join the thread
            if info.get('thread') and info['thread'].is_alive():
                info['thread'].join(timeout=1)
        # 4) Cancel the comodulation updater
        if hasattr(self, '_comod_after_id'):
            self.after_cancel(self._comod_after_id)
        # 5) Stop the LSL data loop as before
        self.data_stop_evt.set()
        for info in self.frames:
            part = info.get('participant')
            if part:
                part.stop_streaming()
        self.exec.shutdown(wait=False, cancel_futures=True)
        self.exec = ThreadPoolExecutor(max_workers=24)

        def _cleanup():
            if any((info['thread'] and info['thread'].is_alive()) or
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
        self.exec.shutdown(wait=False, cancel_futures=True)
        self.exec = ThreadPoolExecutor(max_workers=24)

if __name__ == '__main__':
    YouQuantiPyGUI().mainloop()
