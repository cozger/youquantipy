import tkinter as tk
from tkinter import ttk
from pygrabber.dshow_graph import FilterGraph
from PIL import Image, ImageTk
from multiprocessing import Queue as MPQueue
from multiprocessing import Process
from multiprocessing import Pipe
from concurrent.futures import ThreadPoolExecutor
import queue


import threading
import time
import io
import os, pathlib

import numpy as np
import cv2
import mediapipe as mp

from participant import participant_worker
from facetracker import FaceTracker 
from correlator import ChannelCorrelator
from fastreader import FastScoreReader, NumpySharedBuffer
from videorecorder import VideoRecorderProcess
from audiorecorder import AudioRecorder, VideoAudioRecorder, AudioDeviceManager
from tkinter import filedialog, messagebox
from confighandler import ConfigHandler
from datetime import datetime
from pathlib import Path



# Path to your Mediapipe face_landmarker.task
DEFAULT_MODEL_PATH = r"D:\Projects\MovieSynchrony\face_landmarker.task"


DEFAULT_DESIRED_FPS = 30
CAM_RESOLUTION = (1280, 720)
CAPTURE_FPS = 30  # Capture FPS for video input, can be different from desired FPS
THROTTLE_N = 1    # Only update GUI every 5th frame (~6-7Hz at 30Hz input)
GUI_sleep_time = 0.01 #Also for throttling GUI, time in seconds for refresh, lower is faster
GUI_scheduler_time = 16 #another throttling variable, in milliseconds, lower is faster

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
    """Fast BGR → Tk PhotoImage with performance optimizations."""
    cw, ch = target_wh
    h, w = bgr.shape[:2]
    
    if cw <= 1 or ch <= 1:
        return None

    # Compute scale
    if mode == "cover":
        scale = max(cw / w, ch / h)
    else:
        scale = min(cw / w, ch / h)

    # Fast resize if needed
    if abs(scale - 1.0) > 0.01:  # Only resize if significantly different
        new_w = int(w * scale)
        new_h = int(h * scale)
        # Use INTER_NEAREST for maximum speed
        bgr = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    # Centre-crop for "cover"
    if mode == "cover":
        h, w = bgr.shape[:2]
        x0 = max(0, (w - cw) // 2)
        y0 = max(0, (h - ch) // 2)
        bgr = bgr[y0:y0 + ch, x0:x0 + cw]

    # Convert to RGB and create PhotoImage
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    return ImageTk.PhotoImage(pil_img)

def draw_overlays_combined(
    frame_bgr,
    faces=None,
    pose_landmarks=None,
    labels=None,
    face_mesh=True,
    face_contours=True,
    face_points=True,
    pose_lines=True
):
    """
    Draws face mesh/contours/points for each face, and pose lines if provided.
    - faces: list of dicts, each dict with keys 'landmarks' (normalized), 'id' (optional), 'centroid' (optional)
    - pose_landmarks: MediaPipe pose landmarks (list of normalized coords), or None
    - labels: dict mapping id to name (for multiface), or int->name for holistic
    """
    frame = frame_bgr.copy()
    h, w = frame.shape[:2]

    # Draw face overlays (mesh/contours/points/labels)
    if faces:
        for idx, face in enumerate(faces):
            landmarks = face["landmarks"]
            fid = face.get("id", idx+1)
            face_landmarks_px = [(int(x * w), int(y * h), z) for x, y, z in landmarks]
            if face_mesh:
                for conn in mp.solutions.face_mesh.FACEMESH_TESSELATION:
                    i, j = conn
                    cv2.line(frame, face_landmarks_px[i][:2], face_landmarks_px[j][:2], (64,255,64), 1, cv2.LINE_AA)
            if face_contours:
                for conn in mp.solutions.face_mesh.FACEMESH_CONTOURS:
                    i, j = conn
                    cv2.line(frame, face_landmarks_px[i][:2], face_landmarks_px[j][:2], (0,255,0), 2, cv2.LINE_AA)
            if face_points:
                for pt in face_landmarks_px:
                    cv2.circle(frame, pt[:2], 1, (0,200,255), -1)
            # Draw label above face
            label_text = None
            if labels and fid in labels:
                label_text = labels[fid]
            elif labels and idx in labels:
                label_text = labels[idx]
            elif not labels:
                label_text = f"Face {fid}"
            if label_text:
                if "centroid" in face:
                    cx, cy = int(face["centroid"][0] * w), int(face["centroid"][1] * h) - 50
                else:
                    cx, cy = face_landmarks_px[10][0], face_landmarks_px[10][1] - 50  # use forehead landmark
                cv2.putText(frame, label_text, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2, cv2.LINE_AA)

    # Draw pose overlays (if present)
    if pose_landmarks is not None and len(pose_landmarks) > 0 and pose_lines:
        pose_landmarks_px = [(int(x * w), int(y * h), z) for x, y, z in pose_landmarks]
        for conn in mp.solutions.holistic.POSE_CONNECTIONS:
            i, j = conn
            if i < len(pose_landmarks_px) and j < len(pose_landmarks_px):
                cv2.line(frame, pose_landmarks_px[i][:2], pose_landmarks_px[j][:2], (64,255,255), 2, cv2.LINE_AA)

    return frame

class YouQuantiPyGUI(tk.Tk):
    BAR_HEIGHT = 150

    def __init__(self):
        super().__init__()
        #use handler to import configurations

        self.config = ConfigHandler()
        self.MODEL_PATH = self.config.get('paths.model_path', DEFAULT_MODEL_PATH)
        self.DESIRED_FPS = self.config.get('camera_settings.target_fps', DEFAULT_DESIRED_FPS)

        # Allows for mirroring in holistic mode. Will throw an error if used in multiface
        self.testing_mode = False 
        # Load face_landmarker model buffer
        with open(self.MODEL_PATH, 'rb') as f:
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
        self.canvas_objects = {}  # Cache for reusable canvas objects
        self.transform_cache = {}  # Cache for coordinate transformations
        self.last_frame_time = {}  # Track frame timing per canvas
        

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

        #FPS input
        ttk.Label(self.control_panel, text="FPS:").pack(anchor='w')
        self.desired_fps = tk.IntVar(value=self.DESIRED_FPS)
        self.fps_spin = tk.Spinbox(
            self.control_panel,
            from_=1, to=120,
            textvariable=self.desired_fps,
            width=5,
            command=lambda: None  # no‐op, we just read .get() later
        )
        self.fps_spin.pack(anchor='w', pady=(0,20))

        def _sync_counts(var_name, index, mode):
            """Mirror participants⇆cameras in holistic mode."""
            if not self.multi_face_mode.get():
                # participant changed → update cameras
                if var_name == self.participant_count._name:
                    new_count = self.participant_count.get()
                    self.camera_count.set(new_count)
                    # Update spinbox maximum
                    self.cam_spin.config(to=new_count)
                # camera changed → update participants
                elif var_name == self.camera_count._name:
                    new_count = self.camera_count.get()
                    self.participant_count.set(new_count)

                # attach trace callbacks (write-only)
        self._sync_counts = _sync_counts
        self.participant_count.trace_add('write', self._sync_counts)
        self.camera_count.trace_add('write',      self._sync_counts)

        # ─── Resolution selector ───────────────────────────────────────
        ttk.Label(self.control_panel, text="Resolution:").pack(anchor='w')
        # map label→(w,h)
        self.res_map = {
            "1080p": (1920, 1080),
            "720p":  (1280, 720),
            "480p":  ( 640, 480),
        }
        self.res_choice = tk.StringVar(value="720p")
        self.res_menu = ttk.Combobox(
            self.control_panel,
            textvariable=self.res_choice,
            values=list(self.res_map.keys()),
            state="readonly",
            width=7
        )
        self.res_menu.pack(anchor='w', pady=(0, 20))

        #----- Loading default values -----
        # UI controls with config defaults
        self.participant_count = tk.IntVar(value=self.config.get('startup_mode.participant_count', 2))
        self.camera_count = tk.IntVar(value=self.config.get('startup_mode.camera_count', 2))
        self.enable_mesh = tk.BooleanVar(value=self.config.get('startup_mode.enable_mesh', False))
        self.multi_face_mode = tk.BooleanVar(value=self.config.get('startup_mode.multi_face', False))
        # FPS with config default
        self.desired_fps = tk.IntVar(value=self.config.get('camera_settings.target_fps', 30))
        # Resolution with config default
        default_res = self.config.get('camera_settings.resolution', '720p')
        self.res_choice = tk.StringVar(value=default_res)
        
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
        self.after(GUI_scheduler_time, self.schedule_preview)
        self.after(GUI_scheduler_time, self.continuous_correlation_monitor)
        self.after(1000, self.update_participant_names)

        # Toggles
        ttk.Checkbutton(
            self.control_panel,
            text="Send Complete FaceMesh Data to LSL",
            variable=self.enable_mesh,
            command=self.on_mesh_toggle
        ).pack(anchor='w')

        ttk.Checkbutton(self.control_panel, text="Enable Multi-Face Mode (No Pose Estimation)",
                        variable=self.multi_face_mode, command=self.on_mode_toggle).pack(anchor='w', pady=(0,50))
        
        # ─── Video Recording Section ───────────────────────────────────
        record_frame = ttk.LabelFrame(self.control_panel, text="Video Recording")
        record_frame.pack(fill='x', pady=(20, 0))
        # Record toggle
        self.record_video = tk.BooleanVar(value=self.config.get('video_recording.enabled', False))
        ttk.Checkbutton(
            record_frame,
            text="Enable Video Recording",
            variable=self.record_video,
            command=self.on_record_toggle
        ).pack(anchor='w', padx=5, pady=(5, 0))
        # Save directory
        dir_frame = ttk.Frame(record_frame)
        dir_frame.pack(fill='x', padx=5, pady=5)
        ttk.Label(dir_frame, text="Save to:").pack(side='left')
        self.save_dir = tk.StringVar(value=self.config.get('video_recording.save_directory', './recordings'))
        self.dir_entry = ttk.Entry(dir_frame, textvariable=self.save_dir, width=25)
        self.dir_entry.pack(side='left', padx=(5, 0))
        ttk.Button(
            dir_frame,
            text="Browse",
            command=self.browse_save_directory,
            width=8
        ).pack(side='left', padx=(5, 0))
        # Filename template
        name_frame = ttk.Frame(record_frame)
        name_frame.pack(fill='x', padx=5, pady=(0, 5))
        ttk.Label(name_frame, text="Filename:").pack(side='left')
        self.filename_template = tk.StringVar(
            value=self.config.get('video_recording.filename_template', '{participant}_{timestamp}')
        )
        self.name_entry = ttk.Entry(name_frame, textvariable=self.filename_template, width=25)
        self.name_entry.pack(side='left', padx=(5, 0))
        # Help text
        help_text = ttk.Label(
            record_frame, 
            text="Available: {participant}, {camera}, {timestamp}",
            font=('Arial', 8),
            foreground='gray'
        )
        help_text.pack(anchor='w', padx=5)
        # Recording buttons
        button_frame = ttk.Frame(record_frame)
        button_frame.pack(fill='x', padx=5, pady=5)
        self.record_now_btn = ttk.Button(
            button_frame,
            text="Start Video Recording",
            command=self.toggle_immediate_recording,
            state='disabled'
        )
        self.record_now_btn.pack(side='left', padx=(0, 5))
        self.immediate_recording = False


        self.stop_record_btn = ttk.Button(
            button_frame,
            text="Stop Video Recording",
            command=self.stop_immediate_recording,
            state='disabled'
        )
        self.stop_record_btn.pack(side='left', padx=(0, 5))
        # Recording status
        self.record_status = ttk.Label(record_frame, text="Not recording", foreground='gray')
        self.record_status.pack(anchor='w', padx=5, pady=(5, 5))
        # Initialize recording
        self.video_recorders = {}
        self.recording_active = False
        # Initially hide/show based on toggle
        self.on_record_toggle()


        # ─── Audio Recording Section ───────────────────────────────────
        audio_frame = ttk.LabelFrame(self.control_panel, text="Audio Recording")
        audio_frame.pack(fill='x', pady=(10, 0))
        # Audio recording options
        self.audio_enabled = tk.BooleanVar(value=self.config.get('audio_recording.enabled', False))
        ttk.Checkbutton(
            audio_frame,
            text="Enable Audio Recording",
            variable=self.audio_enabled,
            command=self.on_audio_toggle
        ).pack(anchor='w', padx=5, pady=(5, 0))
        # Audio mode selection
        self.audio_mode_frame = ttk.Frame(audio_frame)
        self.audio_mode_frame.pack(fill='x', padx=20, pady=5)
        self.audio_mode = tk.StringVar(value="standalone")
        ttk.Radiobutton(
            self.audio_mode_frame,
            text="Standalone Audio",
            variable=self.audio_mode,
            value="standalone",
            command=self.on_audio_mode_change
        ).pack(anchor='w')
        ttk.Radiobutton(
            self.audio_mode_frame,
            text="Audio with Video",
            variable=self.audio_mode,
            value="with_video",
            command=self.on_audio_mode_change
        ).pack(anchor='w')

        # Audio control buttons
        audio_button_frame = ttk.Frame(audio_frame)
        audio_button_frame.pack(fill='x', padx=5, pady=5)
        self.start_audio_btn = ttk.Button(
            audio_button_frame,
            text="Start Audio Recording",
            command=self.start_audio_recording,
            state='disabled'
        )
        self.start_audio_btn.pack(side='left', padx=(0, 5))

        self.stop_audio_btn = ttk.Button(
            audio_button_frame,
            text="Stop Audio Recording",
            command=self.stop_audio_recording,
            state='disabled'
        )
        self.stop_audio_btn.pack(side='left')

        # Audio device assignment button
        self.audio_device_btn = ttk.Button(
            audio_frame,
            text="Configure Audio Devices",
            command=self.configure_audio_devices
        )
        self.audio_device_btn.pack(fill='x', padx=5, pady=5)
        # Audio status
        self.audio_status = ttk.Label(audio_frame, text="Audio: Not configured", foreground='gray')
        self.audio_status.pack(anchor='w', padx=5)
        # Initialize audio
        self.audio_recorders = {}
        self.audio_device_assignments = self.config.get('audio_devices', {})
        self.available_audio_devices = []
        self.audio_recording_active = False
        self.refresh_audio_devices()
        # Initially show/hide based on state
        self.on_audio_toggle()

        # Action Buttons
        ttk.Button(self.control_panel, text="Start Data Stream", command=self.start_stream).pack(fill='x', pady=(2))
        ttk.Button(self.control_panel, text="Stop Data Stream", command=self.stop_stream).pack(fill='x', pady=(2))
        ttk.Button(self.control_panel, text="Reset", command=self.reset).pack(fill='x', pady=(2))
        ttk.Button(self.control_panel, text="Save Current Settings", command=self.save_current_settings).pack(fill='x', pady=(10,2))

        # Apply startup mode
        if self.config.get('startup_mode.multi_face', False):
            self.on_mode_toggle()  # This will trigger multiface setup
        
        # Save window geometry on close
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def on_closing(self):
        """Save configuration before closing"""
        # Save current settings
        self.config.set('camera_settings.target_fps', self.desired_fps.get())
        self.config.set('camera_settings.resolution', self.res_choice.get())
        self.config.set('startup_mode.multi_face', self.multi_face_mode.get())
        self.config.set('startup_mode.participant_count', self.participant_count.get())
        self.config.set('startup_mode.camera_count', self.camera_count.get())
        self.config.set('startup_mode.enable_mesh', self.enable_mesh.get())
        
        self.destroy()
    
    def on_mesh_toggle(self):
        """Notify each worker to turn raw‐mesh pushing on/off."""
        val = self.enable_mesh.get()
        for info in self.frames:
            conn = info.get('control_conn')
            if conn:
                conn.send(('set_mesh', val))
                print(f"[GUI] Sent mesh toggle → {val} to worker")

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
        
    def on_record_toggle(self):
        """Handle recording toggle state change"""
        enabled = self.record_video.get()
        state = 'normal' if enabled else 'disabled'
        
        # Enable/disable recording controls
        self.dir_entry.config(state=state)
        self.name_entry.config(state=state)
        
        # Only enable record now button if we have active cameras and not streaming
        if enabled and not self.streaming:
            active_workers = [info for info in self.frames if info.get('proc') and info['proc'].is_alive()]
            if active_workers:
                self.record_now_btn.config(state='normal')
            else:
                self.record_now_btn.config(state='disabled')
        else:
            self.record_now_btn.config(state='disabled')
        
        # Save preference
        self.config.set('video_recording.enabled', enabled)
        
        # Update browse button state
        for widget in self.dir_entry.master.winfo_children():
            if isinstance(widget, ttk.Button) and widget.cget('text') == 'Browse':
                widget.config(state=state)

    def toggle_immediate_recording(self):
        """Start recording immediately without LSL stream"""
        # Check if we have active workers
        active_workers = [info for info in self.frames if info.get('proc') and info['proc'].is_alive()]
        if not active_workers:
            messagebox.showwarning("No Active Cameras", "Please select cameras first")
            return
        
        # Check if canvases have content
        has_content = False
        for info in active_workers:
            canvas = info['canvas']
            if canvas.find_all():  # Check if canvas has any items
                has_content = True
                break
        
        if not has_content:
            messagebox.showwarning("No Video Content", "Please wait for video feed to start before recording")
            return
        
        self.record_now_btn.config(state='disabled')
        self.stop_record_btn.config(state='normal')
        self._start_video_recording(active_workers)

    def stop_immediate_recording(self):
        """Stop immediate recording"""
        self.record_now_btn.config(state='normal' if self.record_video.get() else 'disabled')
        self.stop_record_btn.config(state='disabled')
        self._stop_video_recording()

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


        # 3) destroy all canvas frames and cache objects
        for w in self.container.winfo_children():
            w.destroy()
        self.canvas_objects.clear()
        self.transform_cache.clear()
        self.last_frame_time.clear()

        if hasattr(self, '_photo_images'):
            self._photo_images.clear()
    
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
            self.multi_panel.grid(row=2, column=1, columnspan=3, padx=10, pady=10, sticky='ew')
            
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

            fps = self.auto_detect_optimal_fps(cam_idx)
            res_label = self.res_choice.get()            # e.g. "720p"
            resolution = self.res_map.get(res_label, CAM_RESOLUTION)

            # Update camera resolution for this specific camera
            info['resolution'] = resolution
            info['fps'] = fps

            

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
                args=(cam_idx, self.MODEL_PATH, participant_id, stream_name, fps,
                    enable_mesh, multi_face, pv_q, score_buffer_name, child_conn, resolution),  # Changed score_buffer to score_buffer_name
                daemon=True
            )
            proc.start()
            info['proc'] = proc
            info['meta_label'].config(text=f"Camera → {cam_idx} ({resolution[0]}x{resolution[1]}@{fps}fps)")
            
            # Update record now button state when a camera is selected
            if self.record_video.get() and not self.streaming:
                self.record_now_btn.config(state='normal')

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
            canvas = tk.Canvas(frame, bg='black',highlightthickness=0, borderwidth=0)
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

    def auto_detect_optimal_fps(self, cam_idx):
        """Auto-detect and set optimal FPS for camera"""
        # Quick test capture

        res_label = self.res_choice.get()               # e.g. "720p"
        width, height = self.res_map.get(res_label, CAM_RESOLUTION)

        cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        # Test actual frame rate
        test_duration = 5.0  # seconds
        start_time = time.time()
        frame_count = 0
        last_frame = None
        unique_count = 0
        
        print(f"[GUI] Testing camera {cam_idx} for optimal FPS ETA {test_duration} seconds: ...")
        while time.time() - start_time < test_duration:
            ret, frame = cap.read()
            if ret:
                frame_count += 1
                # Check if unique
                if last_frame is None or not np.array_equal(frame[::10, ::10], last_frame):
                    unique_count += 1
                    last_frame = frame[::10, ::10].copy()
        
        cap.release()
        
        actual_fps = unique_count / test_duration
        print(f"[GUI] Camera {cam_idx} actual FPS: {actual_fps:.1f}")
        
        # Set FPS to slightly below actual to avoid duplicates
        optimal_fps = int(actual_fps * 0.98)  # 98% of actual
        self.desired_fps.set(optimal_fps)
        return optimal_fps
    
    def schedule_preview(self):
        """Optimized preview focusing on video smoothness"""
        # Run at 30ms intervals (33 FPS) for smoother perception
        self.after(GUI_scheduler_time, self.schedule_preview)
        
        # Process all canvases in rotation
        for idx, q in enumerate(self.preview_queues):
            if not q or idx >= len(self.frames):
                continue
            
            # Get latest frame without blocking
            latest_msg = None
            try:
                # Keep only the latest frame
                while not q.empty():
                    latest_msg = q.get_nowait()
            except:
                continue
                
            if latest_msg is None:
                continue

            canvas = self.frames[idx]['canvas']
            W, H = canvas.winfo_width(), canvas.winfo_height()
            
            if W <= 1 or H <= 1:
                continue

            frame_bgr = latest_msg.get('frame_bgr', None)
            if frame_bgr is None:
                continue

            try:
                # Render image with PhotoImage reuse
                img = self._render_image_fast(frame_bgr, W, H, idx)
                if img is not None:
                    # Update image without recreating canvas items
                    if hasattr(canvas, '_image_id'):
                        canvas.itemconfig(canvas._image_id, image=img)
                    else:
                        canvas._image_id = canvas.create_image(W // 2, H // 2, 
                                                            image=img, anchor='center')
                    canvas.image = img  # Keep reference

                    # Draw overlays
                    if latest_msg['mode'] == 'holistic':
                        face_lms = latest_msg.get('face', [])
                        pose_lms = latest_msg.get('pose', [])
                        self._draw_holistic_fast(canvas, face_lms, pose_lms, W, H, idx)
                    else:
                        faces = latest_msg.get('faces', [])
                        self._draw_multiface_fast(canvas, faces, W, H, idx)
            except Exception as e:
                print(f"[GUI] Error in preview for canvas {idx}: {e}")

    def _render_image_fast(self, frame_bgr, W, H, canvas_idx):
        """Optimized image rendering with dynamic quality adjustment"""
        frame_h, frame_w = frame_bgr.shape[:2]
        
        # Calculate scaling to fit within canvas while maintaining aspect ratio
        scale = min(W / frame_w, H / frame_h)
        scaled_w, scaled_h = int(frame_w * scale), int(frame_h * scale)
        
        # PERFORMANCE: Use lower quality interpolation for high resolution
        if frame_w > 1280 or frame_h > 720:
            interpolation = cv2.INTER_NEAREST  # Fastest for high res
        else:
            interpolation = cv2.INTER_LINEAR   # Better quality for lower res
        
        # Calculate offsets to center the image
        x_offset = (W - scaled_w) // 2
        y_offset = (H - scaled_h) // 2
        
        # Store these values for coordinate transformation
        if canvas_idx not in self.transform_cache:
            self.transform_cache[canvas_idx] = {}
        self.transform_cache[canvas_idx]['video_bounds'] = (x_offset, y_offset, scaled_w, scaled_h)
        
        # Resize frame to fit
        if abs(scale - 1.0) > 0.01:
            frame_bgr = cv2.resize(frame_bgr, (scaled_w, scaled_h), interpolation=interpolation)
        
        # PERFORMANCE: For 1080p, consider downsampling the canvas buffer
        if W > 960 or H > 540:  # If canvas is large
            # Create smaller buffer first, then scale up
            small_w, small_h = W // 2, H // 2
            small_canvas = np.zeros((small_h, small_w, 3), dtype=np.uint8)
            
            # Calculate positions in small canvas
            small_x_offset = x_offset // 2
            small_y_offset = y_offset // 2
            small_scaled_w = scaled_w // 2
            small_scaled_h = scaled_h // 2
            
            # Resize frame to smaller size
            small_frame = cv2.resize(frame_bgr, (small_scaled_w, small_scaled_h), 
                                    interpolation=cv2.INTER_NEAREST)
            
            # Place in small canvas
            small_canvas[small_y_offset:small_y_offset+small_scaled_h, 
                        small_x_offset:small_x_offset+small_scaled_w] = small_frame
            
            # Scale up to full size
            canvas_img = cv2.resize(small_canvas, (W, H), interpolation=cv2.INTER_NEAREST)
        else:
            # Normal processing for smaller canvases
            canvas_img = np.zeros((H, W, 3), dtype=np.uint8)
            canvas_img[y_offset:y_offset+scaled_h, x_offset:x_offset+scaled_w] = frame_bgr
        
        # Convert to RGB
        rgb = cv2.cvtColor(canvas_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        
        # Reuse PhotoImage if it exists
        if hasattr(self, '_photo_images') and canvas_idx in self._photo_images:
            self._photo_images[canvas_idx].paste(pil_img)
            return self._photo_images[canvas_idx]
        else:
            if not hasattr(self, '_photo_images'):
                self._photo_images = {}
            photo = ImageTk.PhotoImage(pil_img)
            self._photo_images[canvas_idx] = photo
            return photo

    
    def _draw_connections_fast(self, canvas, landmarks, connections, line_cache, color, width):
        """Draw connections reusing existing line objects when possible"""
        connection_list = list(connections)
        
        # Extend cache if we need more lines
        while len(line_cache) < len(connection_list):
            line_id = canvas.create_line(0, 0, 0, 0, fill=color, width=width)
            line_cache.append(line_id)
        
        # Update existing lines
        for i, (start_idx, end_idx) in enumerate(connection_list):
            if i >= len(line_cache):
                break
                
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                x1, y1, _ = landmarks[start_idx]
                x2, y2, _ = landmarks[end_idx]
                
                # Update existing line coordinates
                canvas.coords(line_cache[i], int(x1), int(y1), int(x2), int(y2))
                canvas.itemconfig(line_cache[i], state='normal')
            else:
                # Hide unused line
                canvas.itemconfig(line_cache[i], state='hidden')
        
        # Hide excess lines if we have more cached than needed
        for i in range(len(connection_list), len(line_cache)):
            canvas.itemconfig(line_cache[i], state='hidden')

   
    def _draw_holistic_fast(self, canvas, face_lms, pose_lms, W, H, canvas_idx):
        # Initialize canvas objects cache for this canvas if not exists
        if canvas_idx not in self.canvas_objects:
            self.canvas_objects[canvas_idx] = {
                'face_lines': [],
                'pose_lines': [],
                'face_points': [],
                'pose_points': []
            }
        
        cache = self.canvas_objects[canvas_idx]
        
        # Get video bounds for proper coordinate transformation
        video_bounds = self.transform_cache.get(canvas_idx, {}).get('video_bounds', (0, 0, W, H))
        x_offset, y_offset, video_w, video_h = video_bounds
    
        def transform_coords_to_video_area(lms):
            """Transform normalized coords to actual video area within canvas"""
            if not lms:
                return []
            # Scale to video dimensions and offset to video position
            return [(x * video_w + x_offset, y * video_h + y_offset, z) for x, y, z in lms]
        
        # Transform coordinates
        face_coords = transform_coords_to_video_area(face_lms) if face_lms else []
        pose_coords = transform_coords_to_video_area(pose_lms) if pose_lms else []
        
        # Draw or hide face mesh
        if face_coords and len(face_coords) > 0:
            if self.enable_mesh.get():
                # Full tessellation with semi-transparency
                connections = mp.solutions.face_mesh.FACEMESH_TESSELATION
                self._draw_connections_fast(canvas, face_coords, connections, 
                                        cache['face_lines'], '#40FF40', 1)
            else:
                # Just contours for performance
                connections = mp.solutions.face_mesh.FACEMESH_CONTOURS
                self._draw_connections_fast(canvas, face_coords, connections, 
                                        cache['face_lines'], '#00FF00', 1)
        else:
            # No face landmarks - hide all face lines
            for line_id in cache['face_lines']:
                canvas.itemconfig(line_id, state='hidden')
        
        # Draw or hide pose
        if pose_coords and len(pose_coords) > 0:
            connections = mp.solutions.holistic.POSE_CONNECTIONS
            self._draw_connections_fast(canvas, pose_coords, connections, 
                                    cache['pose_lines'], '#40FFFF', 2)
        else:
            # No pose landmarks - hide all pose lines
            for line_id in cache['pose_lines']:
                canvas.itemconfig(line_id, state='hidden')


    def _draw_multiface_fast(self, canvas, faces, W, H, canvas_idx):
        """Fixed multiface drawing with names and tessellation"""
        if canvas_idx not in self.canvas_objects:
            self.canvas_objects[canvas_idx] = {
                'face_lines': {},
                'face_labels': {}
            }
        
        cache = self.canvas_objects[canvas_idx]
        
        # Get video bounds
        video_bounds = self.transform_cache.get(canvas_idx, {}).get('video_bounds', (0, 0, W, H))
        x_offset, y_offset, video_w, video_h = video_bounds
        
        active_faces = set()
        
        for face in faces:
            fid = face['id']
            active_faces.add(fid)
            lm_xyz = face['landmarks']
            cx_norm, cy_norm = face['centroid']
            
            if fid not in cache['face_lines']:
                cache['face_lines'][fid] = []
            if fid not in cache['face_labels']:
                cache['face_labels'][fid] = None
            
            # Transform landmarks to video area coordinates
            canvas_landmarks = [(x * video_w + x_offset, y * video_h + y_offset, z) 
                            for x, y, z in lm_xyz]
            
            # Use TESSELLATION for more detailed mesh with transparency
            connections = mp.solutions.face_mesh.FACEMESH_TESSELATION
            self._draw_connections_fast(canvas, canvas_landmarks, connections, 
                                    cache['face_lines'][fid], '#40FF40', 1)
            
            # Transform label position properly and place above face
            label_x = int(cx_norm * video_w + x_offset)
            label_y = int(cy_norm * video_h + y_offset) - 50
            label_text = self.participant_names.get(fid, f"Face {fid}")
            
            if cache['face_labels'][fid] is None:
                cache['face_labels'][fid] = canvas.create_text(
                    label_x, label_y, text=label_text,
                    fill='yellow', font=('Arial', 14, 'bold'),
                    anchor='center'
                )
            else:
                canvas.coords(cache['face_labels'][fid], label_x, label_y)
                canvas.itemconfig(cache['face_labels'][fid], text=label_text, state='normal')  # Always show
        
        # Hide only truly inactive faces
        for fid in list(cache['face_lines'].keys()):
            if fid not in active_faces:
                for line_id in cache['face_lines'][fid]:
                    canvas.itemconfig(line_id, state='hidden')
                if cache['face_labels'].get(fid):
                    canvas.itemconfig(cache['face_labels'][fid], state='hidden')

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
        """Fixed plot update with proper label alignment"""
        c = self.bar_canvas
        W, H = c.winfo_width(), c.winfo_height()
        
        if W <= 1 or H <= 1:
            return
            
        # Throttling
        now = time.time()
        if hasattr(self, '_last_plot') and now - self._last_plot < 0.016:
            return
        self._last_plot = now
        
        # Initialize blend labels if needed
        if getattr(self, 'blend_labels', None) is None:
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
        
        n = len(corr)
        if n == 0:
            c.delete('all')
            c.create_text(W//2, H//2, text="No correlation data", fill='gray', font=('Arial', 12))
            return
        
        # FIX: Calculate proper spacing with margins
        top_margin = 10
        bottom_margin = 10
        available_height = H - top_margin - bottom_margin
        bar_h = available_height / n  # This ensures all bars fit
        
        # Adjust font size based on bar height
        font_size = max(6, min(10, int(bar_h * 0.7)))
        font = ('Arial', font_size)
        label_w = 120
        margin = 5
        
        # Initialize or update canvas items
        if not hasattr(self, 'bar_items') or len(self.bar_items) != n:
            c.delete('all')
            self.bar_items = []
            self.bar_labels = []
            self.value_texts = []
            
            # Create center line
            center_x = label_w + margin
            c.create_line(center_x, top_margin, center_x, H - bottom_margin, fill='lightgray', width=1)
            
            for i in range(n):
                # FIX: Calculate y position with proper margins
                y0 = top_margin + (i * bar_h)
                y1 = y0 + bar_h - 1
                cy = int(y0 + bar_h / 2)
                
                # Create bar rectangle
                bar_id = c.create_rectangle(0, y0, 0, y1, fill='', outline='')
                self.bar_items.append(bar_id)
                
                # Create label with proper positioning
                lbl = self.blend_labels[i] if i < len(self.blend_labels) else f"BS_{i}"
                label_id = c.create_text(label_w - margin, cy, text=lbl, anchor='e', font=font, fill='black')
                self.bar_labels.append(label_id)
                
                # Create value text
                val_id = c.create_text(0, cy, text='', anchor='center', 
                                    font=('Arial', max(6, font_size-1)), fill='white', state='hidden')
                self.value_texts.append(val_id)
        
        # Update existing items
        bar_width = W - label_w - 2 * margin
        center_x = label_w + margin
        
        for i, val in enumerate(corr):
            if i >= len(self.bar_items):
                break
                
            bar_id = self.bar_items[i]
            val_id = self.value_texts[i]
            
            # FIX: Recalculate positions in case of resize
            y0 = top_margin + (i * bar_h)
            y1 = y0 + bar_h - 1
            cy = int(y0 + bar_h / 2)
            
            # Update label position in case of resize
            c.coords(self.bar_labels[i], label_w - margin, cy)
            
            val_clamped = max(-1.0, min(1.0, val))
            
            if abs(val_clamped) > 0.01:
                # Calculate bar position and color
                if val_clamped >= 0:
                    length = val_clamped * bar_width * 0.5
                    x0 = center_x
                    x1 = x0 + length
                    green = int(255 * val_clamped)
                    color = f'#{0:02x}{green:02x}{0:02x}'
                else:
                    length = abs(val_clamped) * bar_width * 0.5
                    x1 = center_x
                    x0 = x1 - length
                    red = int(255 * abs(val_clamped))
                    color = f'#{red:02x}{0:02x}{0:02x}'
                
                # Update bar
                c.coords(bar_id, min(x0, x1), y0, max(x0, x1), y1)
                c.itemconfig(bar_id, fill=color)
                
                # Update value text
                if abs(val_clamped) > 0.05:
                    val_text = f"{val_clamped:.2f}"
                    text_x = center_x + (length / 2 if val_clamped >= 0 else -length / 2)
                    c.coords(val_id, text_x, cy)
                    c.itemconfig(val_id, text=val_text, state='normal')
                else:
                    c.itemconfig(val_id, state='hidden')
            else:
                c.itemconfig(bar_id, fill='')
                c.itemconfig(val_id, state='hidden')


    def _gui_update_loop(self):
        """Separate thread just for updating the GUI plot"""
        print("[GUI] GUI update loop started")
        while self.streaming and not self.stop_evt.is_set():
            if hasattr(self, 'score_reader') and self.score_reader:
                corr = self.score_reader.get_latest_correlation()
                if corr is not None:
                    self.latest_correlation = corr
            time.sleep(GUI_sleep_time)  # 30 FPS for GUI updates
        print("[GUI] GUI update loop stopped")

    def on_record_toggle(self):
        """Handle recording toggle state change"""
        enabled = self.record_video.get()
        state = 'normal' if enabled else 'disabled'
        
        # Enable/disable recording controls
        self.dir_entry.config(state=state)
        self.name_entry.config(state=state)
        
        # Save preference
        self.config.set('video_recording.enabled', enabled)
        
        # Update browse button state
        for widget in self.dir_entry.master.winfo_children():
            if isinstance(widget, ttk.Button):
                widget.config(state=state)
    
    def browse_save_directory(self):
        """Browse for save directory"""
        directory = filedialog.askdirectory(
            initialdir=self.save_dir.get(),
            title="Select Recording Directory"
        )
        if directory:
            self.save_dir.set(directory)
            self.config.set('video_recording.save_directory', directory)

    def _start_video_recording(self, active_workers):
        """Start video recording with optional audio for all active workers"""
        # Set immediate recording flag if not already recording
        if not self.recording_active:
            self.immediate_recording = True
            
        if not self.record_video.get() and not self.immediate_recording:
            return
        self.recording_active = True
            
        # Create save directory if it doesn't exist
        save_path = Path(self.save_dir.get())
        try:
            save_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            messagebox.showerror("Recording Error", f"Cannot create directory: {e}")
            return
        
        # Check if we should record audio with video
        record_audio_with_video = (
            self.audio_enabled.get() and 
            self.audio_mode.get() == "with_video"
        )
        
        # Save current settings
        self.config.set('video_recording.save_directory', str(save_path))
        self.config.set('video_recording.filename_template', self.filename_template.get())
        
        # Create recorder for each active worker
        self.video_recorders.clear()
        recording_count = 0

        for idx, info in enumerate(self.frames):
            # Skip if this frame doesn't have an active worker
            if not (info.get('proc') and info['proc'].is_alive()):
                continue
                
            # Get frame dimensions from stored resolution
            resolution = info.get('resolution', (640, 480))
            actual_width, actual_height = resolution
            
            # Check if preview queue exists for this camera
            if idx >= len(self.preview_queues) or self.preview_queues[idx] is None:
                print(f"[GUI] No preview queue available for camera {idx}, skipping recording")
                continue
            
            try:
                recorder = self._create_video_recorder(idx, record_audio_with_video)
                if recorder is None:
                    print(f"[GUI] Failed to create recorder for cam {idx}")
                    continue
                    
                # Store recorder info
                self.video_recorders[idx] = {
                    'recorder': recorder,
                    'capture_thread': None,
                    'stop_flag': None
                }
                
                # Define the recording thread function with proper scope
                def create_recording_thread(recorder, preview_queue, width, height, idx, participant_names, multi_face_mode):
                    stop_flag = threading.Event()
                    
                    def record_overlay_thread():
                        """Fixed overlay recording thread with proper frame timing"""
                        frame_count = 0
                        last_frame_time = time.time()
                        frame_interval = 1.0 / CAPTURE_FPS  # Target interval between frames
                        
                        while self.recording_active and not stop_flag.is_set():
                            try:
                                # Calculate when next frame should be captured
                                current_time = time.time()
                                time_since_last = current_time - last_frame_time
                                
                                # Skip if we're ahead of schedule
                                if time_since_last < frame_interval:
                                    time.sleep(frame_interval - time_since_last)
                                    continue
                                
                                # Get latest frame (drain queue)
                                latest = None
                                attempts = 0
                                try:
                                    while not preview_queue.empty() and attempts < 10:
                                        latest = preview_queue.get_nowait()
                                        attempts += 1
                                except Exception as e:
                                    print(f"[Recorder Thread {idx}] Error accessing queue: {e}")
                                    continue
                                
                                if latest is None:
                                    continue
                                    
                                frame_bgr = latest.get('frame_bgr')
                                if frame_bgr is None:
                                    continue
                                
                                # Make a copy to avoid modifying the preview
                                frame_bgr = frame_bgr.copy()
                                
                                if multi_face_mode:
                                    # Multiface mode - faces are already in correct format
                                    faces = latest.get('faces', [])
                                    overlayed = draw_overlays_combined(
                                        frame_bgr,
                                        faces=faces,
                                        labels=participant_names,
                                        face_mesh=True,
                                        face_contours=True,
                                        face_points=False,
                                        pose_lines=False
                                    )
                                else:
                                    # Holistic mode
                                    face_lms = latest.get('face', [])
                                    pose_lms = latest.get('pose', [])
                                    
                                    # Prepare faces list for draw_overlays_combined
                                    faces = []
                                    if face_lms and len(face_lms) > 0:
                                        # face_lms is already a list of (x, y, z) tuples
                                        # Calculate centroid from normalized coordinates
                                        x_coords = [lm[0] for lm in face_lms if len(lm) >= 2]
                                        y_coords = [lm[1] for lm in face_lms if len(lm) >= 2]
                                        
                                        if x_coords and y_coords:
                                            face_dict = {
                                                'landmarks': face_lms,  # Already in (x, y, z) format
                                                'id': 0,
                                                'centroid': (
                                                    np.mean(x_coords),
                                                    np.mean(y_coords)
                                                )
                                            }
                                            faces = [face_dict]
                                    
                                    # Draw overlays with proper face and pose data
                                    participant_name = participant_names.get(idx, f"P{idx+1}")
                                    overlayed = draw_overlays_combined(
                                        frame_bgr,
                                        faces=faces,
                                        pose_landmarks=pose_lms if pose_lms else None,
                                        labels={0: participant_name} if faces else None,
                                        face_mesh=False,
                                        face_contours=True,
                                        face_points=False,
                                        pose_lines=True
                                    )
                                
                                # Get actual frame dimensions
                                actual_h, actual_w = overlayed.shape[:2]
                                
                                # Only resize if dimensions don't match recorder expectations
                                if (actual_w, actual_h) != (width, height):
                                    print(f"[Recorder Thread] Resizing from {actual_w}x{actual_h} to {width}x{height}")
                                    overlayed = cv2.resize(overlayed, (width, height), interpolation=cv2.INTER_LINEAR)
                                
                                # Add frame to recorder
                                if recorder.add_frame(overlayed):
                                    frame_count += 1
                                    last_frame_time = current_time  # Update last frame time
                                    
                                    if frame_count % CAPTURE_FPS == 0:  # Log every second
                                        actual_fps = CAPTURE_FPS / (current_time - last_frame_time + frame_interval * CAPTURE_FPS)
                                        print(f"[Recorder Thread {idx}] {frame_count} frames, actual FPS: {actual_fps:.1f}")
                                
                            except Exception as e:
                                print(f"[Recorder Thread {idx}] Error: {e}")
                                import traceback
                                traceback.print_exc()
                        
                        print(f"[Recorder Thread {idx}] Stopped after {frame_count} frames")
                    
                    # Start the thread
                    thread = threading.Thread(target=record_overlay_thread, daemon=True)
                    thread.start()
                    return thread, stop_flag

                # Now use this function to start the recording thread:
                print(f"[GUI] Starting overlay recording thread for cam {idx}")
                preview_queue = self.preview_queues[idx]
                
                capture_thread, stop_flag = create_recording_thread(
                    recorder, 
                    preview_queue,
                    actual_width,
                    actual_height,
                    idx,
                    self.participant_names,
                    self.multi_face_mode.get()
                )

                self.video_recorders[idx]['capture_thread'] = capture_thread
                self.video_recorders[idx]['stop_flag'] = stop_flag
                print(f"[GUI] Overlay capture thread started for cam {idx}")
                recording_count += 1
                
            except Exception as e:
                print(f"[GUI] Failed to start recorder for cam {idx}: {e}")
                import traceback
                traceback.print_exc()

        # Start standalone audio recorders if enabled
        if self.audio_enabled.get() and self.audio_mode.get() == "standalone":
            self._start_standalone_audio_recording()
        
        if recording_count > 0:
            self.record_status.config(
                text=f"Recording {recording_count} stream(s) to {save_path.name}/",
                foreground='red'
            )
            print(f"[GUI] Started recording {recording_count} video stream(s)")
        else:
            self.record_status.config(text="Recording failed", foreground='red')

    def _create_video_recorder(self, idx, record_audio):
        """Create a video recorder process - dimensions auto-detected"""
        try:
            # Generate filename based on template
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            participant_name = self.participant_names.get(idx, f"P{idx+1}")
            
            filename = self.filename_template.get().format(
                participant=participant_name,
                camera=f"cam{idx}",
                timestamp=timestamp
            )
            
            # Ensure .avi extension
            if not filename.endswith('.avi'):
                filename += '.avi'
            
            save_path = Path(self.save_dir.get())
            
            # Create the recorder process
            recorder = VideoRecorderProcess(
                output_dir=str(save_path),
                codec='MJPG',
                fps=CAPTURE_FPS
            )
            
            # Start recording - dimensions will be auto-detected
            if recorder.start_recording(participant_name, filename):
                print(f"[GUI] Created recorder for {participant_name}: {filename}")
                recorder.participant = participant_name
                return recorder
            else:
                print(f"[GUI] Failed to start recorder for {participant_name}")
                return None
                
        except Exception as e:
            print(f"[GUI] Error creating recorder for camera {idx}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _start_overlayed_recording_thread(self, idx, recorder, preview_queue, mode, participant_names, stop_flag):
        """
        idx: camera index
        recorder: VideoRecorderProcess instance
        preview_queue: queue.Queue, delivers dicts with 'frame_bgr' and overlay data
        mode: "holistic" or "multiface"
        participant_names: {id: name} or similar
        stop_flag: threading.Event or other shared flag
        """
        def record_loop():
            while self.recording_active and not stop_flag.is_set():
                try:
                    # Get latest preview (don't block)
                    latest = None
                    while not preview_queue.empty():
                        latest = preview_queue.get_nowait()
                    if latest is None:
                        time.sleep(1 / CAPTURE_FPS)
                        continue

                    frame_bgr = latest['frame_bgr']
                    # Holistic mode (single face, pose)
                    if mode == 'holistic':
                        face_lms = latest.get('face', [])
                        pose_lms = latest.get('pose', [])
                        faces = []
                        if face_lms:
                            # lm.x, lm.y, lm.z for each landmark
                            landmarks = [(lm.x, lm.y, lm.z) for lm in face_lms]
                            face_dict = {
                                'landmarks': landmarks,
                                'id': 0,
                                'centroid': (
                                    np.mean([lm.x for lm in face_lms]),
                                    np.mean([lm.y for lm in face_lms])
                                )
                            }
                            faces = [face_dict]
                        overlayed = draw_overlays_combined(
                            frame_bgr,
                            faces=faces,
                            pose_landmarks=pose_lms,
                            labels={0: participant_names.get(0, "P1")},
                        )
                    else:  # multiface
                        faces = latest.get('faces', [])
                        overlayed = draw_overlays_combined(
                            frame_bgr,
                            faces=faces,
                            labels=participant_names,
                        )
                        
                    recorder.add_frame(overlayed)
                except Exception as e:
                    print(f"[Recorder Thread] Error: {e}")
                time.sleep(1 / CAPTURE_FPS)
        # Start and return the thread object
        t = threading.Thread(target=record_loop, daemon=True)
        t.start()
        return t

    
    def _stop_video_recording(self):
        """Stop all process recorders + capture threads."""
        if not self.recording_active:
            return
            
        print("[GUI] Stopping video recording...")
        self.recording_active = False  # Signal capture threads to exit
        
        # First, wait for capture threads to finish
        for idx, rec_info in list(self.video_recorders.items()):
            if 'capture_thread' in rec_info and rec_info['capture_thread']:
                print(f"[GUI] Waiting for capture thread {idx}...")
                rec_info['capture_thread'].join(timeout=2.0)
            if 'stop_flag' in rec_info and rec_info['stop_flag']:
                rec_info['stop_flag'].set()
        
        # Give recorders time to process remaining frames
        time.sleep(0.5)
        
        # Then stop recorders
        for idx, rec_info in list(self.video_recorders.items()):
            try:
                recorder = rec_info['recorder']
                if hasattr(recorder, 'get_frame_count'):
                    frame_count = recorder.get_frame_count()
                    print(f"[GUI] Stopping recorder {idx} with {frame_count} frames...")
                else:
                    print(f"[GUI] Stopping recorder {idx}...")
                
                recorder.stop_recording()
                
                # Verify recording was saved
                save_path = Path(self.save_dir.get())
                participant = getattr(recorder, 'participant', f'P{idx+1}')
                recordings = list(save_path.glob(f"{participant}_*.avi"))
                if recordings:
                    latest = max(recordings, key=lambda p: p.stat().st_mtime)
                    size_mb = latest.stat().st_size / (1024 * 1024)
                    print(f"[GUI] Saved {latest.name} ({size_mb:.1f} MB)")
                    
            except Exception as e:
                print(f"[GUI] Error stopping recorder {idx}: {e}")
                import traceback
                traceback.print_exc()
        
        self.video_recorders.clear()
        self.immediate_recording = False  # Reset immediate recording flag
        self.record_status.config(text="Recording stopped", foreground='gray')


    def on_audio_toggle(self):
        """Handle audio recording toggle"""
        enabled = self.audio_enabled.get()
        state = 'normal' if enabled else 'disabled'
        
        # Enable/disable audio controls
        for widget in self.audio_mode_frame.winfo_children():
            widget.config(state=state)
        self.audio_device_btn.config(state=state)
        
        # Save preference
        self.config.set('audio_recording.enabled', enabled)
        
        # Update status
        if enabled:
            self.refresh_audio_status()
        else:
            self.audio_status.config(text="Audio: Disabled", foreground='gray')
    
    def on_audio_mode_change(self):
        """Handle audio mode change"""
        mode = self.audio_mode.get()
        self.config.set('audio_recording.standalone_audio', mode == "standalone")
        self.config.set('audio_recording.audio_with_video', mode == "with_video")
        
        # Update status
        self.refresh_audio_status()

    def start_audio_recording(self):
        """Start standalone audio recording"""
        if self.audio_mode.get() != "standalone":
            messagebox.showinfo("Audio Mode", "Audio recording is set to 'with video' mode. Use video recording controls.")
            return
            
        if not self.audio_device_assignments:
            messagebox.showwarning("No Devices", "Please configure audio devices first")
            return
            
        self._start_standalone_audio_recording()
        self.audio_recording_active = True
        self.start_audio_btn.config(state='disabled')
        self.stop_audio_btn.config(state='normal')
        self.audio_status.config(text=self.audio_status.cget("text") + " (recording)", foreground='red')

    def stop_audio_recording(self):
        """Stop standalone audio recording"""
        self.audio_recording_active = False
        
        # Stop all audio recorders
        for assignment, recorder in self.audio_recorders.items():
            recorder.stop_recording()
            print(f"[GUI] Stopped audio recording for {assignment}")
        
        self.audio_recorders.clear()
        self.start_audio_btn.config(state='normal' if self.audio_enabled.get() else 'disabled')
        self.stop_audio_btn.config(state='disabled')
        self.refresh_audio_status()

    def on_audio_toggle(self):
        """Handle audio recording toggle"""
        enabled = self.audio_enabled.get()
        state = 'normal' if enabled else 'disabled'
        
        # Enable/disable audio controls
        for widget in self.audio_mode_frame.winfo_children():
            widget.config(state=state)
        self.audio_device_btn.config(state=state)
        
        # Handle button states based on mode
        if enabled and self.audio_mode.get() == "standalone":
            self.start_audio_btn.config(state='normal' if not self.audio_recording_active else 'disabled')
            self.stop_audio_btn.config(state='normal' if self.audio_recording_active else 'disabled')
        else:
            self.start_audio_btn.config(state='disabled')
            self.stop_audio_btn.config(state='disabled')
        
        # Save preference
        self.config.set('audio_recording.enabled', enabled)
        
        # Update status
        if enabled:
            self.refresh_audio_status()
        else:
            self.audio_status.config(text="Audio: Disabled", foreground='gray')
    
    def refresh_audio_devices(self):
        """Refresh list of available audio devices"""
        try:
            self.available_audio_devices = AudioDeviceManager.list_audio_devices()
            print(f"[GUI] Found {len(self.available_audio_devices)} audio input devices")
        except Exception as e:
            print(f"[GUI] Error listing audio devices: {e}")
            self.available_audio_devices = []
    
    def refresh_audio_status(self):
        """Update audio status label"""
        if not self.audio_enabled.get():
            self.audio_status.config(text="Audio: Disabled", foreground='gray')
            return
            
        mode = self.audio_mode.get()
        assigned_count = len(self.audio_device_assignments)
        
        if assigned_count == 0:
            self.audio_status.config(text="Audio: No devices assigned", foreground='orange')
        else:
            mode_text = "standalone" if mode == "standalone" else "with video"
            self.audio_status.config(
                text=f"Audio: {assigned_count} device(s) assigned ({mode_text})",
                foreground='green'
            )
    
    def configure_audio_devices(self):
        """Open audio device configuration dialog"""
        dialog = tk.Toplevel(self)
        dialog.title("Configure Audio Devices")
        dialog.geometry("600x400")
        dialog.transient(self)
        dialog.grab_set()
        
        # Instructions
        ttk.Label(
            dialog,
            text="Assign audio devices to participants/cameras:",
            font=('Arial', 10, 'bold')
        ).pack(pady=10)
        
        # Create scrollable frame
        canvas = tk.Canvas(dialog)
        scrollbar = ttk.Scrollbar(dialog, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Device assignment widgets
        device_vars = {}
        
        # Get current counts
        if self.multi_face_mode.get():
            count = self.participant_count.get()
        else:
            count = self.camera_count.get()
        
        if self.multi_face_mode.get():
            # Multiface mode - assign devices to participants
            for i in range(count):
                frame = ttk.Frame(scrollable_frame)
                frame.pack(fill='x', padx=10, pady=5)
                
                participant_name = self.participant_names.get(i + 1, f"Face {i + 1}")
                ttk.Label(frame, text=f"{participant_name}:", width=20).pack(side='left')
                
                var = tk.StringVar(value=self.audio_device_assignments.get(f"face{i+1}", "None"))
                device_vars[f"face{i+1}"] = var
                
                devices = ["None"] + [f"{d['index']}: {d['name']}" for d in self.available_audio_devices]
                combo = ttk.Combobox(frame, textvariable=var, values=devices, state='readonly', width=40)
                combo.pack(side='left', padx=(10, 0))
        else:
            # Holistic mode - assign devices to cameras
            for i in range(count):
                frame = ttk.Frame(scrollable_frame)
                frame.pack(fill='x', padx=10, pady=5)
                
                participant_name = self.participant_names.get(i, f"P{i+1}")
                ttk.Label(frame, text=f"Camera {i} ({participant_name}):", width=20).pack(side='left')
                
                var = tk.StringVar(value=self.audio_device_assignments.get(f"cam{i}", "None"))
                device_vars[f"cam{i}"] = var
                
                devices = ["None"] + [f"{d['index']}: {d['name']}" for d in self.available_audio_devices]
                combo = ttk.Combobox(frame, textvariable=var, values=devices, state='readonly', width=40)
                combo.pack(side='left', padx=(10, 0))
                
        canvas.pack(side="left", fill="both", expand=True, padx=(10, 0))
        scrollbar.pack(side="right", fill="y")
        
        # Buttons
        button_frame = ttk.Frame(dialog)
        button_frame.pack(fill='x', pady=10)
        
        def save_assignments():
            # Save device assignments
            self.audio_device_assignments.clear()
            for key, var in device_vars.items():
                value = var.get()
                if value != "None":
                    # Extract device index from string
                    try:
                        device_index = int(value.split(":")[0])
                        self.audio_device_assignments[key] = device_index
                    except:
                        pass
            
            # Save to config
            self.config.set('audio_devices', self.audio_device_assignments)
            self.refresh_audio_status()
            dialog.destroy()
        
        ttk.Button(button_frame, text="Save", command=save_assignments).pack(side='right', padx=5)
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side='right', padx=5)
        
        # Refresh devices button
        ttk.Button(
            button_frame,
            text="Refresh Devices",
            command=lambda: [self.refresh_audio_devices(), dialog.destroy(), self.configure_audio_devices()]
        ).pack(side='left', padx=5)

    def _start_standalone_audio_recording(self):
        """Start standalone audio recording for assigned devices"""
        save_path = Path(self.save_dir.get())
        
        for assignment, device_index in self.audio_device_assignments.items():
            if device_index is None:
                continue
                
            # Determine participant ID
            if assignment.startswith("face"):
                face_num = int(assignment.replace("face", ""))
                participant_id = self.participant_names.get(face_num, f"Face{face_num}")
            else:
                cam_num = int(assignment.replace("cam", ""))
                participant_id = self.participant_names.get(cam_num, f"P{cam_num+1}")
            
            # Create audio recorder
            recorder = AudioRecorder(
                device_index=device_index,
                sample_rate=self.config.get('audio_recording.sample_rate', 44100),
                channels=self.config.get('audio_recording.channels', 1),
                output_dir=str(save_path)
            )
            
            # Start recording
            if recorder.start_recording(participant_id):
                self.audio_recorders[assignment] = recorder
                print(f"[GUI] Started audio recording for {participant_id}")

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

        # Start video/audio recording if enabled
        if self.record_video.get() or self.audio_enabled.get():
            self._start_video_recording(active_workers)


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
                if not hasattr(self, 'stop_evt') or self.stop_evt is None:
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

        #Stop video recording
        if self.recording_active:
            self._stop_video_recording()

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
        if self.stop_evt:
            self.stop_evt.set()
            self.stop_evt = None
        
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

        # 5) Clear the bar plot AND remove the debug rectangle
        self.bar_canvas.delete('all')
        # Don't create the debug rectangle - just show a message
        self.bar_canvas.create_text(
            self.bar_canvas.winfo_width()//2, 
            self.bar_canvas.winfo_height()//2, 
            text="Ready", fill='gray', font=('Arial', 12)
        )

        # 6) Reset streaming flag and cleanup FastScoreReader
        self.streaming = False
        if hasattr(self, 'score_reader') and self.score_reader:
            self.score_reader.stop()
            self.score_reader = None
            
        # 7) Clear any cached correlation data
        self.latest_correlation = None
        if hasattr(self, 'bar_items'):
            delattr(self, 'bar_items')  # Force recreation of plot items
            
        if hasattr(self, '_comod_after_id'):
            try:
                self.after_cancel(self._comod_after_id)
            except tk.TclError:
                pass
        
        # 8) Clear any cached correlation data
        self.latest_correlation = None
        if hasattr(self, 'bar_items'):
            delattr(self, 'bar_items')
            
        if hasattr(self, '_comod_after_id'):
            try:
                self.after_cancel(self._comod_after_id)
            except tk.TclError:
                pass
                
        # 9) Reset recording states
        self.recording_active = False
        self.immediate_recording = False
        self.record_status.config(text="Not recording", foreground='gray')
        self.record_now_btn.config(text="Record Now", state='normal' if self.record_video.get() else 'disabled')
        
        # 10) Reset audio states
        self.audio_recorders.clear()
        self.audio_status.config(text=self.audio_status.cget("text").replace(" (recording)", ""))
                
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

    def save_current_settings(self):
        """Save all current GUI settings to configuration file"""
        # Save all current settings
        self.config.set('camera_settings.target_fps', self.desired_fps.get())
        self.config.set('camera_settings.resolution', self.res_choice.get())
        self.config.set('startup_mode.multi_face', self.multi_face_mode.get())
        self.config.set('startup_mode.participant_count', self.participant_count.get())
        self.config.set('startup_mode.camera_count', self.camera_count.get())
        self.config.set('startup_mode.enable_mesh', self.enable_mesh.get())
        self.config.set('video_recording.enabled', self.record_video.get())
        self.config.set('video_recording.save_directory', self.save_dir.get())
        self.config.set('video_recording.filename_template', self.filename_template.get())
        self.config.set('audio_recording.enabled', self.audio_enabled.get())
        self.config.set('audio_recording.standalone_audio', self.audio_mode.get() == "standalone")
        self.config.set('audio_recording.audio_with_video', self.audio_mode.get() == "with_video")
        self.config.set('audio_devices', self.audio_device_assignments)
        
        # Show confirmation
        messagebox.showinfo("Settings Saved", "Current settings have been saved to configuration file")

if __name__ == '__main__':
    YouQuantiPyGUI().mainloop()
