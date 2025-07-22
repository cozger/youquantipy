import tkinter as tk
from tkinter import ttk
from pygrabber.dshow_graph import FilterGraph
from PIL import Image, ImageTk
from multiprocessing import Queue as MPQueue
from multiprocessing import Process, shared_memory, Pipe
from concurrent.futures import ThreadPoolExecutor
import queue

import threading
import time
import io
import os, pathlib

import numpy as np
import cv2
import mediapipe as mp

from canvasdrawing import CanvasDrawingManager , draw_overlays_combined
from correlator import ChannelCorrelator
from sharedbuffer import NumpySharedBuffer
from LSLHelper import lsl_helper_process
from videorecorder import VideoRecorderProcess
from audiorecorder import AudioRecorder, VideoAudioRecorder, AudioDeviceManager
from guireliability import setup_reliability_monitoring, GUIReliabilityMonitor
from participantmanager_unified import GlobalParticipantManager
from parallelworker_unified import parallel_participant_worker
from tkinter import filedialog, messagebox
from confighandler import ConfigHandler 
from datetime import datetime
from pathlib import Path

# Path to your Mediapipe face_landmarker.task
DEFAULT_MODEL_PATH = r"D:\Projects\youquantipy\face_landmarker.task"
POSE_MODEL_PATH = r"D:\Projects\youquantipy\pose_landmarker_heavy.task"

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
        devices.append((i, names[i], actual_w, actual_h))
        cap.release()
    return devices

def correlation_monitor_process(correlation_queue: MPQueue, 
                               correlation_buffer_name: str,
                               fps: int = 30):
    """
    Separate process for monitoring correlation without LSL streaming.
    Starts automatically when 2+ faces are detected.
    """
    
    correlator = ChannelCorrelator(window_size=60, fps=fps)
    
    # Connect to correlation output buffer
    try:
        corr_buffer = shared_memory.SharedMemory(name=correlation_buffer_name)
        corr_array = np.ndarray((52,), dtype=np.float32, buffer=corr_buffer.buf)
        print("[Correlation Monitor] Connected to correlation buffer")
    except Exception as e:
        print(f"[Correlation Monitor] Failed to connect to correlation buffer: {e}")
        return
    
    participant_scores = {}
    last_update_time = {}
    
    while True:
        try:
            # Get data with timeout
            data = correlation_queue.get(timeout=0.1)
            
            if data is None:  # Shutdown signal
                break
                
            pid = data['participant_id']
            participant_scores[pid] = np.array(data['blend_scores'])
            last_update_time[pid] = time.time()
            
            # Clean up old participants
            current_time = time.time()
            to_remove = [p for p, t in last_update_time.items() 
                        if current_time - t > 2.0]
            for p in to_remove:
                del participant_scores[p]
                del last_update_time[p]
            
            # Calculate correlation if we have 2+ active participants
            if len(participant_scores) >= 2:
                pids = sorted(participant_scores.keys())[:2]
                corr = correlator.update(
                    participant_scores[pids[0]],
                    participant_scores[pids[1]]
                )
                if corr is not None:
                    corr_array[:] = corr
                    
        except queue.Empty:
            continue
        except Exception as e:
            print(f"[Correlation Monitor] Error: {e}")
            
    print("[Correlation Monitor] Stopped")

class YouQuantiPyGUI(tk.Tk):
    BAR_HEIGHT = 150

    def __init__(self):
        super().__init__()
        
        #use handler to import configurations
        self.config = ConfigHandler()
        self.MODEL_PATH = self.config.get('paths.model_path', DEFAULT_MODEL_PATH)
        self.POSE_MODEL_PATH = self.config.get('paths.pose_model_path',POSE_MODEL_PATH)
        self.DESIRED_FPS = self.config.get('camera_settings.target_fps', DEFAULT_DESIRED_FPS)
        
        # Advanced detection will be automatically used if configured

        # Load face_landmarker model buffer
        with open(self.MODEL_PATH, 'rb') as f:
            self.model_buf = f.read()

        # For tracker
        self.tracker_id_to_slot = {}      # Maps tracker_id -> slot index (for "P#")
        self.slot_to_tracker_id = {}
        
        # For advanced detection bboxes
        self.participant_bboxes = {}  # {(camera_idx, participant_id): bbox}  

        print("[DEBUG] GUI __init__")
        self.title("YouQuantiPy")
        self.geometry("1200x1050")
        self.grid_columnconfigure(0, weight=0)   # fixed width  (left)
        self.grid_columnconfigure(1, weight=1)   # expands      (middle  ♦)
        self.grid_columnconfigure(2, weight=0)   # fixed width  (right)
        # give the first row weight so everything grows vertically
        self.grid_rowconfigure(0, weight=1)

        # get drawing manager
        self.drawingmanager = CanvasDrawingManager()
        
        # DEBUG_RETINAFACE: Enable debug drawing of raw detections
        self.debug_retinaface = True  # Set to False to disable debug drawing
        # Synchrony plot correlator & face tracker
        self.correlator = ChannelCorrelator(window_size=60, fps=30)
        self.correlation_queue = MPQueue()
        self.correlation_monitor_proc = None
        
        # Initialize active camera procs early
        self.active_camera_procs = {}  
            
        # Load config values BEFORE creating UI controls
        self.participant_count = tk.IntVar(value=self.config.get('startup_mode.participant_count', 2))
        self.camera_count = tk.IntVar(value=self.config.get('startup_mode.camera_count', 2))
        self.enable_mesh = tk.BooleanVar(value=self.config.get('startup_mode.enable_mesh', False))
        self.enable_pose = tk.BooleanVar(value=self.config.get('startup_mode.enable_pose', True))
        self.desired_fps = tk.IntVar(value=self.config.get('camera_settings.target_fps', 30))
        default_res = self.config.get('camera_settings.resolution', '720p')
        self.res_choice = tk.StringVar(value=default_res)

        # Global participant management
        self.participant_update_queue = MPQueue()
        self.participant_mapping_pipes = {} 
        
        # Initialize participant manager with face recognition
        self.global_participant_manager = GlobalParticipantManager(
            max_participants=self.participant_count.get(),
            shape_weight=0.5,
            position_weight=0.2,
            recognition_weight=0.3
        )
        print("[GUI] Using participant manager with face recognition")

        # NOW start participant update processor after queue is created
        self.participant_update_thread = threading.Thread(
            target=self._process_participant_updates, 
            daemon=True
        )
        self.participant_update_thread.start()
        
        self.participant_monitor_thread = None
        self.monitoring_active = False

        self.performance_stats = {
            'face_fps': {},
            'pose_fps': {},
            'fusion_fps': {}
        }

        self.data_thread = None
        self.streaming = False
        self.blend_labels = None 
        self.worker_procs = []
        self.preview_queues = []
        self.score_queues = []
        self.recording_queues = []
        self.participant_names = {}
        self.score_reader = None
        self.gui_update_thread = None
        self.stop_evt = None

        # Left-side control panel
        self.control_panel = ttk.Frame(self)
        self.control_panel.grid(row=0, column=0, rowspan=3, sticky='ns', padx=10, pady=10)
        self.grid_columnconfigure(0, weight=0)
        
        # Participant count
        ttk.Label(self.control_panel, text="Participants:").pack(anchor='w')
        self.part_spin = tk.Spinbox(self.control_panel, from_=1, to=6, textvariable=self.participant_count,
                                    command=self.on_participant_count_change, width=5)
        self.part_spin.pack(anchor='w', pady=(0,20))

        # Camera count
        ttk.Label(self.control_panel, text="Cameras:").pack(anchor='w')
        self.cam_spin = tk.Spinbox(self.control_panel, from_=1, to=self.participant_count.get(),
                                    textvariable=self.camera_count,
                                    command=self.on_camera_count_change, width=5)
        self.cam_spin.pack(anchor='w', pady=(0,15))

        #FPS input
        ttk.Label(self.control_panel, text="FPS:").pack(anchor='w')
        self.fps_spin = tk.Spinbox(
            self.control_panel,
            from_=1, to=120,
            textvariable=self.desired_fps,
            width=5,
            command=lambda: None  # no‐op, we just read .get() later
        )
        self.fps_spin.pack(anchor='w', pady=(0,20))

        # ─── Resolution selector ───────────────────────────────────────
        ttk.Label(self.control_panel, text="Resolution:").pack(anchor='w')
        # map label→(w,h)
        self.res_map = {
            "1080p": (1920, 1080),
            "720p":  (1280, 720),
            "480p":  ( 640, 480),
            "240p":  ( 320, 240),
        }
        self.res_menu = ttk.Combobox(
            self.control_panel,
            textvariable=self.res_choice,
            values=list(self.res_map.keys()),
            state="readonly",
            width=7
        )
        self.res_menu.pack(anchor='w', pady=(0, 20))
        self.res_menu.bind("<<ComboboxSelected>>", lambda e: self.on_resolution_change())

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

        ttk.Checkbutton(
            self.control_panel,
            text="Enable Pose Estimation",
            variable=self.enable_pose,
            command=self.on_pose_toggle
        ).pack(anchor='w', pady=(0,10))

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

        # Save window geometry on close
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        # === RELIABILITY MONITORING SETUP ===
        # Configure reliability monitoring (optional - uses defaults if not provided)
        reliability_config = {
            'memory_growth_threshold': 800,    # memory threshold in MB
            'max_queue_size': 8,               # Smaller queues for better responsiveness  
            'gui_freeze_threshold': 3.0,       # More sensitive freeze detection
            'stats_report_interval': 60,      # Report every N seconds
            'resource_check_interval': 10,      # More frequent memory checks (default: 10)
            'queue_check_interval': 5,         # More frequent queue checks (default: 5)
            'gui_check_interval': 1,         # More sensitive GUI freeze detection (default: 1)
        }
        
        # Setup monitoring - this adds convenience methods to self
        self.reliability_monitor, self.recording_protection = setup_reliability_monitoring(
            self, reliability_config
        )
        
        # Start monitoring after GUI is fully initialized
        self.after(1000, self._start_monitoring)  # Start 1 second after GUI loads
        self._last_cache_cleanup = time.time()

    def _start_monitoring(self):
        """Start reliability monitoring after GUI is ready"""
        self.reliability_monitor.start_monitoring()
        
        # The recording protection is already checking recovery state on init
        # but we can add periodic state saves during recording
        print("[GUI] Reliability monitoring and recording protection active")

    def _process_participant_updates(self):
        """Process participant updates from workers in the main thread"""
        last_cleanup = time.time()
        update_count = 0  # Add counter for debugging

        while True:
            try:
                update = self.participant_update_queue.get(timeout=0.1)
                if update is None:  # Shutdown signal
                    break
                    
                # Update global participant manager
                camera_idx = update['camera_idx']
                local_id = update['local_id']
                centroid = update['centroid']
                shape = update.get('shape')
                
                # # Debug print
                # update_count += 1
                # if update_count % 30 == 0:  # Every 30 updates
                #     print(f"[GUI] Processing update #{update_count}: camera={camera_idx}, local_id={local_id}, centroid={centroid}")
                
                # Get global ID from manager
                # Check if we're using advanced manager with face recognition data
                if hasattr(self.global_participant_manager, 'update_from_advanced_detection') and 'bbox' in update:
                    # Advanced detection provides more data
                    face_data = [{
                        'track_id': local_id,
                        'participant_id': update.get('participant_id', -1),
                        'centroid': centroid,
                        'landmarks': update.get('landmarks', []),
                        'bbox': update.get('bbox'),
                        'quality_score': update.get('confidence', 0.5)
                    }]
                    track_to_global = self.global_participant_manager.update_from_advanced_detection(
                        camera_idx, face_data
                    )
                    global_id = track_to_global.get(local_id, 1)
                else:
                    # Standard update
                    global_id = self.global_participant_manager.update_participant(
                        camera_idx, local_id, centroid, shape  
                    )
                
                # Debug print the result
                # if update_count % 30 == 0:
                #     print(f"[GUI] Assigned global_id={global_id} to local_id={local_id}")
                
                # Store bbox if available
                if 'bbox' in update:
                    self.participant_bboxes[(camera_idx, global_id)] = update['bbox']
                    if update_count % 240 == 0:
                        print(f"[GUI] Stored bbox for participant {global_id}: {update['bbox']}")
                
                # Send response back to worker
                if camera_idx in self.participant_mapping_pipes:
                    try:
                        response = {
                            'local_id': local_id,
                            'global_id': global_id
                        }
                        self.participant_mapping_pipes[camera_idx].send(response)
                        # if update_count % 30 == 0:
                        #     print(f"[GUI] Sent mapping response: {response}")
                    except Exception as e:
                        print(f"[GUI] Error sending response to camera {camera_idx}: {e}")

                # Periodically call cleanup:
                now = time.time()
                if now - last_cleanup > 1.0:
                    self.global_participant_manager.cleanup_old_participants()
                    last_cleanup = now
                        
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[GUI] Error processing participant update: {e}")
                import traceback
                traceback.print_exc()
                
        print("[GUI] Participant update processor stopped")
        
    def update_performance_stats(self):
        """Monitor and display performance statistics"""
        if not self.streaming:
            return
            
        # Collect stats from workers via control connection
        for idx, info in enumerate(self.frames):
            if info.get('control_conn'):
                info['control_conn'].send('get_stats')
        
        # Get reliability monitor stats
        if hasattr(self, 'reliability_monitor'):
            stats = self.reliability_monitor.get_stats()
            if stats.get('should_report'):
                print("\n[Reliability Monitor] Performance Report:")
                print(f"  Uptime: {stats['uptime']}")
                print(f"  Preview FPS: {stats['preview_fps']:.1f}")
                print(f"  Total frames: {stats['total_preview_updates']}")
                print(f"  Dropped frames: {stats['total_dropped_frames']} ({stats['drop_rate']:.1%})")
                print(f"  Queue overflows: {stats['total_queue_overflows']}")
                print(f"  Memory warnings: {stats['total_memory_warnings']}")
                print(f"  GUI freeze warnings: {stats['total_gui_freeze_warnings']}")
                print(f"  Emergency cleanups: {stats['total_emergency_cleanups']}")
                
        # Schedule next update
        self.after(5000, self.update_performance_stats) 
    
    def on_participant_count_change(self):
        """Handle participant count change"""
        # Update camera spinbox maximum
        self.cam_spin.config(to=self.participant_count.get())
        #update max participants in global manager
        self.global_participant_manager.set_max_participants(self.participant_count.get())

        # Ensure camera count doesn't exceed participant count
        if self.camera_count.get() > self.participant_count.get():
            self.camera_count.set(self.participant_count.get())
        
        # Rebuild frames
        self.build_frames()

    def on_camera_count_change(self):
        """Handle camera count change"""
        # Ensure camera count doesn't exceed participant count
        if self.camera_count.get() > self.participant_count.get():
            self.camera_count.set(self.participant_count.get())
        
        # Rebuild frames
        self.build_frames()

    def on_resolution_change(self):
        """
        Called when the user picks a new resolution from the dropdown. This does not restart running cameras.
        """
        res_label = self.res_choice.get()
        print(f"[GUI] Resolution changed to {res_label} ({self.res_map[res_label]})")
        for info in self.frames:
            if info.get('meta_label'):
                fps = self.desired_fps.get()
                info['meta_label'].config(text=f"Camera → {info['camera_index']} (Capture: {res_label[0]}x{res_label[1]}@{fps}fps)")

    def on_closing(self):
        """Save configuration before closing"""

        # Stop reliability monitoring first
        if hasattr(self, 'reliability_monitor'):
            self.reliability_monitor.stop_monitoring()
        
        # Stop participant update processor
        if hasattr(self, 'participant_update_queue'):
            try:
                self.participant_update_queue.put(None)
            except:
                pass
        
        # Stop correlation monitor
        if hasattr(self, 'correlation_queue'):
            try:
                self.correlation_queue.put(None)
            except:
                pass

        # Clean up drawing manager
        if hasattr(self, 'drawingmanager '):
            for idx in range(len(self.frames)):
                self.drawingmanager .cleanup_canvas(idx)
        
        if hasattr(self, 'correlation_monitor_proc') and self.correlation_monitor_proc:
            self.correlation_monitor_proc.terminate()
            self.correlation_monitor_proc.join(timeout=1.0)
        
        # Save current settings
        self.config.set('camera_settings.target_fps', self.desired_fps.get())
        self.config.set('camera_settings.resolution', self.res_choice.get())
        self.config.set('startup_mode.participant_count', self.participant_count.get())
        self.config.set('startup_mode.camera_count', self.camera_count.get())
        self.config.set('startup_mode.enable_mesh', self.enable_mesh.get())
        self.config.set('startup_mode.enable_pose', self.enable_pose.get())
        
        # Clean up shared memory
        if hasattr(self, 'correlation_buffer'):
            try:
                self.correlation_buffer.close()
                self.correlation_buffer.unlink()
            except:
                pass
        
        # Stop LSL helper
        if hasattr(self, 'lsl_helper_proc') and self.lsl_helper_proc is not None and self.lsl_helper_proc.is_alive():
            self.lsl_command_queue.put({'type': 'stop'})
            self.lsl_helper_proc.terminate()
            self.lsl_helper_proc.join(timeout=1.0)
        
        self.destroy()

    def on_mesh_toggle(self):
        """Notify each worker to turn raw‐mesh pushing on/off."""
        val = self.enable_mesh.get()
        print(f"[GUI] Mesh toggle changed to: {val}")
        
        # First, notify all workers
        for info in self.frames:
            conn = info.get('control_conn')
            if conn:
                conn.send(('set_mesh', val))
                print(f"[GUI] Sent mesh toggle → {val} to worker")
        
        # If streaming, force recreation of all streams
        if self.streaming and hasattr(self, 'lsl_command_queue'):
            # Send a special command to force recreation
            self.lsl_command_queue.put({
                'type': 'force_recreate_streams',
                'mesh_enabled': val
            })
    def on_pose_toggle(self):
        enabled = self.enable_pose.get()
        for info in self.frames:
            if info.get('control_conn'):
                info['control_conn'].send(('enable_pose', enabled))
        # LSL streams
        if self.streaming and hasattr(self, 'lsl_command_queue'):
            for idx, info in enumerate(self.frames):
                participant_id = self.participant_names.get(idx, f"P{idx+1}")
                if enabled:
                    print(f"[GUI] ({'Enabling' if enabled else 'Disabling'}) pose stream for:", participant_id)

                    self.lsl_command_queue.put({
                        'type': 'create_pose_stream',
                        'participant_id': participant_id,
                        'fps': self.desired_fps.get()
                    })
                else:
                    self.lsl_command_queue.put({
                        'type': 'close_pose_stream',
                        'participant_id': participant_id
                    })
        print(f"[GUI] Pose estimation: {'enabled' if enabled else 'disabled'}")

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
        self.after(1000, self.update_participant_names)
        
        # Update participant names from entries
        if hasattr(self, 'participant_entries'):
            names_changed = False
            for idx, entry in enumerate(self.participant_entries):
                name = entry.get().strip()
                old_name = self.participant_names.get(idx, f"P{idx + 1}")
                if name:
                    if name != old_name:
                        names_changed = True
                    self.participant_names[idx] = name
                else:
                    if f"P{idx + 1}" != old_name:
                        names_changed = True
                    self.participant_names[idx] = f"P{idx + 1}"
        
        # Update the global participant manager with current names
        if hasattr(self, 'global_participant_manager'):
            self.global_participant_manager.set_participant_names(self.participant_names)
        
        # If streaming and names changed, update all fusion processes
        if self.streaming and names_changed:
            for cam_idx, pipe in self.participant_mapping_pipes.items():
                try:
                    pipe.send({
                        'type': 'participant_names',
                        'names': self.global_participant_manager.participant_names.copy()
                    })
                except:
                    pass

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
        
        # Cancel any lingering preview callbacks
        for info in self.frames:
            if aid := info.get('after_id'):
                self.after_cancel(aid)

        # Stop & join threads
        for info in self.frames:
            if info.get('proc'):
                info['proc'].terminate()
                info['proc'].join(timeout=1.0)

        # Destroy all canvas frames and cache objects
        for w in self.container.winfo_children():
            w.destroy()

        if hasattr(self, '_photo_images'):
            self._photo_images.clear()

        # Clear the list
        self.frames.clear()
        self.cams = list_video_devices()
        
        # Clear IPC state
        self.worker_procs = []
        self.preview_queues = []
        self.score_queues = []
        self.recording_queues = []

        # Get camera count (limited by participant count)
        cam_count = min(self.camera_count.get(), self.participant_count.get())
        self.camera_count.set(cam_count)
        self.cam_spin.config(to=self.participant_count.get())

        # Remove any old multiface panel if it exists
        if hasattr(self, 'multi_panel'):
            self.multi_panel.destroy()

        # Create participant name panel
        self.participant_panel = ttk.LabelFrame(self, text="Participant Names")
        self.participant_panel.grid(row=2, column=1, columnspan=3, padx=10, pady=10, sticky='ew')
        
        self.participant_entries = []
        for i in range(self.participant_count.get()):
            row = ttk.Frame(self.participant_panel)
            row.pack(fill='x', padx=5, pady=2)
            ttk.Label(row, text=f"Participant {i+1}:").pack(side='left', padx=(0,5))
            ent = ttk.Entry(row, width=20)
            ent.insert(0, f"P{i+1}")  # Default name
            ent.pack(side='left', padx=5)
            self.participant_entries.append(ent)
            self.participant_names[i] = f"P{i+1}"
            ent.bind("<KeyRelease>", lambda event, idx=i: self.on_participant_name_change(idx, event.widget.get().strip()))


        # Build camera selection values
        cam_vals = [f"{i}: {name})" for i,name,w,h in self.cams]

        # Initialize LSL helper if needed
        if not hasattr(self, 'lsl_helper_proc') or self.lsl_helper_proc is None or not self.lsl_helper_proc.is_alive():
            self._initialize_lsl_helper()

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

            #Kill any existing process using the same camera index (elsewhere) 
            for other_idx, other_info in enumerate(self.frames):
                if other_idx == idx:
                    continue
                other_proc = other_info.get('proc')
                other_sel = other_info.get('combo').get() if other_info.get('combo') else None
                if other_sel:
                    other_cam_idx = int(other_sel.split(":", 1)[0])
                    if other_cam_idx == cam_idx and other_proc and other_proc.is_alive():
                        print(f"[GUI] Terminating process for camera {other_cam_idx} in slot {other_idx} (duplicate usage)")
                        other_proc.terminate()
                        other_proc.join(timeout=2.0)
                        other_info['proc'] = None

            pv_q = MPQueue(maxsize=1)  # Reduced to 1 for real-time display
            rec_q = MPQueue(maxsize=10)

            # Update queues
            if idx >= len(self.preview_queues):
                self.preview_queues.append(pv_q)
                self.recording_queues.append(rec_q)
            else:
                self.preview_queues[idx] = pv_q
                self.recording_queues[idx] = rec_q

            # Create score buffer
            score_buffer = NumpySharedBuffer(52)
            info['score_buffer'] = score_buffer
            score_buffer_name = score_buffer.name

            # Get FPS and resolution
            if info['use_auto_fps'].get():
                best = self.auto_detect_optimal_fps(cam_idx)
                fps = int(best['actual_fps'])
                resolution = tuple(map(int, best['resolution'].split('x')))
            else:
                fps = self.desired_fps.get()
                res_label = self.res_choice.get()
                resolution = self.res_map.get(res_label, CAM_RESOLUTION)

            info['resolution'] = resolution
            info['fps'] = fps

            # Create participant mapping pipe for this camera
            main_pipe, worker_pipe = Pipe()
            self.participant_mapping_pipes[cam_idx] = main_pipe

            max_participants = self.participant_count.get()

            # Start correlation monitor if not running
            if not self.correlation_monitor_proc or not self.correlation_monitor_proc.is_alive():
                self._start_correlation_monitor()

            # Start worker process with queues and pipes
            # Get model paths from configuration
            retinaface_model = self.config.get('advanced_detection.retinaface_model')
            arcface_model = self.config.get('advanced_detection.arcface_model')
            
            proc = Process(
                target=parallel_participant_worker,
                args=(cam_idx, self.MODEL_PATH, self.POSE_MODEL_PATH,
                    fps, self.enable_mesh.get(), self.enable_pose.get(),
                    pv_q, score_buffer_name, child_conn,
                    rec_q, self.lsl_data_queue,
                    self.participant_update_queue,
                    worker_pipe, self.correlation_queue,
                    max_participants,
                    resolution,
                    retinaface_model,  # retinaface_model_path
                    arcface_model,     # arcface_model_path
                    True),             # enable_recognition=True
                daemon=False
            )
            proc.start()
            info['proc'] = proc
            info['meta_label'].config(text=f"Camera → {cam_idx} ({resolution[0]}x{resolution[1]}@{fps}fps)")
            # Update record button state
            if self.record_video.get() and not self.streaming:
                self.record_now_btn.config(state='normal')
            # Update global active camera mapping
            self.active_camera_procs[cam_idx] = proc


        # Create camera frames
        for i in range(cam_count):
            frame = ttk.LabelFrame(self.container, text=f"Camera {i}")
            frame.grid(row=0, column=i, padx=5, sticky='nsew')
            self.container.grid_columnconfigure(i, weight=1)
            self.container.grid_rowconfigure(0, weight=1)
            frame.grid_columnconfigure(0, weight=1)
            frame.grid_rowconfigure(1, weight=1)

            # Canvas
            canvas = tk.Canvas(frame, bg='black', highlightthickness=0, borderwidth=0)
            canvas.grid(row=1, column=0, sticky='nsew', padx=2, pady=2)
            self.drawingmanager.initialize_canvas(canvas, i)
            # Force initial transform cache
            canvas.update_idletasks()
            W, H = canvas.winfo_width(), canvas.winfo_height()
            if W > 1 and H > 1:
                if i not in self.drawingmanager.transform_cache:
                    self.drawingmanager.transform_cache[i] = {}
                self.drawingmanager.transform_cache[i]['video_bounds'] = (0, 0, W, H)


            # Meta label
            meta = ttk.Label(frame, text="Camera not selected")
            meta.grid(row=2, column=0, sticky='w', padx=2, pady=(4,0))

            # Camera selector
            cmb = ttk.Combobox(frame, values=cam_vals, state='readonly', width=30)
            if cam_vals: 
                cmb.current(0)
            cmb.grid(row=4, column=0, sticky='we', padx=2, pady=(0,4))

            # Auto FPS toggle
            use_auto_fps = tk.BooleanVar(value=False)
            chk = ttk.Checkbutton(frame, text="Auto-detect optimal FPS", variable=use_auto_fps)
            chk.grid(row=5, column=0, sticky='w', padx=2, pady=(0,4))

            # Store references
            info = {
                'frame': frame,
                'canvas': canvas,
                'meta_label': meta,
                'combo': cmb,
                'proc': None,
                'preview_queue': None,
                'score_buffer': None,
                'use_auto_fps': use_auto_fps,
                'camera_index': i
            }
            self.frames.append(info)

            # Bind selection event
            cmb.bind("<<ComboboxSelected>>", lambda event, idx=i: on_select(idx, event))

            self.diagnose_pipeline() #call the diagnostic

    def _initialize_lsl_helper(self):
        """Initialize the LSL helper process"""
        self.lsl_command_queue = MPQueue()
        self.lsl_data_queue = MPQueue()
        
        # Create correlation buffer
        if hasattr(self, 'correlation_buffer'):
            try:
                self.correlation_buffer.close()
                self.correlation_buffer.unlink()
            except:
                pass
        
        self.correlation_buffer = shared_memory.SharedMemory(
            create=True, 
            size=52 * 4  # float32 array
        )
        self.corr_array = np.ndarray((52,), dtype=np.float32, 
                                    buffer=self.correlation_buffer.buf)
        
        # Start LSL helper process
        self.lsl_helper_proc = Process(
            target=lsl_helper_process,
            args=(self.lsl_command_queue, self.lsl_data_queue, 
                self.correlation_buffer.name, self.desired_fps.get()),
            daemon=True
        )
        self.lsl_helper_proc.start()
        print("[GUI] LSL helper process started")

    def auto_detect_optimal_fps(self, cam_idx):
        """Test camera to find best resolution/FPS combination"""
        test_configs = [
            # (width, height, target_fps)
            (320, 240, 120),
            (320, 240, 60),
            (640, 480, 60),
            (640, 480, 30),
            (1280, 720, 60),
            (1280, 720, 30),
            (1920, 1080, 30),
        ]
        
        camera_opt_results = []
        
        for width, height, target_fps in test_configs:
            cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
            if not cap.isOpened():
                continue

            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, target_fps)

            # Warm-up
            start_time = time.time()
            while time.time() - start_time < 0.5:  # 0.5 sec warm-up
                cap.read()

            # Measure for longer
            measure_time = 2.0
            frames = 0
            unique = 0
            last_hash = None
            measure_start = time.time()
            while time.time() - measure_start < measure_time:
                ret, frame = cap.read()
                if ret:
                    frames += 1
                    h = hash(frame[::10, ::10].tobytes())
                    if h != last_hash:
                        unique += 1
                        last_hash = h

            cap.release()
            actual_fps = unique / measure_time
            camera_opt_results.append({
                'resolution': f"{width}x{height}",
                'target_fps': target_fps,
                'actual_fps': actual_fps,
                'efficiency': actual_fps / target_fps if target_fps > 0 else 0
            })
            
            print(f"[Camera Test] {width}x{height}@{target_fps}: {actual_fps:.1f} FPS actual")
        
        def select_smart_config(configs):
            """Prefers higher resolution unless it costs ≥10 fps vs next lowest."""
            # Parse width, height for sorting
            for cfg in configs:
                w, h = map(int, cfg['resolution'].split('x'))
                cfg['width'] = w
                cfg['height'] = h
                cfg['pixels'] = w * h

            # Sort by pixel count (ascending)
            configs = sorted(configs, key=lambda x: x['pixels'])

            best = configs[0]
            for i in range(1, len(configs)):
                prev = configs[i-1]
                curr = configs[i]
                if prev['actual_fps'] - curr['actual_fps'] >= 10:
                    # Too much fps drop, pick previous
                    break
                else:
                    best = curr  # Move up to higher res

            return best

        
        # Find best configuration
        best = select_smart_config(camera_opt_results)
        print(f"[Camera Test] Best config: {best['resolution']} @ {best['actual_fps']:.1f} FPS")
        
        return best
    
        
    def diagnose_pipeline(self):
        """Diagnostic method to test if the pipeline is working"""
        print("\n=== PIPELINE DIAGNOSIS ===")
        
        # Check if workers are running
        active_workers = 0
        for idx, info in enumerate(self.frames):
            if info.get('proc') and info['proc'].is_alive():
                active_workers += 1
                print(f"Worker {idx}: RUNNING")
            else:
                print(f"Worker {idx}: NOT RUNNING")
        
        print(f"Active workers: {active_workers}")
        
        # Check if preview queues have data
        for idx, q in enumerate(self.preview_queues):
            if q:
                print(f"Preview queue {idx}: {q.qsize()} items")
            else:
                print(f"Preview queue {idx}: None")
        
        # Check LSL helper
        if not hasattr(self, 'lsl_helper_proc') or self.lsl_helper_proc is None or not self.lsl_helper_proc.is_alive():
            print("LSL Helper: RUNNING")
        else:
            print("LSL Helper: NOT RUNNING")
        
        # Check streaming state
        print(f"Streaming active: {self.streaming}")
        
        print("=== END DIAGNOSIS ===\n")
    
    def schedule_preview(self):
        """Preview updates using the drawing manager."""
        # Schedule next update - balanced for low FPS cameras
        self.after(33, self.schedule_preview)  # ~30 FPS checking rate
        
        # Update GUI responsiveness timestamp
        if hasattr(self, 'reliability_monitor'):
            self.reliability_monitor.update_gui_timestamp()

        if not hasattr(self, '_preview_debug_counter'):
            self._preview_debug_counter = 0
    
        # Add canvas diagnostic
        if not hasattr(self, '_canvas_diagnostic_counter'):
            self._canvas_diagnostic_counter = 0
        
        self._canvas_diagnostic_counter += 1
        if self._canvas_diagnostic_counter % 300 == 0:  # Every 10 seconds at 30fps
            for idx, info in enumerate(self.frames[:1]):  # Just first camera
                canvas = info['canvas']
                print(f"\n[GUI Canvas Diagnostic] Canvas {idx}:")
                print(f"  Canvas size: {canvas.winfo_width()}x{canvas.winfo_height()}")
                print(f"  Canvas exists: {canvas.winfo_exists()}")
                
                # Check what's actually on the canvas
                all_items = canvas.find_all()
                print(f"  Total items on canvas: {len(all_items)}")
                
                # Count by tag
                overlay_items = canvas.find_withtag('overlay')
                face_items = canvas.find_withtag('face_line')
                print(f"  Overlay items: {len(overlay_items)}")
                print(f"  Face line items: {len(face_items)}")
                
                # Check item visibility
                if len(overlay_items) > 0:
                    for item in overlay_items[:5]:  # First 5 items
                        coords = canvas.coords(item)
                        state = canvas.itemcget(item, 'state')
                        print(f"    Item {item}: coords={coords[:4] if len(coords) >= 4 else coords}, state={state}")
        
        
        # Process multiple cameras in one update cycle
        cameras_processed = 0
        max_cameras_per_cycle = 4  # Process up to 4 cameras per cycle
        
        for idx, q in enumerate(self.preview_queues):
            if not q or idx >= len(self.frames):
                continue
            
            if cameras_processed >= max_cameras_per_cycle:
                break
                
            # Check if we should skip this frame (FPS limiting)
            if self.drawingmanager.should_skip_frame(idx):
                continue
            
            # Get latest frame (keep only the most recent)
            latest_msg = None
            drop_count = 0
            
            try:
                # Get the most recent frame, dropping older ones
                temp_msg = None
                while True:
                    try:
                        temp_msg = q.get_nowait()
                        if latest_msg:
                            drop_count += 1
                        latest_msg = temp_msg
                    except queue.Empty:
                        break
                        
                # Log excessive drops only
                if drop_count > 5:
                    if not hasattr(self, '_drop_warning_count'):
                        self._drop_warning_count = {}
                    if idx not in self._drop_warning_count:
                        self._drop_warning_count[idx] = 0
                    self._drop_warning_count[idx] += 1
                    if self._drop_warning_count[idx] % 10 == 0:
                        print(f"[GUI Preview {idx}] Excessive drops: {drop_count} frames")
            except Exception as e:
                print(f"[GUI Preview {idx}] Queue error: {e}")
                continue
            
            if latest_msg is None:
                continue
            
            # Track dropped frames
            if drop_count > 1 and hasattr(self, 'reliability_monitor'):
                self.reliability_monitor.track_dropped_frames(drop_count - 1)
            
            try:
                canvas = self.frames[idx]['canvas']

                self._preview_debug_counter += 1
                if self._preview_debug_counter % 600 == 0 and idx == 0:  # Only for first camera
                    faces = latest_msg.get('faces', [])
                    print(f"\n[GUI DIAGNOSTIC] Frame {self._preview_debug_counter}")
                    print(f"  Canvas {idx}: {len(faces)} faces in preview data")
                    # for i, face in enumerate(faces[:2]):  # First 2 faces only
                    #     print(f"  Face {i}:")
                    #     print(f"    ID: {face.get('id', 'NO_ID')}")
                    #     print(f"    Global ID: {face.get('global_id', 'NO_GLOBAL_ID')}")
                    #     print(f"    Track ID: {face.get('track_id', 'NO_TRACK_ID')}")
                    #     print(f"    Has landmarks: {len(face.get('landmarks', [])) > 0}")
                    #     print(f"    Landmark count: {len(face.get('landmarks', []))}")
                    #     print(f"    Has centroid: {'centroid' in face}")
                    #     print(f"    Has bbox: {'bbox' in face}")
                    #     if 'bbox' in face:
                    #         print(f"    Bbox: {face['bbox']}")
                    
                    # Check transform cache
                    transform_info = self.drawingmanager.transform_cache.get(idx, {})
                    print(f"  Transform cache for canvas {idx}: {transform_info.keys()}")
                    if 'video_bounds' in transform_info:
                        print(f"    Video bounds: {transform_info['video_bounds']}")
                
                
                # CRITICAL: Render frame BEFORE drawing overlays
                frame_bgr = latest_msg.get('frame_bgr')
                original_res = latest_msg.get('original_resolution')  
                if frame_bgr is not None:
                    photo = self.drawingmanager.render_frame_to_canvas(
                        frame_bgr, canvas, idx, 
                        original_resolution=original_res  # Pass it through
                    )
                    # Only draw overlays if frame was successfully rendered
                    if photo is not None:
                        # Draw face overlays
                        faces = latest_msg.get('faces', [])
                        if faces:
                            # Get labels
                            labels = {}
                            for face in faces:
                                global_id = face.get('id')
                                if global_id and not isinstance(global_id, str):
                                    labels[global_id] = self.global_participant_manager.get_participant_name(global_id)
                            
                            # Draw faces with optimization
                            self.drawingmanager.draw_faces_optimized(
                                canvas, faces, idx, 
                                labels=labels,
                                participant_count=self.participant_count.get(),
                                participant_names=self.participant_names
                            )
                        
                        # Draw pose overlays
                        if self.enable_pose.get():
                            all_poses = latest_msg.get('all_poses', [])
                            self.drawingmanager.draw_poses_optimized(
                                canvas, all_poses, idx, 
                                enabled=True
                            )
                        
                        # DEBUG_RETINAFACE: Draw raw detections if enabled
                        if self.debug_retinaface:
                            debug_detections = latest_msg.get('debug_detections')
                            if debug_detections:
                                self.drawingmanager.draw_debug_detections(canvas, debug_detections, idx)
                else:
                    # Debug when frame is missing
                    if not hasattr(self, '_frame_missing_count'):
                        self._frame_missing_count = {}
                    if idx not in self._frame_missing_count:
                        self._frame_missing_count[idx] = 0
                    self._frame_missing_count[idx] += 1
                    if self._frame_missing_count[idx] % 30 == 0:
                        print(f"[GUI DEBUG] Camera {idx}: No frame_bgr in preview data. Keys: {list(latest_msg.keys())}")
                
                # Track successful preview update
                if hasattr(self, 'reliability_monitor'):
                    self.reliability_monitor.track_preview_update()
                
                # Increment cameras processed
                cameras_processed += 1
                    
            except Exception as e:
                print(f"[GUI Preview {idx}] Error: {e}")
                import traceback
                traceback.print_exc()

    def debug_canvas_state(self, canvas_idx=None):
        """Debug canvas state using drawing manager stats."""
        if canvas_idx is None:
            for idx in range(len(self.frames)):
                self.debug_canvas_state(idx)
            return
        
        if canvas_idx >= len(self.frames):
            return
        
        print(f"\n[Canvas Debug] Canvas {canvas_idx}:")
        stats = self.drawingmanager.get_stats(canvas_idx)
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    def on_participant_name_change(self, slot_idx, new_name):
        """Update participant_names and force overlays to update."""
        if new_name:
            self.participant_names[slot_idx] = new_name
        else:
            self.participant_names[slot_idx] = f"P{slot_idx+1}"

        # Optionally: force overlays to update now
        self.schedule_preview()

    def get_label_for_face(self, global_id):
        """
        Given a global participant ID (integer), return display label:
        - Use GUI participant name if set, else default to 'P#'.
        - Mapping is 1-based (global_id starts at 1).
        """
        idx = global_id - 1  # zero-based for lists/dicts
        name = self.participant_names.get(idx)
        if name and name.strip() and not name.strip().startswith('P'):
            return name.strip()
        return f"P{global_id}"

    def continuous_correlation_monitor(self):
        """Monitor correlation from shared buffer"""
        self.after(33, self.continuous_correlation_monitor)
        if hasattr(self, 'corr_array'):
            try:
                # Read from shared memory
                corr = self.corr_array.copy()
                if np.any(corr):
                    self.update_plot(corr)
            except:
                pass
                
    def _start_correlation_monitor(self):
        """Start the correlation monitoring process"""
        if hasattr(self, 'correlation_monitor_proc') and self.correlation_monitor_proc and self.correlation_monitor_proc.is_alive():
            return
            
        self.correlation_monitor_proc = Process(
            target=correlation_monitor_process,
            args=(self.correlation_queue, self.correlation_buffer.name, self.desired_fps.get()),
            daemon=True
        )
        self.correlation_monitor_proc.start()
        print("[GUI] Started correlation monitor")  

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
        
        # Save recording state for recovery protection
        if hasattr(self, 'recording_protection'):
            self.recording_protection._save_emergency_state()
            
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
                def create_recording_thread(recorder, recording_queue, preview_queue, width, height, idx, participant_names):
                    stop_flag = threading.Event()
                
                    def add_recording_info_overlay(frame, frame_count, lsl_timestamp, recording_fps=None):
                        """
                        Add recording information overlay to frame.
                        
                        Args:
                            frame: The video frame to overlay on
                            frame_count: Current frame number
                            lsl_timestamp: LSL timestamp (seconds since epoch)
                            recording_fps: Optional actual recording FPS
                        """
                        h, w = frame.shape[:2]
                        
                        # Format timestamp for display
                        dt = datetime.fromtimestamp(lsl_timestamp)
                        time_str = dt.strftime("%H:%M:%S.%f")[:-3]  # HH:MM:SS.mmm
                        
                        # Prepare text lines
                        text_lines = [
                            f"Frame: {frame_count:06d}",
                            f"Time: {time_str}",
                            f"LSL: {lsl_timestamp:.6f}",
                        ]
                        
                        if recording_fps is not None:
                            text_lines.append(f"FPS: {recording_fps:.1f}")
                        
                        # Style settings
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.5
                        thickness = 1
                        text_color = (0, 255, 0)  # Green
                        bg_color = (0, 0, 0)  # Black background
                        padding = 5
                        line_height = 20
                        
                        # Calculate text dimensions
                        max_width = 0
                        for text in text_lines:
                            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                            max_width = max(max_width, text_size[0])
                        
                        # Position in top-right corner
                        x_start = w - max_width - 15
                        y_start = 15
                        
                        # Draw semi-transparent background
                        overlay = frame.copy()
                        cv2.rectangle(overlay,
                                    (x_start - padding, y_start - padding),
                                    (w - 5, y_start + len(text_lines) * line_height + padding),
                                    bg_color, -1)
                        
                        # Blend with original (semi-transparent background)
                        alpha = 0.7
                        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
                        
                        # Draw text lines
                        for i, text in enumerate(text_lines):
                            y = y_start + (i + 1) * line_height
                            cv2.putText(frame, text, (x_start, y),
                                        font, font_scale, text_color, thickness, cv2.LINE_AA)
                        
                        return frame

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
                                                            
                                try:
                                    latest = recording_queue.get(timeout=0.1)
                                except:
                                    continue
                                
                                if latest is None:
                                    continue
                                    
                                frame_bgr = latest.get('frame_bgr')
                                if frame_bgr is None:
                                    continue
                                
                                # Make a copy to avoid modifying the preview
                                frame_bgr = frame_bgr.copy()

                                # Get ALL faces data (multiple faces)
                                faces = latest.get('faces', [])
                                
                                # Get ALL poses data (multiple poses)
                                all_poses = latest.get('all_poses', [])
                                
                                # Build labels dictionary for faces using global IDs
                                labels = {}
                                for face in faces:
                                    global_id = face.get('id')
                                    if global_id and (not isinstance(global_id, str) or not str(global_id).startswith('local_')):
                                        # Get participant name from global manager
                                        labels[global_id] = self.global_participant_manager.get_participant_name(global_id)

                                # Draw overlays with all faces and poses
                                overlayed = draw_overlays_combined(
                                    frame_bgr,
                                    faces=faces,  # Pass all faces
                                    pose_landmarks=None,  # We'll handle poses separately
                                    labels=labels,
                                    face_mesh=False,
                                    face_contours=True,
                                    face_points=False,
                                    pose_lines=False  # We'll draw poses manually
                                )
                                
                                # Draw bboxes from advanced detection
                                for face in faces:
                                    global_id = face.get('id')
                                    if global_id and (cam_idx, global_id) in self.participant_bboxes:
                                        bbox = self.participant_bboxes[(cam_idx, global_id)]
                                        if bbox and len(bbox) == 4:
                                            x1, y1, x2, y2 = [int(v) for v in bbox]
                                            # Draw green bbox
                                            cv2.rectangle(overlayed, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                            # Draw label background
                                            label = labels.get(global_id, f"P{global_id}")
                                            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                                            cv2.rectangle(overlayed, (x1, y1-25), (x1+label_size[0]+10, y1), (0, 255, 0), -1)
                                            # Draw label text
                                            cv2.putText(overlayed, label, (x1+5, y1-8), 
                                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                                
                                # Draw all poses if enabled
                                if self.enable_pose.get() and all_poses:
                                    for pose_idx, pose in enumerate(all_poses):
                                        if pose and 'landmarks' in pose:
                                            # Draw each pose
                                            h, w = overlayed.shape[:2]
                                            pose_landmarks_px = [(int(x * w), int(y * h), z) 
                                                            for x, y, z in pose['landmarks']]
                                            
                                            # Draw pose connections
                                            for conn in mp.solutions.pose.POSE_CONNECTIONS:
                                                i, j = conn
                                                if i < len(pose_landmarks_px) and j < len(pose_landmarks_px):
                                                    cv2.line(overlayed, 
                                                        pose_landmarks_px[i][:2], 
                                                        pose_landmarks_px[j][:2], 
                                                        (64, 255, 255), 2, cv2.LINE_AA)
                                
                                # Add recording info overlay
                                overlayed = add_recording_info_overlay(
                                    overlayed, 
                                    frame_count, 
                                    current_time,
                                    1.0 / (current_time - last_frame_time) if last_frame_time else None
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
                recording_queue = self.recording_queues[idx]

                
                capture_thread, stop_flag = create_recording_thread(
                    recorder, 
                    recording_queue,
                    preview_queue,
                    actual_width,
                    actual_height,
                    idx,
                    self.participant_names,
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
            text="Assign audio devices to participants:",
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
        
        # Create assignment for each participant
        for i in range(self.participant_count.get()):
            frame = ttk.Frame(scrollable_frame)
            frame.pack(fill='x', padx=10, pady=5)
            
            # Get participant name from the entry field
            participant_name = self.participant_names.get(i, f"P{i+1}")
            
            ttk.Label(frame, text=f"{participant_name}:", width=20).pack(side='left')
            
            # Store by participant index
            var = tk.StringVar(value=self.audio_device_assignments.get(f"participant{i}", "None"))
            device_vars[f"participant{i}"] = var
            
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
                
            # Determine participant ID from assignment
            if assignment.startswith("participant"):
                participant_num = int(assignment.replace("participant", ""))
                participant_id = self.participant_names.get(participant_num, f"P{participant_num+1}")
            else:
                # Fallback for old format
                participant_id = assignment
            
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
        """Start unified streaming (LSL only, correlation already running)"""
        if self.streaming:
            print("[GUI] Already streaming, returning")
            return
        
        active_workers = [info for info in self.frames if info.get('control_conn')]
        print(f"[GUI] Found {len(active_workers)} active workers")
        
        if not active_workers:
            print("[GUI] No active workers found")
            return
        
        # Re-initialize LSL helper if needed
        if not hasattr(self, 'lsl_helper_proc') or self.lsl_helper_proc is None or not self.lsl_helper_proc.is_alive():
            self._initialize_lsl_helper()
        
        self.streaming = True
        
        # Start performance monitoring
        self.after(1000, self.update_performance_stats)
        
        # Notify LSL helper that streaming started
        self.lsl_command_queue.put({
            'type': 'streaming_started',
            'max_participants': self.participant_count.get()
        })
        
        # Send streaming state AND participant names to fusion processes
        for cam_idx, pipe in self.participant_mapping_pipes.items():
            try:
                # Send streaming state
                pipe.send({
                    'type': 'streaming_state',
                    'active': True
                })
                # Send participant names mapping
                pipe.send({
                    'type': 'participant_names',
                    'names': self.global_participant_manager.participant_names.copy()
                })
                print(f"[GUI] Sent streaming=True and participant names to fusion for camera {cam_idx}")
            except Exception as e:
                print(f"[GUI] ERROR sending streaming state to fusion for camera {cam_idx}: {e}")
        
        # Notify workers about streaming state AND current mesh state
        for idx, info in enumerate(self.frames):
            if info.get('control_conn'):
                try:
                    # Send streaming state
                    info['control_conn'].send(('streaming_state', True))
                    print(f"[GUI] Sent streaming=True to worker {idx}")
                    
                    # Also ensure mesh state is synced
                    info['control_conn'].send(('set_mesh', self.enable_mesh.get()))
                    print(f"[GUI] Sent mesh state ({self.enable_mesh.get()}) to worker {idx}")
                except Exception as e:
                    print(f"[GUI] ERROR sending to worker {idx}: {e}")
        
        # Start comodulation LSL stream
        self.lsl_command_queue.put({
            'type': 'start_comodulation'
        })
        
        # If pose is enabled, ensure pose streams are created
        if self.enable_pose.get():
            for idx, info in enumerate(self.frames):
                if info.get('control_conn'):
                    participant_id = self.participant_names.get(idx, f"P{idx+1}")
                    self.lsl_command_queue.put({
                        'type': 'create_pose_stream',
                        'participant_id': participant_id,
                        'fps': self.desired_fps.get()
                    })
        
        # Auto-start video recording if enabled
        if self.record_video.get() and active_workers:
            print("[GUI] Auto-starting video recording with stream...")
            # Disable the immediate recording button since we're recording with stream
            self.record_now_btn.config(state='disabled')
            self.stop_record_btn.config(state='disabled')  # Will be controlled by stop stream
            # Start recording
            self._start_video_recording(active_workers)
            # Update status to indicate recording is tied to streaming
            if self.recording_active:
                self.record_status.config(
                    text=f"Recording with stream to {Path(self.save_dir.get()).name}/",
                    foreground='red'
                )
        print(f"[GUI] Started LSL streaming - streams will be created dynamically with mesh={'enabled' if self.enable_mesh.get() else 'disabled'}")
    
    def _shutdown_all_processes(self, timeout=3.0):
        """Shutdown with reliability monitoring"""
        print("\n[GUI] === COMPREHENSIVE SHUTDOWN INITIATED ===")
        
        # Stop reliability monitoring first
        if hasattr(self, 'reliability_monitor'):
            # Perform emergency cleanup if needed
            self.reliability_monitor.emergency_cleanup()
            self.reliability_monitor.stop_monitoring()
        
        # 1. Stop streaming flag first
        self.streaming = False
        
        # 2. Stop all recordings
        if self.recording_active:
            print("[GUI] Stopping video recording...")
            self._stop_video_recording()
        
        if self.audio_recording_active:
            print("[GUI] Stopping audio recording...")
            self.stop_audio_recording()
        
        # 3. Send shutdown signals to all worker control connections
        print("[GUI] Sending shutdown signals to workers...")
        for idx, info in enumerate(self.frames):
            if info.get('control_conn'):
                try:
                    for _ in range(3):
                        info['control_conn'].send('stop')
                    print(f"[GUI] Sent stop signal to worker {idx}")
                except Exception as e:
                    print(f"[GUI] Error sending stop to worker {idx}: {e}")
        
        # 4. Close all participant mapping pipes
        print("[GUI] Closing participant mapping pipes...")
        for cam_idx, pipe in list(self.participant_mapping_pipes.items()):
            try:
                pipe.send({'type': 'shutdown'})
                pipe.close()
            except:
                pass
        self.participant_mapping_pipes.clear()
        
        # 5. Send shutdown to LSL helper
        if hasattr(self, 'lsl_command_queue'):
            print("[GUI] Shutting down LSL helper...")
            try:
                for _ in range(3):
                    self.lsl_command_queue.put({'type': 'stop'})
            except:
                pass
        
        # 6. Clear all queues - FIXED: Handle both dict and list cases
        print("[GUI] Clearing all queues...")
        queues_to_clear = [
            ('participant_update_queue', self.participant_update_queue),
            ('correlation_queue', self.correlation_queue),
            ('lsl_command_queue', getattr(self, 'lsl_command_queue', None)),
            ('lsl_data_queue', getattr(self, 'lsl_data_queue', None))
        ]
        
        for name, q in queues_to_clear:
            if q:
                try:
                    q.put(None)
                    while not q.empty():
                        try:
                            q.get_nowait()
                        except:
                            break
                    print(f"[GUI] Cleared {name}")
                except:
                    pass
        
        # 7. Terminate worker processes
        print("[GUI] Terminating worker processes...")
        for idx, info in enumerate(self.frames):
            if info.get('proc'):
                proc = info['proc']
                if proc.is_alive():
                    print(f"[GUI] Terminating worker {idx}...")
                    proc.terminate()
                    proc.join(timeout=1.0)
                    if proc.is_alive():
                        print(f"[GUI] Force killing worker {idx}...")
                        proc.kill()
                        proc.join(timeout=0.5)
                info['proc'] = None
        
        # 8. Terminate LSL helper process
        if hasattr(self, 'lsl_helper_proc') and self.lsl_helper_proc:
            if self.lsl_helper_proc.is_alive():
                print("[GUI] Terminating LSL helper...")
                self.lsl_helper_proc.terminate()
                self.lsl_helper_proc.join(timeout=1.0)
                if self.lsl_helper_proc.is_alive():
                    print("[GUI] Force killing LSL helper...")
                    self.lsl_helper_proc.kill()
                    self.lsl_helper_proc.join(timeout=0.5)
            self.lsl_helper_proc = None
        
        # 9. Terminate correlation monitor
        if hasattr(self, 'correlation_monitor_proc') and self.correlation_monitor_proc:
            if self.correlation_monitor_proc.is_alive():
                print("[GUI] Terminating correlation monitor...")
                self.correlation_monitor_proc.terminate()
                self.correlation_monitor_proc.join(timeout=1.0)
                if self.correlation_monitor_proc.is_alive():
                    print("[GUI] Force killing correlation monitor...")
                    self.correlation_monitor_proc.kill()
                    self.correlation_monitor_proc.join(timeout=0.5)
            self.correlation_monitor_proc = None
        
        # 10. Clean up shared memory
        print("[GUI] Cleaning up shared memory...")
        # Score buffers
        for info in self.frames:
            if info.get('score_buffer'):
                try:
                    info['score_buffer'].cleanup()
                except:
                    pass
                info['score_buffer'] = None
        
        # Correlation buffer
        if hasattr(self, 'correlation_buffer'):
            try:
                self.correlation_buffer.close()
                self.correlation_buffer.unlink()
            except:
                pass
            self.correlation_buffer = None
        
        # 11. Terminate active camera processes
        print("[GUI] Terminating active camera processes...")
        for cam_idx, proc in list(self.active_camera_procs.items()):
            try:
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=1.0)
                    if proc.is_alive():
                        proc.kill()
                        proc.join(timeout=0.5)
            except:
                pass
        self.active_camera_procs.clear()
        
        # 12. Reset state variables
        self.streaming = False
        self.recording_active = False
        self.immediate_recording = False
        self.audio_recording_active = False
        
        # 13. Final cleanup - ensure all child processes are gone
        import psutil
        try:
            current_process = psutil.Process()
            children = current_process.children(recursive=True)
            if children:
                print(f"[GUI] Found {len(children)} remaining child processes, terminating...")
                for child in children:
                    try:
                        child.terminate()
                    except:
                        pass
                # Give them time to terminate
                gone, alive = psutil.wait_procs(children, timeout=1)
                # Force kill any remaining
                for p in alive:
                    try:
                        print(f"[GUI] Force killing process {p.pid}")
                        p.kill()
                    except:
                        pass
        except ImportError:
            print("[GUI] psutil not available, skipping child process cleanup")
        except Exception as e:
            print(f"[GUI] Error during child process cleanup: {e}")
        
        print("[GUI] === SHUTDOWN COMPLETE ===\n")


    def stop_stream(self):
        """Stop unified streaming with comprehensive cleanup"""
        if not self.streaming:
            return
        
        print("\n[GUI] === STOPPING STREAM ===")
        
        # Use the comprehensive shutdown
        self._shutdown_all_processes()
        
        # Update UI elements
        for info in self.frames:
            if info.get('meta_label'):
                info['meta_label'].config(text="LSL → OFF")
        
        # Update button states
        if self.record_video.get():
            active_workers = [info for info in self.frames if info.get('proc') and info['proc'].is_alive()]
            if active_workers:
                self.record_now_btn.config(state='normal')
        self.stop_record_btn.config(state='disabled')
        
        print("[GUI] Stream stopped\n")


    def reset(self):
        """Complete reset with comprehensive cleanup"""
        print("\n[GUI] === FULL RESET INITIATED ===")
        
        # 1. Comprehensive shutdown first
        self._shutdown_all_processes()
        
        # 2. Stop and wait for participant update thread
        if hasattr(self, 'participant_update_thread') and self.participant_update_thread.is_alive():
            print("[GUI] Stopping participant update thread...")
            if hasattr(self, 'participant_update_queue'):
                self.participant_update_queue.put(None)
            self.participant_update_thread.join(timeout=2.0)
        
        # 3. Reset global participant manager
        if hasattr(self, 'global_participant_manager'):
            self.global_participant_manager.reset()
        
        # 4. Clear all data structures
        self.frames.clear()
        self.worker_procs.clear()
        self.preview_queues.clear()
        self.score_queues.clear()
        self.recording_queues.clear()
        self.participant_mapping_pipes.clear()
        self.video_recorders.clear()
        self.audio_recorders.clear()
        
        # 5. Clear drawing manager state
        for idx in range(len(self.frames)):
            self.drawingmanager .cleanup_canvas(idx)
        
        # 6. Clear the bar plot
        self.bar_canvas.delete('all')
        self.bar_canvas.create_text(
            self.bar_canvas.winfo_width()//2, 
            self.bar_canvas.winfo_height()//2, 
            text="Ready", fill='gray', font=('Arial', 12)
        )
        if hasattr(self, 'bar_items'):
            delattr(self, 'bar_items')
        
        # 7. Reset all flags
        self.streaming = False
        self.recording_active = False
        self.immediate_recording = False
        self.audio_recording_active = False
        self.monitoring_active = False
        self.latest_correlation = None
        
        # 8. Update UI states
        self.record_status.config(text="Not recording", foreground='gray')
        self.record_now_btn.config(
            text="Start Video Recording", 
            state='normal' if self.record_video.get() else 'disabled'
        )
        self.stop_record_btn.config(state='disabled')
        self.start_audio_btn.config(
            state='normal' if self.audio_enabled.get() and self.audio_mode.get() == "standalone" else 'disabled'
        )
        self.stop_audio_btn.config(state='disabled')
        self.audio_status.config(text=self.audio_status.cget("text").replace(" (recording)", ""))
        
        # 9. Recreate essential structures
        self.participant_update_queue = MPQueue()
        self.correlation_queue = MPQueue()
        
        # 10. Restart participant update processor
        self.participant_update_thread = threading.Thread(
            target=self._process_participant_updates, 
            daemon=True
        )
        self.participant_update_thread.start()
        
        # 11. Re-initialize correlation buffer
        self.correlation_buffer = shared_memory.SharedMemory(
            create=True, 
            size=52 * 4
        )
        self.corr_array = np.ndarray((52,), dtype=np.float32, 
                                    buffer=self.correlation_buffer.buf)
        
        # 12. Rebuild frames
        self.build_frames()
        
        print("[GUI] === RESET COMPLETE ===\n")
        
    def save_current_settings(self):
        """Save all current GUI settings to configuration file"""
        # Save all current settings
        self.config.set('camera_settings.target_fps', self.desired_fps.get())
        self.config.set('camera_settings.resolution', self.res_choice.get())
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
    
