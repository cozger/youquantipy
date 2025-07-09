"""
Multiprocessing version of Landmark Worker Pool with proper daemon handling.
Spawns workers from the main process level to avoid daemon spawning issues.
"""

import cv2
import numpy as np
import time
import multiprocessing as mp
from multiprocessing import Process, Queue, Pipe, Manager
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import mediapipe as mp_lib
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import threading
import queue
from queue import Empty  # Import Empty exception

@dataclass
class LandmarkTask:
    """Task for landmark detection."""
    roi_image: np.ndarray
    track_id: int
    frame_id: int
    task_id: int

@dataclass 
class LandmarkResult:
    """Result from landmark detection."""
    track_id: int
    frame_id: int
    task_id: int
    landmarks: np.ndarray  # 478x3 for face
    blendshapes: Optional[np.ndarray] = None  # 52 blend scores
    mesh_data: Optional[np.ndarray] = None  # Full mesh if enabled
    processing_time: float = 0.0
    error: Optional[str] = None
    success: bool = True

def landmark_worker_process(worker_id: int, task_queue: Queue, result_queue: Queue,
                          control_pipe, face_model_path: str, enable_mesh: bool = False):
    """Worker process for landmark detection."""
    print(f"[Landmark Worker {worker_id}] Starting")
    
    # Initialize MediaPipe
    try:
        base_options = python.BaseOptions(model_asset_path=face_model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=False
        )
        face_landmarker = vision.FaceLandmarker.create_from_options(options)
    except Exception as e:
        print(f"[Landmark Worker {worker_id}] Failed to initialize: {e}")
        return
    
    # Process tasks
    while True:
        # Check for control messages
        if control_pipe.poll(0):
            msg = control_pipe.recv()
            if msg == 'stop':
                break
            elif msg == 'toggle_mesh':
                enable_mesh = not enable_mesh
        
        try:
            # Get task with timeout
            task = task_queue.get(timeout=0.1)
            if task is None:  # Shutdown signal
                break
            
            # Process ROI
            start_time = time.time()
            try:
                # Convert to MediaPipe image
                mp_image = mp_lib.Image(image_format=mp_lib.ImageFormat.SRGB, data=task.roi_image)
                
                # Detect landmarks
                detection_result = face_landmarker.detect(mp_image)
                
                if detection_result.face_landmarks:
                    # Extract landmarks (first face only)
                    face_landmarks = detection_result.face_landmarks[0]
                    landmarks = np.array([[lm.x, lm.y, lm.z] for lm in face_landmarks])
                    
                    # Extract blendshapes
                    blendshapes = None
                    if detection_result.face_blendshapes:
                        blend_scores = [b.score for b in detection_result.face_blendshapes[0]][:52]
                        blend_scores += [0.0] * (52 - len(blend_scores))
                        blendshapes = np.array(blend_scores)
                    
                    # Extract mesh if enabled
                    mesh_data = None
                    if enable_mesh:
                        mesh_data = landmarks.flatten()
                    
                    # Create result
                    result = LandmarkResult(
                        track_id=task.track_id,
                        frame_id=task.frame_id,
                        task_id=task.task_id,
                        landmarks=landmarks,
                        blendshapes=blendshapes,
                        mesh_data=mesh_data,
                        processing_time=time.time() - start_time,
                        success=True
                    )
                else:
                    # No face detected
                    result = LandmarkResult(
                        track_id=task.track_id,
                        frame_id=task.frame_id,
                        task_id=task.task_id,
                        landmarks=np.zeros((478, 3)),
                        error="No face detected",
                        processing_time=time.time() - start_time,
                        success=False
                    )
                
                # Send result
                try:
                    result_queue.put_nowait(result)
                except:
                    pass
                    
            except Exception as e:
                # Error processing
                result = LandmarkResult(
                    track_id=task.track_id,
                    frame_id=task.frame_id,
                    task_id=task.task_id,
                    landmarks=np.zeros((478, 3)),
                    error=str(e),
                    processing_time=time.time() - start_time,
                    success=False
                )
                try:
                    result_queue.put_nowait(result)
                except:
                    pass
                    
        except Empty:
            continue
        except Exception as e:
            print(f"[Landmark Worker {worker_id}] Error: {e}")
    
    # Cleanup
    face_landmarker.close()
    print(f"[Landmark Worker {worker_id}] Stopped")


class LandmarkWorkerPoolMP:
    """
    Multiprocessing pool of landmark detection workers.
    Uses a coordinator thread to manage the pool from within a daemon process.
    """
    
    def __init__(self, face_model_path: str, num_workers: int = 4, 
                 enable_mesh: bool = False, max_queue_size: int = 100):
        self.face_model_path = face_model_path
        self.num_workers = num_workers
        self.enable_mesh = enable_mesh
        
        # Internal thread-safe queues for coordination
        self.task_queue_internal = queue.Queue(maxsize=max_queue_size)
        self.result_queue_internal = queue.Queue(maxsize=max_queue_size * 2)
        
        # Multiprocessing queues (created by manager)
        self.manager = None
        self.task_queue_mp = None
        self.result_queue_mp = None
        
        # Workers and control
        self.workers = []
        self.control_pipes = []
        self.is_running = False
        
        # Coordinator thread
        self.coordinator_thread = None
        
        # Task tracking
        self.next_task_id = 0
        self.pending_tasks = {}
        
    def start(self):
        """Start worker pool with coordinator thread."""
        if not self.is_running:
            self.is_running = True
            
            # Start coordinator thread
            self.coordinator_thread = threading.Thread(target=self._coordinator_loop, daemon=True)
            self.coordinator_thread.start()
            
            print(f"[Landmark Pool] Started coordinator for {self.num_workers} workers")
    
    def _coordinator_loop(self):
        """Coordinator thread that manages the multiprocessing pool."""
        # Create manager for shared queues
        self.manager = Manager()
        self.task_queue_mp = self.manager.Queue(maxsize=100)
        self.result_queue_mp = self.manager.Queue(maxsize=200)
        
        # Create and start workers
        for i in range(self.num_workers):
            parent_conn, child_conn = Pipe()
            self.control_pipes.append(parent_conn)
            
            worker = Process(
                target=landmark_worker_process,
                args=(i, self.task_queue_mp, self.result_queue_mp, 
                      child_conn, self.face_model_path, self.enable_mesh)
            )
            worker.start()
            self.workers.append(worker)
        
        print(f"[Landmark Pool] Started {self.num_workers} worker processes")
        
        # Coordination loop
        while self.is_running:
            # Forward tasks from internal queue to multiprocessing queue
            try:
                task = self.task_queue_internal.get(timeout=0.01)
                if task is not None:
                    self.task_queue_mp.put(task, timeout=0.1)
            except queue.Empty:
                pass
            except:
                pass
            
            # Forward results from multiprocessing queue to internal queue
            try:
                result = self.result_queue_mp.get(timeout=0.01)
                self.result_queue_internal.put_nowait(result)
            except:
                pass
        
        # Cleanup
        self._cleanup_workers()
    
    def _cleanup_workers(self):
        """Clean up worker processes."""
        # Send stop signal to all workers
        for pipe in self.control_pipes:
            try:
                pipe.send('stop')
            except:
                pass
        
        # Add None to queue to unblock workers
        for _ in range(self.num_workers):
            try:
                self.task_queue_mp.put(None, timeout=0.1)
            except:
                pass
        
        # Wait for workers
        for worker in self.workers:
            worker.join(timeout=2.0)
            if worker.is_alive():
                worker.terminate()
        
        self.workers.clear()
        self.control_pipes.clear()
        
        # Shutdown manager
        if self.manager:
            self.manager.shutdown()
    
    def stop(self):
        """Stop worker pool."""
        if self.is_running:
            self.is_running = False
            
            # Wait for coordinator thread
            if self.coordinator_thread:
                self.coordinator_thread.join(timeout=5.0)
            
            print("[Landmark Pool] Stopped")
    
    def submit_roi(self, roi_image: np.ndarray, track_id: int, frame_id: int) -> bool:
        """Submit ROI for landmark detection."""
        if not self.is_running:
            return False
        
        try:
            task = LandmarkTask(
                roi_image=roi_image,
                track_id=track_id,
                frame_id=frame_id,
                task_id=self.next_task_id
            )
            
            self.task_queue_internal.put_nowait(task)
            self.pending_tasks[self.next_task_id] = (track_id, frame_id)
            self.next_task_id += 1
            return True
            
        except queue.Full:
            return False
    
    def get_results(self, timeout: float = 0.001) -> List[LandmarkResult]:
        """Get available results."""
        results = []
        deadline = time.time() + timeout
        
        while time.time() < deadline:
            try:
                result = self.result_queue_internal.get_nowait()
                results.append(result)
                
                # Remove from pending
                if result.task_id in self.pending_tasks:
                    del self.pending_tasks[result.task_id]
                    
            except queue.Empty:
                break
        
        return results
    
    def set_mesh_enabled(self, enabled: bool):
        """Toggle mesh extraction for all workers."""
        self.enable_mesh = enabled
        for pipe in self.control_pipes:
            try:
                pipe.send('toggle_mesh')
            except:
                pass
    
    def get_stats(self) -> Dict:
        """Get pool statistics."""
        return {
            'num_workers': self.num_workers,
            'pending_tasks': len(self.pending_tasks),
            'task_queue_size': self.task_queue_internal.qsize(),
            'result_queue_size': self.result_queue_internal.qsize()
        }