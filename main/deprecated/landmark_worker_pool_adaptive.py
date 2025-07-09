"""
Adaptive Landmark Worker Pool with fixed worker count but dynamic task distribution.
Uses multiprocessing for true parallelism while avoiding dynamic process spawning.
"""

import cv2
import numpy as np
import time
import multiprocessing as mp
from multiprocessing import Process, Queue, Pipe
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import mediapipe as mp_lib
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import threading
import queue
from queue import Empty, Full

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

def landmark_worker_process(worker_id: int, task_queue: mp.Queue, result_queue: mp.Queue,
                          control_pipe, face_model_path: str, enable_mesh: bool = False):
    """Worker process for landmark detection."""
    print(f"[Landmark Worker {worker_id}] Starting")
    
    # Initialize MediaPipe
    face_landmarker = None
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
    
    idle_count = 0
    max_idle_iterations = 100  # Idle for up to 10 seconds before considering shutdown
    
    # Process tasks
    while True:
        # Check for control messages (non-blocking)
        if control_pipe.poll(0):
            msg = control_pipe.recv()
            if msg == 'stop':
                break
            elif msg == 'toggle_mesh':
                enable_mesh = not enable_mesh
        
        try:
            # Get task with short timeout
            task = task_queue.get(timeout=0.1)
            idle_count = 0  # Reset idle counter
            
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
                    result_queue.put(result, timeout=1.0)
                except Full:
                    print(f"[Landmark Worker {worker_id}] Result queue full, dropping result")
                    
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
                    result_queue.put(result, timeout=1.0)
                except Full:
                    pass
                    
        except Empty:
            # No task available - increment idle counter
            idle_count += 1
            if idle_count >= max_idle_iterations:
                # Worker has been idle too long - this is normal when no faces detected
                idle_count = 0  # Reset and continue
        except Exception as e:
            print(f"[Landmark Worker {worker_id}] Unexpected error: {e}")
            import traceback
            traceback.print_exc()
    
    # Cleanup
    if face_landmarker:
        face_landmarker.close()
    print(f"[Landmark Worker {worker_id}] Stopped")


class LandmarkWorkerPoolAdaptive:
    """
    Adaptive multiprocessing pool for landmark detection.
    Uses fixed number of workers but distributes tasks dynamically based on load.
    """
    
    def __init__(self, face_model_path: str, num_workers: int = 4, 
                 enable_mesh: bool = False, max_queue_size: int = 100):
        self.face_model_path = face_model_path
        self.num_workers = num_workers
        self.enable_mesh = enable_mesh
        
        # Multiprocessing queues
        self.task_queue = mp.Queue(maxsize=max_queue_size)
        self.result_queue = mp.Queue(maxsize=max_queue_size * 2)
        
        # Workers and control
        self.workers = []
        self.control_pipes = []
        self.is_running = False
        
        # Task tracking
        self.next_task_id = 0
        self.pending_tasks = {}
        
        # Statistics
        self.stats_lock = threading.Lock()
        self.active_tasks = 0
        self.completed_tasks = 0
        self.failed_tasks = 0
        
    def start(self):
        """Start worker pool."""
        if not self.is_running:
            self.is_running = True
            
            # Create and start workers
            for i in range(self.num_workers):
                parent_conn, child_conn = Pipe()
                self.control_pipes.append(parent_conn)
                
                worker = Process(
                    target=landmark_worker_process,
                    args=(i, self.task_queue, self.result_queue, 
                          child_conn, self.face_model_path, self.enable_mesh),
                    daemon=False  # Not daemon to allow clean shutdown
                )
                worker.start()
                self.workers.append(worker)
            
            print(f"[Landmark Pool] Started {self.num_workers} workers")
    
    def stop(self):
        """Stop worker pool."""
        if self.is_running:
            self.is_running = False
            
            # Send stop signal to all workers
            for pipe in self.control_pipes:
                try:
                    pipe.send('stop')
                except:
                    pass
            
            # Add None to queue to unblock workers
            for _ in range(self.num_workers):
                try:
                    self.task_queue.put(None, timeout=0.1)
                except:
                    pass
            
            # Wait for workers
            for worker in self.workers:
                worker.join(timeout=2.0)
                if worker.is_alive():
                    worker.terminate()
                    worker.join(timeout=1.0)
            
            self.workers.clear()
            self.control_pipes.clear()
            
            # Clear queues
            while not self.task_queue.empty():
                try:
                    self.task_queue.get_nowait()
                except:
                    break
                    
            while not self.result_queue.empty():
                try:
                    self.result_queue.get_nowait()
                except:
                    break
            
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
            
            # Try to put task in queue (non-blocking)
            self.task_queue.put_nowait(task)
            
            with self.stats_lock:
                self.pending_tasks[self.next_task_id] = (track_id, frame_id)
                self.next_task_id += 1
                self.active_tasks += 1
                
            return True
            
        except Full:
            # Queue is full - this is expected when processing is slower than submission
            return False
        except Exception as e:
            print(f"[Landmark Pool] Error submitting task: {e}")
            return False
    
    def get_results(self, timeout: float = 0.001) -> List[LandmarkResult]:
        """Get available results."""
        results = []
        deadline = time.time() + timeout
        
        while time.time() < deadline:
            try:
                result = self.result_queue.get_nowait()
                results.append(result)
                
                # Update statistics
                with self.stats_lock:
                    self.active_tasks -= 1
                    if result.success:
                        self.completed_tasks += 1
                    else:
                        self.failed_tasks += 1
                    
                    # Remove from pending
                    if result.task_id in self.pending_tasks:
                        del self.pending_tasks[result.task_id]
                    
            except Empty:
                break
            except Exception as e:
                print(f"[Landmark Pool] Error getting result: {e}")
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
        with self.stats_lock:
            try:
                task_queue_size = self.task_queue.qsize()
            except:
                task_queue_size = -1
                
            try:
                result_queue_size = self.result_queue.qsize()
            except:
                result_queue_size = -1
                
            return {
                'num_workers': self.num_workers,
                'pending_tasks': len(self.pending_tasks),
                'active_tasks': self.active_tasks,
                'completed_tasks': self.completed_tasks,
                'failed_tasks': self.failed_tasks,
                'task_queue_size': task_queue_size,
                'result_queue_size': result_queue_size,
                'workers_alive': sum(1 for w in self.workers if w.is_alive())
            }
    
    def adjust_workers(self, target_faces: int):
        """
        Adjust worker count based on detected faces.
        This is a no-op in the current design as we use fixed workers with dynamic distribution.
        Future enhancement could implement worker hibernation/activation.
        """
        # For now, we keep all workers active and let them idle when no work
        # This avoids the expensive process spawn/kill cycle
        pass