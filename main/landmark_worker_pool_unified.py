"""
Landmark Worker Pool for parallel landmark detection on ROIs.
Pool of processes for parallel landmark detection using enhanced mode.
"""

import multiprocessing as mp
from multiprocessing import Process, Queue, Pipe
import mediapipe as mp_module
import numpy as np
from typing import Dict, List, Optional, Tuple
import time
from dataclasses import dataclass
import threading
import queue
from queue import Empty, Full
import cv2

# Import MediaPipe components
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

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
    landmarks: Optional[np.ndarray]  # Nx3 array (x, y, z)
    blendshapes: Optional[List[float]]
    mesh_data: Optional[np.ndarray] = None  # Full mesh if enabled
    success: bool = True
    processing_time: float = 0.0
    error: Optional[str] = None

def landmark_worker_process(worker_id: int, 
                          task_queue: Queue,
                          result_queue: Queue,
                          control_pipe,
                          face_model_path: str,
                          enable_mesh: bool = False):
    """
    Individual landmark worker process.
    Runs MediaPipe on single face ROIs.
    
    Args:
        worker_id: Worker identifier
        task_queue: Queue for receiving tasks
        result_queue: Queue for sending results
        control_pipe: Pipe for control messages
        face_model_path: Path to face landmarker model
        enable_mesh: Whether to output mesh data
    """
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
    
    task_count = 0
    idle_count = 0
    max_idle_iterations = 100
    
    try:
        while True:
            # Check for control commands
            if control_pipe.poll(0):
                cmd = control_pipe.recv()
                if cmd == 'stop':
                    break
                elif cmd == 'toggle_mesh' or cmd == ('set_mesh', True):
                    enable_mesh = True
                elif cmd == ('set_mesh', False):
                    enable_mesh = False
            
            # Get task
            try:
                task = task_queue.get(timeout=0.1)
                idle_count = 0  # Reset idle counter
            except:
                idle_count += 1
                if idle_count > max_idle_iterations:
                    # Consider shutting down after being idle
                    print(f"[Landmark Worker {worker_id}] Idle for too long, considering shutdown")
                continue
            
            if task is None:
                break
            
            start_time = time.time()
            
            try:
                # Create MediaPipe image
                mp_image = mp_module.Image(
                    image_format=mp_module.ImageFormat.SRGB, 
                    data=task.roi_image
                )
                
                # Detect landmarks
                detection_result = face_landmarker.detect(mp_image)
                
                if detection_result.face_landmarks:
                    # Extract landmarks (first face only)
                    face_landmarks = detection_result.face_landmarks[0]
                    
                    # Convert to numpy array
                    landmarks = np.array([[lm.x, lm.y, lm.z] for lm in face_landmarks])
                    
                    # Extract blendshapes
                    blendshapes = None
                    if detection_result.face_blendshapes:
                        blend_list = detection_result.face_blendshapes[0]
                        blendshapes = np.array([b.score for b in blend_list[:52]], dtype=np.float32)
                    
                    # Mesh data if enabled
                    mesh_data = None
                    if enable_mesh:
                        mesh_data = landmarks.flatten()  # Flatten for transmission
                    
                    # Create result
                    result = LandmarkResult(
                        track_id=task.track_id,
                        frame_id=task.frame_id,
                        task_id=task.task_id,
                        landmarks=landmarks,
                        blendshapes=blendshapes,
                        mesh_data=mesh_data,
                        success=True,
                        processing_time=time.time() - start_time
                    )
                else:
                    # No face detected
                    result = LandmarkResult(
                        track_id=task.track_id,
                        frame_id=task.frame_id,
                        task_id=task.task_id,
                        landmarks=None,
                        blendshapes=None,
                        success=False,
                        processing_time=time.time() - start_time,
                        error="No face detected"
                    )
                
                # Send result
                result_queue.put(result)
                task_count += 1
                
                # Progress reporting
                if task_count % 100 == 0:
                    print(f"[Landmark Worker {worker_id}] Processed {task_count} tasks")
                
            except Exception as e:
                # Error result
                result = LandmarkResult(
                    track_id=task.track_id,
                    frame_id=task.frame_id,
                    task_id=task.task_id,
                    landmarks=None,
                    blendshapes=None,
                    success=False,
                    processing_time=time.time() - start_time,
                    error=str(e)
                )
                result_queue.put(result)
                print(f"[Landmark Worker {worker_id}] Error processing task: {e}")
    
    except Exception as e:
        print(f"[Landmark Worker {worker_id}] Fatal error: {e}")
    
    finally:
        print(f"[Landmark Worker {worker_id}] Shutting down after {task_count} tasks")


class LandmarkWorkerPool:
    """
    Pool of landmark detection workers using enhanced mode.
    """
    def __init__(self, num_workers: int = 4, face_model_path: str = None,
                 enable_mesh: bool = False,
                 max_queue_size: int = 100):
        """
        Initialize landmark worker pool.
        
        Args:
            num_workers: Number of worker processes
            face_model_path: Path to face landmarker model
            enable_mesh: Whether to output mesh data
            max_queue_size: Maximum size of task/result queues
        """
        self.num_workers = num_workers
        self.face_model_path = face_model_path
        self.enable_mesh = enable_mesh
        self.max_queue_size = max_queue_size
        
        # Queues and processes
        self.task_queue = mp.Queue(maxsize=max_queue_size)
        self.result_queue = mp.Queue(maxsize=max_queue_size * 2)
        self.workers = []
        self.control_pipes = []
        
        # State
        self.running = False
        self.task_counter = 0
        self._lock = threading.Lock()
        
        # Stats
        self.stats = {
            'tasks_submitted': 0,
            'tasks_completed': 0,
            'tasks_failed': 0,
            'avg_processing_time': 0.0
        }
        
        # Result collection thread
        self.result_thread = None
        self.result_callbacks = {}
        
    def start(self):
        """Start the worker pool."""
        if self.running:
            return
            
        print(f"[LandmarkWorkerPool] Starting {self.num_workers} workers")
        
        self.running = True
        
        # Start worker processes
        for i in range(self.num_workers):
            parent_pipe, child_pipe = mp.Pipe()
            worker = mp.Process(
                target=landmark_worker_process,
                args=(i, self.task_queue, self.result_queue, child_pipe,
                      self.face_model_path, self.enable_mesh),
                daemon=False
            )
            worker.start()
            self.workers.append(worker)
            self.control_pipes.append(parent_pipe)
        
        # Start result collection thread
        self.result_thread = threading.Thread(
            target=self._result_collector_thread,
            daemon=True
        )
        self.result_thread.start()
        
        print(f"[LandmarkWorkerPool] Started successfully")
    
    def stop(self):
        """Stop the worker pool."""
        if not self.running:
            return
            
        print(f"[LandmarkWorkerPool] Stopping...")
        self.running = False
        
        # Send stop signal to all workers
        for pipe in self.control_pipes:
            try:
                pipe.send('stop')
            except:
                pass
        
        # Send None to task queue to unblock workers
        for _ in range(self.num_workers):
            try:
                self.task_queue.put(None, timeout=0.1)
            except:
                pass
        
        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=2.0)
            if worker.is_alive():
                worker.terminate()
                worker.join()
        
        # Clear queues
        self._clear_queue(self.task_queue)
        self._clear_queue(self.result_queue)
        
        # Stop result thread
        if self.result_thread and self.result_thread.is_alive():
            self.result_thread.join(timeout=1.0)
        
        self.workers.clear()
        self.control_pipes.clear()
        
        print(f"[LandmarkWorkerPool] Stopped")
    
    def submit_task(self, roi_image: np.ndarray, track_id: int, 
                    frame_id: int, callback=None) -> bool:
        """
        Submit a landmark detection task.
        
        Args:
            roi_image: Face ROI image
            track_id: Track identifier
            frame_id: Frame identifier
            callback: Optional callback for result (adaptive mode)
            
        Returns:
            True if task was submitted, False if queue is full
        """
        if not self.running:
            return False
        
        with self._lock:
            task_id = self.task_counter
            self.task_counter += 1
        
        task = LandmarkTask(
            roi_image=roi_image,
            track_id=track_id,
            frame_id=frame_id,
            task_id=task_id
        )
        
        try:
            self.task_queue.put_nowait(task)
            self.stats['tasks_submitted'] += 1
            
            # Register callback if provided
            if callback:
                self.result_callbacks[task_id] = callback
            
            return True
        except Full:
            return False
    
    def get_result(self, timeout: float = 0.001) -> Optional[LandmarkResult]:
        """
        Get a result from the pool.
        Note: This is primarily for backward compatibility as callbacks are preferred.
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            LandmarkResult or None if no result available
        """
        try:
            result = self.result_queue.get(timeout=timeout)
            self._update_stats(result)
            return result
        except Empty:
            return None
    
    def get_all_results(self, max_results: int = 100) -> List[LandmarkResult]:
        """Get all available results."""
        results = []
        for _ in range(max_results):
            result = self.get_result(timeout=0.001)
            if result is None:
                break
            results.append(result)
        return results
    
    def set_mesh_enabled(self, enabled: bool):
        """Enable or disable mesh output."""
        self.enable_mesh = enabled
        for pipe in self.control_pipes:
            try:
                pipe.send(('set_mesh', enabled))
            except:
                pass
    
    def get_stats(self) -> Dict:
        """Get pool statistics."""
        return self.stats.copy()
    
    def _result_collector_thread(self):
        """Collect results and invoke callbacks."""
        while self.running:
            try:
                result = self.result_queue.get(timeout=0.1)
                self._update_stats(result)
                
                # Invoke callback if registered
                callback = self.result_callbacks.pop(result.task_id, None)
                if callback:
                    try:
                        callback(result)
                    except Exception as e:
                        print(f"[LandmarkWorkerPool] Callback error: {e}")
                        
            except Empty:
                continue
            except Exception as e:
                print(f"[LandmarkWorkerPool] Result collector error: {e}")
    
    def _update_stats(self, result: LandmarkResult):
        """Update statistics based on result."""
        if result.success:
            self.stats['tasks_completed'] += 1
        else:
            self.stats['tasks_failed'] += 1
        
        # Update average processing time
        if self.stats['tasks_completed'] > 0:
            alpha = 0.1  # Exponential moving average
            self.stats['avg_processing_time'] = (
                (1 - alpha) * self.stats['avg_processing_time'] +
                alpha * result.processing_time
            )
    
    def _clear_queue(self, q):
        """Clear a queue."""
        try:
            while True:
                q.get_nowait()
        except Empty:
            pass


# Backward compatibility alias
LandmarkWorkerPoolStandard = LandmarkWorkerPool
LandmarkWorkerPoolAdaptive = LandmarkWorkerPool