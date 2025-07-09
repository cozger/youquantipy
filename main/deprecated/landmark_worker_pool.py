"""
Landmark Worker Pool
Pool of processes for parallel landmark detection on ROIs
"""

import multiprocessing as mp
from multiprocessing import Process, Queue, Pipe
import mediapipe as mp_module
import numpy as np
from typing import Dict, List, Optional, Tuple
import time
from dataclasses import dataclass

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
    success: bool
    processing_time: float

def landmark_worker_process(worker_id: int, 
                          task_queue: Queue,
                          result_queue: Queue,
                          control_pipe,
                          face_model_path: str,
                          enable_mesh: bool = False):
    """
    Individual landmark worker process.
    Runs MediaPipe on single face ROIs.
    """
    print(f"[Landmark Worker {worker_id}] Starting")
    
    # Initialize MediaPipe
    BaseOptions = mp_module.tasks.BaseOptions
    FaceLandmarker = mp_module.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp_module.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp_module.tasks.vision.RunningMode
    
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=face_model_path),
        running_mode=VisionRunningMode.VIDEO,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=1  # Single face per ROI
    )
    
    face_landmarker = FaceLandmarker.create_from_options(options)
    
    # Timestamp generator for MediaPipe
    class TimestampGenerator:
        def __init__(self):
            self.timestamp = 0
        def next(self):
            self.timestamp += 1
            return self.timestamp
    
    ts_gen = TimestampGenerator()
    task_count = 0
    
    try:
        while True:
            # Check for control commands
            if control_pipe.poll():
                cmd = control_pipe.recv()
                if cmd == 'stop':
                    break
                elif cmd == 'toggle_mesh':
                    enable_mesh = not enable_mesh
            
            # Get task
            try:
                task = task_queue.get(timeout=0.1)
            except:
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
                timestamp_ms = ts_gen.next()
                face_result = face_landmarker.detect_for_video(mp_image, timestamp_ms)
                
                # Process results
                if face_result.face_landmarks and len(face_result.face_landmarks) > 0:
                    # Get first (and should be only) face
                    face_landmarks = face_result.face_landmarks[0]
                    
                    # Convert to numpy array
                    landmarks = np.array([(lm.x, lm.y, lm.z) for lm in face_landmarks])
                    
                    # Get blendshapes
                    blendshapes = None
                    if face_result.face_blendshapes and len(face_result.face_blendshapes) > 0:
                        blendshapes = [b.score for b in face_result.face_blendshapes[0]]
                    
                    result = LandmarkResult(
                        track_id=task.track_id,
                        frame_id=task.frame_id,
                        task_id=task.task_id,
                        landmarks=landmarks,
                        blendshapes=blendshapes,
                        success=True,
                        processing_time=time.time() - start_time
                    )
                else:
                    # No face detected in ROI
                    result = LandmarkResult(
                        track_id=task.track_id,
                        frame_id=task.frame_id,
                        task_id=task.task_id,
                        landmarks=None,
                        blendshapes=None,
                        success=False,
                        processing_time=time.time() - start_time
                    )
                
                # Send result
                result_queue.put(result)
                task_count += 1
                
                if task_count % 100 == 0:
                    print(f"[Landmark Worker {worker_id}] Processed {task_count} ROIs")
                    
            except Exception as e:
                print(f"[Landmark Worker {worker_id}] Error: {e}")
                # Send failure result
                result = LandmarkResult(
                    track_id=task.track_id,
                    frame_id=task.frame_id,
                    task_id=task.task_id,
                    landmarks=None,
                    blendshapes=None,
                    success=False,
                    processing_time=time.time() - start_time
                )
                result_queue.put(result)
    
    finally:
        face_landmarker.close()
        print(f"[Landmark Worker {worker_id}] Stopped")

class LandmarkWorkerPool:
    """Manages a pool of landmark detection workers."""
    
    def __init__(self, face_model_path: str, num_workers: int = 4, enable_mesh: bool = False):
        """
        Initialize landmark worker pool.
        
        Args:
            face_model_path: Path to MediaPipe face model
            num_workers: Number of worker processes
            enable_mesh: Enable mesh data extraction
        """
        self.face_model_path = face_model_path
        self.num_workers = num_workers
        self.enable_mesh = enable_mesh
        
        # Queues
        self.task_queue = mp.Queue(maxsize=num_workers * 10)
        self.result_queue = mp.Queue(maxsize=num_workers * 10)
        
        # Workers
        self.workers = []
        self.control_pipes = []
        
        # Task tracking
        self.next_task_id = 0
        self.pending_tasks = {}
        
        # Running state
        self.is_running = False
        
    def start(self):
        """Start worker pool."""
        if not self.is_running:
            self.is_running = True
            
            # Start workers
            for i in range(self.num_workers):
                parent_conn, child_conn = mp.Pipe()
                self.control_pipes.append(parent_conn)
                
                worker = Process(
                    target=landmark_worker_process,
                    args=(i, self.task_queue, self.result_queue, 
                          child_conn, self.face_model_path, self.enable_mesh)
                )
                worker.daemon = True
                worker.start()
                self.workers.append(worker)
            
            print(f"[Landmark Pool] Started {self.num_workers} workers")
    
    def stop(self):
        """Stop worker pool."""
        if self.is_running:
            self.is_running = False
            
            # Send stop signal to all workers
            for pipe in self.control_pipes:
                pipe.send('stop')
            
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
            
            self.workers.clear()
            self.control_pipes.clear()
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
            
            self.task_queue.put_nowait(task)
            self.pending_tasks[self.next_task_id] = (track_id, frame_id)
            self.next_task_id += 1
            return True
            
        except mp.queues.Full:
            return False
    
    def get_results(self, timeout: float = 0.001) -> List[LandmarkResult]:
        """Get available results."""
        results = []
        deadline = time.time() + timeout
        
        while time.time() < deadline:
            try:
                result = self.result_queue.get_nowait()
                results.append(result)
                
                # Remove from pending
                if result.task_id in self.pending_tasks:
                    del self.pending_tasks[result.task_id]
                    
            except:
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
            'task_queue_size': self.task_queue.qsize(),
            'result_queue_size': self.result_queue.qsize()
        }