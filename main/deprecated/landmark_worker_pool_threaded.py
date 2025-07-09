"""
Threaded version of Landmark Worker Pool to avoid daemon process issues.
Uses threading instead of multiprocessing for landmark detection workers.
"""

import cv2
import numpy as np
import time
import threading
from queue import Queue, Empty, Full
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import mediapipe as mp
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
    landmarks: np.ndarray  # 478x3 for face
    blendshapes: Optional[np.ndarray] = None  # 52 blend scores
    mesh_data: Optional[np.ndarray] = None  # Full mesh if enabled
    processing_time: float = 0.0
    error: Optional[str] = None
    success: bool = True  # Added for compatibility

class LandmarkWorkerThread:
    """Thread-based landmark detection worker."""
    
    def __init__(self, worker_id: int, task_queue: Queue, result_queue: Queue,
                 face_model_path: str, enable_mesh: bool = False):
        self.worker_id = worker_id
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.face_model_path = face_model_path
        self.enable_mesh = enable_mesh
        self.running = False
        self.thread = None
        self.face_landmarker = None
        
    def start(self):
        """Start the worker thread."""
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        
    def stop(self):
        """Stop the worker thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
            
    def _run(self):
        """Worker thread main loop."""
        print(f"[Landmark Worker {self.worker_id}] Starting")
        
        # Initialize MediaPipe
        try:
            base_options = python.BaseOptions(model_asset_path=self.face_model_path)
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
            self.face_landmarker = vision.FaceLandmarker.create_from_options(options)
        except Exception as e:
            print(f"[Landmark Worker {self.worker_id}] Failed to initialize: {e}")
            return
            
        # Process tasks
        while self.running:
            try:
                # Get task with timeout
                task = self.task_queue.get(timeout=0.1)
                if task is None:  # Shutdown signal
                    break
                    
                # Process ROI
                start_time = time.time()
                try:
                    # Convert to MediaPipe image
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=task.roi_image)
                    
                    # Detect landmarks
                    detection_result = self.face_landmarker.detect(mp_image)
                    
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
                        if self.enable_mesh:
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
                        self.result_queue.put_nowait(result)
                    except Full:
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
                        self.result_queue.put_nowait(result)
                    except Full:
                        pass
                        
            except Empty:
                continue
            except Exception as e:
                print(f"[Landmark Worker {self.worker_id}] Error: {e}")
                
        # Cleanup
        if self.face_landmarker:
            self.face_landmarker.close()
        print(f"[Landmark Worker {self.worker_id}] Stopped")

class LandmarkWorkerPoolThreaded:
    """Thread-based pool of landmark detection workers."""
    
    def __init__(self, face_model_path: str, num_workers: int = 4, 
                 enable_mesh: bool = False, max_queue_size: int = 100):
        self.face_model_path = face_model_path
        self.num_workers = num_workers
        self.enable_mesh = enable_mesh
        
        # Queues
        self.task_queue = Queue(maxsize=max_queue_size)
        self.result_queue = Queue(maxsize=max_queue_size * 2)
        
        # Workers
        self.workers = []
        self.is_running = False
        
        # Task tracking
        self.next_task_id = 0
        self.pending_tasks = {}
        
    def start(self):
        """Start worker pool."""
        if not self.is_running:
            self.is_running = True
            
            # Create and start workers
            for i in range(self.num_workers):
                worker = LandmarkWorkerThread(
                    i, self.task_queue, self.result_queue,
                    self.face_model_path, self.enable_mesh
                )
                worker.start()
                self.workers.append(worker)
                
            print(f"[Landmark Pool] Started {self.num_workers} workers")
            
    def stop(self):
        """Stop worker pool."""
        if self.is_running:
            self.is_running = False
            
            # Send stop signal to all workers
            for _ in range(self.num_workers):
                try:
                    self.task_queue.put(None, timeout=0.1)
                except:
                    pass
                    
            # Stop all workers
            for worker in self.workers:
                worker.stop()
                
            self.workers.clear()
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
            
        except Full:
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
                    
            except Empty:
                break
                
        return results
        
    def set_mesh_enabled(self, enabled: bool):
        """Toggle mesh extraction for all workers."""
        self.enable_mesh = enabled
        # Note: Would need to restart workers to take effect
        
    def get_stats(self) -> Dict:
        """Get pool statistics."""
        return {
            'num_workers': self.num_workers,
            'pending_tasks': len(self.pending_tasks),
            'task_queue_size': self.task_queue.qsize(),
            'result_queue_size': self.result_queue.qsize()
        }