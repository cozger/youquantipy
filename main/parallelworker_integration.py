"""
Seamless integration of advanced face detection into existing parallelworker.
This module provides a drop-in replacement that automatically uses advanced
detection when high-resolution video is detected.
"""

import cv2
import numpy as np
from parallelworker import face_worker_process, pose_worker_process
from parallelworker_advanced import face_worker_advanced_process, ParallelWorkerAdvanced
from multiprocessing import Process, Queue as MPQueue, Pipe
import os
import json
from confighandler import ConfigHandler

class ParallelWorkerAuto:
    """
    Automatic worker selection based on resolution and configuration.
    Drop-in replacement for ParallelWorker that seamlessly upgrades to
    advanced detection when needed.
    """
    
    def __init__(self, participant_index, face_model_path, pose_model_path=None):
        self.participant_index = participant_index
        self.face_model_path = face_model_path
        self.pose_model_path = pose_model_path
        
        # Load configuration
        self.config = self._load_config()
        
        # Determine if advanced detection should be used
        self.use_advanced = self._should_use_advanced()
        
        # Create appropriate worker
        if self.use_advanced:
            print(f"[ParallelWorker] Using advanced detection for participant {participant_index}")
            self._create_advanced_worker()
        else:
            print(f"[ParallelWorker] Using standard detection for participant {participant_index}")
            self._create_standard_worker()
    
    def _load_config(self):
        """Load configuration from file"""
        try:
            config_handler = ConfigHandler()
            return config_handler.config
        except:
            return {}
    
    def _should_use_advanced(self):
        """Determine if advanced detection should be used"""
        # Check if face recognition is enabled
        startup_mode = self.config.get('startup_mode', {})
        if not startup_mode.get('enable_face_recognition', False):
            return False
        
        # Check if models exist
        advanced_config = self.config.get('advanced_detection', {})
        retinaface_model = advanced_config.get('retinaface_model')
        
        if retinaface_model and os.path.exists(retinaface_model):
            return True
        
        return False
    
    def _create_advanced_worker(self):
        """Create advanced worker with all features"""
        advanced_config = self.config.get('advanced_detection', {})
        
        self.worker = ParallelWorkerAdvanced(
            participant_index=self.participant_index,
            face_model_path=self.face_model_path,
            pose_model_path=self.pose_model_path,
            retinaface_model_path=advanced_config.get('retinaface_model'),
            arcface_model_path=advanced_config.get('arcface_model'),
            enable_recognition=True
        )
    
    def _create_standard_worker(self):
        """Create standard worker for compatibility"""
        # Mimic the standard ParallelWorker structure
        self.frame_queue = MPQueue(maxsize=5)
        self.result_queue = MPQueue(maxsize=10)
        
        self.face_control_send, self.face_control_recv = Pipe()
        self.pose_control_send, self.pose_control_recv = Pipe()
        
        self.face_process = None
        self.pose_process = None
        
        self.running = False
        self.enable_mesh = False
        self.enable_pose = True
        
        # Create a wrapper to match advanced worker interface
        self.worker = self
    
    # Delegate all methods to the appropriate worker
    def start(self):
        if hasattr(self.worker, 'start'):
            self.worker.start()
        else:
            # Standard worker start logic
            if not self.running:
                self.running = True
                
                self.face_process = Process(
                    target=face_worker_process,
                    args=(self.frame_queue, self.result_queue, self.face_model_path,
                          self.face_control_recv)
                )
                self.face_process.daemon = True
                self.face_process.start()
                
                if self.pose_model_path and self.enable_pose:
                    self.pose_process = Process(
                        target=pose_worker_process,
                        args=(self.frame_queue, self.result_queue, self.pose_model_path,
                              self.pose_control_recv)
                    )
                    self.pose_process.daemon = True
                    self.pose_process.start()
    
    def stop(self):
        if hasattr(self.worker, 'stop'):
            self.worker.stop()
        else:
            # Standard worker stop logic
            if self.running:
                self.running = False
                
                self.face_control_send.send('stop')
                if self.pose_process:
                    self.pose_control_send.send('stop')
                
                if self.face_process:
                    self.face_process.join(timeout=2.0)
                    if self.face_process.is_alive():
                        self.face_process.terminate()
                
                if self.pose_process:
                    self.pose_process.join(timeout=2.0)
                    if self.pose_process.is_alive():
                        self.pose_process.terminate()
    
    def submit_frame(self, frame_data):
        if hasattr(self.worker, 'submit_frame'):
            return self.worker.submit_frame(frame_data)
        else:
            if self.running:
                try:
                    self.frame_queue.put_nowait(frame_data)
                    return True
                except:
                    return False
            return False
    
    def get_results(self, timeout=0.001):
        if hasattr(self.worker, 'get_results'):
            return self.worker.get_results(timeout)
        else:
            results = []
            import time
            deadline = time.time() + timeout
            
            while time.time() < deadline:
                try:
                    result = self.result_queue.get_nowait()
                    results.append(result)
                except:
                    break
            
            return results
    
    def set_mesh_enabled(self, enabled):
        if hasattr(self.worker, 'set_mesh_enabled'):
            self.worker.set_mesh_enabled(enabled)
        else:
            self.enable_mesh = enabled
            self.face_control_send.send(('set_mesh', enabled))
    
    def set_pose_enabled(self, enabled):
        if hasattr(self.worker, 'set_pose_enabled'):
            self.worker.set_pose_enabled(enabled)
        else:
            self.enable_pose = enabled
            if self.pose_process:
                self.pose_control_send.send(('enable_pose', enabled))
    
    def get_stats(self):
        if hasattr(self.worker, 'get_stats'):
            return self.worker.get_stats()
        else:
            return {}


def create_parallel_worker(participant_index, face_model_path, pose_model_path=None):
    """
    Factory function to create appropriate parallel worker.
    This can be used as a drop-in replacement for ParallelWorker creation.
    """
    return ParallelWorkerAuto(participant_index, face_model_path, pose_model_path)