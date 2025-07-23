"""
Integration layer for the new GPU pipeline with existing YouQuantiPy components.

CRITICAL: This shows how to replace the existing multiprocess architecture
with the new GPU-first design while maintaining compatibility.
"""

import threading
import multiprocessing as mp
from queue import Queue
import time
from typing import Dict, Optional

from gpu_frame_distributor import GPUFrameDistributor
from gpu_face_processor import GPUFaceProcessor

class GPUPipelineIntegration:
    """
    Integrates GPU pipeline with existing YouQuantiPy components.
    
    Key changes from original:
    1. Face processing happens in thread (not process)
    2. Frame distributor is GPU-first
    3. Only final results cross process boundaries
    """
    
    def __init__(self, camera_index: int, config: Dict):
        self.camera_index = camera_index
        self.config = config
        
        # Output queues for existing components
        self.preview_queue = mp.Queue(maxsize=2)  # For GUI
        self.pose_queue = mp.Queue(maxsize=2)     # For pose process
        self.lsl_queue = mp.Queue(maxsize=10)     # For LSL streaming
        
        # Internal queues (thread-safe, not multiprocess)
        self.face_results_queue = Queue(maxsize=10)
        
        # Components
        self.distributor = None
        self.face_processor = None
        self.result_forwarder_thread = None
        
    def start(self):
        """Start the GPU pipeline"""
        print(f"[GPU Pipeline] Starting for camera {self.camera_index}")
        
        # Create frame distributor
        self.distributor = GPUFrameDistributor(
            camera_index=self.camera_index,
            resolution=(1920, 1080),
            fps=30
        )
        
        # Create face processor
        self.face_processor = GPUFaceProcessor(
            retinaface_engine=self.config['retinaface_trt_path'],
            landmark_engine=self.config['landmark_trt_path'],
            max_faces=self.config.get('max_participants', 6),
            confidence_threshold=self.config.get('detection_confidence', 0.9)
        )
        
        # Add GPU stream for face processing
        self.distributor.add_gpu_stream(
            name='face',
            gpu_processor=self.face_processor.process_frame,
            output_queue=self.face_results_queue
        )
        
        # Add CPU streams
        self.distributor.add_cpu_stream(
            name='preview',
            output_queue=self.preview_queue,
            resolution=(960, 540),
            format='bgr'
        )
        
        self.distributor.add_cpu_stream(
            name='pose',
            output_queue=self.pose_queue,
            resolution=(640, 480),
            format='rgb'
        )
        
        # Start result forwarder thread
        self.result_forwarder_thread = threading.Thread(
            target=self._forward_results,
            name="ResultForwarder",
            daemon=True
        )
        self.result_forwarder_thread.start()
        
        # Start frame distribution
        self.distributor.start()
        
        print("[GPU Pipeline] Started successfully")
    
    def _forward_results(self):
        """
        Forward face results to multiprocess queues.
        This is the bridge between thread-based GPU processing
        and process-based downstream components.
        """
        while True:
            try:
                # Get face results from GPU pipeline
                result = self.face_results_queue.get(timeout=1.0)
                
                # Forward to LSL queue
                if not self.lsl_queue.full():
                    lsl_data = {
                        'type': 'face_data',
                        'camera_index': self.camera_index,
                        'faces': result['faces'],
                        'timestamp': result['timestamp'],
                        'frame_id': result['frame_id']
                    }
                    self.lsl_queue.put(lsl_data)
                
                # Also update preview queue with face data
                # (Preview frames are sent separately by distributor)
                
            except:
                continue
    
    def stop(self):
        """Stop the GPU pipeline"""
        print("[GPU Pipeline] Stopping...")
        
        if self.distributor:
            self.distributor.stop()
        
        print("[GPU Pipeline] Stopped")

# Integration with existing parallel_participant_worker
def create_gpu_worker(cam_idx: int, config: Dict, 
                     preview_queue: mp.Queue, lsl_queue: mp.Queue) -> GPUPipelineIntegration:
    """
    Create GPU worker to replace existing parallel_participant_worker.
    
    This is the main integration point - replace the complex multiprocess
    worker with this simpler GPU pipeline.
    """
    pipeline = GPUPipelineIntegration(cam_idx, config)
    
    # Connect to existing queues
    pipeline.preview_queue = preview_queue
    pipeline.lsl_queue = lsl_queue
    
    # Start pipeline
    pipeline.start()
    
    return pipeline

# Example of how to modify the main GUI code
def modify_gui_for_gpu_pipeline():
    """
    Example modifications needed in gui.py:
    
    1. Replace parallel_participant_worker with create_gpu_worker
    2. Remove face_frame_queue, face_result_queue (internal to GPU pipeline)
    3. Keep preview_queue, pose_queue for existing components
    4. Simplify participant management (no fusion process needed)
    """
    pass