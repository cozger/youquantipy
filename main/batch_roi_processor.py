"""
Batch ROI Processor for efficient GPU utilization.

This module processes multiple ROIs in batches to reduce GPU kernel launch overhead
and improve throughput.
"""

import numpy as np
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import threading
import queue

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    cp = None

from gpu_frame_cache import get_gpu_frame_cache


@dataclass
class BatchROIRequest:
    """Container for batch ROI extraction request."""
    frame: np.ndarray
    frame_id: Optional[int]
    rois: List[Dict]  # List of {'bbox': [...], 'track_id': ..., 'timestamp': ...}
    callback: Optional[callable] = None


class BatchROIProcessor:
    """
    Processes ROIs in batches for better GPU efficiency.
    """
    
    def __init__(self, target_size: Tuple[int, int] = (256, 256),
                 batch_size: int = 8, 
                 padding_percent: float = 25.0,
                 max_queue_size: int = 100):
        """
        Initialize batch ROI processor.
        
        Args:
            target_size: Target size for ROIs (width, height)
            batch_size: Number of ROIs to process in parallel
            padding_percent: Padding percentage for ROI extraction
            max_queue_size: Maximum queue size for requests
        """
        self.target_size = target_size
        self.batch_size = batch_size
        self.padding_percent = padding_percent
        
        # Request queue
        self.request_queue = queue.Queue(maxsize=max_queue_size)
        
        # Processing thread
        self.is_running = False
        self.processing_thread = None
        
        # Statistics
        self.stats = {
            'batches_processed': 0,
            'rois_processed': 0,
            'gpu_time_ms': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # GPU memory pool
        if HAS_CUPY:
            mempool = cp.get_default_memory_pool()
            mempool.set_mempool_allocator(cp.cuda.MemoryPool(cp.cuda.malloc_managed))
        
        self.enabled = HAS_CUPY
        
    def start(self):
        """Start the batch processing thread."""
        if not self.enabled:
            print("[BatchROIProcessor] CuPy not available, processor disabled")
            return
            
        if self.is_running:
            return
            
        self.is_running = True
        self.processing_thread = threading.Thread(
            target=self._processing_loop,
            name="BatchROIProcessor",
            daemon=True
        )
        self.processing_thread.start()
        print(f"[BatchROIProcessor] Started with batch_size={self.batch_size}")
    
    def stop(self):
        """Stop the batch processing thread."""
        self.is_running = False
        
        # Send stop signal
        try:
            self.request_queue.put(None, timeout=0.1)
        except:
            pass
            
        if self.processing_thread:
            self.processing_thread.join(timeout=1.0)
            
        # Print statistics
        print(f"[BatchROIProcessor] Stopped. Stats: {self.stats}")
    
    def process_batch(self, request: BatchROIRequest):
        """
        Add a batch request to the processing queue.
        
        Args:
            request: BatchROIRequest object
        """
        if not self.enabled:
            return
            
        try:
            self.request_queue.put(request, timeout=0.01)
        except queue.Full:
            print("[BatchROIProcessor] Queue full, dropping request")
    
    def _processing_loop(self):
        """Main processing loop for batch ROI extraction."""
        cache = get_gpu_frame_cache()
        
        while self.is_running:
            try:
                # Get request
                request = self.request_queue.get(timeout=0.1)
                if request is None:
                    break
                    
                start_time = time.time()
                
                # Get GPU frame from cache or transfer
                gpu_frame = None
                if request.frame_id is not None:
                    gpu_frame = cache.get(request.frame_id)
                    if gpu_frame is not None:
                        self.stats['cache_hits'] += 1
                    else:
                        self.stats['cache_misses'] += 1
                
                if gpu_frame is None:
                    # Transfer frame to GPU
                    if not request.frame.flags['C_CONTIGUOUS']:
                        request.frame = np.ascontiguousarray(request.frame)
                    gpu_frame = cp.asarray(request.frame)
                    
                    # Cache for future use
                    if request.frame_id is not None:
                        cache.put(request.frame_id, gpu_frame)
                
                # Process ROIs in batches
                results = []
                frame_h, frame_w = gpu_frame.shape[:2]
                
                for i in range(0, len(request.rois), self.batch_size):
                    batch_rois = request.rois[i:i + self.batch_size]
                    batch_results = self._process_roi_batch(gpu_frame, batch_rois, frame_w, frame_h)
                    results.extend(batch_results)
                
                # Update statistics
                gpu_time = (time.time() - start_time) * 1000
                self.stats['gpu_time_ms'] += gpu_time
                self.stats['batches_processed'] += 1
                self.stats['rois_processed'] += len(request.rois)
                
                # Call callback if provided
                if request.callback:
                    request.callback(results)
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[BatchROIProcessor] Error in processing loop: {e}")
                import traceback
                traceback.print_exc()
    
    def _process_roi_batch(self, gpu_frame: cp.ndarray, rois: List[Dict], 
                          frame_w: int, frame_h: int) -> List[Dict]:
        """
        Process a batch of ROIs on GPU.
        
        Args:
            gpu_frame: Frame on GPU
            rois: List of ROI dictionaries
            frame_w: Frame width
            frame_h: Frame height
            
        Returns:
            List of processed ROI results
        """
        results = []
        
        # Allocate batch array on GPU
        batch_size = len(rois)
        batch_array = cp.zeros((batch_size, self.target_size[1], self.target_size[0], 3), 
                              dtype=cp.float32)
        
        # Process each ROI
        valid_indices = []
        for idx, roi_info in enumerate(rois):
            bbox = roi_info['bbox']
            
            # Calculate padded bbox
            x1, y1, x2, y2 = bbox
            w = x2 - x1
            h = y2 - y1
            
            # Add padding
            pad_w = w * self.padding_percent / 100
            pad_h = h * self.padding_percent / 100
            
            # Padded coordinates
            px1 = int(max(0, x1 - pad_w))
            py1 = int(max(0, y1 - pad_h))
            px2 = int(min(frame_w, x2 + pad_w))
            py2 = int(min(frame_h, y2 + pad_h))
            
            if px2 <= px1 or py2 <= py1:
                continue
                
            # Extract ROI
            roi_gpu = gpu_frame[py1:py2, px1:px2]
            
            # Resize to target size
            roi_resized = self._resize_gpu_batch(roi_gpu, self.target_size)
            
            # Normalize to [-1, 1]
            roi_normalized = (roi_resized.astype(cp.float32) - 127.5) / 127.5
            
            # Add to batch
            batch_array[idx] = roi_normalized
            valid_indices.append(idx)
            
            # Prepare result
            results.append({
                'roi': cp.asnumpy(roi_normalized),
                'track_id': roi_info['track_id'],
                'timestamp': roi_info['timestamp'],
                'bbox': bbox,
                'padded_bbox': [px1, py1, px2, py2],
                'transform': {
                    'scale': (px2 - px1) / self.target_size[0],
                    'offset_x': px1,
                    'offset_y': py1
                }
            })
        
        return results
    
    def _resize_gpu_batch(self, roi: cp.ndarray, target_size: Tuple[int, int]) -> cp.ndarray:
        """
        Resize ROI on GPU using CuPy.
        
        Args:
            roi: ROI array on GPU
            target_size: Target size (width, height)
            
        Returns:
            Resized ROI
        """
        # Use CuPy's resize function
        h, w = roi.shape[:2]
        target_w, target_h = target_size
        
        # Calculate scaling factors
        scale_x = target_w / w
        scale_y = target_h / h
        
        # Create output array
        output = cp.zeros((target_h, target_w, 3), dtype=roi.dtype)
        
        # Perform bilinear interpolation
        y_coords = cp.arange(target_h) / scale_y
        x_coords = cp.arange(target_w) / scale_x
        
        y_int = y_coords.astype(cp.int32)
        x_int = x_coords.astype(cp.int32)
        
        y_frac = y_coords - y_int
        x_frac = x_coords - x_int
        
        # Clip coordinates
        y_int = cp.clip(y_int, 0, h - 2)
        x_int = cp.clip(x_int, 0, w - 2)
        
        # Bilinear interpolation
        for c in range(3):
            tl = roi[y_int[:, None], x_int[None, :], c]
            tr = roi[y_int[:, None], x_int[None, :] + 1, c]
            bl = roi[y_int[:, None] + 1, x_int[None, :], c]
            br = roi[y_int[:, None] + 1, x_int[None, :] + 1, c]
            
            top = tl * (1 - x_frac[None, :]) + tr * x_frac[None, :]
            bottom = bl * (1 - x_frac[None, :]) + br * x_frac[None, :]
            
            output[:, :, c] = top * (1 - y_frac[:, None]) + bottom * y_frac[:, None]
        
        return output