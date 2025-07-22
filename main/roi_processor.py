import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
import time
import cupy as cp
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    print("WARNING: CuPy not installed, GPU cropping will fall back to CPU")
    HAS_CUPY = False

try:
    # For NVIDIA GPU acceleration with OpenCV
    cv2.cuda.getCudaEnabledDeviceCount()
    HAS_OPENCV_CUDA = True
except:
    HAS_OPENCV_CUDA = False
    print("WARNING: OpenCV not built with CUDA support")

@dataclass
class ROIData:
    """Container for ROI data with transformation info."""
    roi_image: np.ndarray
    original_bbox: np.ndarray
    padded_bbox: np.ndarray
    transform_matrix: np.ndarray
    track_id: int
    frame_id: int
    quality_score: float
    
    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """Transform points from ROI coordinates to original frame coordinates."""
        if points is None or len(points) == 0:
            return points
            
        # Ensure points are in homogeneous coordinates
        if points.shape[-1] == 2:
            ones = np.ones((points.shape[0], 1))
            points_h = np.hstack([points, ones])
        else:
            points_h = points
            
        # Apply transformation
        transformed = points_h @ self.transform_matrix.T
        
        # Convert back to 2D
        return transformed[:, :2] / transformed[:, 2:3]


class ROIProcessor:
    """
    GPU-accelerated ROI processor using CuPy and OpenCV CUDA.
    Keeps frames on GPU throughout the pipeline.
    """
    
    def __init__(self,
                 target_size: Tuple[int, int] = (256, 256),
                 padding_ratio: float = 0.3,
                 min_quality_score: float = 0.5,
                 max_workers: int = 1,  # GPU processing is typically single-threaded
                 batch_size: int = 8,
                 use_gpu: bool = True):
        """
        Initialize GPU ROI processor.
        
        Args:
            target_size: Target size for ROI extraction
            padding_ratio: Padding ratio relative to bbox size
            min_quality_score: Minimum quality score for processing
            max_workers: Number of parallel workers (usually 1 for GPU)
            batch_size: Batch size for GPU processing
            use_gpu: Whether to use GPU acceleration
        """
        self.target_size = target_size
        self.padding_ratio = padding_ratio
        self.min_quality_score = min_quality_score
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.use_gpu = use_gpu and (HAS_CUPY or HAS_OPENCV_CUDA)
        
        # Processing queues
        self.input_queue = queue.Queue(maxsize=100)
        self.output_queue = queue.Queue(maxsize=100)
        
        # Thread pool
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Processing thread
        self.is_running = False
        self.processing_thread = None
        
        # GPU memory pool for CuPy
        if self.use_gpu and HAS_CUPY:
            # Set memory pool for efficient allocation
            mempool = cp.get_default_memory_pool()
            mempool.set_limit(size=2 * 1024**3)  # 2GB limit
            
        # Pre-allocate GPU arrays for batch processing
        if self.use_gpu:
            self._preallocate_gpu_memory()
            
        # Performance tracking
        self.gpu_time_accum = 0
        self.cpu_time_accum = 0
        self.roi_count = 0
        
        print(f"[ROI Processor] Initialized with GPU: {self.use_gpu}")
        if self.use_gpu:
            print(f"[ROI Processor] Using: CuPy={HAS_CUPY}, OpenCV CUDA={HAS_OPENCV_CUDA}")
    
    def _preallocate_gpu_memory(self):
        """Pre-allocate GPU memory for batch processing."""
        if HAS_CUPY:
            # Pre-allocate arrays for batch ROI extraction
            self.gpu_batch_rois = cp.zeros(
                (self.batch_size, self.target_size[1], self.target_size[0], 3), 
                dtype=cp.uint8
            )
            self.gpu_temp_buffer = cp.zeros(
                (self.target_size[1] * 2, self.target_size[0] * 2, 3), 
                dtype=cp.uint8
            )
    
    def start(self):
        """Start the ROI processing thread."""
        if not self.is_running:
            self.is_running = True
            self.processing_thread = threading.Thread(target=self._processing_loop)
            self.processing_thread.daemon = True
            self.processing_thread.start()
            print("[ROI Processor] Started")
    
    def stop(self):
        """Stop the ROI processing thread."""
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=1.0)
        self.executor.shutdown(wait=False)
        
        # Clear GPU memory
        if self.use_gpu and HAS_CUPY:
            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()
            
        print("[ROI Processor] Stopped")
        print(f"[ROI Processor] Performance - GPU time: {self.gpu_time_accum:.2f}s, "
              f"CPU time: {self.cpu_time_accum:.2f}s, ROIs: {self.roi_count}")
    
    def extract_roi(self, frame: np.ndarray, bbox: List[float], track_id: int, 
                    timestamp: float) -> Optional[Dict]:
        """
        Synchronous ROI extraction with GPU acceleration.
        """
        if self.use_gpu:
            return self._extract_roi_gpu_sync(frame, bbox, track_id, timestamp)
        else:
            # Fallback to CPU version
            return self._extract_roi_cpu(frame, bbox, track_id, timestamp)
    
    def _extract_roi_gpu_sync(self, frame: np.ndarray, bbox: List[float], 
                            track_id: int, timestamp: float) -> Optional[Dict]:
        """GPU-accelerated ROI extraction (synchronous)."""
        start_time = time.time()
        
        try:
            # Validate inputs
            if not isinstance(bbox, (list, np.ndarray)) or len(bbox) != 4:
                print(f"[ROI Processor] Invalid bbox: {bbox}")
                return self._extract_roi_cpu(frame, bbox, track_id, timestamp)
            
            # Ensure bbox values are valid
            bbox = np.array(bbox, dtype=np.float32)
            if np.any(np.isnan(bbox)) or np.any(np.isinf(bbox)):
                print(f"[ROI Processor] Invalid bbox values: {bbox}")
                return self._extract_roi_cpu(frame, bbox, track_id, timestamp)
            
            # Convert frame to GPU if needed
            if HAS_CUPY:
                try:
                    # Clear any previous CUDA errors
                    cp.cuda.runtime.getLastError()
                except:
                    pass
                
                # FIX: Keep frame on GPU if it's already there
                if isinstance(frame, cp.ndarray):
                    gpu_frame = frame
                    print(f"[ROI Processor] Frame already on GPU")
                else:
                    # Ensure frame is contiguous before GPU transfer
                    if not frame.flags['C_CONTIGUOUS']:
                        frame = np.ascontiguousarray(frame)
                    
                    try:
                        gpu_frame = cp.asarray(frame)
                    except Exception as e:
                        print(f"[ROI Processor] Failed to transfer frame to GPU: {e}")
                        return self._extract_roi_cpu(frame, bbox, track_id, timestamp)
                
                # Calculate padded bbox
                frame_shape = frame.shape if isinstance(frame, np.ndarray) else (gpu_frame.shape[0], gpu_frame.shape[1], gpu_frame.shape[2])
                padded_bbox = self._calculate_padded_bbox(bbox, frame_shape)
                
                # Check quality
                quality_score = self._calculate_quality_score(bbox, frame_shape)
                if quality_score < self.min_quality_score:
                    return None
                
                # Validate padded bbox
                x1, y1, x2, y2 = padded_bbox.astype(int)
                
                # Ensure coordinates are within frame bounds
                x1 = max(0, min(x1, frame_shape[1] - 1))
                y1 = max(0, min(y1, frame_shape[0] - 1))
                x2 = max(x1 + 1, min(x2, frame_shape[1]))
                y2 = max(y1 + 1, min(y2, frame_shape[0]))
                
                if x2 <= x1 or y2 <= y1:
                    print(f"[ROI Processor] Invalid ROI dimensions: ({x1},{y1}) to ({x2},{y2})")
                    return None
                
                try:
                    # Extract ROI with padding on GPU
                    roi_gpu = gpu_frame[y1:y2, x1:x2]
                    
                    if roi_gpu.size == 0:
                        print(f"[ROI Processor] Empty ROI")
                        return None
                    
                    # FIX: Resize and normalize on GPU in one step
                    roi_resized_gpu = self._resize_gpu_cupy(roi_gpu, self.target_size)
                    
                    # FIX: Transfer to CPU only because MediaPipe/TensorRT needs CPU input
                    # The data is already normalized to [-1, 1] range
                    roi_resized = cp.asnumpy(roi_resized_gpu)
                    
                    # Debug output
                    if self.roi_count % 50 == 0:
                        print(f"[ROI Processor] ROI {self.roi_count}: "
                            f"shape={roi_resized.shape}, dtype={roi_resized.dtype}, "
                            f"range=[{roi_resized.min():.3f}, {roi_resized.max():.3f}]")
                    
                    # Clear GPU memory periodically
                    if self.roi_count % 100 == 0:
                        mempool = cp.get_default_memory_pool()
                        used_bytes = mempool.used_bytes()
                        total_bytes = mempool.total_bytes()
                        print(f"[ROI Processor] GPU memory: {used_bytes/1024/1024:.1f}MB used, "
                            f"{total_bytes/1024/1024:.1f}MB total")
                        mempool.free_all_blocks()
                    
                except Exception as e:
                    print(f"[ROI Processor] GPU processing failed: {e}")
                    # Fallback to CPU
                    return self._extract_roi_cpu(frame, bbox, track_id, timestamp)
                
                # Ensure proper dtype and contiguous memory for downstream processing
                roi_resized = np.ascontiguousarray(roi_resized, dtype=np.float32)
                                
            elif HAS_OPENCV_CUDA:
                # Use OpenCV CUDA
                if isinstance(frame, np.ndarray):
                    gpu_frame = cv2.cuda_GpuMat()
                    gpu_frame.upload(frame)
                else:
                    gpu_frame = frame
                    
                # Calculate padded bbox
                padded_bbox = self._calculate_padded_bbox(bbox, (gpu_frame.rows, gpu_frame.cols, 3))
                
                # Check quality
                quality_score = self._calculate_quality_score(bbox, (gpu_frame.rows, gpu_frame.cols))
                if quality_score < self.min_quality_score:
                    return None
                
                # Extract ROI
                x1, y1, x2, y2 = padded_bbox.astype(int)
                roi_gpu = cv2.cuda_GpuMat(gpu_frame, (x1, y1, x2-x1, y2-y1))
                
                # Resize on GPU
                roi_resized_gpu = cv2.cuda_GpuMat()
                cv2.cuda.resize(roi_gpu, self.target_size, roi_resized_gpu)
                
                # Download to CPU
                roi_resized = roi_resized_gpu.download()
                
                # Convert to float32 and normalize
                roi_resized = roi_resized.astype(np.float32)
                roi_resized = (roi_resized - 127.5) / 127.5
                
                # Ensure proper format
                roi_resized = np.ascontiguousarray(roi_resized, dtype=np.float32)
            
            # Calculate transformation matrix
            transform_matrix = self._calculate_transform_matrix(padded_bbox, self.target_size)
            
            # Track performance
            self.gpu_time_accum += time.time() - start_time
            self.roi_count += 1
            
            # Return result with normalized CPU array
            return {
                'roi': roi_resized,  # Normalized float32 array in [-1, 1] range
                'transform': {
                    'scale': transform_matrix[0, 0],
                    'offset_x': transform_matrix[0, 2],
                    'offset_y': transform_matrix[1, 2],
                    'frame_width': frame_shape[1],
                    'frame_height': frame_shape[0]
                },
                'quality_score': quality_score,
                'original_bbox': np.array(bbox),
                'padded_bbox': padded_bbox
            }
            
        except Exception as e:
            print(f"[ROI Processor] GPU extraction error: {e}")
            # Fallback to CPU
            return self._extract_roi_cpu(frame, bbox, track_id, timestamp)

    def _resize_gpu_cupy(self, roi_gpu: cp.ndarray, target_size: Tuple[int, int]) -> cp.ndarray:
        """Resize ROI on GPU using CuPy's optimized resize."""
        try:
            import cupyx.scipy.ndimage as ndimage
            
            h, w = roi_gpu.shape[:2]
            target_h, target_w = target_size[1], target_size[0]
            
            # Calculate zoom factors
            zoom_y = target_h / h
            zoom_x = target_w / w
            
            # Use CuPy's optimized zoom function
            if len(roi_gpu.shape) == 3:
                # For RGB images, apply zoom to spatial dimensions only
                zoom_factors = (zoom_y, zoom_x, 1)
            else:
                zoom_factors = (zoom_y, zoom_x)
            
            # Use order=1 for bilinear interpolation
            resized = ndimage.zoom(roi_gpu, zoom_factors, order=1, prefilter=False)
            
            # Convert to float32 and normalize ONCE here on GPU
            # This avoids doing it again in the landmark process
            resized = resized.astype(cp.float32)
            # Normalize to [-1, 1] range expected by TensorRT landmark model
            resized = (resized - 127.5) / 127.5
            
            return resized
            
        except ImportError:
            # Fallback to OpenCV-based resize if cupyx is not available
            # Transfer to CPU, resize, and transfer back
            roi_cpu = cp.asnumpy(roi_gpu)
            resized_cpu = cv2.resize(roi_cpu, target_size, interpolation=cv2.INTER_LINEAR)
            # Normalize on CPU before transfer back
            resized_cpu = resized_cpu.astype(np.float32)
            resized_cpu = (resized_cpu - 127.5) / 127.5
            return cp.asarray(resized_cpu)


    def _extract_roi_cpu(self, frame: np.ndarray, bbox: List[float], 
                        track_id: int, timestamp: float) -> Optional[Dict]:
        """CPU fallback for ROI extraction."""
        start_time = time.time()
        
        # Existing CPU implementation with normalization added
        padded_bbox = self._calculate_padded_bbox(bbox, frame.shape)
        quality_score = self._calculate_quality_score(bbox, frame.shape)
        
        if quality_score < self.min_quality_score:
            return None
        
        x1, y1, x2, y2 = padded_bbox.astype(int)
        roi = frame[y1:y2, x1:x2]
        
        if roi.size == 0:
            return None
        
        roi_resized = cv2.resize(roi, self.target_size, interpolation=cv2.INTER_LINEAR)
        
        # FIX: Normalize CPU output to match GPU output
        roi_resized = roi_resized.astype(np.float32)
        roi_resized = (roi_resized - 127.5) / 127.5
        
        transform_matrix = self._calculate_transform_matrix(padded_bbox, self.target_size)
        
        self.cpu_time_accum += time.time() - start_time
        self.roi_count += 1
        
        return {
            'roi': roi_resized,  # Now normalized like GPU version
            'transform': {
                'scale': transform_matrix[0, 0],
                'offset_x': transform_matrix[0, 2],
                'offset_y': transform_matrix[1, 2],
                'frame_width': frame.shape[1],
                'frame_height': frame.shape[0]
            },
            'quality_score': quality_score,
            'original_bbox': np.array(bbox),
            'padded_bbox': padded_bbox
        }

    def _processing_loop(self):
        """Main processing loop with batch GPU processing."""
        batch_frames = []
        batch_tracks = []
        batch_metadata = []
        last_batch_time = time.time()
        batch_timeout = 0.01  # 10ms
        
        while self.is_running:
            try:
                # Collect data for batch
                timeout = batch_timeout if not batch_frames else 0.001
                data = self.input_queue.get(timeout=timeout)
                
                if data is None:
                    continue
                    
                frame, tracks, frame_id = data
                
                # Add to batch
                for track in tracks:
                    batch_frames.append(frame)
                    batch_tracks.append(track)
                    batch_metadata.append(frame_id)
                
                # Process batch if full or timeout
                current_time = time.time()
                should_process = (
                    len(batch_frames) >= self.batch_size or
                    (len(batch_frames) > 0 and current_time - last_batch_time > batch_timeout)
                )
                
                if should_process:
                    # Process batch on GPU
                    if self.use_gpu and HAS_CUPY:
                        rois = self._process_batch_gpu(batch_frames, batch_tracks, batch_metadata)
                    else:
                        # Process individually on CPU
                        rois = []
                        for frame, track, meta in zip(batch_frames, batch_tracks, batch_metadata):
                            roi_data = self._extract_roi_cpu(frame, track['bbox'], 
                                                            track['track_id'], 0)
                            if roi_data:
                                rois.append(ROIData(
                                    roi_image=roi_data['roi'],
                                    original_bbox=roi_data['original_bbox'],
                                    padded_bbox=roi_data['padded_bbox'],
                                    transform_matrix=self._calculate_transform_matrix(
                                        roi_data['padded_bbox'], self.target_size
                                    ),
                                    track_id=track['track_id'],
                                    frame_id=meta,
                                    quality_score=roi_data['quality_score']
                                ))
                    
                    # Output results
                    if rois:
                        self.output_queue.put(rois)
                    
                    # Clear batch
                    batch_frames.clear()
                    batch_tracks.clear()
                    batch_metadata.clear()
                    last_batch_time = current_time
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[ROI Processor] Processing loop error: {e}")
    
    def _process_batch_gpu(self, frames: List[np.ndarray], tracks: List[Dict], 
                          metadata: List[int]) -> List[ROIData]:
        """Process a batch of ROIs on GPU."""
        if not HAS_CUPY:
            return []
            
        rois = []
        
        # Transfer all frames to GPU at once
        gpu_frames = [cp.asarray(frame) for frame in frames]
        
        # Process each ROI
        for gpu_frame, track, frame_id in zip(gpu_frames, tracks, metadata):
            try:
                bbox = track['bbox']
                track_id = track['track_id']
                
                # Calculate padded bbox
                padded_bbox = self._calculate_padded_bbox(bbox, gpu_frame.shape)
                
                # Check quality
                quality_score = self._calculate_quality_score(bbox, gpu_frame.shape)
                if quality_score < self.min_quality_score:
                    continue
                
                # Extract ROI on GPU
                x1, y1, x2, y2 = padded_bbox.astype(int)
                roi_gpu = gpu_frame[y1:y2, x1:x2]
                
                if roi_gpu.size == 0:
                    continue
                
                # Resize on GPU
                roi_resized_gpu = self._resize_gpu_cupy(roi_gpu, self.target_size)
                
                # Keep on GPU if next stage supports it, otherwise transfer to CPU
                roi_resized = cp.asnumpy(roi_resized_gpu)
                
                # Create ROI data
                transform_matrix = self._calculate_transform_matrix(padded_bbox, self.target_size)
                
                roi_data = ROIData(
                    roi_image=roi_resized,
                    original_bbox=np.array(bbox),
                    padded_bbox=padded_bbox,
                    transform_matrix=transform_matrix,
                    track_id=track_id,
                    frame_id=frame_id,
                    quality_score=quality_score
                )
                
                rois.append(roi_data)
                
            except Exception as e:
                print(f"[ROI Processor] Batch GPU error: {e}")
                continue

        if self.roi_count % 100 == 0:
            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()
                
                
        return rois
    
    def _calculate_padded_bbox(self, bbox: np.ndarray, frame_shape: Tuple) -> np.ndarray:
        """Calculate bbox with padding."""
        if isinstance(bbox, list):
            bbox = np.array(bbox)
            
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        
        # Calculate padding
        pad_w = w * self.padding_ratio
        pad_h = h * self.padding_ratio
        
        # Apply padding
        x1_pad = max(0, x1 - pad_w)
        y1_pad = max(0, y1 - pad_h)
        x2_pad = min(frame_shape[1], x2 + pad_w)
        y2_pad = min(frame_shape[0], y2 + pad_h)
        
        # Ensure square aspect ratio if needed
        w_pad = x2_pad - x1_pad
        h_pad = y2_pad - y1_pad
        
        if w_pad > h_pad:
            diff = (w_pad - h_pad) / 2
            y1_pad = max(0, y1_pad - diff)
            y2_pad = min(frame_shape[0], y2_pad + diff)
        elif h_pad > w_pad:
            diff = (h_pad - w_pad) / 2
            x1_pad = max(0, x1_pad - diff)
            x2_pad = min(frame_shape[1], x2_pad + diff)
        
        return np.array([x1_pad, y1_pad, x2_pad, y2_pad])
    
    def _calculate_quality_score(self, bbox: np.ndarray, frame_shape: Tuple) -> float:
        """Calculate quality score for ROI."""
        if isinstance(bbox, list):
            bbox = np.array(bbox)
            
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        
        # Size score
        size_score = min(1.0, (w * h) / (frame_shape[0] * frame_shape[1] * 0.1))
        
        # Aspect ratio score
        aspect_ratio = w / h if h > 0 else 0
        ar_score = 1.0 - abs(1.0 - aspect_ratio) * 0.5
        
        # Position score
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        pos_score = 1.0 - abs(cx - frame_shape[1]/2) / frame_shape[1] * 0.5
        pos_score *= 1.0 - abs(cy - frame_shape[0]/2) / frame_shape[0] * 0.5
        
        # Boundary score
        margin = min(x1, y1, frame_shape[1] - x2, frame_shape[0] - y2)
        boundary_score = min(1.0, margin / 50.0)
        
        # Combined score
        quality_score = (size_score * 0.4 + ar_score * 0.2 + 
                        pos_score * 0.2 + boundary_score * 0.2)
        
        return quality_score
    
    def _calculate_transform_matrix(self, padded_bbox: np.ndarray,
                                   target_size: Tuple[int, int]) -> np.ndarray:
        """Calculate transformation matrix from ROI to original coordinates."""
        x1, y1, x2, y2 = padded_bbox
        w = x2 - x1
        h = y2 - y1
        
        # Scale factors
        scale_x = w / target_size[0]
        scale_y = h / target_size[1]
        
        # Translation
        tx = x1
        ty = y1
        
        # Build transformation matrix
        transform = np.array([
            [scale_x, 0, tx],
            [0, scale_y, ty],
            [0, 0, 1]
        ], dtype=np.float32)
        
        return transform
    
    def get_stats(self) -> Dict:
        """Get performance statistics."""
        total_time = self.gpu_time_accum + self.cpu_time_accum
        
        return {
            'roi_count': self.roi_count,
            'gpu_time': self.gpu_time_accum,
            'cpu_time': self.cpu_time_accum,
            'gpu_percentage': (self.gpu_time_accum / total_time * 100) if total_time > 0 else 0,
            'avg_roi_time_ms': (total_time / self.roi_count * 1000) if self.roi_count > 0 else 0,
            'using_gpu': self.use_gpu
        }
    
    def _ensure_cpu_array(self, array):
        """Ensure array is on CPU and in correct format for MediaPipe."""
        if HAS_CUPY and isinstance(array, cp.ndarray):
            # Transfer from GPU to CPU
            cpu_array = cp.asnumpy(array)
        else:
            cpu_array = array
        
        # Ensure contiguous memory layout and uint8 dtype
        if not cpu_array.flags['C_CONTIGUOUS'] or cpu_array.dtype != np.uint8:
            cpu_array = np.ascontiguousarray(cpu_array, dtype=np.uint8)
        
        return cpu_array