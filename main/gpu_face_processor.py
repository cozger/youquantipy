"""
GPU Face Processing Pipeline
CRITICAL: This runs in a thread, not a process, to share GPU context.
All processing from detection through landmarks happens on GPU.
"""

import cupy as cp
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import time
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass

@dataclass
class FaceDetection:
    """Face detection result"""
    bbox: np.ndarray  # [x1, y1, x2, y2] in original image coordinates
    confidence: float
    landmarks_5: Optional[np.ndarray] = None  # 5 key points if available

class GPUFaceProcessor:
    """
    End-to-end face processing on GPU.
    Detection → ROI Extraction → Landmarks → Blendshapes
    
    CRITICAL DESIGN:
    1. All processing stays on GPU
    2. Only final results transfer to CPU
    3. Batch processing for efficiency
    4. Pre-allocated buffers
    """
    
    def __init__(self, retinaface_engine: str, landmark_engine: str,
                 max_faces: int = 10, confidence_threshold: float = 0.9):
        """
        Initialize GPU face processor.
        
        Args:
            retinaface_engine: Path to RetinaFace TensorRT engine
            landmark_engine: Path to face landmark TensorRT engine
            max_faces: Maximum faces to process
            confidence_threshold: Detection confidence threshold
        """
        self.max_faces = max_faces
        self.confidence_threshold = confidence_threshold
        
        # Load TensorRT engines
        self._load_engines(retinaface_engine, landmark_engine)
        
        # Pre-allocate GPU buffers
        self._allocate_buffers()
        
        # Generate RetinaFace anchors
        self._generate_anchors()
        
        # Store ROI transformation info for landmark coordinate mapping
        self.roi_transforms = []
        
        # Performance tracking
        self.frame_count = 0
        self.timings = {
            'detection': [],
            'roi_extraction': [],
            'landmarks': [],
            'total': []
        }
        
        print("[GPU Face Processor] Initialized")
    
    def _load_engines(self, retinaface_path: str, landmark_path: str):
        """Load TensorRT engines"""
        trt_logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(trt_logger)
        
        # Load detection engine
        print(f"[GPU Face Processor] Loading RetinaFace engine from {retinaface_path}")
        with open(retinaface_path, 'rb') as f:
            self.detect_engine = runtime.deserialize_cuda_engine(f.read())
        self.detect_context = self.detect_engine.create_execution_context()
        
        # Load landmark engine
        print(f"[GPU Face Processor] Loading landmark engine from {landmark_path}")
        with open(landmark_path, 'rb') as f:
            self.landmark_engine = runtime.deserialize_cuda_engine(f.read())
        self.landmark_context = self.landmark_engine.create_execution_context()
        
        # Analyze engine dimensions
        self._analyze_engines()
    
    def _analyze_engines(self):
        """Analyze TensorRT engine configurations"""
        # Detection engine
        input_name = self.detect_engine.get_tensor_name(0)
        input_shape = self.detect_engine.get_tensor_shape(input_name)
        print(f"[GPU Face Processor] Detection input shape: {input_shape}")
        
        # Get detection dimensions
        if input_shape[-1] == 3:  # NHWC format
            self.detect_height = input_shape[1]
            self.detect_width = input_shape[2]
        else:  # NCHW format
            self.detect_height = input_shape[2]
            self.detect_width = input_shape[3]
        
        # Landmark engine
        landmark_input = self.landmark_engine.get_tensor_name(0)
        landmark_shape = self.landmark_engine.get_tensor_shape(landmark_input)
        print(f"[GPU Face Processor] Landmark input shape: {landmark_shape}")
        
        # Determine number of landmarks from output
        for i in range(self.landmark_engine.num_io_tensors):
            name = self.landmark_engine.get_tensor_name(i)
            if self.landmark_engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                shape = self.landmark_engine.get_tensor_shape(name)
                # Common landmark counts: 468 or 478
                if any(dim in [468*3, 478*3, 1404, 1434] for dim in shape):
                    if 1434 in shape or 478*3 in shape:
                        self.num_landmarks = 478
                    else:
                        self.num_landmarks = 468
                    break
        
        print(f"[GPU Face Processor] Detected {self.num_landmarks} landmarks model")
    
    def _allocate_buffers(self):
        """Pre-allocate all GPU buffers"""
        # Detection buffers
        self.gpu_detect_input = cuda.mem_alloc(640 * 640 * 3 * 4)  # float32
        self.gpu_detect_outputs = []
        
        # Allocate detection outputs
        for i in range(self.detect_engine.num_io_tensors):
            name = self.detect_engine.get_tensor_name(i)
            if self.detect_engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                shape = self.detect_engine.get_tensor_shape(name)
                size = int(abs(np.prod(shape)) * 4)  # float32, convert to Python int
                self.gpu_detect_outputs.append(cuda.mem_alloc(size))
        
        # ROI buffers (batch processing)
        self.gpu_rois = cp.zeros((self.max_faces, 256, 256, 3), dtype=cp.float32)
        self.gpu_roi_batch = cuda.mem_alloc(self.max_faces * 256 * 256 * 3 * 4)
        
        # Landmark outputs
        self.gpu_landmark_outputs = []
        for i in range(self.landmark_engine.num_io_tensors):
            name = self.landmark_engine.get_tensor_name(i)
            if self.landmark_engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                shape = list(self.landmark_engine.get_tensor_shape(name))
                shape[0] = self.max_faces  # Set batch size
                size = int(abs(np.prod(shape)) * 4)  # float32, convert to Python int
                self.gpu_landmark_outputs.append(cuda.mem_alloc(size))
        
        # Host buffers for final results only
        self.host_landmarks = np.zeros((self.max_faces, self.num_landmarks, 3), dtype=np.float32)
        
        print(f"[GPU Face Processor] Allocated buffers for {self.max_faces} faces")
    
    def _generate_anchors(self):
        """Generate RetinaFace anchor boxes"""
        # Based on model configuration
        if self.detect_width == 640 and self.detect_height == 640:
            min_sizes = [[16, 32], [64, 128], [256, 512]]
            steps = [8, 16, 32]
        elif self.detect_width == 640 and self.detect_height == 608:
            min_sizes = [[16, 32], [64, 128], [256, 512]]
            steps = [8, 16, 32]
        else:
            # Default configuration
            min_sizes = [[16, 32], [64, 128], [256, 512]]
            steps = [8, 16, 32]
        
        anchors = []
        for k, step in enumerate(steps):
            f_h = self.detect_height // step
            f_w = self.detect_width // step
            
            for i in range(f_h):
                for j in range(f_w):
                    for min_size in min_sizes[k]:
                        cx = (j + 0.5) * step / self.detect_width
                        cy = (i + 0.5) * step / self.detect_height
                        w = min_size / self.detect_width
                        h = min_size / self.detect_height
                        anchors.append([cx, cy, w, h])
        
        self.anchors = cp.array(anchors, dtype=cp.float32)
        print(f"[GPU Face Processor] Generated {len(anchors)} anchor boxes")
    
    def process_frame(self, detection_frame: cp.ndarray, full_frame: cp.ndarray,
                     frame_id: int, timestamp: float, detection_params: Dict) -> Dict:
        """
        Process a frame entirely on GPU.
        
        Args:
            detection_frame: 640x640 detection input (GPU array)
            full_frame: Full resolution frame (GPU array)
            frame_id: Frame identifier
            timestamp: Frame timestamp
            detection_params: Detection scaling parameters
            
        Returns:
            Dict with processed faces (only this transfers to CPU)
        """
        total_start = time.time()
        self.frame_count += 1
        
        # Run detection
        detect_start = time.time()
        detections = self._run_detection(detection_frame, detection_params)
        detect_time = time.time() - detect_start
        
        if len(detections) == 0:
            return {
                'faces': [],
                'processing_time_ms': (time.time() - total_start) * 1000
            }
        
        # Extract ROIs on GPU
        roi_start = time.time()
        valid_faces = self._extract_rois_gpu(full_frame, detections[:self.max_faces])
        roi_time = time.time() - roi_start
        
        if valid_faces == 0:
            return {
                'faces': [],
                'processing_time_ms': (time.time() - total_start) * 1000
            }
        
        # Run landmark detection
        landmark_start = time.time()
        landmarks = self._run_landmarks_batch(valid_faces)
        landmark_time = time.time() - landmark_start
        
        # Build results (only now transfer to CPU)
        faces = []
        for i in range(valid_faces):
            # Transform landmarks from ROI space to frame space
            transformed_landmarks = self._transform_landmarks_to_frame(
                landmarks[i], self.roi_transforms[i]
            )
            
            faces.append({
                'id': i + 1,  # 1-based ID
                'bbox': detections[i].bbox.tolist(),  # Already CPU from detection
                'confidence': float(detections[i].confidence),
                'landmarks': transformed_landmarks.tolist(),  # Transformed landmark data
                'centroid': self._compute_centroid(transformed_landmarks)
            })
        
        # Update timing stats
        total_time = time.time() - total_start
        self.timings['detection'].append(detect_time * 1000)
        self.timings['roi_extraction'].append(roi_time * 1000)
        self.timings['landmarks'].append(landmark_time * 1000)
        self.timings['total'].append(total_time * 1000)
        
        self.frame_count += 1
        
        # Print performance every 300 frames
        if self.frame_count % 300 == 0:
            self._print_performance()
        
        return {
            'faces': faces,
            'processing_time_ms': total_time * 1000,
            'timings': {
                'detection_ms': detect_time * 1000,
                'roi_ms': roi_time * 1000,
                'landmarks_ms': landmark_time * 1000
            }
        }
    
    def _run_detection(self, detection_frame: cp.ndarray, detection_params: Dict) -> List[FaceDetection]:
        """
        Run RetinaFace detection on GPU.
        
        CRITICAL: Detection happens entirely on GPU, only final boxes transfer to CPU.
        """
        # Convert to float32 (TRT model expects raw pixel values 0-255)
        normalized = detection_frame.astype(cp.float32)
        
        # Copy to TRT input buffer (GPU to GPU)
        cuda.memcpy_dtod(
            self.gpu_detect_input,
            normalized.data.ptr,
            normalized.nbytes
        )
        
        # Check TensorRT API version and execute accordingly
        if hasattr(self.detect_context, 'set_input_shape'):
            # New API (TensorRT 8.5+)
            # Set input shape
            input_name = self.detect_engine.get_tensor_name(0)
            self.detect_context.set_input_shape(input_name, (1, self.detect_height, self.detect_width, 3))
            
            # Set tensor addresses
            self.detect_context.set_tensor_address(input_name, int(self.gpu_detect_input))
            
            # Set output addresses
            output_idx = 0
            for i in range(self.detect_engine.num_io_tensors):
                name = self.detect_engine.get_tensor_name(i)
                if self.detect_engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                    self.detect_context.set_tensor_address(name, int(self.gpu_detect_outputs[output_idx]))
                    output_idx += 1
            
            # Execute
            self.detect_context.execute_async_v3(0)
        else:
            # Old API - build bindings
            bindings = [int(self.gpu_detect_input)]
            for buf in self.gpu_detect_outputs:
                bindings.append(int(buf))
            
            try:
                self.detect_context.execute_async_v2(bindings=bindings, stream_handle=0)
            except AttributeError:
                # Fallback to synchronous execution
                self.detect_context.execute_v2(bindings)
        
        cuda.Context.synchronize()
        
        # Process outputs on GPU
        # Copy outputs to CuPy arrays for processing
        outputs = []
        output_idx = 0
        for i in range(self.detect_engine.num_io_tensors):
            name = self.detect_engine.get_tensor_name(i)
            if self.detect_engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                shape = self.detect_context.get_tensor_shape(name)
                size = abs(np.prod(shape))
                
                # Create CuPy array pointing to GPU memory
                gpu_array = cp.ndarray(
                    shape=shape[1:],  # Remove batch dimension
                    dtype=cp.float32,
                    memptr=cp.cuda.MemoryPointer(
                        cp.cuda.UnownedMemory(
                            int(self.gpu_detect_outputs[output_idx]),
                            size * 4,
                            self
                        ),
                        0
                    )
                )
                outputs.append(gpu_array)
                output_idx += 1
        
        # Decode detections on GPU
        detections = self._decode_detections_gpu(outputs, detection_params)
        
        # Debug logging
        if self.frame_count % 10 == 0:  # Log every 10 frames for debugging
            print(f"[GPU Face] Frame {self.frame_count}: Found {len(detections)} detections")
            if len(outputs) > 0:
                print(f"[GPU Face] Output shapes: {[o.shape for o in outputs]}")
        
        return detections
    
    def _decode_detections_gpu(self, outputs: List[cp.ndarray], 
                              detection_params: Dict) -> List[FaceDetection]:
        """
        Decode RetinaFace outputs entirely on GPU.
        Only final filtered detections transfer to CPU.
        """
        # Identify outputs by shape
        boxes_output = None
        scores_output = None
        
        for output in outputs:
            if output.shape[-1] == 4:
                boxes_output = output
            elif output.shape[-1] == 2:
                scores_output = output
        
        if boxes_output is None or scores_output is None:
            return []
        
        # Apply sigmoid to face class scores (binary classification)
        # The model outputs logits for [background, face]
        face_logits = scores_output[:, 1]  # Face class logits
        face_scores = 1 / (1 + cp.exp(-face_logits))  # Sigmoid activation
        
        # Debug: Check score distribution
        if self.frame_count % 10 == 0:
            max_score = float(cp.max(face_scores)) if len(face_scores) > 0 else 0
            print(f"[GPU Face] Max face score: {max_score:.3f}, threshold: {self.confidence_threshold}")
            if max_score > 0.5:  # If there's a decent score
                top_scores = cp.sort(face_scores)[-5:]  # Top 5 scores
                print(f"[GPU Face] Top 5 scores: {top_scores.get()}")
        
        # Filter by confidence
        valid_mask = face_scores > self.confidence_threshold
        valid_indices = cp.where(valid_mask)[0]
        
        if len(valid_indices) == 0:
            return []
        
        # Decode boxes on GPU
        valid_boxes = boxes_output[valid_indices]
        valid_scores = face_scores[valid_indices]
        valid_anchors = self.anchors[valid_indices]
        
        # RetinaFace box decoding
        decoded_boxes = cp.zeros_like(valid_boxes)
        
        # Decode center
        decoded_boxes[:, 0] = valid_anchors[:, 0] + valid_boxes[:, 0] * 0.1 * valid_anchors[:, 2]
        decoded_boxes[:, 1] = valid_anchors[:, 1] + valid_boxes[:, 1] * 0.1 * valid_anchors[:, 3]
        
        # Decode size
        decoded_boxes[:, 2] = valid_anchors[:, 2] * cp.exp(valid_boxes[:, 2] * 0.2)
        decoded_boxes[:, 3] = valid_anchors[:, 3] * cp.exp(valid_boxes[:, 3] * 0.2)
        
        # Convert to corners
        x1 = decoded_boxes[:, 0] - decoded_boxes[:, 2] / 2
        y1 = decoded_boxes[:, 1] - decoded_boxes[:, 3] / 2
        x2 = decoded_boxes[:, 0] + decoded_boxes[:, 2] / 2
        y2 = decoded_boxes[:, 1] + decoded_boxes[:, 3] / 2
        
        # Scale to detection frame coordinates
        x1 *= self.detect_width
        y1 *= self.detect_height
        x2 *= self.detect_width
        y2 *= self.detect_height
        
        # Apply NMS on GPU
        keep_indices = self._nms_gpu(
            cp.stack([x1, y1, x2, y2], axis=1),
            valid_scores,
            threshold=0.3
        )
        
        # Convert to original image coordinates
        scale = detection_params['scale']
        offset_x = detection_params['offset_x']
        offset_y = detection_params['offset_y']
        orig_w = detection_params['original_width']
        orig_h = detection_params['original_height']
        
        # Only transfer final results to CPU
        detections = []
        for idx in keep_indices:
            # Transform to original coordinates
            box_x1 = float((x1[idx] - offset_x) / scale)
            box_y1 = float((y1[idx] - offset_y) / scale)
            box_x2 = float((x2[idx] - offset_x) / scale)
            box_y2 = float((y2[idx] - offset_y) / scale)
            
            # Clip to image bounds
            box_x1 = max(0, min(box_x1, orig_w))
            box_y1 = max(0, min(box_y1, orig_h))
            box_x2 = max(0, min(box_x2, orig_w))
            box_y2 = max(0, min(box_y2, orig_h))
            
            detections.append(FaceDetection(
                bbox=np.array([box_x1, box_y1, box_x2, box_y2], dtype=np.float32),
                confidence=float(valid_scores[idx])
            ))
        
        return detections
    
    def _nms_gpu(self, boxes: cp.ndarray, scores: cp.ndarray, threshold: float) -> List[int]:
        """
        Non-Maximum Suppression on GPU.
        Returns indices of boxes to keep.
        """
        # Sort by score
        order = cp.argsort(scores)[::-1]
        
        keep = []
        while len(order) > 0:
            i = order[0]
            keep.append(int(i))
            
            if len(order) == 1:
                break
            
            # Compute IoU with remaining boxes
            xx1 = cp.maximum(boxes[i, 0], boxes[order[1:], 0])
            yy1 = cp.maximum(boxes[i, 1], boxes[order[1:], 1])
            xx2 = cp.minimum(boxes[i, 2], boxes[order[1:], 2])
            yy2 = cp.minimum(boxes[i, 3], boxes[order[1:], 3])
            
            w = cp.maximum(0, xx2 - xx1)
            h = cp.maximum(0, yy2 - yy1)
            inter = w * h
            
            area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
            area_rest = (boxes[order[1:], 2] - boxes[order[1:], 0]) * \
                       (boxes[order[1:], 3] - boxes[order[1:], 1])
            
            iou = inter / (area_i + area_rest - inter)
            
            # Keep boxes with IoU less than threshold
            inds = cp.where(iou <= threshold)[0]
            order = order[inds + 1]
        
        return keep
    
    def _extract_rois_gpu(self, full_frame: cp.ndarray, detections: List[FaceDetection]) -> int:
        """
        Extract and resize ROIs entirely on GPU.
        
        Returns:
            Number of valid ROIs extracted
        """
        valid_count = 0
        self.roi_transforms = []  # Reset transforms for this batch
        
        for i, det in enumerate(detections):
            if i >= self.max_faces:
                break
            
            # Get bbox with padding
            x1, y1, x2, y2 = det.bbox
            w = x2 - x1
            h = y2 - y1
            
            # Add 30% padding
            pad_w = w * 0.15
            pad_h = h * 0.15
            
            x1_padded = int(max(0, x1 - pad_w))
            y1_padded = int(max(0, y1 - pad_h))
            x2_padded = int(min(full_frame.shape[1], x2 + pad_w))
            y2_padded = int(min(full_frame.shape[0], y2 + pad_h))
            
            if x2_padded <= x1_padded or y2_padded <= y1_padded:
                continue
            
            # Store transformation info for this ROI
            roi_w = x2_padded - x1_padded
            roi_h = y2_padded - y1_padded
            self.roi_transforms.append({
                'x1': x1_padded,
                'y1': y1_padded,
                'scale_x': roi_w / 256.0,
                'scale_y': roi_h / 256.0
            })
            
            # Extract ROI
            roi = full_frame[y1_padded:y2_padded, x1_padded:x2_padded]
            
            # Resize to 256x256 on GPU
            roi_resized = self._gpu_resize_high_quality(roi, (256, 256))
            
            # Normalize for landmark model
            roi_normalized = roi_resized.astype(cp.float32) / 255.0
            
            # Store in batch
            self.gpu_rois[valid_count] = roi_normalized
            valid_count += 1
        
        return valid_count
    
    def _gpu_resize_high_quality(self, img: cp.ndarray, size: Tuple[int, int]) -> cp.ndarray:
        """
        High quality resize on GPU using bicubic interpolation.
        """
        from cupyx.scipy import ndimage
        
        h_new, w_new = size
        h_old, w_old = img.shape[:2]
        
        zoom_factors = [h_new / h_old, w_new / w_old, 1]
        
        # Use order=3 for bicubic interpolation
        resized = ndimage.zoom(img, zoom_factors, order=3)
        
        return resized.astype(cp.uint8)
    
    def _run_landmarks_batch(self, batch_size: int) -> np.ndarray:
        """
        Run landmark detection on batch of ROIs.
        
        Returns:
            Landmarks array (CPU) - only this transfers from GPU
        """
        # Copy batch to TRT buffer
        cuda.memcpy_dtod(
            self.gpu_roi_batch,
            self.gpu_rois[:batch_size].data.ptr,
            batch_size * 256 * 256 * 3 * 4
        )
        
        # Set batch size
        input_name = self.landmark_engine.get_tensor_name(0)
        self.landmark_context.set_input_shape(input_name, [batch_size, 256, 256, 3])
        
        # Check TensorRT API version and execute accordingly
        if hasattr(self.landmark_context, 'set_tensor_address'):
            # New API (TensorRT 8.5+)
            # Set tensor addresses
            self.landmark_context.set_tensor_address(input_name, int(self.gpu_roi_batch))
            
            # Set output addresses
            output_idx = 0
            for i in range(self.landmark_engine.num_io_tensors):
                name = self.landmark_engine.get_tensor_name(i)
                if self.landmark_engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                    self.landmark_context.set_tensor_address(name, int(self.gpu_landmark_outputs[output_idx]))
                    output_idx += 1
            
            # Execute
            self.landmark_context.execute_async_v3(0)
        else:
            # Old API - build bindings
            bindings = [int(self.gpu_roi_batch)]
            for buf in self.gpu_landmark_outputs:
                bindings.append(int(buf))
            
            try:
                self.landmark_context.execute_async_v2(bindings=bindings, stream_handle=0)
            except AttributeError:
                # Fallback to synchronous execution
                self.landmark_context.execute_v2(bindings)
        
        cuda.Context.synchronize()
        
        # Get landmarks output (first output is landmarks)
        # Only transfer the exact amount we need
        landmarks_size = batch_size * self.num_landmarks * 3 * 4
        cuda.memcpy_dtoh(
            self.host_landmarks[:batch_size].ravel(),
            self.gpu_landmark_outputs[0]
        )
        
        return self.host_landmarks[:batch_size]
    
    def _compute_centroid(self, landmarks: np.ndarray) -> Tuple[float, float]:
        """Compute face centroid from landmarks"""
        x_mean = float(np.mean(landmarks[:, 0]))
        y_mean = float(np.mean(landmarks[:, 1]))
        return (x_mean, y_mean)
    
    def _transform_landmarks_to_frame(self, landmarks: np.ndarray, transform: Dict) -> np.ndarray:
        """
        Transform landmarks from ROI space (256x256) to original frame space.
        
        Args:
            landmarks: Landmarks in ROI space (N, 3) with x, y, z
            transform: Dictionary with transformation parameters
        
        Returns:
            Transformed landmarks in frame space
        """
        transformed = landmarks.copy()
        
        # Apply scale and translation to x and y coordinates
        transformed[:, 0] = landmarks[:, 0] * transform['scale_x'] + transform['x1']
        transformed[:, 1] = landmarks[:, 1] * transform['scale_y'] + transform['y1']
        
        # Z coordinate remains unchanged
        return transformed
    
    def _print_performance(self):
        """Print performance statistics"""
        print(f"\n[GPU Face Processor] Performance Report (frame {self.frame_count}):")
        
        for stage, times in self.timings.items():
            if times:
                recent = times[-300:]  # Last 300 samples
                avg = np.mean(recent)
                std = np.std(recent)
                print(f"  {stage}: {avg:.2f} ± {std:.2f} ms")
        
        # Clear old samples to prevent memory growth
        for times in self.timings.values():
            if len(times) > 1000:
                del times[:500]