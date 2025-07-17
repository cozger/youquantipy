import sys
import numpy as np
import threading
import time
from queue import Queue, Empty, Full
from typing import Optional, Tuple, List, Dict
from concurrent.futures import ThreadPoolExecutor

try:
    import onnxruntime as ort
except ImportError:
    print("ERROR: onnxruntime is not installed!")
    print("Please install it using: pip install onnxruntime")
    sys.exit(1)


class RetinaFaceDetector:
    def __init__(self, 
                 model_path: str,
                 tile_size: int = 640,          # Kept for compatibility, not used
                 overlap: float = 0.2,          # Kept for compatibility, not used
                 confidence_threshold: float = 0.3,
                 nms_threshold: float = 0.4,
                 max_workers: int = 4,          # Kept for compatibility, not used
                 debug_queue=None):
        
        # Model components
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.variances = [0.1, 0.2]
        
        # Async processing
        self.input_queue = Queue(maxsize=5)
        self.output_queue = Queue(maxsize=5)
        self.debug_queue = debug_queue
        self.is_running = False
        self.detection_thread = None
        self.executor = ThreadPoolExecutor(max_workers=1)  # Compatibility
        
        # Performance tracking
        self.fps_tracker = []
        self.detection_count = 0
        self.parse_count = 0
        
        # Load model
        self._load_model()
        
    def _load_model(self):
        """Load the ONNX model and extract metadata"""
        import os
        if not os.path.exists(self.model_path):
            raise RuntimeError(f"Model file not found: {self.model_path}")
            
        try:
            self.model = ort.InferenceSession(self.model_path)
            print(f"Loaded RetinaFace model from {self.model_path}")
            
            # Get input info
            input_info = self.model.get_inputs()[0]
            self.input_name = input_info.name
            
            # Extract model dimensions
            shape = input_info.shape
            if len(shape) == 4:
                if shape[1] == 3:  # NCHW format
                    self.model_format = 'NCHW'
                    self.model_height = shape[2]
                    self.model_width = shape[3]
                else:  # NHWC format
                    self.model_format = 'NHWC'
                    self.model_height = shape[1]
                    self.model_width = shape[2]
            else:
                raise RuntimeError(f"Unexpected input shape: {shape}")
                
            print(f"Model input: {self.input_name}, format: {self.model_format}, size: {self.model_width}x{self.model_height}")
            
            # Get output names
            self.output_names = [output.name for output in self.model.get_outputs()]
            print(f"Model outputs: {self.output_names}")
            
            # Generate anchors based on model size
            self.anchors = self._generate_anchors()
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
            
    def _generate_anchors(self):
        """Generate anchor boxes for the model"""
        # Configuration based on model size
        if self.model_width == 640 and self.model_height == 640:
            min_sizes = [[16, 32], [64, 128], [256, 512]]
            steps = [8, 16, 32]
        else:
            # Auto-detect based on output shape
            outputs = self.model.get_outputs()
            anchor_count = 0
            for output in outputs:
                if 'boxes' in output.name.lower() or 'loc' in output.name.lower():
                    shape = output.shape
                    if len(shape) > 1:
                        anchor_count = shape[1] if shape[1] > 1000 else shape[-1]
                        break
                        
            if anchor_count == 16800:
                min_sizes = [[16, 32], [64, 128], [256, 512]]
                steps = [8, 16, 32]
            elif anchor_count == 25600:
                min_sizes = [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]]
                steps = [8, 16, 32, 64]
            else:
                # Default configuration
                min_sizes = [[16, 32], [64, 128], [256, 512]]
                steps = [8, 16, 32]
                
        # Generate anchors
        anchors = []
        for k, step in enumerate(steps):
            f_h = self.model_height // step
            f_w = self.model_width // step
            for i in range(f_h):
                for j in range(f_w):
                    for min_size in min_sizes[k]:
                        cx = (j + 0.5) * step / self.model_width
                        cy = (i + 0.5) * step / self.model_height
                        w = min_size / self.model_width
                        h = min_size / self.model_height
                        anchors.append([cx, cy, w, h])
                        
        return np.array(anchors, dtype=np.float32)
        
    def _preprocess_image(self, image):
        """Preprocess image for model input"""
        # Resize to model dimensions
        import cv2
        resized = cv2.resize(image, (self.model_width, self.model_height))
        
        # Convert to float32
        img = resized.astype(np.float32)
        
        # Apply ImageNet mean subtraction
        img -= np.array([104, 117, 123], dtype=np.float32)
        
        # Format based on model requirements
        if self.model_format == 'NCHW':
            img = np.transpose(img, (2, 0, 1))  # HWC to CHW
            img = np.expand_dims(img, axis=0)    # Add batch dimension
        else:  # NHWC
            img = np.expand_dims(img, axis=0)    # Add batch dimension
            
        return img
        
    def _softmax(self, x):
        """Apply softmax to scores"""
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)
        
    def _decode_boxes(self, raw_boxes, anchors):
        """Decode bounding boxes from anchor-relative format"""
        # Check if boxes are already in corner format
        if raw_boxes.shape[-1] == 4:
            # Check if values suggest anchor-relative encoding
            if np.max(np.abs(raw_boxes)) < 10:
                # Anchor-relative decoding
                boxes = np.zeros_like(raw_boxes)
                boxes[:, 0] = anchors[:, 0] + raw_boxes[:, 0] * self.variances[0] * anchors[:, 2]
                boxes[:, 1] = anchors[:, 1] + raw_boxes[:, 1] * self.variances[0] * anchors[:, 3]
                boxes[:, 2] = anchors[:, 2] * np.exp(raw_boxes[:, 2] * self.variances[1])
                boxes[:, 3] = anchors[:, 3] * np.exp(raw_boxes[:, 3] * self.variances[1])
                
                # Convert from center to corner format
                boxes[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
                boxes[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
                boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
                boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
                
                return boxes
                
        return raw_boxes
        
    def _nms(self, boxes, scores):
        """Apply Non-Maximum Suppression"""
        if len(boxes) == 0:
            return []
            
        # Sort by score
        indices = np.argsort(scores)[::-1]
        
        keep = []
        while len(indices) > 0:
            # Take the first (highest score) box
            i = indices[0]
            keep.append(i)
            
            if len(indices) == 1:
                break
                
            # Calculate IoU with remaining boxes
            current_box = boxes[i]
            remaining_boxes = boxes[indices[1:]]
            
            # Intersection
            x1 = np.maximum(current_box[0], remaining_boxes[:, 0])
            y1 = np.maximum(current_box[1], remaining_boxes[:, 1])
            x2 = np.minimum(current_box[2], remaining_boxes[:, 2])
            y2 = np.minimum(current_box[3], remaining_boxes[:, 3])
            
            w = np.maximum(0, x2 - x1)
            h = np.maximum(0, y2 - y1)
            intersection = w * h
            
            # Union
            current_area = (current_box[2] - current_box[0]) * (current_box[3] - current_box[1])
            remaining_areas = (remaining_boxes[:, 2] - remaining_boxes[:, 0]) * \
                            (remaining_boxes[:, 3] - remaining_boxes[:, 1])
            union = current_area + remaining_areas - intersection
            
            # IoU
            iou = intersection / (union + 1e-6)
            
            # Keep boxes with IoU below threshold
            remaining_indices = indices[1:][iou < self.nms_threshold]
            indices = remaining_indices
            
        return keep
        
    def _process_outputs(self, outputs, orig_height, orig_width):
        """Process model outputs to get detections"""
        # Identify outputs by name or shape
        boxes_output = None
        scores_output = None
        
        for i, name in enumerate(self.output_names):
            output = outputs[i]
            name_lower = name.lower()
            
            if 'box' in name_lower or 'loc' in name_lower:
                boxes_output = output
            elif 'score' in name_lower or 'conf' in name_lower or 'cls' in name_lower:
                scores_output = output
                
        # Fallback to shape-based detection
        if boxes_output is None or scores_output is None:
            for output in outputs:
                shape = output.shape
                if len(shape) >= 2:
                    last_dim = shape[-1]
                    if last_dim == 4:
                        boxes_output = output
                    elif last_dim == 2 or last_dim == 1:
                        scores_output = output
                        
        if boxes_output is None or scores_output is None:
            return []
            
        # Remove batch dimension if present
        if boxes_output.ndim > 2:
            boxes_output = boxes_output.squeeze(0)
        if scores_output.ndim > 2:
            scores_output = scores_output.squeeze(0)
            
        # Process scores
        if scores_output.shape[-1] == 2:
            # Two-class output, apply softmax and get face class
            scores = self._softmax(scores_output)[:, 1]
        else:
            # Single class output
            scores = scores_output.squeeze(-1)
            
        # Decode boxes
        boxes = self._decode_boxes(boxes_output, self.anchors[:len(boxes_output)])
        
        # Scale to original image size
        scale_x = orig_width / self.model_width
        scale_y = orig_height / self.model_height
        
        boxes[:, 0] *= self.model_width * scale_x
        boxes[:, 1] *= self.model_height * scale_y
        boxes[:, 2] *= self.model_width * scale_x
        boxes[:, 3] *= self.model_height * scale_y
        
        # Filter by confidence
        valid_indices = scores > self.confidence_threshold
        boxes = boxes[valid_indices]
        scores = scores[valid_indices]
        
        # Filter by size and aspect ratio
        if len(boxes) > 0:
            widths = boxes[:, 2] - boxes[:, 0]
            heights = boxes[:, 3] - boxes[:, 1]
            areas = widths * heights
            aspect_ratios = widths / (heights + 1e-6)
            
            valid = (areas > 400) & (aspect_ratios > 0.4) & (aspect_ratios < 2.5)
            boxes = boxes[valid]
            scores = scores[valid]
            
        # Clamp to image bounds
        boxes[:, 0] = np.clip(boxes[:, 0], 0, orig_width)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, orig_height)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, orig_width)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, orig_height)
        
        # Apply NMS
        if len(boxes) > 0:
            keep_indices = self._nms(boxes, scores)
            boxes = boxes[keep_indices]
            scores = scores[keep_indices]
            
        # Format output
        detections = []
        for box, score in zip(boxes, scores):
            detections.append({
                'bbox': box.tolist(),
                'confidence': float(score)
            })
            
        return detections
        
    def _detection_loop(self):
        """Main detection loop running in separate thread"""
        first_frame = True
        
        while self.is_running:
            try:
                # Get frame from input queue
                frame_data = self.input_queue.get(timeout=0.1)
                if frame_data is None:
                    break
                    
                frame, frame_id = frame_data
                start_time = time.time()
                
                # Preprocess
                preprocessed = self._preprocess_image(frame)
                
                # Run inference
                outputs = self.model.run(self.output_names, {self.input_name: preprocessed})
                
                # Process outputs
                detections = self._process_outputs(outputs, frame.shape[0], frame.shape[1])
                
                # Update statistics
                self.detection_count += len(detections)
                self.parse_count += 1
                
                # Calculate FPS
                process_time = time.time() - start_time
                if process_time > 0:
                    self.fps_tracker.append(1.0 / process_time)
                    if len(self.fps_tracker) > 30:
                        self.fps_tracker.pop(0)
                        
                # Debug output
                if first_frame:
                    print(f"First frame shape: {frame.shape}")
                    print(f"Preprocessed shape: {preprocessed.shape}")
                    for i, name in enumerate(self.output_names):
                        print(f"Output {name}: {outputs[i].shape}")
                    first_frame = False
                    
                if self.parse_count % 30 == 0:
                    scores = [d['confidence'] for d in detections]
                    if scores:
                        print(f"Frame {frame_id}: {len(detections)} detections above threshold")
                        print(f"Max confidence: {max(scores):.3f}")
                        print(f"Top 5 scores: {sorted(scores, reverse=True)[:5]}")
                        
                # Show first few detections
                if self.parse_count <= 3 and detections:
                    print(f"Frame {frame_id} detections: {detections[:2]}")
                    
                # Send to output queue
                try:
                    self.output_queue.put((frame_id, detections), timeout=0.001)
                except Full:
                    pass
                    
                # Send debug info if requested
                if self.debug_queue is not None:
                    debug_info = {
                        'frame_id': frame_id,
                        'raw_detections': detections,
                        'processing_mode': 'direct',
                        'frame_shape': frame.shape
                    }
                    try:
                        self.debug_queue.put(debug_info, timeout=0.001)
                    except:
                        pass
                        
            except Empty:
                continue
            except Exception as e:
                print(f"Error in detection loop: {e}")
                continue
                
    def start(self):
        """Start the async detection thread"""
        if self.is_running:
            return
            
        self.is_running = True
        self.detection_thread = threading.Thread(target=self._detection_loop)
        self.detection_thread.start()
        print("RetinaFaceDetector started")
        
    def stop(self):
        """Stop the detection thread and cleanup resources"""
        self.is_running = False
        
        # Send stop signal
        try:
            self.input_queue.put(None, timeout=0.1)
        except:
            pass
            
        # Wait for thread to finish
        if self.detection_thread:
            self.detection_thread.join(timeout=1.0)
            
        # Cleanup
        self.executor.shutdown(wait=False)
        
        # Clear queues
        while not self.input_queue.empty():
            try:
                self.input_queue.get_nowait()
            except:
                break
                
        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except:
                break
                
        print("RetinaFaceDetector stopped")
        
    def submit_frame(self, frame: np.ndarray, frame_id: int) -> bool:
        """Submit an RGB frame for detection"""
        try:
            self.input_queue.put((frame, frame_id), timeout=0.001)
            return True
        except Full:
            return False
            
    def get_detections(self, timeout: float = 0.001) -> Optional[Tuple[int, List[Dict]]]:
        """Get detection results"""
        try:
            return self.output_queue.get(timeout=timeout)
        except Empty:
            return None
            
    def get_stats(self) -> Dict:
        """Get performance statistics"""
        avg_fps = 0.0
        if self.fps_tracker:
            avg_fps = sum(self.fps_tracker) / len(self.fps_tracker)
            
        return {
            'avg_fps': avg_fps,
            'input_queue_size': self.input_queue.qsize(),
            'output_queue_size': self.output_queue.qsize(),
            'processing_mode': 'direct'
        }