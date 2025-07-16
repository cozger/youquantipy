import cv2
import numpy as np
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
import time
from typing import List, Tuple, Optional, Dict
import logging
import os
import sys

class RetinaFaceDetector:
    def __init__(self, 
                 model_path: str,
                 tile_size: int = 640,
                 overlap: float = 0.2,
                 confidence_threshold: float = 0.3,
                 nms_threshold: float = 0.4,
                 max_workers: int = 4,
                 debug_queue=None):
        """
        Async RetinaFace detector based on working standalone version.
        
        Args:
            model_path: Path to RetinaFace ONNX model
            confidence_threshold: Detection confidence threshold
            nms_threshold: NMS IoU threshold
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.debug_queue = debug_queue
        
        # Model dimensions (will be detected from model)
        self.model_width = 640
        self.model_height = 640  
        self.model_format = 'NCHW'  # Default to NCHW
        
        # Anchor parameters
        self.anchors = None
        self.variances = [0.1, 0.2]
        
        # Initialize model
        self.model = None
        self._init_model()
        
        # Async processing components
        self.input_queue = queue.Queue(maxsize=5)
        self.output_queue = queue.Queue(maxsize=5)
        self.is_running = False
        self.detection_thread = None
        
        # Performance tracking
        self.fps_tracker = []
        
        # Initialize executor (not used in current implementation but needed for shutdown)
        self.executor = ThreadPoolExecutor(max_workers=1)
        
    def _init_model(self):
        """Initialize ONNX model and detect format."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
        print(f"Loading RetinaFace model from: {self.model_path}")
        
        try:
            import onnxruntime as ort
        except ImportError:
            print("ERROR: onnxruntime is not installed!")
            print("Please install it using: pip install onnxruntime")
            sys.exit(1)
            
        self.model = ort.InferenceSession(self.model_path)
        self.input_name = self.model.get_inputs()[0].name
        self.output_names = [out.name for out in self.model.get_outputs()]
        
        # Analyze model input format
        input_info = self.model.get_inputs()[0]
        input_shape = input_info.shape
        print(f"Model input: {self.input_name}, shape: {input_shape}, type: {input_info.type}")
        
        # Detect format from shape
        if isinstance(input_shape, list) and len(input_shape) == 4:
            # Look for dimension with value 3 (channels)
            for i, dim in enumerate(input_shape):
                if isinstance(dim, (int, float)) and dim == 3:
                    if i == 1:
                        self.model_format = 'NCHW'
                        print("Detected NCHW format (channels at index 1)")
                    elif i == 3:
                        self.model_format = 'NHWC'
                        print("Detected NHWC format (channels at index 3)")
                    break
            
            # Extract height and width
            numeric_dims = []
            for dim in input_shape:
                if isinstance(dim, (int, float)) and dim > 3:
                    numeric_dims.append(int(dim))
                elif isinstance(dim, str) and dim.isdigit():
                    numeric_dims.append(int(dim))
            
            if len(numeric_dims) >= 2:
                sorted_dims = sorted(numeric_dims, reverse=True)
                self.model_width = sorted_dims[0]
                self.model_height = sorted_dims[1] if len(sorted_dims) > 1 else sorted_dims[0]
            elif len(numeric_dims) == 1:
                self.model_width = numeric_dims[0]
                self.model_height = numeric_dims[0]
            
            # Handle common RetinaFace input sizes
            if self.model_format == 'NCHW':
                if len(input_shape) == 4:
                    if isinstance(input_shape[2], (int, float)) and input_shape[2] > 0:
                        self.model_height = int(input_shape[2])
                    if isinstance(input_shape[3], (int, float)) and input_shape[3] > 0:
                        self.model_width = int(input_shape[3])
        
        print(f"Detected model format: {self.model_format}")
        print(f"Model dimensions: {self.model_width}x{self.model_height}")
        print(f"Model outputs: {[(out.name, out.shape) for out in self.model.get_outputs()]}")
        
        # Generate anchors based on model dimensions
        self._generate_anchors()
        print(f"Generated {len(self.anchors)} anchors")
    
    def _generate_anchors(self):
        """Generate anchors for RetinaFace."""
        # Try to detect anchor configuration from model outputs
        output_shapes = [out.shape for out in self.model.get_outputs()]
        print(f"Output shapes: {output_shapes}")
        
        # Default configuration for 640x640 model
        if self.model_width == 640 and self.model_height == 640:
            min_sizes = [[16, 32], [64, 128], [256, 512]]
            steps = [8, 16, 32]
        else:
            # Original configuration for other sizes
            min_sizes = [[64, 96, 128], [128, 192], [256, 384], [384, 512, 640]]
            steps = [8, 16, 32, 64]
        
        # Check if we can determine anchor count from output shapes
        if output_shapes and len(output_shapes) > 0:
            first_output_shape = output_shapes[0]
            if isinstance(first_output_shape, list) and len(first_output_shape) >= 2:
                # Try to find the anchor dimension
                for dim in first_output_shape:
                    if isinstance(dim, (int, float)) and dim > 1000:
                        anchor_count = int(dim)
                        print(f"Detected {anchor_count} anchors from output shape")
                        
                        # Adjust configuration based on anchor count
                        if anchor_count == 16800:  # Common for 640x640
                            min_sizes = [[16, 32], [64, 128], [256, 512]]
                            steps = [8, 16, 32]
                        elif anchor_count == 25600:  # Another common configuration
                            min_sizes = [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]]
                            steps = [8, 16, 32, 64]
                        break
        
        anchors = []
        
        for k, step in enumerate(steps):
            feature_h = self.model_height // step
            feature_w = self.model_width // step
            
            for i in range(feature_h):
                for j in range(feature_w):
                    for min_size in min_sizes[k]:
                        cx = (j + 0.5) * step / self.model_width
                        cy = (i + 0.5) * step / self.model_height
                        w = min_size / self.model_width
                        h = min_size / self.model_height
                        anchors.append([cx, cy, w, h])
        
        self.anchors = np.array(anchors, dtype=np.float32)
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for RetinaFace."""
        # Resize to model input size
        img = cv2.resize(image, (self.model_width, self.model_height))
        
        # Input is already RGB from frame distributor
        # Convert to float32 and normalize
        img = img.astype(np.float32)
        # Use RGB mean values since input is RGB (not BGR)
        img -= (123, 117, 104)  # RGB mean (reversed from BGR)
        
        # Format based on model requirements
        if self.model_format == 'NCHW':
            # Transpose from HWC to CHW
            img = img.transpose(2, 0, 1)
            # Add batch dimension -> [1, 3, H, W]
            img = np.expand_dims(img, axis=0)
        else:  # NHWC
            # Add batch dimension -> [1, H, W, 3]
            img = np.expand_dims(img, axis=0)
        
        # Only print shape on first call
        if not hasattr(self, '_preprocess_count'):
            self._preprocess_count = 0
        self._preprocess_count += 1
        
        if self._preprocess_count == 1:
            print(f"Preprocessed image shape: {img.shape} (format: {self.model_format})")
            print(f"Expected format: {self.model_format}")
        
        return img
    
    def start(self):
        """Start the async detection thread."""
        if not self.is_running:
            self.is_running = True
            self.detection_thread = threading.Thread(target=self._detection_loop)
            self.detection_thread.daemon = True
            self.detection_thread.start()
            logging.info("RetinaFace detector started")
    
    def stop(self):
        """Stop the async detection thread."""
        self.is_running = False
        if self.detection_thread:
            self.detection_thread.join(timeout=1.0)
        self.executor.shutdown(wait=False)
        logging.info("RetinaFace detector stopped")
    
    def submit_frame(self, frame: np.ndarray, frame_id: int) -> bool:
        """
        Submit a frame for detection.
        
        Args:
            frame: Input frame (4K or any resolution)
            frame_id: Unique frame identifier
            
        Returns:
            True if frame was queued, False if queue is full
        """
        if not self.is_running:
            print(f"[RETINAFACE] Cannot submit frame {frame_id}: detector not running")
            return False
            
        try:
            self.input_queue.put_nowait((frame, frame_id))
            print(f"[RETINAFACE] Submitted frame {frame_id} for detection, shape: {frame.shape}")
            return True
        except queue.Full:
            print(f"[RETINAFACE] Frame queue full, dropping frame {frame_id}")
            return False
    
    def get_detections(self, timeout: float = 0.001) -> Optional[Tuple[int, List[Dict]]]:
        """
        Get detection results.
        
        Returns:
            Tuple of (frame_id, detections) or None if no results available
        """
        try:
            return self.output_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def _detection_loop(self):
        """Main detection loop running in separate thread."""
        while self.is_running:
            try:
                # Get frame from queue
                frame_data = self.input_queue.get(timeout=0.1)
                if frame_data is None:
                    continue
                    
                frame, frame_id = frame_data
                start_time = time.time()
                
                # Run detection on full frame
                detections = self.detect(frame)
                
                # Apply NMS
                final_detections = self._apply_nms(detections)
                
                # Send to debug queue if available
                if self.debug_queue is not None:
                    try:
                        debug_data = {
                            'frame_id': frame_id,
                            'raw_detections': detections,
                            'final_detections': final_detections,
                            'frame_shape': frame.shape
                        }
                        self.debug_queue.put_nowait(debug_data)
                    except:
                        pass
                
                # Calculate FPS
                detection_time = time.time() - start_time
                self.fps_tracker.append(1.0 / detection_time)
                if len(self.fps_tracker) > 30:
                    self.fps_tracker.pop(0)
                
                # Output results
                self.output_queue.put((frame_id, final_detections))
                
                # Log detection results more frequently for debugging
                if frame_id % 10 == 0 or len(final_detections) > 0:
                    print(f"[RETINAFACE] Frame {frame_id}: {len(detections)} raw detections, {len(final_detections)} after NMS")
                    if len(final_detections) > 0:
                        for i, det in enumerate(final_detections[:3]):
                            print(f"  Face {i}: bbox={det['bbox']}, conf={det['confidence']:.3f}")
                
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Detection error: {e}")
    
    def detect(self, image: np.ndarray) -> List[Dict]:
        """Run detection on image."""
        h, w = image.shape[:2]
        
        # Debug: Check image format periodically
        if not hasattr(self, '_detect_count'):
            self._detect_count = 0
        self._detect_count += 1
        
        # Log that detect is being called
        if self._detect_count <= 5 or self._detect_count % 50 == 0:
            print(f"[RETINAFACE] detect() called for frame #{self._detect_count}, image shape: {image.shape}")
        
        if self._detect_count % 30 == 0:
            # Sample a few pixels to check if it looks like RGB
            center_y, center_x = h // 2, w // 2
            sample = image[center_y, center_x]
            print(f"[RETINAFACE DEBUG] Image shape: {image.shape}, center pixel RGB: {sample}, dtype: {image.dtype}")
        
        # Preprocess
        input_tensor = self.preprocess(image)
        
        # Run inference
        try:
            outputs = self.model.run(self.output_names, {self.input_name: input_tensor})
        except Exception as e:
            print(f"Inference error: {e}")
            print(f"Input tensor shape: {input_tensor.shape}")
            print(f"Expected input name: {self.input_name}")
            raise
        
        # Parse detections
        detections = self._parse_detections(outputs, w, h)
        
        # Log detection results for debugging
        if self._detect_count % 10 == 0 or len(detections) > 0:
            print(f"[RETINAFACE DEBUG] Frame detect #{self._detect_count}: {len(detections)} raw detections")
        
        # Apply NMS
        final_detections = self._apply_nms(detections)
        
        if self._detect_count % 10 == 0 or len(final_detections) > 0:
            print(f"[RETINAFACE DEBUG] Frame detect #{self._detect_count}: {len(final_detections)} detections after NMS")
        
        return final_detections
    
    def _apply_nms(self, detections: List[Dict]) -> List[Dict]:
        """Apply Non-Maximum Suppression."""
        if not detections:
            return []
            
        boxes = np.array([d['bbox'] for d in detections])
        scores = np.array([d['confidence'] for d in detections])
        
        # NMS
        indices = self._nms(boxes, scores, self.nms_threshold)
        
        return [detections[i] for i in indices]
    
    def _nms(self, boxes: np.ndarray, scores: np.ndarray, threshold: float) -> List[int]:
        """Non-Maximum Suppression."""
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            
            ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            
            inds = np.where(ovr <= threshold)[0]
            order = order[inds + 1]
            
        return keep
    
    def _parse_detections(self, outputs: List[np.ndarray], img_w: int, img_h: int) -> List[Dict]:
        """Parse model outputs."""
        detections = []
        
        # Only print output info on first detection
        if not hasattr(self, '_parse_count'):
            self._parse_count = 0
        self._parse_count += 1
        
        if self._parse_count == 1:
            print(f"\nParsing outputs: {len(outputs)} arrays")
            for i, out in enumerate(outputs):
                print(f"Output {i}: shape={out.shape}, dtype={out.dtype}, name={self.model.get_outputs()[i].name}")
        
        if len(outputs) >= 2:
            # Try to identify which output is boxes and which is scores
            boxes = None
            scores = None
            landmarks = None
            
            for i, out in enumerate(outputs):
                out_name = self.model.get_outputs()[i].name.lower()
                
                # Remove batch dimension if present
                if len(out.shape) == 3 and out.shape[0] == 1:
                    out = out[0]
                
                # Identify by name or shape
                if 'box' in out_name or 'loc' in out_name or 'bbox' in out_name:
                    boxes = out
                    if self._parse_count == 1:
                        print(f"Identified boxes by name: shape={boxes.shape}")
                elif 'score' in out_name or 'conf' in out_name or 'cls' in out_name or 'class' in out_name:
                    scores = out
                    if self._parse_count == 1:
                        print(f"Identified scores by name: shape={scores.shape}")
                elif 'landmark' in out_name or 'ldmk' in out_name:
                    landmarks = out
                    if self._parse_count == 1:
                        print(f"Identified landmarks by name: shape={landmarks.shape}")
            
            # If name matching fails, use shape-based detection
            if boxes is None or scores is None:
                for i, out in enumerate(outputs):
                    # Remove batch dimension if present
                    if len(out.shape) == 3 and out.shape[0] == 1:
                        out = out[0]
                    
                    # Check shape patterns
                    if boxes is None and len(out.shape) == 2 and out.shape[-1] == 4:
                        boxes = out
                        if self._parse_count == 1:
                            print(f"Identified boxes by shape: shape={boxes.shape}")
                    elif scores is None and len(out.shape) == 2 and out.shape[-1] in [1, 2]:
                        scores = out
                        if self._parse_count == 1:
                            print(f"Identified scores by shape: shape={scores.shape}")
                    elif landmarks is None and len(out.shape) == 2 and out.shape[-1] == 10:
                        landmarks = out
                        if self._parse_count == 1:
                            print(f"Identified landmarks by shape: shape={landmarks.shape}")
            
            # Final fallback to positional
            if boxes is None or scores is None:
                boxes = outputs[0]
                scores = outputs[1] if len(outputs) > 1 else None
                
                # Remove batch dimensions
                if len(boxes.shape) == 3:
                    boxes = boxes[0]
                if scores is not None and len(scores.shape) == 3:
                    scores = scores[0]
            
            if boxes is None or scores is None:
                print("Could not identify boxes and scores in outputs")
                return detections
                
            # Process scores
            if len(scores.shape) == 2 and scores.shape[1] == 2:
                # Apply softmax for two-class scores
                scores_exp = np.exp(scores - np.max(scores, axis=1, keepdims=True))
                scores_softmax = scores_exp / np.sum(scores_exp, axis=1, keepdims=True)
                scores = scores_softmax[:, 1]  # Face class
            elif len(scores.shape) == 2 and scores.shape[1] == 1:
                # Single score per detection
                scores = scores.flatten()
            elif len(scores.shape) == 1:
                # Already single scores
                pass
            else:
                if self._parse_count == 1:
                    print(f"Unexpected score shape: {scores.shape}")
                # Try to use as-is
                if scores.shape[-1] > 1:
                    scores = scores[:, -1]  # Take last column
                else:
                    scores = scores.flatten()
                
            # Scale factors
            scale_x = img_w / self.model_width
            scale_y = img_h / self.model_height
            
            if self._parse_count % 30 == 1:  # Print every 30 frames
                print(f"Processing {len(scores)} detections, threshold={self.confidence_threshold}")
                if len(scores) > 0:
                    print(f"Max score: {np.max(scores):.3f}, scores > threshold: {np.sum(scores > self.confidence_threshold)}")
                    # Show distribution of scores
                    if np.max(scores) > 0.1:
                        top_scores = np.sort(scores)[-5:][::-1]
                        print(f"Top 5 scores: {top_scores}")
            
            # Process high-confidence detections
            for i in range(len(scores)):
                if scores[i] > self.confidence_threshold:
                    box = boxes[i]
                    
                    # Check if box values look like they need anchor decoding
                    box_range = np.max(np.abs(box))
                    
                    if box_range < 10 and self.anchors is not None and i < len(self.anchors):
                        # Small values suggest anchor-relative encoding
                        anchor = self.anchors[i]
                        
                        # Decode box using anchors
                        cx = anchor[0] + box[0] * self.variances[0] * anchor[2]
                        cy = anchor[1] + box[1] * self.variances[0] * anchor[3]
                        w = anchor[2] * np.exp(box[2] * self.variances[1])
                        h = anchor[3] * np.exp(box[3] * self.variances[1])
                        
                        # Convert to corner format
                        x1 = (cx - w / 2) * self.model_width
                        y1 = (cy - h / 2) * self.model_height
                        x2 = (cx + w / 2) * self.model_width
                        y2 = (cy + h / 2) * self.model_height
                    else:
                        # Values look like direct coordinates or normalized coordinates
                        if box_range <= 1.0:
                            # Normalized coordinates [0, 1]
                            x1, y1, x2, y2 = box
                            x1 *= self.model_width
                            y1 *= self.model_height
                            x2 *= self.model_width
                            y2 *= self.model_height
                        else:
                            # Direct pixel coordinates
                            x1, y1, x2, y2 = box
                    
                    # Scale to original image size
                    x1 *= scale_x
                    y1 *= scale_y
                    x2 *= scale_x
                    y2 *= scale_y
                    
                    # Clamp to image bounds
                    x1 = max(0, min(x1, img_w - 1))
                    y1 = max(0, min(y1, img_h - 1))
                    x2 = max(0, min(x2, img_w - 1))
                    y2 = max(0, min(y2, img_h - 1))
                    
                    # Filter by size and aspect ratio
                    face_w = x2 - x1
                    face_h = y2 - y1
                    
                    if face_w < 20 or face_h < 20:  # Too small
                        continue
                        
                    aspect_ratio = face_w / face_h if face_h > 0 else 0
                    if aspect_ratio < 0.4 or aspect_ratio > 2.5:  # Bad aspect ratio
                        continue
                        
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': float(scores[i])
                    })
                    
                    # Debug first few detections
                    if len(detections) <= 3 and self._parse_count == 1:
                        print(f"Detection {i}: score={scores[i]:.3f}, box={box}, decoded=[{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]")
                        
        return detections
    
    
    
    def get_stats(self) -> Dict:
        """Get detector statistics."""
        avg_fps = np.mean(self.fps_tracker) if self.fps_tracker else 0
        return {
            'avg_fps': avg_fps,
            'input_queue_size': self.input_queue.qsize(),
            'output_queue_size': self.output_queue.qsize()
        }