import cv2
import numpy as np
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
import time
from typing import List, Tuple, Optional, Dict
import logging

class RetinaFaceDetector:
    def __init__(self, 
                 model_path: str,
                 tile_size: int = 640,
                 overlap: float = 0.2,
                 confidence_threshold: float = 0.7,
                 nms_threshold: float = 0.4,
                 max_workers: int = 4):
        """
        Async RetinaFace detector for 4K video processing.
        
        Args:
            model_path: Path to RetinaFace ONNX model
            tile_size: Size of tiles for processing (default 640x640)
            overlap: Overlap percentage between tiles (default 0.2)
            confidence_threshold: Detection confidence threshold
            nms_threshold: NMS IoU threshold
            max_workers: Number of parallel workers for tile processing
        """
        self.model_path = model_path
        self.tile_size = tile_size
        self.tile_width = tile_size  # Will be updated based on model
        self.tile_height = tile_size  # Will be updated based on model
        self.overlap = overlap
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.max_workers = max_workers
        
        # Initialize model (placeholder - will use ONNX runtime)
        self.model = None
        self._init_model()
        
        # Async processing components
        self.input_queue = queue.Queue(maxsize=5)
        self.output_queue = queue.Queue(maxsize=5)
        self.is_running = False
        self.detection_thread = None
        
        # Thread pool for parallel tile processing
        # Use threads instead of processes since we're already in a child process
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Performance tracking
        self.last_detection_time = 0
        self.fps_tracker = []
        
    def _init_model(self):
        """Initialize RetinaFace model using ONNX runtime."""
        try:
            import onnxruntime as ort
            self.model = ort.InferenceSession(self.model_path)
            self.input_name = self.model.get_inputs()[0].name
            self.output_names = [out.name for out in self.model.get_outputs()]
            
            # Get detailed model info
            input_info = self.model.get_inputs()[0]
            self.input_shape = input_info.shape
            self.input_type = input_info.type
            
            print("\n" + "="*60)
            print("[RETINAFACE] Model loaded successfully!")
            print(f"[RETINAFACE] Input name: {self.input_name}")
            print(f"[RETINAFACE] Input shape: {self.input_shape}")
            print(f"[RETINAFACE] Input type: {self.input_type}")
            
            # Determine expected dimensions
            if len(self.input_shape) == 4:
                # Extract height and width based on format
                if self.input_shape[1] == 3:  # NCHW format
                    self.model_format = 'NCHW'
                    self.model_height = self.input_shape[2]
                    self.model_width = self.input_shape[3]
                else:  # NHWC format
                    self.model_format = 'NHWC'
                    self.model_height = self.input_shape[1]
                    self.model_width = self.input_shape[2]
                
                print(f"[RETINAFACE] Model format: {self.model_format}")
                print(f"[RETINAFACE] Expected dimensions: {self.model_width}x{self.model_height}")
            
            print(f"[RETINAFACE] Outputs: {[(out.name, out.shape) for out in self.model.get_outputs()]}")
            print("="*60 + "\n")
            
            # Update tile dimensions to match model
            if hasattr(self, 'model_width') and hasattr(self, 'model_height'):
                self.tile_width = self.model_width
                self.tile_height = self.model_height
                print(f"[RETINAFACE] Updated tile dimensions to {self.tile_width}x{self.tile_height} to match model")
                
        except Exception as e:
            logging.warning(f"ONNX runtime not available, using mock detector: {e}")
            self.model = None
            self.model_format = 'NHWC'
            self.model_width = 640
            self.model_height = 640
    
    def start(self):
        """Start the async detection thread."""
        if not self.is_running:
            self.is_running = True
            self.detection_thread = threading.Thread(target=self._detection_loop)
            self.detection_thread.daemon = True
            self.detection_thread.start()
            print("[RETINAFACE] Detector started with tile_size={}x{}, overlap={}".format(
                self.tile_width, self.tile_height, self.overlap))
    
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
        try:
            self.input_queue.put_nowait((frame, frame_id))
            return True
        except queue.Full:
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
                
                # Generate tiles
                tiles, tile_coords = self._generate_tiles(frame)
                print(f"[RETINAFACE] Processing frame {frame_id}: {frame.shape} -> {len(tiles)} tiles")
                
                # Process tiles in parallel
                futures = []
                for i, (tile, coords) in enumerate(zip(tiles, tile_coords)):
                    future = self.executor.submit(self._detect_in_tile, tile, coords, i)
                    futures.append(future)
                
                # Collect results
                all_detections = []
                for future in futures:
                    tile_detections = future.result()
                    all_detections.extend(tile_detections)
                
                # Apply global NMS
                final_detections = self._global_nms(all_detections)
                
                # Calculate FPS
                detection_time = time.time() - start_time
                self.fps_tracker.append(1.0 / detection_time)
                if len(self.fps_tracker) > 30:
                    self.fps_tracker.pop(0)
                
                # Output results
                self.output_queue.put((frame_id, final_detections))
                
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Detection error: {e}")
    
    def _generate_tiles(self, frame: np.ndarray) -> Tuple[List[np.ndarray], List[Tuple[int, int, int, int]]]:
        """
        Generate overlapping tiles from the input frame.
        
        Returns:
            Tuple of (tiles, tile_coordinates)
        """
        h, w = frame.shape[:2]
        tiles = []
        coords = []
        
        # Use model's expected tile dimensions
        tile_width = self.tile_width
        tile_height = self.tile_height
        
        # Calculate strides for both dimensions
        stride_x = int(tile_width * (1 - self.overlap))
        stride_y = int(tile_height * (1 - self.overlap))
        
        # Ensure we cover the entire frame
        y_positions = list(range(0, h - tile_height + 1, stride_y))
        if y_positions and y_positions[-1] + tile_height < h:
            y_positions.append(h - tile_height)
        elif not y_positions:  # Frame smaller than tile
            y_positions = [0]
            
        x_positions = list(range(0, w - tile_width + 1, stride_x))
        if x_positions and x_positions[-1] + tile_width < w:
            x_positions.append(w - tile_width)
        elif not x_positions:  # Frame smaller than tile
            x_positions = [0]
        
        for y in y_positions:
            for x in x_positions:
                # Extract tile
                x_end = min(x + tile_width, w)
                y_end = min(y + tile_height, h)
                
                tile = frame[y:y_end, x:x_end]
                
                # Pad tile if necessary (edge cases)
                if tile.shape[0] < tile_height or tile.shape[1] < tile_width:
                    tile = cv2.copyMakeBorder(
                        tile,
                        0, tile_height - tile.shape[0],
                        0, tile_width - tile.shape[1],
                        cv2.BORDER_CONSTANT,
                        value=(0, 0, 0)
                    )
                
                tiles.append(tile)
                coords.append((x, y, x_end, y_end))
        
        return tiles, coords
    
    def _detect_in_tile(self, tile: np.ndarray, coords: Tuple[int, int, int, int], tile_id: int) -> List[Dict]:
        """
        Run detection on a single tile.
        
        Returns:
            List of detections with global coordinates
        """
        x_offset, y_offset, x_end, y_end = coords
        tile_w = x_end - x_offset
        tile_h = y_end - y_offset
        
        if self.model is None:
            # Mock detector for testing
            return self._mock_detect(tile, x_offset, y_offset)
        
        # Preprocess tile (resizes to 640x608)
        input_tensor = self._preprocess_tile(tile)
        
        # Run inference
        outputs = self.model.run(self.output_names, {self.input_name: input_tensor})
        
        # Parse detections with scale factors
        scale_x = tile_w / float(self.tile_width)   # Scale from model width back to tile width
        scale_y = tile_h / float(self.tile_height)  # Scale from model height back to tile height
        detections = self._parse_detections(outputs, x_offset, y_offset, scale_x, scale_y)
        
        return detections
    
    def _preprocess_tile(self, tile: np.ndarray) -> np.ndarray:
        """Preprocess tile for RetinaFace model."""
        # Get expected dimensions
        target_width = self.tile_width
        target_height = self.tile_height
        
        # Debug input tile
        if hasattr(self, '_preprocess_count'):
            self._preprocess_count += 1
        else:
            self._preprocess_count = 1
            
        if self._preprocess_count % 100 == 1:  # Log every 100th tile
            print(f"[RETINAFACE] Preprocessing tile {self._preprocess_count}: input shape={tile.shape}, target={target_width}x{target_height}")
        
        # Resize to model's expected dimensions (width, height for cv2.resize)
        if tile.shape[:2] != (target_height, target_width):
            img = cv2.resize(tile, (target_width, target_height))
        else:
            img = tile.copy()
        
        # Standard RetinaFace preprocessing
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        img -= (104, 117, 123)  # ImageNet mean subtraction
        
        # Format based on model requirements
        if hasattr(self, 'model_format'):
            if self.model_format == 'NCHW':
                # Transpose to CHW format
                img = img.transpose(2, 0, 1)
                img = np.expand_dims(img, axis=0)  # Add batch dimension
                if self._preprocess_count % 100 == 1:
                    print(f"[RETINAFACE] Output shape (NCHW): {img.shape}")
            else:  # NHWC
                # Keep HWC format
                img = np.expand_dims(img, axis=0)  # Add batch dimension
                if self._preprocess_count % 100 == 1:
                    print(f"[RETINAFACE] Output shape (NHWC): {img.shape}")
        else:
            # Default to NHWC if not determined
            img = np.expand_dims(img, axis=0)
            
        return img
    
    def _parse_detections(self, outputs: List[np.ndarray], x_offset: int, y_offset: int, 
                         scale_x: float = 1.0, scale_y: float = 1.0) -> List[Dict]:
        """Parse model outputs and convert to global coordinates."""
        detections = []
        
        try:
            # This is a simplified version - actual parsing depends on model output format
            # Typical RetinaFace outputs: boxes, scores, landmarks
            if len(outputs) >= 2:
                boxes = outputs[0]      # Shape: [15960, 4]
                scores = outputs[1]     # Shape: [15960, 2] 
                landmarks = outputs[2] if len(outputs) > 2 else None  # Shape: [15960, 10]
                
                # Handle batch dimension if present
                if len(boxes.shape) == 3:
                    boxes = boxes[0]
                if len(landmarks.shape) == 3 and landmarks is not None:
                    landmarks = landmarks[0]
                
                # Extract face confidence scores (column 1)
                # Column 0 is background, column 1 is face
                if len(scores.shape) == 2 and scores.shape[1] == 2:
                    scores = scores[:, 1]  # Get face confidence scores
                elif len(scores.shape) == 3:
                    scores = scores[0, :, 1]  # Remove batch dim and get face scores
                    
                # Now iterate through detections
                # Debug: Check score distribution
                if len(scores) > 0:
                    max_score = np.max(scores)
                    num_above_threshold = np.sum(scores > self.confidence_threshold)
                    if self._preprocess_count % 100 == 1:
                        print(f"[RETINAFACE] Scores: max={max_score:.3f}, above {self.confidence_threshold}: {num_above_threshold}/{len(scores)}")
                
                for i in range(len(scores)):
                    if scores[i] > self.confidence_threshold:
                        box = boxes[i]
                        # Scale coordinates from model size back to tile size, then add offset
                        detection = {
                            'bbox': [
                                box[0] * scale_x + x_offset,
                                box[1] * scale_y + y_offset,
                                box[2] * scale_x + x_offset,
                                box[3] * scale_y + y_offset
                            ],
                            'confidence': float(scores[i]),
                            'landmarks': None
                        }
                        
                        if landmarks is not None and i < len(landmarks):
                            lm = landmarks[i].reshape(-1, 2)  # Reshape to 5x2
                            # Scale landmarks
                            detection['landmarks'] = lm * np.array([scale_x, scale_y]) + np.array([x_offset, y_offset])
                        
                        detections.append(detection)
        except Exception as e:
            print(f"[RETINAFACE] Error parsing detections: {e}")
            print(f"[RETINAFACE] Output shapes: {[out.shape for out in outputs]}")
            import traceback
            traceback.print_exc()
        
        return detections
    
    def _mock_detect(self, tile: np.ndarray, x_offset: int, y_offset: int) -> List[Dict]:
        """Mock detector for testing without model."""
        # Simple face detection using OpenCV for testing
        gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
        
        # Use simple edge detection to simulate faces
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # Minimum area threshold
                x, y, w, h = cv2.boundingRect(contour)
                if w > 30 and h > 30:  # Minimum size
                    detection = {
                        'bbox': [
                            x + x_offset,
                            y + y_offset,
                            x + w + x_offset,
                            y + h + y_offset
                        ],
                        'confidence': 0.8,
                        'landmarks': None
                    }
                    detections.append(detection)
        
        return detections
    
    def _global_nms(self, detections: List[Dict]) -> List[Dict]:
        """Apply Non-Maximum Suppression across all detections."""
        if not detections:
            return []
        
        # Extract boxes and scores
        boxes = np.array([d['bbox'] for d in detections])
        scores = np.array([d['confidence'] for d in detections])
        
        # Apply NMS
        indices = self._nms(boxes, scores, self.nms_threshold)
        
        # Return filtered detections
        return [detections[i] for i in indices]
    
    def _nms(self, boxes: np.ndarray, scores: np.ndarray, threshold: float) -> List[int]:
        """Non-Maximum Suppression implementation."""
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
    
    def get_stats(self) -> Dict:
        """Get detector statistics."""
        avg_fps = np.mean(self.fps_tracker) if self.fps_tracker else 0
        return {
            'avg_fps': avg_fps,
            'input_queue_size': self.input_queue.qsize(),
            'output_queue_size': self.output_queue.qsize()
        }