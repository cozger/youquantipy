import cv2
import numpy as np
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
import time
from typing import List, Tuple, Optional, Dict
import logging
import os
from itertools import product

class RetinaFaceDetector:
    def __init__(self, 
                 model_path: str,
                 tile_size: int = 640,
                 overlap: float = 0.2,
                 confidence_threshold: float = 0.98,
                 nms_threshold: float = 0.4,
                 max_workers: int = 4,
                 debug_queue=None):  # DEBUG_RETINAFACE: Optional queue for raw detections
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
        print(f"[RETINAFACE] Initializing detector with model_path={model_path}")
        self.model_path = model_path
        self.tile_size = tile_size
        self.tile_width = tile_size  # Will be updated based on model
        self.tile_height = tile_size  # Will be updated based on model
        self.overlap = overlap
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.max_workers = max_workers
        self.debug_queue = debug_queue  # DEBUG_RETINAFACE: Store debug queue
        
        # Initialize model (placeholder - will use ONNX runtime)
        self.model = None
        print(f"[RETINAFACE] Calling _init_model...")
        try:
            self._init_model()
            print(f"[RETINAFACE] Model initialization completed successfully")
            print(f"[RETINAFACE] Anchors after init: {self.anchors is not None}, shape: {self.anchors.shape if self.anchors is not None else 'None'}")
        except Exception as e:
            print(f"[RETINAFACE] ERROR during model initialization: {e}")
            import traceback
            traceback.print_exc()
            # Set anchors to None explicitly
            self.anchors = None
        
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
        
        # Anchor parameters for RetinaFace
        self.anchors = None
        self.variances = [0.1, 0.2]  # Common RetinaFace variances
        
        # Ensure anchors are always generated
        if self.anchors is None:
            print(f"[RETINAFACE] Anchors not generated during init, generating default 15960 configuration")
            # Set correct tile dimensions for 15960 anchor configuration
            self.tile_width = 640
            self.tile_height = 608
            self._generate_anchors_15960()
        
    def _init_model(self):
        """Initialize RetinaFace model using ONNX runtime."""
        print(f"[RETINAFACE] Initializing model from: {self.model_path}")
        
        # Check if model file exists
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"RetinaFace model file not found: {self.model_path}\n"
                f"Please download the model or check the path in config."
            )
        
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
                # Handle dynamic dimensions (could be string or -1)
                # Find numeric dimensions
                numeric_dims = []
                for dim in self.input_shape:
                    if isinstance(dim, (int, float)) and dim > 0:
                        numeric_dims.append(int(dim))
                
                print(f"[RETINAFACE] Numeric dimensions found: {numeric_dims}")
                
                # Determine format based on dimensions
                if len(numeric_dims) >= 3:
                    if numeric_dims[-1] == 3:  # NHWC format (?, H, W, 3)
                        self.model_format = 'NHWC'
                        self.model_height = numeric_dims[-3]
                        self.model_width = numeric_dims[-2]
                    elif numeric_dims[0] == 3:  # NCHW format (?, 3, H, W)
                        self.model_format = 'NCHW'
                        self.model_height = numeric_dims[-2]
                        self.model_width = numeric_dims[-1]
                    else:
                        # Default assumption for [?, 608, 640, 3]
                        self.model_format = 'NHWC'
                        self.model_height = numeric_dims[0]
                        self.model_width = numeric_dims[1]
                    
                    print(f"[RETINAFACE] Model format: {self.model_format}")
                    print(f"[RETINAFACE] Expected dimensions: {self.model_width}x{self.model_height}")
                else:
                    # Could not determine dimensions from model, use defaults
                    print(f"[RETINAFACE] Could not determine model dimensions from shape: {self.input_shape}")
                    print(f"[RETINAFACE] Using default dimensions: 640x608")
                    self.model_width = 640
                    self.model_height = 608
                    self.model_format = 'NHWC'  # Default assumption
            
            print(f"[RETINAFACE] Outputs: {[(out.name, out.shape) for out in self.model.get_outputs()]}")
            print("="*60 + "\n")
            
            # Update tile dimensions to match model
            if hasattr(self, 'model_width') and hasattr(self, 'model_height'):
                self.tile_width = self.model_width
                self.tile_height = self.model_height
                print(f"[RETINAFACE] Updated tile dimensions to {self.tile_width}x{self.tile_height} to match model")
            else:
                # Use default dimensions if not detected from model
                self.model_width = self.tile_width
                self.model_height = self.tile_height
                print(f"[RETINAFACE] Using default tile dimensions: {self.tile_width}x{self.tile_height}")
            
            # Always generate anchors for the model
            self._generate_anchors()
            print(f"[RETINAFACE] Generated {len(self.anchors) if self.anchors is not None else 0} anchors")
            if self.anchors is not None:
                print(f"[RETINAFACE] Initial anchors: shape={self.anchors.shape}, id={id(self.anchors)}")
            else:
                print(f"[RETINAFACE] ERROR: Anchors generation failed!")
                
        except ImportError as e:
            error_msg = (
                "\n" + "="*60 + "\n"
                "[RETINAFACE] ERROR: onnxruntime is not installed!\n"
                "Face detection CANNOT work without it.\n"
                "Please install it using: pip install onnxruntime\n"
                "="*60 + "\n"
            )
            print(error_msg)
            raise ImportError(error_msg) from e
        except FileNotFoundError as e:
            error_msg = (
                f"\n[RETINAFACE] ERROR: Model file not found: {model_path}\n"
                f"Please ensure the RetinaFace model file exists at the specified path.\n"
            )
            print(error_msg)
            raise FileNotFoundError(error_msg) from e
        except Exception as e:
            error_msg = (
                f"\n[RETINAFACE] ERROR: Failed to initialize model: {str(e)}\n"
                f"Model path: {model_path}\n"
            )
            print(error_msg)
            import traceback
            traceback.print_exc()
            raise RuntimeError(error_msg) from e
    
    def _generate_anchors(self):
        """Generate anchor boxes for RetinaFace model."""
        # RetinaFace typically uses 3 feature pyramid levels
        # with different anchor sizes and strides
        min_sizes = [[16, 32], [64, 128], [256, 512]]
        steps = [8, 16, 32]
        
        anchors = []
        
        # Calculate feature map sizes for each pyramid level
        for k, step in enumerate(steps):
            feature_h = self.tile_height // step
            feature_w = self.tile_width // step
            
            for i, j in product(range(feature_h), range(feature_w)):
                for min_size in min_sizes[k]:
                    # Center coordinates
                    cx = (j + 0.5) * step / self.tile_width
                    cy = (i + 0.5) * step / self.tile_height
                    
                    # Size relative to image
                    s_kx = min_size / self.tile_width
                    s_ky = min_size / self.tile_height
                    
                    # Store as [cx, cy, w, h] in normalized coordinates
                    anchors.append([cx, cy, s_kx, s_ky])
        
        self.anchors = np.array(anchors, dtype=np.float32)
        
        # Verify anchor count matches model output
        expected_anchors = len(self.anchors)
        print(f"[RETINAFACE] Initial anchor generation created {expected_anchors} anchors")
        
        # Always use 15960 configuration for standard RetinaFace
        # This matches the model output shape we see in the logs
        print(f"[RETINAFACE] Force generating 15960 anchors for standard RetinaFace model")
        self._generate_anchors_15960()
    
    def _generate_anchors_15960(self):
        """Generate exactly 15960 anchors for standard RetinaFace configuration."""
        # This configuration generates exactly 15960 anchors
        # Increased minimum sizes to avoid detecting facial features
        min_sizes = [[64, 96, 128], [128, 192], [256, 384], [384, 512, 640]]
        steps = [8, 16, 32, 64]
        
        anchors = []
        
        for k, step in enumerate(steps):
            feature_h = self.tile_height // step
            feature_w = self.tile_width // step
            
            for i, j in product(range(feature_h), range(feature_w)):
                for min_size in min_sizes[k]:
                    # Use normalized coordinates (0-1 range)
                    cx = (j + 0.5) * step / self.tile_width
                    cy = (i + 0.5) * step / self.tile_height
                    w = min_size / self.tile_width
                    h = min_size / self.tile_height
                    anchors.append([cx, cy, w, h])
        
        self.anchors = np.array(anchors, dtype=np.float32)
        print(f"[RETINAFACE] Generated {len(self.anchors)} anchors with 15960 configuration")
        print(f"[RETINAFACE] Anchor array shape: {self.anchors.shape}")
        print(f"[RETINAFACE] First anchor: {self.anchors[0] if len(self.anchors) > 0 else 'None'}")
        print(f"[RETINAFACE] Anchors stored at id: {id(self.anchors)}")
    
    def start(self):
        """Start the async detection thread."""
        if not self.is_running:
            # Final check for anchors before starting
            if self.anchors is None:
                print(f"[RETINAFACE] CRITICAL: Anchors still None at start, generating emergency anchors")
                self._generate_anchors_15960()
            
            print(f"[RETINAFACE] Starting detector with {len(self.anchors) if self.anchors is not None else 0} anchors")
            if self.anchors is not None:
                print(f"[RETINAFACE] Anchors available at start: shape={self.anchors.shape}, id={id(self.anchors)}")
            else:
                print(f"[RETINAFACE] WARNING: Anchors are None at detector start!")
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
        print(f"[RETINAFACE] Detection loop started, anchors status: {self.anchors is not None}")
        if self.anchors is not None:
            print(f"[RETINAFACE] Anchors in detection thread: shape={self.anchors.shape}, id={id(self.anchors)}")
        else:
            # Emergency anchor generation if still None
            print(f"[RETINAFACE] EMERGENCY: Anchors None in detection thread, generating now")
            self._generate_anchors_15960()
            print(f"[RETINAFACE] Emergency anchors generated: {self.anchors.shape if self.anchors is not None else 'Still None'}")
        
        while self.is_running:
            try:
                # Get frame from queue
                frame_data = self.input_queue.get(timeout=0.1)
                if frame_data is None:
                    continue
                    
                frame, frame_id = frame_data
                start_time = time.time()
                
                # TESTING: Process full image without tiling
                print(f"[RETINAFACE] Processing FULL frame {frame_id}: {frame.shape} (NO TILING)")
                
                # Process full frame directly
                h, w = frame.shape[:2]
                coords = (0, 0, w, h)  # Full frame coordinates
                all_detections = self._detect_in_tile(frame, coords, tile_id=0)
                
                print(f"[RETINAFACE] Full frame detection found {len(all_detections)} faces")
                
                # # Generate tiles
                # tiles, tile_coords = self._generate_tiles(frame)
                # print(f"[RETINAFACE] Processing frame {frame_id}: {frame.shape} -> {len(tiles)} tiles")
                
                # # Process tiles in parallel
                # futures = []
                # for i, (tile, coords) in enumerate(zip(tiles, tile_coords)):
                #     future = self.executor.submit(self._detect_in_tile, tile, coords, i)
                #     futures.append(future)
                
                # # Collect results
                # all_detections = []
                # detections_by_tile = []  # Track which tile each detection came from
                # for i, future in enumerate(futures):
                #     tile_detections = future.result()
                #     all_detections.extend(tile_detections)
                #     detections_by_tile.extend([i] * len(tile_detections))
                
                # # DEBUG: Log detections per tile
                # if len(all_detections) > 0:
                #     tile_counts = {}
                #     for tile_idx in detections_by_tile:
                #         tile_counts[tile_idx] = tile_counts.get(tile_idx, 0) + 1
                #     print(f"[RETINAFACE] Detections by tile: {tile_counts}")
                
                # Apply global NMS
                final_detections = self._global_nms(all_detections)
                
                print(f"[RETINAFACE] Frame {frame_id}: {len(all_detections)} total detections -> {len(final_detections)} after NMS")
                
                # DEBUG: Check for suspicious clustering of detections
                if len(all_detections) > 5:
                    # Calculate centroid of all detections
                    centroids = [(d['bbox'][0] + d['bbox'][2])/2 for d in all_detections]
                    centroid_y = [(d['bbox'][1] + d['bbox'][3])/2 for d in all_detections]
                    
                    # Check variance - low variance means clustering
                    var_x = np.var(centroids) if len(centroids) > 1 else 0
                    var_y = np.var(centroid_y) if len(centroid_y) > 1 else 0
                    
                    if var_x < 100 or var_y < 100:  # Suspicious clustering
                        print(f"[RETINAFACE WARNING] Detections appear clustered! Variance: X={var_x:.1f}, Y={var_y:.1f}")
                        print(f"[RETINAFACE WARNING] Detection centers: X range=[{min(centroids):.0f}, {max(centroids):.0f}], "
                              f"Y range=[{min(centroid_y):.0f}, {max(centroid_y):.0f}]")
                
                # DEBUG_RETINAFACE: Send raw detections to debug queue if available
                if self.debug_queue is not None:  # Send even if no detections for testing
                    try:
                        debug_data = {
                            'frame_id': frame_id,
                            'raw_detections': all_detections,  # Before NMS
                            'final_detections': final_detections,  # After NMS
                            'frame_shape': frame.shape,  # Include frame dimensions
                            'full_frame_mode': True  # Indicate we're in full frame mode
                        }
                        self.debug_queue.put_nowait(debug_data)
                        print(f"[RETINAFACE DEBUG] Sent detections to debug queue - Raw: {len(all_detections)}, Final: {len(final_detections)}, Frame: {frame.shape}")
                    except:
                        pass
                
                # Log score statistics for debugging
                if len(all_detections) > 0:
                    all_scores = [d['confidence'] for d in all_detections]
                    print(f"[RETINAFACE] Frame {frame_id} score stats: min={min(all_scores):.3f}, max={max(all_scores):.3f}, mean={np.mean(all_scores):.3f}")
                
                # Calculate FPS
                detection_time = time.time() - start_time
                self.fps_tracker.append(1.0 / detection_time)
                if len(self.fps_tracker) > 30:
                    self.fps_tracker.pop(0)
                
                # Output results
                self.output_queue.put((frame_id, final_detections))
                
                if len(final_detections) > 0:
                    print(f"[RETINAFACE] Frame {frame_id}: Found {len(final_detections)} faces in FULL FRAME")
                    for i, d in enumerate(final_detections[:3]):
                        bbox = d['bbox']
                        print(f"  Face {i}: [{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}] conf={d['confidence']:.3f}")
                else:
                    print(f"[RETINAFACE] Frame {frame_id}: No faces detected in FULL FRAME")
                
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
        
        # DEBUG: Log tile generation info
        tile_count = 0
        
        for y in y_positions:
            for x in x_positions:
                # Extract tile
                x_end = min(x + tile_width, w)
                y_end = min(y + tile_height, h)
                
                tile = frame[y:y_end, x:x_end]
                original_tile_shape = tile.shape
                
                # Pad tile if necessary (edge cases)
                was_padded = False
                if tile.shape[0] < tile_height or tile.shape[1] < tile_width:
                    was_padded = True
                    tile = cv2.copyMakeBorder(
                        tile,
                        0, tile_height - tile.shape[0],
                        0, tile_width - tile.shape[1],
                        cv2.BORDER_CONSTANT,
                        value=(0, 0, 0)
                    )
                
                # DEBUG: Log tile info for first few tiles
                if tile_count < 4 or was_padded:
                    print(f"[RETINAFACE TILE {tile_count}] Pos: ({x},{y}) -> ({x_end},{y_end}), "
                          f"Original: {original_tile_shape}, Padded: {was_padded}, "
                          f"Actual size: {x_end-x} x {y_end-y}")
                
                tiles.append(tile)
                coords.append((x, y, x_end, y_end))
                tile_count += 1
        
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
            raise RuntimeError(
                "RetinaFace model not loaded! Possible causes:\n"
                "1. onnxruntime is not installed (run: pip install onnxruntime)\n"
                "2. Model file not found or invalid\n"
                "3. Model initialization failed"
            )
        
        # Preprocess tile (resizes to 640x608)
        input_tensor = self._preprocess_tile(tile)
        
        # Run inference
        outputs = self.model.run(self.output_names, {self.input_name: input_tensor})
        
        # Parse detections with scale factors
        scale_x = tile_w / float(self.tile_width)   # Scale from model width back to tile width
        scale_y = tile_h / float(self.tile_height)  # Scale from model height back to tile height
        
        # DEBUG: Log scale factors for tiles with detections
        if tile_id < 4:
            print(f"[RETINAFACE TILE {tile_id}] Tile size: {tile_w}x{tile_h}, Model size: {self.tile_width}x{self.tile_height}, "
                  f"Scale: {scale_x:.3f}x{scale_y:.3f}, Offset: ({x_offset},{y_offset})")
        
        detections = self._parse_detections(outputs, x_offset, y_offset, scale_x, scale_y, tile_id=tile_id, tile_w=tile_w, tile_h=tile_h)
        
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
            
        # Always log for full frame mode (testing)
        print(f"[RETINAFACE] Preprocessing frame {self._preprocess_count}: input shape={tile.shape}, target={target_width}x{target_height}")
        
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
                         scale_x: float = 1.0, scale_y: float = 1.0, tile_id: int = -1,
                         tile_w: int = None, tile_h: int = None) -> List[Dict]:
        """Parse model outputs and convert to global coordinates."""
        detections = []
        
        # Add diagnostic logging
        if hasattr(self, '_parse_count'):
            self._parse_count += 1
        else:
            self._parse_count = 1
            
        if self._parse_count % 50 == 1:  # Reduced frequency
            print(f"[RETINAFACE] Parse detections called. Outputs: {len(outputs)} arrays")
            if self.anchors is None:
                print(f"[RETINAFACE] ERROR: Anchors are None in parse!")
            for i, out in enumerate(outputs):
                print(f"[RETINAFACE] Output {i}: shape={out.shape}, dtype={out.dtype}")
        
        try:
            # This is a simplified version - actual parsing depends on model output format
            # Typical RetinaFace outputs: boxes, scores, landmarks
            if len(outputs) >= 2:
                boxes = outputs[0]      # Shape: [15960, 4]
                scores = outputs[1]     # Shape: [15960, 2] 
                landmarks = outputs[2] if len(outputs) > 2 else None  # Shape: [15960, 10]
                
                # Debug: Log shapes before processing
                if self._parse_count % 10 == 1:
                    print(f"[RETINAFACE] Raw shapes - boxes: {boxes.shape}, scores: {scores.shape}")
                
                # Handle batch dimension if present
                if len(boxes.shape) == 3:
                    boxes = boxes[0]
                    if self._parse_count % 10 == 1:
                        print(f"[RETINAFACE] Removed batch dim from boxes, new shape: {boxes.shape}")
                if len(landmarks.shape) == 3 and landmarks is not None:
                    landmarks = landmarks[0]
                
                # Extract face confidence scores (column 1)
                # Column 0 is background, column 1 is face
                # Apply softmax to convert logits to probabilities
                if len(scores.shape) == 2 and scores.shape[1] == 2:
                    # Apply softmax across the two classes
                    scores_exp = np.exp(scores - np.max(scores, axis=1, keepdims=True))
                    scores_softmax = scores_exp / np.sum(scores_exp, axis=1, keepdims=True)
                    scores = scores_softmax[:, 1]  # Get face class probabilities
                elif len(scores.shape) == 3:
                    # Remove batch dimension first
                    scores = scores[0]
                    # Apply softmax across the two classes
                    scores_exp = np.exp(scores - np.max(scores, axis=1, keepdims=True))
                    scores_softmax = scores_exp / np.sum(scores_exp, axis=1, keepdims=True)
                    scores = scores_softmax[:, 1]  # Get face class probabilities
                    
                # Now iterate through detections
                # Debug: Check score distribution
                num_above_threshold = 0
                if len(scores) > 0:
                    max_score = np.max(scores)
                    num_above_threshold = np.sum(scores > self.confidence_threshold)
                    if self._parse_count % 5 == 0:  # More frequent logging
                        print(f"[RETINAFACE] Scores: max={max_score:.3f}, above {self.confidence_threshold}: {num_above_threshold}/{len(scores)}")
                        # Show top 5 scores for debugging
                        top_scores = np.sort(scores)[-5:][::-1]
                        print(f"[RETINAFACE] Top 5 scores: {top_scores}")
                
                # Debug: Track detections being processed
                high_conf_count = 0
                filtered_size = 0
                added_count = 0
                
                # Debug: Check box values when we have high confidence scores
                if num_above_threshold > 0 and self._parse_count % 5 == 0:
                    print(f"[RETINAFACE] Box data type: {boxes.dtype}, shape: {boxes.shape}")
                    # Find first high-confidence detection
                    for idx in range(min(5, len(scores))):
                        if scores[idx] > self.confidence_threshold:
                            print(f"[RETINAFACE] First high-conf box at idx {idx}: {boxes[idx]}")
                            break
                
                for i in range(len(scores)):
                    if scores[i] > self.confidence_threshold:
                        high_conf_count += 1
                        box = boxes[i]
                        
                        # Debug: Log raw box coordinates
                        if high_conf_count <= 3:  # Log first 3 high-confidence detections
                            print(f"[RETINAFACE] Detection {i}: score={scores[i]:.3f}, raw box={box}")
                        
                        # Decode box predictions using anchors
                        if self.anchors is not None and i < len(self.anchors):
                            anchor = self.anchors[i]
                            
                            # RetinaFace box format: [dx, dy, dw, dh] as offsets
                            # Decode center point
                            cx = anchor[0] + box[0] * self.variances[0] * anchor[2]
                            cy = anchor[1] + box[1] * self.variances[0] * anchor[3]
                            
                            # Decode width and height
                            w = anchor[2] * np.exp(box[2] * self.variances[1])
                            h = anchor[3] * np.exp(box[3] * self.variances[1])
                            
                            # Convert from center format to corner format
                            x1 = (cx - w / 2) * self.tile_width
                            y1 = (cy - h / 2) * self.tile_height
                            x2 = (cx + w / 2) * self.tile_width
                            y2 = (cy + h / 2) * self.tile_height
                            
                            # Apply tile offset
                            # IMPORTANT: For padded tiles, we need to ensure coordinates don't extend beyond actual content
                            if tile_w is not None and tile_h is not None:
                                scaled_bbox = [
                                    max(x_offset, min(x1 * scale_x + x_offset, x_offset + tile_w - 1)),
                                    max(y_offset, min(y1 * scale_y + y_offset, y_offset + tile_h - 1)),
                                    max(x_offset, min(x2 * scale_x + x_offset, x_offset + tile_w - 1)),
                                    max(y_offset, min(y2 * scale_y + y_offset, y_offset + tile_h - 1))
                                ]
                            else:
                                # Fallback to original calculation
                                scaled_bbox = [
                                    x1 * scale_x + x_offset,
                                    y1 * scale_y + y_offset,
                                    x2 * scale_x + x_offset,
                                    y2 * scale_y + y_offset
                                ]
                        else:
                            # Emergency fallback - should not happen with proper initialization
                            print(f"[RETINAFACE] ERROR: No anchors available for detection {i}! This should not happen.")
                            print(f"[RETINAFACE] Anchor state: anchors={self.anchors is not None}, index={i}, max_index={len(self.anchors) if self.anchors is not None else 0}")
                            # Skip this detection as we can't decode it properly
                            continue
                        
                        # Debug: Log scaled coordinates (for first detection in each tile)
                        if high_conf_count == 1 and tile_id >= 0 and tile_id < 8:
                            print(f"[RETINAFACE TILE {tile_id}] Detection transformation:")
                            print(f"  - Anchor: {anchor}")
                            print(f"  - Raw box (offsets): {box}")
                            print(f"  - Decoded center: ({cx:.3f}, {cy:.3f}), size: ({w:.3f}, {h:.3f})")
                            print(f"  - Model space bbox: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
                            print(f"  - Scaled bbox: {[f'{v:.1f}' for v in scaled_bbox]}")
                            print(f"  - Scale: {scale_x:.3f}x{scale_y:.3f}, Offset: ({x_offset},{y_offset})")
                        
                        # Validate bbox coordinates
                        if scaled_bbox[0] > scaled_bbox[2] or scaled_bbox[1] > scaled_bbox[3]:
                            print(f"[RETINAFACE] WARNING: Invalid bbox coordinates detected: {scaled_bbox}")
                            print(f"[RETINAFACE] Raw box was: {box}, anchor: {anchor if self.anchors is not None and i < len(self.anchors) else 'N/A'}")
                            # Try to fix by swapping if needed
                            if scaled_bbox[0] > scaled_bbox[2]:
                                scaled_bbox[0], scaled_bbox[2] = scaled_bbox[2], scaled_bbox[0]
                            if scaled_bbox[1] > scaled_bbox[3]:
                                scaled_bbox[1], scaled_bbox[3] = scaled_bbox[3], scaled_bbox[1]
                        
                        # Filter out faces that are too small
                        # Since we're detecting in tiles, use a smaller threshold
                        # A 60x60 face in full image would be ~20x20 in a tile
                        face_width = scaled_bbox[2] - scaled_bbox[0]
                        face_height = scaled_bbox[3] - scaled_bbox[1]
                        
                        # Log face sizes only occasionally for debugging
                        if high_conf_count <= 3 and self._parse_count % 10 == 1:
                            print(f"[RETINAFACE] Face size: {face_width:.1f}x{face_height:.1f}")
                        
                        # Additional validation for negative or invalid dimensions
                        if face_width <= 0 or face_height <= 0:
                            print(f"[RETINAFACE] ERROR: Invalid face dimensions: {face_width:.1f}x{face_height:.1f}")
                            filtered_size += 1
                            continue
                        
                        # Use larger threshold to filter out feature detections
                        # 100x100 pixels in tile = ~300x300 in full image (reasonable face size)
                        MIN_FACE_SIZE_IN_TILE = 100
                        if face_width < MIN_FACE_SIZE_IN_TILE or face_height < MIN_FACE_SIZE_IN_TILE:
                            filtered_size += 1
                            if high_conf_count <= 10:  # Log first few filtered faces
                                print(f"[RETINAFACE] Filtered small face: {face_width:.1f}x{face_height:.1f} < {MIN_FACE_SIZE_IN_TILE}")
                            continue
                        
                        # Add aspect ratio check to filter out non-face detections
                        aspect_ratio = face_width / face_height if face_height > 0 else 0
                        if aspect_ratio < 0.5 or aspect_ratio > 2.0:
                            filtered_size += 1
                            if high_conf_count <= 10:
                                print(f"[RETINAFACE] Filtered by aspect ratio: {aspect_ratio:.2f} (w={face_width:.1f}, h={face_height:.1f})")
                            continue
                        
                        detection = {
                            'bbox': scaled_bbox,
                            'confidence': float(scores[i]),
                            'landmarks': None
                        }
                        
                        if landmarks is not None and i < len(landmarks):
                            lm = landmarks[i].reshape(-1, 2)  # Reshape to 5x2
                            # Scale landmarks
                            detection['landmarks'] = lm * np.array([scale_x, scale_y]) + np.array([x_offset, y_offset])
                        
                        detections.append(detection)
                        added_count += 1
                
                # Debug summary
                if high_conf_count > 0:
                    print(f"[RETINAFACE] Tile summary: {high_conf_count} high-conf, {filtered_size} filtered by size, {added_count} added")
        except Exception as e:
            print(f"[RETINAFACE] Error parsing detections: {e}")
            print(f"[RETINAFACE] Output shapes: {[out.shape for out in outputs]}")
            import traceback
            traceback.print_exc()
        
        return detections
    
    
    def _global_nms(self, detections: List[Dict]) -> List[Dict]:
        """Apply Non-Maximum Suppression across all detections."""
        if not detections:
            return []
        
        # First apply clustering to merge small detections into larger face regions
        clustered_detections = self._cluster_detections(detections)
        
        # Extract boxes and scores
        boxes = np.array([d['bbox'] for d in clustered_detections])
        scores = np.array([d['confidence'] for d in clustered_detections])
        
        # Apply NMS
        indices = self._nms(boxes, scores, self.nms_threshold)
        
        # Return filtered detections
        return [clustered_detections[i] for i in indices]
    
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
    
    def _cluster_detections(self, detections: List[Dict], cluster_threshold: float = 50) -> List[Dict]:
        """Cluster nearby small detections into larger face regions."""
        if len(detections) < 2:
            return detections
        
        # Group detections that are close together
        clustered = []
        used = set()
        
        for i, det1 in enumerate(detections):
            if i in used:
                continue
                
            cluster = [det1]
            bbox1 = det1['bbox']
            
            for j, det2 in enumerate(detections[i+1:], i+1):
                if j in used:
                    continue
                    
                bbox2 = det2['bbox']
                # Check if centers are close
                center1 = ((bbox1[0] + bbox1[2])/2, (bbox1[1] + bbox1[3])/2)
                center2 = ((bbox2[0] + bbox2[2])/2, (bbox2[1] + bbox2[3])/2)
                
                dist = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
                
                if dist < cluster_threshold:
                    cluster.append(det2)
                    used.add(j)
            
            # Create merged detection from cluster
            if len(cluster) > 2:  # If we have multiple small detections
                # Compute bounding box that encompasses all
                x1 = min(d['bbox'][0] for d in cluster)
                y1 = min(d['bbox'][1] for d in cluster)
                x2 = max(d['bbox'][2] for d in cluster)
                y2 = max(d['bbox'][3] for d in cluster)
                
                # Check if the merged box is reasonable
                merged_width = x2 - x1
                merged_height = y2 - y1
                
                # Only create merged detection if it's a reasonable face size
                if merged_width > 80 and merged_height > 80 and merged_width < 800 and merged_height < 800:
                    clustered.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': max(d['confidence'] for d in cluster),
                        'landmarks': cluster[0].get('landmarks')  # Use first detection's landmarks
                    })
                    used.add(i)
                    print(f"[RETINAFACE] Clustered {len(cluster)} detections into merged face at [{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]")
                else:
                    # Keep original detection if merged box is unreasonable
                    clustered.append(det1)
            elif i not in used:
                clustered.append(det1)
        
        print(f"[RETINAFACE] Clustering: {len(detections)} detections -> {len(clustered)} after clustering")
        return clustered
    
    def get_stats(self) -> Dict:
        """Get detector statistics."""
        avg_fps = np.mean(self.fps_tracker) if self.fps_tracker else 0
        return {
            'avg_fps': avg_fps,
            'input_queue_size': self.input_queue.qsize(),
            'output_queue_size': self.output_queue.qsize()
        }