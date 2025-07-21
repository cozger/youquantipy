import sys
import numpy as np
import threading
import time
from queue import Queue, Empty, Full
from typing import Optional, Tuple, List, Dict
import cv2

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit 
except ImportError:
    print("ERROR: TensorRT and PyCUDA are required for GPU acceleration!")
    print("Install with: pip install tensorrt pycuda")
    sys.exit(1)


class RetinaFaceDetectorGPU:
    """
    GPU-accelerated RetinaFace detector using TensorRT.
    Drop-in replacement for RetinaFaceDetector.
    """
    
    def __init__(self, 
                model_path: str,  # Path to ONNX or TRT engine
                tile_size: int = 640,
                overlap: float = 0.2,
                confidence_threshold: float = 0.99,
                nms_threshold: float = 0.4,
                max_workers: int = 4,
                debug_queue=None,
                max_batch_size: int = 4):
        # Initialize CUDA context first
        cuda.init()
        self.cuda_device = cuda.Device(0)
        self.cuda_context = self.cuda_device.make_context()
        
        # Model components
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.variances = [0.1, 0.2]
        self.max_batch_size = max_batch_size
        
        # Async processing
        self.input_queue = Queue(maxsize=5)
        self.output_queue = Queue(maxsize=5)
        self.debug_queue = debug_queue
        self.is_running = False
        self.detection_thread = None
        
        # Performance tracking
        self.fps_tracker = []
        self.detection_count = 0
        self.parse_count = 0
        
        # TensorRT components
        self.trt_logger = trt.Logger(trt.Logger.WARNING)
        self.engine = None
        self.context = None
        self.stream = None
        
        # Buffers
        self.device_inputs = []
        self.device_outputs = []
        self.host_inputs = []
        self.host_outputs = []
        self.bindings = []
        
        # Load or build TensorRT engine
        self._initialize_trt()
        
        # Pop context after initialization
        self.cuda_context.pop()

    def _initialize_trt(self):
        """Initialize TensorRT engine from ONNX or existing engine."""
        import os
        
        # Check if we have a serialized engine
        engine_path = self.model_path.replace('.onnx', '.trt')
        
        if os.path.exists(engine_path) and engine_path.endswith('.trt'):
            print(f"[GPU Detector] Loading TensorRT engine from {engine_path}")
            self._load_engine(engine_path)
        else:
            print(f"[GPU Detector] Building TensorRT engine from {self.model_path}")
            self._build_engine_from_onnx()
            # Save the engine
            self._save_engine(engine_path)
        
        # Create execution context
        self.context = self.engine.create_execution_context()
        
        # Allocate buffers
        self._allocate_buffers()
        
        # Create CUDA stream
        self.stream = cuda.Stream()

        # Warm up
        self._warmup_engine()
        
        print(f"[GPU Detector] TensorRT initialized successfully")

    def _warmup_engine(self):
        """Warmup TensorRT engine to avoid first-frame issues."""
        print("[GPU Detector] Warming up TensorRT engine...")
        
        # Create dummy input
        if self.model_format == 'NCHW':
            dummy_input = np.random.randn(1, 3, self.model_height, self.model_width).astype(np.float32)
        else:
            dummy_input = np.random.randn(1, self.model_height, self.model_width, 3).astype(np.float32)
        
        # Run inference
        try:
            _ = self._run_inference_batch(dummy_input)
            print("[GPU Detector] Warmup complete")
        except Exception as e:
            print(f"[GPU Detector] Warmup failed: {e}")
        
    def _build_engine_from_onnx(self):
        """Build TensorRT engine from ONNX model."""
        builder = trt.Builder(self.trt_logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, self.trt_logger)
        
        # Parse ONNX
        with open(self.model_path, 'rb') as f:
            if not parser.parse(f.read()):
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                raise RuntimeError("Failed to parse ONNX model")
        
        # Build engine config
        config = builder.create_builder_config()
        
        # Set memory pool limit (new API)
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
        
        # Enable FP16 if available
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("[GPU Detector] Enabled FP16 precision")
        
        # Set batch size
        profile = builder.create_optimization_profile()
        input_shape = network.get_input(0).shape
        
        # Dynamic batch size support
        profile.set_shape(
            network.get_input(0).name,
            (1, input_shape[1], input_shape[2], input_shape[3]),      # min
            (self.max_batch_size, input_shape[1], input_shape[2], input_shape[3]),  # opt
            (self.max_batch_size, input_shape[1], input_shape[2], input_shape[3])   # max
        )
        config.add_optimization_profile(profile)
        
        # Build engine
        print("[GPU Detector] Building TensorRT engine... (this may take a few minutes)")
        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            raise RuntimeError("Failed to build TensorRT engine")
            
        # Deserialize to get engine
        runtime = trt.Runtime(self.trt_logger)
        self.engine = runtime.deserialize_cuda_engine(serialized_engine)
        
        if self.engine is None:
            raise RuntimeError("Failed to build TensorRT engine")
            
        # Get model info
        self.model_height = input_shape[2]
        self.model_width = input_shape[3]
        self.model_format = 'NCHW' if input_shape[1] == 3 else 'NHWC'
        
        print(f"[GPU Detector] Engine built: input shape {input_shape}, format {self.model_format}")
        
        # Generate anchors
        self.anchors = self._generate_anchors()
        
    def _load_engine(self, engine_path: str):
        """Load serialized TensorRT engine."""
        with open(engine_path, 'rb') as f:
            self.engine = trt.Runtime(self.trt_logger).deserialize_cuda_engine(f.read())
        
        # Get model dimensions from engine - using new API
        # Get input tensor name first
        input_name = self.engine.get_tensor_name(0)
        input_shape = list(self.engine.get_tensor_shape(input_name))
        
        # Handle dynamic batch dimension
        if input_shape[0] == -1:
            input_shape[0] = 1  # Use 1 for shape extraction
        
        # The model expects NHWC format: [-1, 608, 640, 3]
        if len(input_shape) == 4 and input_shape[-1] == 3:
            self.model_format = 'NHWC'
            self.model_height = input_shape[1]  # 608
            self.model_width = input_shape[2]   # 640
        else:
            self.model_format = 'NCHW'
            self.model_height = input_shape[2]
            self.model_width = input_shape[3]
        
        print(f"[GPU Detector] Loaded engine with input shape {input_shape}, "
            f"model size: {self.model_width}x{self.model_height}, format: {self.model_format}")
        
        # Generate anchors
        self.anchors = self._generate_anchors()

    def _save_engine(self, engine_path: str):
        """Save TensorRT engine to file."""
        with open(engine_path, 'wb') as f:
            f.write(self.engine.serialize())
        print(f"[GPU Detector] Saved TensorRT engine to {engine_path}")
        
    def _allocate_buffers(self):
        """Allocate host and device buffers."""
        # Get number of bindings
        num_bindings = self.engine.num_io_tensors
        
        print(f"[GPU Detector] Allocating buffers for {num_bindings} tensors")
        total_host_memory = 0
        
        for i in range(num_bindings):
            # Get tensor name and properties
            tensor_name = self.engine.get_tensor_name(i)
            shape = list(self.engine.get_tensor_shape(tensor_name))  # Convert to list for modification
            dtype = trt.nptype(self.engine.get_tensor_dtype(tensor_name))
            
            # Check if input or output
            is_input = self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT
            
            # Replace -1 (dynamic dimension) with max batch size
            if shape[0] == -1:
                shape[0] = self.max_batch_size
            
            # Calculate size
            size = int(np.prod(shape))
            
            # Calculate memory in MB
            bytes_per_element = np.dtype(dtype).itemsize
            memory_mb = (size * bytes_per_element) / (1024 * 1024)
            total_host_memory += memory_mb
            
            print(f"[GPU Detector] Tensor '{tensor_name}': shape={shape}, dtype={dtype}, "
                f"size={size}, memory={memory_mb:.2f}MB, is_input={is_input}")
            
            # Allocate host buffer
            try:
                host_buffer = cuda.pagelocked_empty(size, dtype)
            except cuda.MemoryError as e:
                print(f"[GPU Detector] Failed to allocate {memory_mb:.2f}MB for tensor '{tensor_name}'")
                print(f"[GPU Detector] Total host memory attempted so far: {total_host_memory:.2f}MB")
                raise
            
            # Allocate device buffer
            device_buffer = cuda.mem_alloc(host_buffer.nbytes)
            
            # Store
            self.bindings.append(int(device_buffer))
            
            if is_input:
                self.host_inputs.append(host_buffer)
                self.device_inputs.append(device_buffer)
            else:
                self.host_outputs.append(host_buffer)
                self.device_outputs.append(device_buffer)
        
        print(f"[GPU Detector] Successfully allocated {len(self.device_inputs)} input and "
            f"{len(self.device_outputs)} output buffers, total host memory: {total_host_memory:.2f}MB")

    def _generate_anchors(self):
        """Generate anchor boxes for the model."""
        # Configuration based on model size
        if self.model_width == 640 and self.model_height == 640:
            min_sizes = [[16, 32], [64, 128], [256, 512]]
            steps = [8, 16, 32]
        elif self.model_width == 640 and self.model_height == 608:  # Add this case
            min_sizes = [[16, 32], [64, 128], [256, 512]]
            steps = [8, 16, 32]
        else:
            min_sizes = [[16, 32], [64, 128], [256, 512]]
            steps = [8, 16, 32]
            
        anchors = []
        anchor_counts = []
        
        for k, step in enumerate(steps):
            f_h = self.model_height // step
            f_w = self.model_width // step
            layer_anchors = 0
            
            for i in range(f_h):
                for j in range(f_w):
                    for min_size in min_sizes[k]:
                        cx = (j + 0.5) * step / self.model_width
                        cy = (i + 0.5) * step / self.model_height
                        w = min_size / self.model_width
                        h = min_size / self.model_height
                        anchors.append([cx, cy, w, h])
                        layer_anchors += 1
            
            anchor_counts.append(layer_anchors)
            print(f"[GPU Detector] Layer {k}: {f_h}x{f_w} grid, {len(min_sizes[k])} sizes = {layer_anchors} anchors")
        
        anchors_array = np.array(anchors, dtype=np.float32)
        print(f"[GPU Detector] Total anchors: {anchors_array.shape[0]} (expected ~16000)")
        print(f"[GPU Detector] Anchor distribution by layer: {anchor_counts}")
        
        return anchors_array
    
    def _preprocess_batch_gpu(self, frames: List[np.ndarray]) -> np.ndarray:
        """Preprocess batch of frames on GPU."""
        batch_size = len(frames)
        
        # Allocate batch array
        if self.model_format == 'NCHW':
            batch = np.zeros((batch_size, 3, self.model_height, self.model_width), dtype=np.float32)
        else:
            batch = np.zeros((batch_size, self.model_height, self.model_width, 3), dtype=np.float32)
        
        for i, frame in enumerate(frames):
            # Use GPU for resize if available
            if hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0:
                gpu_frame = cv2.cuda_GpuMat()
                gpu_frame.upload(frame)
                gpu_resized = cv2.cuda.resize(gpu_frame, (self.model_width, self.model_height))
                resized = gpu_resized.download()
            else:
                # Fallback to CPU resize
                resized = cv2.resize(frame, (self.model_width, self.model_height))
            
            # Convert to float and normalize
            img = resized.astype(np.float32)
            img -= np.array([104, 117, 123], dtype=np.float32)
            
            # Format conversion
            if self.model_format == 'NCHW':
                img = np.transpose(img, (2, 0, 1))
                batch[i] = img
            else:
                batch[i] = img
                
        return batch
        
    def _run_inference_batch(self, batch: np.ndarray) -> List[np.ndarray]:
        """Run TensorRT inference on batch."""
        batch_size = batch.shape[0]
        
        # Get input tensor name
        input_name = self.engine.get_tensor_name(0)
        
        # Set batch size - fix the shape order
        # The engine expects [-1, 608, 640, 3] so we need to match that format
        self.context.set_input_shape(input_name, batch.shape)
        
        # Copy input to device
        np.copyto(self.host_inputs[0][:batch.size], batch.flatten())
        cuda.memcpy_htod_async(self.device_inputs[0], self.host_inputs[0], self.stream)
        # Set tensor addresses
        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            self.context.set_tensor_address(tensor_name, self.bindings[i])
        
        # Run inference
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        
        # Copy outputs back
        outputs = []
        for device_output, host_output in zip(self.device_outputs, self.host_outputs):
            cuda.memcpy_dtoh_async(host_output, device_output, self.stream)
            outputs.append(host_output.copy())
        
        # Synchronize
        self.stream.synchronize()
        
        # Reshape outputs
        output_shapes = []
        output_idx = 0
        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.OUTPUT:
                shape = self.context.get_tensor_shape(tensor_name)
                output_shapes.append(shape)
        
        reshaped_outputs = []
        for output, shape in zip(outputs, output_shapes):
            # Adjust shape for batch size
            shape = (batch_size, *shape[1:])
            reshaped = output[:np.prod(shape)].reshape(shape)
            reshaped_outputs.append(reshaped)
            
        return reshaped_outputs


    def _detection_loop(self):
        """Main detection loop with batch processing."""
        
        # Push the CUDA context for this thread
        self.cuda_context.push()
        
        batch_frames = []
        batch_metadata = []
        last_batch_time = time.time()
        batch_timeout = 0.05  # 50ms max wait for batch
        frames_processed = 0
        last_debug_time = time.time()
        
        try:
            while self.is_running:
                try:
                    # Collect frames for batch
                    timeout = batch_timeout if not batch_frames else 0.001
                    frame_data = self.input_queue.get(timeout=timeout)
                    
                    if frame_data is None:
                        break
                        
                    batch_frames.append(frame_data[0])
                    batch_metadata.append(frame_data)
                    
                    # Debug: Track frames received
                    frames_processed += 1
                    if frames_processed % 30 == 0:
                        print(f"[GPU Detector] Received frame {frame_data[1]}, batch size: {len(batch_frames)}")
                    
                    # Process batch if full or timeout
                    current_time = time.time()
                    should_process = (
                        len(batch_frames) >= self.max_batch_size or
                        (len(batch_frames) > 0 and current_time - last_batch_time > batch_timeout)
                    )
                    
                    if should_process:
                        start_time = current_time
                        
                        print(f"[GPU Detector] Processing batch of {len(batch_frames)} frames")
                        
                        # Preprocess batch
                        batch = self._preprocess_batch_gpu(batch_frames)
                        
                        # Run inference
                        outputs = self._run_inference_batch(batch)
                        
                        # Process each result
                        detections_found = 0
                        for i in range(len(batch_frames)):
                            frame = batch_frames[i]
                            frame_id = batch_metadata[i][1]
                            
                            # Extract outputs for this sample
                            sample_outputs = [output[i] for output in outputs]
                            
                            # Process outputs
                            detections = self._process_outputs(
                                sample_outputs, 
                                frame.shape[0], 
                                frame.shape[1]
                            )
                            
                            detections_found += len(detections)
                            
                            # Always send result, even if empty
                            try:
                                self.output_queue.put((frame_id, detections), timeout=0.01)
                                if len(detections) > 0:
                                    print(f"[GPU Detector] Frame {frame_id}: Found {len(detections)} faces")
                            except Full:
                                print(f"[GPU Detector] Output queue full, dropping result for frame {frame_id}")
                            
                            # Update statistics
                            self.detection_count += len(detections)
                            self.parse_count += 1
                        
                        # Performance tracking
                        process_time = time.time() - start_time
                        print(f"[GPU Detector] Batch processed in {process_time*1000:.1f}ms, found {detections_found} total faces")
                        
                        if process_time > 0:
                            batch_fps = len(batch_frames) / process_time
                            self.fps_tracker.append(batch_fps)
                            if len(self.fps_tracker) > 30:
                                self.fps_tracker.pop(0)
                        
                        # Clear batch
                        batch_frames.clear()
                        batch_metadata.clear()
                        last_batch_time = current_time
                        
                except Empty:
                    # Process any remaining frames
                    if batch_frames:
                        continue
                except Exception as e:
                    print(f"[GPU Detector] Error in detection loop: {e}")
                    import traceback
                    traceback.print_exc()
                    
            # Process any remaining frames before exit
            if batch_frames:
                print(f"[GPU Detector] Processing final batch of {len(batch_frames)} frames")
                # ... process remaining batch ...
                
        finally:
            # Clean up CUDA context
            self.cuda_context.pop()
            print(f"[GPU Detector] Detection loop ended. Processed {frames_processed} frames total")

    def _process_outputs(self, outputs, orig_height, orig_width):
        """Process model outputs with proper coordinate scaling and validation."""
        
        # Debug output shapes and values
        if self.parse_count <= 10:  # More samples for debugging
            print(f"\n[GPU Detector DEBUG] Processing outputs for frame (h={orig_height}, w={orig_width}):")
            print(f"  Number of outputs: {len(outputs)}")
            for i, output in enumerate(outputs):
                print(f"  Output {i}: shape={output.shape}, dtype={output.dtype}")
                print(f"    Stats: min={output.min():.6f}, max={output.max():.6f}, "
                    f"mean={output.mean():.6f}, std={output.std():.6f}")
                # Check for all zeros
                if np.allclose(output, 0.0):
                    print(f"    WARNING: Output {i} is all zeros!")
                # Sample some values
                flat = output.flatten()
                print(f"    First 5 values: {flat[:5]}")
                print(f"    Last 5 values: {flat[-5:]}")
        
        boxes_output = None
        scores_output = None
        landmarks_output = None
        
        # Identify outputs by their last dimension
        for i, output in enumerate(outputs):
            shape = output.shape
            if len(shape) >= 2:
                last_dim = shape[-1]
                if last_dim == 4:
                    boxes_output = output
                    print(f"[GPU Detector DEBUG] Identified output {i} as boxes (shape={shape})")
                elif last_dim == 2:
                    scores_output = output
                    print(f"[GPU Detector DEBUG] Identified output {i} as scores (shape={shape})")
                elif last_dim == 10:
                    landmarks_output = output
                    print(f"[GPU Detector DEBUG] Identified output {i} as landmarks (shape={shape})")
        
        if boxes_output is None or scores_output is None:
            print(f"[GPU Detector] ERROR: Could not identify outputs correctly")
            return []
        
        # Remove batch dimension if present
        if boxes_output.ndim > 2:
            boxes_output = boxes_output.squeeze(0)
        if scores_output.ndim > 2:
            scores_output = scores_output.squeeze(0)
        
        # CHECK 1: Validate outputs are not degenerate
        boxes_all_zero = np.allclose(boxes_output, 0.0)
        scores_all_zero = np.allclose(scores_output, 0.0)
        
        if self.parse_count <= 10:
            print(f"\n[GPU Detector DEBUG] Output validation:")
            print(f"  Boxes all zero: {boxes_all_zero}")
            print(f"  Scores all zero: {scores_all_zero}")
        
        # If both are zero, no detections
        if boxes_all_zero and scores_all_zero:
            if self.parse_count <= 10:
                print(f"[GPU Detector DEBUG] Both outputs are zero - no faces detected")
            return []
        
        # CHECK 2: Determine score format and process accordingly
        # Check if scores are in logit or probability format
        scores_min = scores_output.min()
        scores_max = scores_output.max()
        scores_range = scores_max - scores_min
        
        if self.parse_count <= 10:
            print(f"\n[GPU Detector DEBUG] Score analysis:")
            print(f"  Raw scores range: [{scores_min:.6f}, {scores_max:.6f}]")
            print(f"  Range: {scores_range:.6f}")
        
        # Determine format based on value range
        if scores_output.shape[-1] == 2:
            # Two-class output
            if scores_all_zero:
                # Special case: all zeros
                print(f"[GPU Detector DEBUG] All scores are zero - using zero confidence")
                face_scores = np.zeros(len(scores_output))
            elif scores_range < 0.01 and abs(scores_min) < 0.01:
                # Very small range near zero - likely needs softmax
                print(f"[GPU Detector DEBUG] Applying softmax (small range near zero)")
                scores = self._softmax(scores_output)
                face_scores = scores[:, 1]
            elif scores_min >= 0 and scores_max <= 1 and scores_range > 0.1:
                # Already probabilities
                print(f"[GPU Detector DEBUG] Scores appear to be probabilities")
                face_scores = scores_output[:, 1]
            else:
                # Logits - apply softmax
                print(f"[GPU Detector DEBUG] Applying softmax (logit range)")
                scores = self._softmax(scores_output)
                face_scores = scores[:, 1]
        else:
            # Single score output
            face_scores = scores_output.squeeze(-1)
            if scores_min < -1 or scores_max > 1:
                # Logits - apply sigmoid
                print(f"[GPU Detector DEBUG] Applying sigmoid")
                face_scores = 1 / (1 + np.exp(-face_scores))
        
        # CHECK 3: Analyze score distribution
        valid_scores = face_scores[face_scores > 0.001]  # Ignore near-zero scores
        
        if self.parse_count <= 10:
            print(f"\n[GPU Detector DEBUG] Score distribution:")
            print(f"  Total anchors: {len(face_scores)}")
            print(f"  Non-zero scores: {len(valid_scores)}")
            if len(valid_scores) > 0:
                print(f"  Valid score range: [{valid_scores.min():.6f}, {valid_scores.max():.6f}]")
                print(f"  Valid score mean: {valid_scores.mean():.6f}")
            
            # Histogram of scores
            hist_bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            hist, _ = np.histogram(face_scores, bins=hist_bins)
            print(f"  Score histogram: {dict(zip([f'{b:.1f}-{hist_bins[i+1]:.1f}' for i, b in enumerate(hist_bins[:-1])], hist))}")
        
        # CHECK 4: Apply confidence threshold with validation
        # Adjust threshold based on score distribution
        effective_threshold = self.confidence_threshold
        
        # If too many detections pass threshold, something is wrong
        initial_valid = face_scores > effective_threshold
        initial_count = np.sum(initial_valid)
        
        if initial_count > 100:  # More than 100 faces is suspicious
            print(f"[GPU Detector WARNING] {initial_count} detections above threshold {effective_threshold:.3f} - adjusting threshold")
            # Dynamically adjust threshold
            sorted_scores = np.sort(face_scores)[::-1]
            if len(sorted_scores) > 100:
                effective_threshold = max(sorted_scores[100], 0.5)  # Keep top 100 or 0.5 minimum
                print(f"[GPU Detector] Adjusted threshold to {effective_threshold:.3f}")
        
        valid_indices = face_scores > effective_threshold
        
        if self.parse_count <= 10:
            print(f"  Count > {effective_threshold:.3f}: {np.sum(valid_indices)}")
        
        if not np.any(valid_indices):
            return []
        
        # CHECK 5: Validate boxes before decoding
        valid_boxes_raw = boxes_output[valid_indices]
        valid_scores = face_scores[valid_indices]
        valid_anchors = self.anchors[:len(boxes_output)][valid_indices]
        
        # Check if boxes are reasonable
        if self.parse_count <= 10 and len(valid_boxes_raw) > 0:
            print(f"\n[GPU Detector DEBUG] Box validation (first 5):")
            for i in range(min(5, len(valid_boxes_raw))):
                box = valid_boxes_raw[i]
                print(f"  Raw box {i}: {box}, score: {valid_scores[i]:.3f}")
        
        # Decode boxes
        decoded_boxes = self._decode_boxes(valid_boxes_raw, valid_anchors)
        
        # CHECK 6: Validate decoded boxes
        if self.parse_count <= 10 and len(decoded_boxes) > 0:
            print(f"\n[GPU Detector DEBUG] Decoded boxes (normalized, first 5):")
            for i in range(min(5, len(decoded_boxes))):
                print(f"  Box {i}: {decoded_boxes[i]}")
        
        # Scale to pixel coordinates
        decoded_boxes[:, 0] *= self.model_width
        decoded_boxes[:, 1] *= self.model_height
        decoded_boxes[:, 2] *= self.model_width
        decoded_boxes[:, 3] *= self.model_height
        
        # Scale to original image size
        scale_x = orig_width / self.model_width
        scale_y = orig_height / self.model_height
        
        decoded_boxes[:, 0] *= scale_x
        decoded_boxes[:, 1] *= scale_y
        decoded_boxes[:, 2] *= scale_x
        decoded_boxes[:, 3] *= scale_y
        
        # CHECK 7: Filter out invalid boxes
        widths = decoded_boxes[:, 2] - decoded_boxes[:, 0]
        heights = decoded_boxes[:, 3] - decoded_boxes[:, 1]
        
        # More strict filtering
        min_size = 20  # Minimum 20x20 pixels
        max_size = min(orig_width, orig_height) * 0.9  # Max 90% of image
        
        valid_size = (widths > min_size) & (heights > min_size) & (widths < max_size) & (heights < max_size)
        valid_aspect = (widths / heights > 0.5) & (widths / heights < 2.0)  # Reasonable aspect ratio
        valid_position = (decoded_boxes[:, 0] >= -10) & (decoded_boxes[:, 1] >= -10) & \
                        (decoded_boxes[:, 2] <= orig_width + 10) & (decoded_boxes[:, 3] <= orig_height + 10)
        
        valid_boxes_mask = valid_size & valid_aspect & valid_position
        
        if self.parse_count <= 10:
            print(f"\n[GPU Detector DEBUG] Box filtering:")
            print(f"  Valid size: {np.sum(valid_size)}/{len(valid_size)}")
            print(f"  Valid aspect: {np.sum(valid_aspect)}/{len(valid_aspect)}")
            print(f"  Valid position: {np.sum(valid_position)}/{len(valid_position)}")
            print(f"  Total valid: {np.sum(valid_boxes_mask)}/{len(valid_boxes_mask)}")
        
        decoded_boxes = decoded_boxes[valid_boxes_mask]
        valid_scores = valid_scores[valid_boxes_mask]
        
        # CHECK 8: Apply NMS with debugging
        if len(decoded_boxes) > 0:
            pre_nms_count = len(decoded_boxes)
            keep_indices = self._nms(decoded_boxes, valid_scores)
            final_boxes = decoded_boxes[keep_indices]
            final_scores = valid_scores[keep_indices]
            
            if self.parse_count <= 10:
                print(f"\n[GPU Detector DEBUG] NMS: {pre_nms_count} -> {len(final_boxes)} boxes")
        else:
            return []
        
        # Format output
        detections = []
        for box, score in zip(final_boxes, final_scores):
            detections.append({
                'bbox': box.tolist(),
                'confidence': float(score)
            })
        
        # Increment parse count
        self.parse_count += 1
        
        return detections

    def _softmax(self, x):
        """Apply softmax to scores."""
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)
        
    def _decode_boxes(self, raw_boxes, anchors):
        """Decode bounding boxes from anchor-relative format."""
        # Ensure we're working with the right shapes
        if raw_boxes.shape[0] != anchors.shape[0]:
            print(f"[GPU Detector] WARNING: Box count mismatch: {raw_boxes.shape[0]} vs {anchors.shape[0]} anchors")
            anchors = anchors[:raw_boxes.shape[0]]
        
        boxes = np.zeros_like(raw_boxes)
        
        # RetinaFace box encoding:
        # raw_boxes contains [dx, dy, dw, dh] as deltas from anchors
        # We need to decode to [cx, cy, w, h] then convert to [x1, y1, x2, y2]
        
        # Decode center coordinates
        boxes[:, 0] = anchors[:, 0] + raw_boxes[:, 0] * self.variances[0] * anchors[:, 2]
        boxes[:, 1] = anchors[:, 1] + raw_boxes[:, 1] * self.variances[0] * anchors[:, 3]
        
        # Decode width and height
        boxes[:, 2] = anchors[:, 2] * np.exp(raw_boxes[:, 2] * self.variances[1])
        boxes[:, 3] = anchors[:, 3] * np.exp(raw_boxes[:, 3] * self.variances[1])
        
        # Convert from [cx, cy, w, h] to [x1, y1, x2, y2]
        boxes_corner = np.zeros_like(boxes)
        boxes_corner[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
        boxes_corner[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
        boxes_corner[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2
        boxes_corner[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2
        
        return boxes_corner
        
    def _nms(self, boxes, scores):
        """Apply Non-Maximum Suppression (same as CPU)."""
        if len(boxes) == 0:
            return []
            
        # Sort by score
        indices = np.argsort(scores)[::-1]
        
        keep = []
        while len(indices) > 0:
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
        
    def start(self):
        """Start the async detection thread."""
        if self.is_running:
            return
            
        self.is_running = True
        self.detection_thread = threading.Thread(target=self._detection_loop)
        self.detection_thread.start()
        print("[GPU Detector] Started")

    def _with_cuda_context(self, func, *args, **kwargs):
        """Execute a function with CUDA context active."""
        self.cuda_context.push()
        try:
            return func(*args, **kwargs)
        finally:
            self.cuda_context.pop()
 
    def stop(self):
        """Stop the detection thread and cleanup resources."""
        self.is_running = False
        
        # Send stop signal
        try:
            self.input_queue.put(None, timeout=0.1)
        except:
            pass
            
        # Wait for thread to finish
        if self.detection_thread:
            self.detection_thread.join(timeout=1.0)
            
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
        
        # Destroy CUDA context
        if hasattr(self, 'cuda_context'):
            try:
                self.cuda_context.push()
                self.cuda_context.pop()
                self.cuda_context.detach()
            except:
                pass
                
        print("[GPU Detector] Stopped")

    def submit_frame(self, frame: np.ndarray, frame_id: int) -> bool:
        """Submit an RGB frame for detection."""
        try:
            self.input_queue.put((frame, frame_id), timeout=0.001)
            return True
        except Full:
            return False
            
    def get_detections(self, timeout: float = 0.001) -> Optional[Tuple[int, List[Dict]]]:
        """Get detection results."""
        try:
            return self.output_queue.get(timeout=timeout)
        except Empty:
            return None
            
    def get_stats(self) -> Dict:
        """Get performance statistics."""
        avg_fps = 0.0
        if self.fps_tracker:
            avg_fps = sum(self.fps_tracker) / len(self.fps_tracker)
            
        return {
            'avg_fps': avg_fps,
            'input_queue_size': self.input_queue.qsize(),
            'output_queue_size': self.output_queue.qsize(),
            'processing_mode': 'gpu_batch',
            'max_batch_size': self.max_batch_size
        }