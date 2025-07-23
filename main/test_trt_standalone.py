#!/usr/bin/env python
"""
Standalone test for TensorRT RetinaFace model
Tests if the TRT model is producing correct detection scores
"""

import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import time

class SimpleTRTDetector:
    def __init__(self, engine_path):
        self.engine_path = engine_path
        
        # Load TRT engine
        self.trt_logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(self.trt_logger)
        
        print(f"Loading TRT engine from {engine_path}")
        with open(engine_path, 'rb') as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        
        # Analyze engine
        self._analyze_engine()
        
        # Allocate buffers
        self._allocate_buffers()
        
        # Generate anchors (standard RetinaFace anchors)
        self._generate_anchors()
        
    def _analyze_engine(self):
        """Analyze engine inputs/outputs"""
        print("\nEngine Analysis:")
        # New TensorRT API
        print(f"Number of IO tensors: {self.engine.num_io_tensors}")
        
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(name)
            dtype = self.engine.get_tensor_dtype(name)
            mode = self.engine.get_tensor_mode(name)
            is_input = mode == trt.TensorIOMode.INPUT
            
            print(f"Tensor {i}: {name}")
            print(f"  Shape: {shape}")
            print(f"  Type: {dtype}")
            print(f"  Mode: {mode}")
            print(f"  Is Input: {is_input}")
            
            if is_input:
                self.input_shape = shape
                self.input_name = name
                # Determine input dimensions
                if shape[-1] == 3:  # NHWC
                    self.input_height = shape[1]
                    self.input_width = shape[2]
                else:  # NCHW
                    self.input_height = shape[2]
                    self.input_width = shape[3]
        
        print(f"\nInput dimensions: {self.input_width}x{self.input_height}")
    
    def _allocate_buffers(self):
        """Allocate GPU memory"""
        self.tensors = {}
        self.outputs = []
        
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = list(self.engine.get_tensor_shape(name))
            dtype = self.engine.get_tensor_dtype(name)
            mode = self.engine.get_tensor_mode(name)
            
            # Replace dynamic dimension with batch size 1
            shape = [1 if s == -1 else s for s in shape]
            
            size = int(np.prod(shape)) * np.dtype(np.float32).itemsize
            
            # Allocate device memory
            device_mem = cuda.mem_alloc(size)
            self.tensors[name] = device_mem
            
            # For outputs, also create host buffer
            if mode == trt.TensorIOMode.OUTPUT:
                host_mem = cuda.pagelocked_empty(int(np.prod(shape)), dtype=np.float32)
                self.outputs.append((name, host_mem, device_mem, shape))
        
        # Create stream
        self.stream = cuda.Stream()
    
    def _generate_anchors(self):
        """Generate RetinaFace anchors"""
        # Standard RetinaFace anchor configuration
        min_sizes = [[16, 32], [64, 128], [256, 512]]
        steps = [8, 16, 32]
        
        anchors = []
        
        for k, step in enumerate(steps):
            grid_h = self.input_height // step
            grid_w = self.input_width // step
            
            for i in range(grid_h):
                for j in range(grid_w):
                    for min_size in min_sizes[k]:
                        cx = (j + 0.5) * step / self.input_width
                        cy = (i + 0.5) * step / self.input_height
                        w = min_size / self.input_width
                        h = min_size / self.input_height
                        anchors.append([cx, cy, w, h])
        
        self.anchors = np.array(anchors, dtype=np.float32)
        print(f"Generated {len(self.anchors)} anchors")
    
    def preprocess(self, image, method='subtract_mean'):
        """Preprocess image for RetinaFace"""
        # Resize to model input size
        resized = cv2.resize(image, (self.input_width, self.input_height))
        
        # Convert to float32
        input_data = resized.astype(np.float32)
        
        # Try different preprocessing methods
        if method == 'normalize':
            # Option 1: Simple normalization to [0,1]
            input_data = input_data / 255.0
        elif method == 'imagenet':
            # Option 2: ImageNet normalization
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            input_data = (input_data / 255.0 - mean) / std
        elif method == 'subtract_mean':
            # Option 3: Subtract mean (common for RetinaFace)
            input_data = input_data - 127.5
            input_data = input_data / 128.0
        elif method == 'raw':
            # Option 4: Raw pixel values
            pass
        
        print(f"Preprocessing method: {method}, Data range: [{input_data.min():.2f}, {input_data.max():.2f}]")
        
        # Add batch dimension
        if self.input_shape[-1] == 3:  # NHWC
            input_data = input_data[np.newaxis, ...]
        else:  # NCHW
            input_data = input_data.transpose(2, 0, 1)[np.newaxis, ...]
        
        return input_data
    
    def detect(self, image, preprocess_method='subtract_mean'):
        """Run detection on image"""
        # Preprocess
        input_data = self.preprocess(image, method=preprocess_method)
        
        # Set input shape for dynamic batch
        input_shape = list(input_data.shape)
        if self.input_shape[-1] == 3:  # NHWC
            self.context.set_input_shape(self.input_name, input_shape)
        else:
            self.context.set_input_shape(self.input_name, input_shape)
        
        # Copy to GPU
        cuda.memcpy_htod_async(self.tensors[self.input_name], input_data, self.stream)
        
        # Set tensor addresses for new API
        for name, mem in self.tensors.items():
            self.context.set_tensor_address(name, int(mem))
        
        # Run inference
        if not self.context.execute_async_v3(stream_handle=self.stream.handle):
            print("ERROR: Inference failed!")
            return []
        
        # Copy outputs back
        outputs_data = []
        for name, host_mem, device_mem, shape in self.outputs:
            cuda.memcpy_dtoh_async(host_mem, device_mem, self.stream)
            outputs_data.append(host_mem.reshape(shape))
        
        # Synchronize
        self.stream.synchronize()
        
        # Process outputs
        return self._process_outputs(outputs_data)
    
    def _process_outputs(self, outputs):
        """Process RetinaFace outputs"""
        print(f"\nOutput shapes: {[out.shape for out in outputs]}")
        
        # RetinaFace typically outputs:
        # - bbox regression (num_anchors, 4)
        # - face confidence (num_anchors, 2) or (num_anchors, 1)
        # - landmarks (num_anchors, 10)
        
        # Find confidence scores based on shape
        conf_output = None
        bbox_output = None
        landmark_output = None
        
        for out in outputs:
            if out.shape[-1] == 2 and out.shape[1] == 15960:  # Binary classification
                conf_output = out[0]  # Remove batch dimension
                scores = conf_output[:, 1]  # Foreground scores
            elif out.shape[-1] == 4 and out.shape[1] == 15960:  # Bounding boxes
                bbox_output = out[0]
            elif out.shape[-1] == 10 and out.shape[1] == 15960:  # Landmarks
                landmark_output = out[0]
        
        if conf_output is None:
            print("WARNING: Could not find confidence output!")
            return []
        
        # Apply sigmoid to get probabilities if needed
        if scores.min() < 0 or scores.max() > 1:
            print("Applying sigmoid to scores...")
            scores = 1 / (1 + np.exp(-scores))
        
        # Debug: show score statistics
        print(f"Score stats - Min: {scores.min():.4f}, Max: {scores.max():.4f}, Mean: {scores.mean():.4f}")
        print(f"Scores > 0.9: {(scores > 0.9).sum()}")
        print(f"Scores > 0.5: {(scores > 0.5).sum()}")
        print(f"Scores > 0.3: {(scores > 0.3).sum()}")
        print(f"Scores > 0.1: {(scores > 0.1).sum()}")
        
        # Get top scores
        if len(scores) > 0:
            top_k = min(10, len(scores))
            top_indices = np.argsort(scores)[-top_k:][::-1]
            print(f"\nTop {top_k} scores:")
            for idx in top_indices:
                print(f"  Anchor {idx}: {scores[idx]:.4f}")
                
            # Decode top detections
            if bbox_output is not None and scores.max() > 0.5:
                print("\nDecoding top detection:")
                best_idx = np.argmax(scores)
                anchor = self.anchors[best_idx]
                bbox_delta = bbox_output[best_idx]
                
                # Decode bbox (RetinaFace style)
                cx = anchor[0] + bbox_delta[0] * 0.1 * anchor[2]
                cy = anchor[1] + bbox_delta[1] * 0.1 * anchor[3]
                w = anchor[2] * np.exp(bbox_delta[2] * 0.2)
                h = anchor[3] * np.exp(bbox_delta[3] * 0.2)
                
                x1 = (cx - w/2) * self.input_width
                y1 = (cy - h/2) * self.input_height
                x2 = (cx + w/2) * self.input_width
                y2 = (cy + h/2) * self.input_height
                
                print(f"  Best detection: score={scores[best_idx]:.4f}, bbox=[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
        else:
            print("No scores available!")
        
        return scores

def main():
    # Initialize camera
    print("Opening camera...")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    # Initialize detector
    engine_path = "d:/Projects/youquantipy/retinaface.trt"
    detector = SimpleTRTDetector(engine_path)
    
    print("\nStarting detection loop (press 'q' to quit)...")
    frame_count = 0
    
    # Try different preprocessing methods
    methods = ['subtract_mean', 'normalize', 'imagenet', 'raw']
    method_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break
        
        frame_count += 1
        
        # Run detection every 10 frames
        if frame_count % 10 == 0:
            method = methods[method_idx % len(methods)]
            print(f"\n--- Frame {frame_count} ---")
            scores = detector.detect(frame, preprocess_method=method)
            method_idx += 1
        
        # Display
        cv2.imshow("Test TRT Detection", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()