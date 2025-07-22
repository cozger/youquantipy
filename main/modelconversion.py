#!/usr/bin/env python3
"""
MediaPipe Face Landmarker to TensorRT Converter
Converts MediaPipe's face landmark and blendshape models to TensorRT engines.

Usage:
    python convert_mediapipe_to_trt.py --model face_landmarker_v2_with_blendshapes.task --output_dir ./engines
"""

import os
import sys
import argparse
import zipfile
import shutil
import numpy as np
import tensorflow as tf
import onnx
import struct
from pathlib import Path

try:
    import tensorrt as trt
except ImportError:
    print("ERROR: TensorRT is required. Install with: pip install tensorrt")
    sys.exit(1)

try:
    import tf2onnx
except ImportError:
    print("ERROR: tf2onnx is required. Install with: pip install tf2onnx")
    sys.exit(1)


class MediaPipeToTensorRTConverter:
    def __init__(self, task_file_path, output_dir, combine_models=False):
        self.task_file_path = Path(task_file_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.combine_models = combine_models
        
        # Model names in MediaPipe task files
        self.model_names = {
            'detector': 'face_detection_short_range.tflite',
            'landmark': 'face_landmarks_detector.tflite', 
            'blendshape': 'face_blendshapes.tflite'
        }
        
        # TensorRT logger
        self.trt_logger = trt.Logger(trt.Logger.INFO)
        
        # Temp directory for extraction
        self.temp_dir = self.output_dir / 'temp_extraction'
        self.temp_dir.mkdir(exist_ok=True)
        
    def extract_models(self):
        """Extract individual TFLite models from MediaPipe task file"""
        print(f"\n[1/4] Extracting models from {self.task_file_path}")
        
        # MediaPipe task files are ZIP archives
        with zipfile.ZipFile(self.task_file_path, 'r') as task_zip:
            task_zip.extractall(self.temp_dir)
        
        # Find the actual model files
        extracted_models = {}
        for model_type, expected_name in self.model_names.items():
            # Search for the model file
            found = False
            for root, dirs, files in os.walk(self.temp_dir):
                for file in files:
                    if file.endswith('.tflite') and model_type in file.lower():
                        model_path = Path(root) / file
                        extracted_models[model_type] = model_path
                        print(f"  Found {model_type}: {file}")
                        found = True
                        break
                if found:
                    break
            
            if not found and model_type != 'detector':  # detector is optional since we use RetinaFace
                # Try alternative names
                alt_names = {
                    'landmark': ['face_landmark.tflite', 'face_landmarks.tflite'],
                    'blendshape': ['face_blendshape.tflite', 'blendshapes.tflite']
                }
                
                for alt_name in alt_names.get(model_type, []):
                    for root, dirs, files in os.walk(self.temp_dir):
                        if alt_name in files:
                            model_path = Path(root) / alt_name
                            extracted_models[model_type] = model_path
                            print(f"  Found {model_type}: {alt_name}")
                            found = True
                            break
                    if found:
                        break
        
        # Verify we have the required models
        if 'landmark' not in extracted_models:
            raise FileNotFoundError("Could not find face landmark model in task file")
        
        return extracted_models
    
    def analyze_tflite_model(self, tflite_path):
        """Analyze TFLite model to understand inputs/outputs"""
        print(f"\n  Analyzing {tflite_path.name}:")
        
        interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
        interpreter.allocate_tensors()
        
        # Get input details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"    Inputs:")
        for inp in input_details:
            print(f"      - {inp['name']}: shape={inp['shape']}, dtype={inp['dtype']}")
        
        print(f"    Outputs:")
        for out in output_details:
            print(f"      - {out['name']}: shape={out['shape']}, dtype={out['dtype']}")
        
        return input_details, output_details
    
    def convert_tflite_to_onnx(self, tflite_path, onnx_path):
        """Convert TFLite model to ONNX format"""
        print(f"\n  Converting {tflite_path.name} to ONNX...")
        
        # First analyze the model
        input_details, output_details = self.analyze_tflite_model(tflite_path)
        
        # Use tf2onnx command line tool (more reliable)
        cmd = f"""
        python -m tf2onnx.convert \
            --tflite {tflite_path} \
            --output {onnx_path} \
            --opset 13 \
            --verbose
        """
        
        result = os.system(cmd)
        if result != 0:
            raise RuntimeError(f"Failed to convert {tflite_path} to ONNX")
        
        print(f"  ✓ Saved ONNX to {onnx_path}")
        
        # Verify ONNX model
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
        
        return True
    
    def build_tensorrt_engine(self, onnx_path, engine_path, fp16=True, int8=False,
                            max_batch_size=8, max_workspace_size=1<<30):
        """Build TensorRT engine from ONNX model"""
        print(f"\n  Building TensorRT engine from {onnx_path.name}...")
        
        builder = trt.Builder(self.trt_logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, self.trt_logger)
        
        # Parse ONNX
        print("    Parsing ONNX model...")
        with open(onnx_path, 'rb') as f:
            if not parser.parse(f.read()):
                for error in range(parser.num_errors):
                    print(f"    ERROR: {parser.get_error(error)}")
                raise RuntimeError("Failed to parse ONNX model")
        
        # Get network input shape for dynamic batch size
        network_input = network.get_input(0)
        input_shape = network_input.shape
        print(f"    Network input shape: {input_shape}")
        
        # Create builder config
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, max_workspace_size)
        
        # Set precision flags
        if fp16 and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("    ✓ Enabled FP16 precision")
        
        if int8 and builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            print("    ✓ Enabled INT8 precision")
            # Note: INT8 requires calibration data
        
        # Create optimization profile for dynamic batch size
        profile = builder.create_optimization_profile()
        
        # Handle different model types
        model_name = onnx_path.stem
        if 'landmark' in model_name:
            # Face landmark model: [batch, 256, 256, 3] - MediaPipe uses 256x256!
            profile.set_shape(
                network_input.name,
                (1, 256, 256, 3),    # min
                (4, 256, 256, 3),    # optimal
                (max_batch_size, 256, 256, 3)  # max
            )
        elif 'blendshape' in model_name:
            # Blendshape model: [batch, 146, 2] (146 2D points)
            # Check if the model has static batch size
            if input_shape[0] == 1:
                # Static batch size - use 1 for all profiles
                profile.set_shape(
                    network_input.name,
                    (1, 146, 2),    # min
                    (1, 146, 2),    # optimal  
                    (1, 146, 2)     # max
                )
                print("    Note: Blendshape model has static batch size of 1")
            else:
                # Dynamic batch size
                profile.set_shape(
                    network_input.name,
                    (1, 146, 2),    # min
                    (4, 146, 2),    # optimal  
                    (max_batch_size, 146, 2)  # max
                )
        
        config.add_optimization_profile(profile)
        
        # Build engine
        print("    Building engine (this may take a few minutes)...")
        serialized_engine = builder.build_serialized_network(network, config)
        
        if serialized_engine is None:
            raise RuntimeError("Failed to build TensorRT engine")
        
        # Save engine
        with open(engine_path, 'wb') as f:
            f.write(serialized_engine)
        
        print(f"  ✓ Saved TensorRT engine to {engine_path}")
        
        # Print engine stats
        runtime = trt.Runtime(self.trt_logger)
        with open(engine_path, 'rb') as f:
            engine_bytes = f.read()
            engine = runtime.deserialize_cuda_engine(engine_bytes)
            print(f"    Engine size: {len(engine_bytes) / 1024 / 1024:.2f} MB")
            # Use new API for TensorRT 8.5+
            try:
                print(f"    Num tensors: {engine.num_io_tensors}")
            except AttributeError:
                # Fallback for older TensorRT
                print(f"    Num bindings: {engine.num_bindings}")
        
        return True
    
    def create_combined_engine(self, landmark_engine_path, blendshape_engine_path, combined_path):
        """Create a combined engine that runs both models in sequence"""
        print(f"\n[EXPERIMENTAL] Creating combined engine...")
        
        # This is complex because we need to create a custom ONNX graph
        # that combines both models. For now, we'll create a wrapper class instead.
        
        wrapper_code = f'''
"""
Combined MediaPipe Face Processor Engine Wrapper
Auto-generated by convert_mediapipe_to_trt.py
"""

import tensorrt as trt
import pycuda.driver as cuda
import numpy as np

class CombinedFaceEngine:
    """Wrapper that runs landmark and blendshape models in sequence"""
    
    def __init__(self, landmark_engine_path="{landmark_engine_path.name}", 
                 blendshape_engine_path="{blendshape_engine_path.name}"):
        # Load both engines
        self.logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(self.logger)
        
        with open(landmark_engine_path, 'rb') as f:
            self.landmark_engine = runtime.deserialize_cuda_engine(f.read())
        self.landmark_context = self.landmark_engine.create_execution_context()
        
        if blendshape_engine_path:
            with open(blendshape_engine_path, 'rb') as f:
                self.blendshape_engine = runtime.deserialize_cuda_engine(f.read())
            self.blendshape_context = self.blendshape_engine.create_execution_context()
        
        # Allocate buffers
        self._allocate_buffers()
        
        # Create CUDA stream
        self.stream = cuda.Stream()
        
        # MediaPipe landmark indices for blendshapes (subset of 468)
        self.blendshape_indices = self._get_blendshape_indices()
    
    def _get_blendshape_indices(self):
        """Get the 146 landmark indices used for blendshape computation"""
        # These are the specific landmarks MediaPipe uses for blendshapes
        # This is a simplified version - you may need to adjust based on actual model
        indices = []
        
        # Key facial features (this is approximate - verify with actual model)
        # Mouth region
        indices.extend(range(0, 17))    # Outer lips
        indices.extend(range(17, 22))   # Upper lip
        indices.extend(range(22, 27))   # Lower lip
        
        # Eyes region  
        indices.extend(range(33, 42))   # Right eye
        indices.extend(range(42, 51))   # Left eye
        
        # Eyebrows
        indices.extend(range(52, 68))   # Eyebrows
        
        # Nose
        indices.extend(range(68, 85))   # Nose
        
        # Face oval
        indices.extend(range(85, 117))  # Face contour
        
        # Fill to 146 if needed
        while len(indices) < 146:
            indices.append(indices[-1])
            
        return indices[:146]
    
    def process_batch(self, images_batch):
        """Process a batch of face images"""
        batch_size = images_batch.shape[0]
        
        # Run landmark detection
        landmarks = self._run_landmarks(images_batch)
        
        # Extract subset for blendshapes
        if hasattr(self, 'blendshape_engine'):
            # Extract the specific landmarks needed
            blendshape_input = landmarks[:, self.blendshape_indices, :2]  # Only x,y
            blendshapes = self._run_blendshapes(blendshape_input)
        else:
            blendshapes = None
        
        return {{
            'landmarks': landmarks,
            'blendshapes': blendshapes
        }}
'''
        
        # Save wrapper code
        wrapper_path = combined_path.with_suffix('.py')
        with open(wrapper_path, 'w') as f:
            f.write(wrapper_code)
        
        print(f"  ✓ Created combined engine wrapper at {wrapper_path}")
        print(f"    Note: This is a Python wrapper, not a single TRT engine")
        
        return True
    
    def cleanup(self):
        """Clean up temporary files"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            print("\n✓ Cleaned up temporary files")
    
    def convert(self):
        """Main conversion pipeline"""
        print(f"\n{'='*60}")
        print(f"MediaPipe to TensorRT Converter")
        print(f"{'='*60}")
        print(f"Input: {self.task_file_path}")
        print(f"Output: {self.output_dir}")
        print(f"{'='*60}")
        
        try:
            # Step 1: Extract models
            models = self.extract_models()
            
            # Step 2: Convert to ONNX
            print(f"\n[2/4] Converting models to ONNX format")
            onnx_models = {}
            
            for model_type, tflite_path in models.items():
                if model_type == 'detector':
                    continue  # Skip detector since we use RetinaFace
                    
                onnx_path = self.output_dir / f"{model_type}.onnx"
                self.convert_tflite_to_onnx(tflite_path, onnx_path)
                onnx_models[model_type] = onnx_path
            
            # Step 3: Build TensorRT engines
            print(f"\n[3/4] Building TensorRT engines")
            trt_engines = {}
            
            for model_type, onnx_path in onnx_models.items():
                engine_path = self.output_dir / f"{model_type}.trt"
                self.build_tensorrt_engine(onnx_path, engine_path)
                trt_engines[model_type] = engine_path
            
            # Step 4: Optionally create combined engine
            if self.combine_models and 'landmark' in trt_engines and 'blendshape' in trt_engines:
                print(f"\n[4/4] Creating combined engine")
                combined_path = self.output_dir / "combined_face_engine"
                self.create_combined_engine(
                    trt_engines['landmark'],
                    trt_engines['blendshape'],
                    combined_path
                )
            else:
                print(f"\n[4/4] Skipping combined engine creation")
            
            # Summary
            print(f"\n{'='*60}")
            print(f"✓ Conversion completed successfully!")
            print(f"\nGenerated files:")
            for file in self.output_dir.glob("*.trt"):
                print(f"  - {file.name} ({file.stat().st_size / 1024 / 1024:.2f} MB)")
            for file in self.output_dir.glob("*.onnx"):
                print(f"  - {file.name} ({file.stat().st_size / 1024 / 1024:.2f} MB)")
            print(f"{'='*60}")
            print(f"\n⚠️  IMPORTANT: These engines require GPU for inference")
            print(f"Ensure CUDA and TensorRT runtime are properly installed")
            
            # Create example usage code
            self._create_example_usage()
            
        finally:
            self.cleanup()
    
    def _create_example_usage(self):
        """Create example code for using the generated engines"""
        example_code = '''
"""
Example usage of the converted TensorRT engines
"""

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2

class MediaPipeTRTInference:
    def __init__(self, landmark_engine_path, blendshape_engine_path=None):
        # Initialize TensorRT
        self.logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(self.logger)
        
        # Load engines
        with open(landmark_engine_path, 'rb') as f:
            self.landmark_engine = runtime.deserialize_cuda_engine(f.read())
        self.landmark_ctx = self.landmark_engine.create_execution_context()
        
        # Allocate memory
        self.landmark_input = cuda.mem_alloc(1 * 192 * 192 * 3 * 4)  # float32
        self.landmark_output = cuda.mem_alloc(1 * 468 * 3 * 4)  # 468 3D points
        
        self.stream = cuda.Stream()
    
    def preprocess(self, face_roi):
        """MediaPipe-compatible preprocessing"""
        # Resize to 192x192
        resized = cv2.resize(face_roi, (192, 192))
        
        # Normalize to [-1, 1]
        normalized = (resized.astype(np.float32) - 127.5) / 127.5
        
        # Ensure NHWC format
        if normalized.ndim == 3:
            normalized = np.expand_dims(normalized, 0)
        
        return normalized
    
    def detect_landmarks(self, face_roi):
        """Run landmark detection"""
        # Preprocess
        input_data = self.preprocess(face_roi)
        
        # Copy to GPU
        cuda.memcpy_htod_async(self.landmark_input, input_data, self.stream)
        
        # Run inference
        self.landmark_ctx.execute_async_v2(
            bindings=[int(self.landmark_input), int(self.landmark_output)],
            stream_handle=self.stream.handle
        )
        
        # Copy back results
        output = np.empty((468, 3), dtype=np.float32)
        cuda.memcpy_dtoh_async(output, self.landmark_output, self.stream)
        self.stream.synchronize()
        
        return output

# Usage
if __name__ == "__main__":
    # Initialize
    detector = MediaPipeTRTInference("landmark.trt", "blendshape.trt")
    
    # Load test image
    image = cv2.imread("face.jpg")
    roi = image  # Assume face is already cropped
    
    # Detect
    landmarks = detector.detect_landmarks(roi)
    print(f"Detected {len(landmarks)} landmarks")
'''
        
        example_path = self.output_dir / "example_usage.py"
        with open(example_path, 'w') as f:
            f.write(example_code)
        
        print(f"\n✓ Created example usage code at {example_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert MediaPipe Face Landmarker to TensorRT"
    )
    parser.add_argument(
        "--model", "-m",
        required=True,
        help="Path to MediaPipe face landmarker task file (.task)"
    )
    parser.add_argument(
        "--output_dir", "-o",
        default="./trt_engines",
        help="Output directory for TensorRT engines"
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=True,
        help="Enable FP16 precision (default: True)"
    )
    parser.add_argument(
        "--int8",
        action="store_true",
        help="Enable INT8 precision (requires calibration)"
    )
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=8,
        help="Maximum batch size for TensorRT engines"
    )
    parser.add_argument(
        "--combine",
        action="store_true",
        help="Create combined engine wrapper"
    )
    
    args = parser.parse_args()
    
    # Verify input file exists
    if not os.path.exists(args.model):
        print(f"ERROR: Model file not found: {args.model}")
        sys.exit(1)
    
    # Create converter
    converter = MediaPipeToTensorRTConverter(
        args.model,
        args.output_dir,
        combine_models=args.combine
    )
    
    # Run conversion
    try:
        converter.convert()
    except Exception as e:
        print(f"\nERROR: Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()