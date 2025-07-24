#!/usr/bin/env python
"""Quick analysis of TensorRT engine to debug shape issues"""

import tensorrt as trt
import sys

def analyze_engine(engine_path):
    """Analyze TensorRT engine inputs/outputs"""
    trt_logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(trt_logger)
    
    print(f"\nAnalyzing engine: {engine_path}")
    print("=" * 60)
    
    try:
        with open(engine_path, 'rb') as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        
        if engine is None:
            print("Failed to deserialize engine!")
            return
            
        print(f"Number of IO tensors: {engine.num_io_tensors}")
        
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            shape = engine.get_tensor_shape(name)
            dtype = engine.get_tensor_dtype(name)
            mode = engine.get_tensor_mode(name)
            is_input = mode == trt.TensorIOMode.INPUT
            
            print(f"\nTensor {i}: {name}")
            print(f"  Shape: {shape}")
            print(f"  Type: {dtype}")
            print(f"  Mode: {'INPUT' if is_input else 'OUTPUT'}")
            
            if is_input:
                # Determine format
                if len(shape) == 4:
                    if shape[-1] == 3:  # NHWC
                        print(f"  Format: NHWC")
                        print(f"  Height: {shape[1]}, Width: {shape[2]}")
                    else:  # NCHW
                        print(f"  Format: NCHW")
                        print(f"  Height: {shape[2]}, Width: {shape[3]}")
        
    except Exception as e:
        print(f"Error analyzing engine: {e}")

if __name__ == "__main__":
    # Analyze all TRT engines
    engines = [
        "D:/Projects/youquantipy/retinaface.trt",
        "D:/Projects/youquantipy/landmark.trt",
        "D:/Projects/youquantipy/blendshape.trt"
    ]
    
    for engine_path in engines:
        analyze_engine(engine_path)