#!/usr/bin/env python
"""
Diagnostic script for TensorRT execution issues
Tests different API methods to find what works
"""

import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

def test_trt_execution(engine_path):
    """Test different TensorRT execution methods"""
    print(f"\n{'='*60}")
    print(f"Testing TensorRT execution methods for: {engine_path}")
    print(f"{'='*60}\n")
    
    # Load engine
    trt_logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(trt_logger)
    
    with open(engine_path, 'rb') as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    
    if engine is None:
        print("ERROR: Failed to load engine!")
        return
    
    context = engine.create_execution_context()
    stream = cuda.Stream()
    
    # Analyze engine
    print("Engine Info:")
    print(f"  Number of tensors: {engine.num_io_tensors}")
    
    # Find input/output info
    input_name = None
    input_shape = None
    buffers = {}
    
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = engine.get_tensor_shape(name)
        mode = engine.get_tensor_mode(name)
        is_input = mode == trt.TensorIOMode.INPUT
        
        print(f"  Tensor '{name}': shape={shape}, input={is_input}")
        
        # Handle dynamic dimensions
        shape_list = list(shape)
        for j, dim in enumerate(shape_list):
            if dim == -1:
                shape_list[j] = 1
        shape = tuple(shape_list)
        
        # Allocate buffer
        size = int(np.prod(shape)) * 4  # float32
        buffer = cuda.mem_alloc(size)
        buffers[name] = {'buffer': buffer, 'shape': shape, 'is_input': is_input}
        
        if is_input and input_name is None:
            input_name = name
            input_shape = shape
    
    if input_name is None:
        print("ERROR: No input tensor found!")
        return
    
    print(f"\nUsing input tensor: '{input_name}' with shape {input_shape}")
    
    # Create dummy input data
    if len(input_shape) == 4 and input_shape[-1] == 3:  # NHWC
        print("  Format: NHWC")
        h, w = input_shape[1], input_shape[2]
    elif len(input_shape) == 4:  # NCHW
        print("  Format: NCHW")
        h, w = input_shape[2], input_shape[3]
    else:
        h, w = 256, 256  # Default
    
    # Test data with raw pixel values (proven to work)
    input_data = np.random.randint(0, 255, size=input_shape, dtype=np.uint8).astype(np.float32)
    cuda.memcpy_htod_async(buffers[input_name]['buffer'], input_data, stream)
    
    print(f"\nInput data: shape={input_data.shape}, range=[{input_data.min():.1f}, {input_data.max():.1f}]")
    
    # Test 1: execute_async_v3 with stream_handle parameter
    print("\n1. Testing execute_async_v3(stream_handle=...):")
    try:
        # Set input shape first
        context.set_input_shape(input_name, input_shape)
        
        # Set tensor addresses
        for name, buf in buffers.items():
            context.set_tensor_address(name, int(buf['buffer']))
        
        success = context.execute_async_v3(stream_handle=stream.handle)
        print(f"   Result: {'SUCCESS' if success else 'FAILED'}")
    except Exception as e:
        print(f"   Exception: {e}")
    
    # Test 2: execute_async_v3 without parameters
    print("\n2. Testing execute_async_v3() without parameters:")
    try:
        success = context.execute_async_v3()
        print(f"   Result: {'SUCCESS' if success else 'FAILED'}")
    except Exception as e:
        print(f"   Exception: {e}")
    
    # Test 3: execute_async_v2
    print("\n3. Testing execute_async_v2:")
    try:
        # Build bindings array
        bindings = []
        for i in range(engine.num_io_tensors):
            tensor_name = engine.get_tensor_name(i)
            bindings.append(int(buffers[tensor_name]['buffer']))
        
        success = context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        print(f"   Result: {'SUCCESS' if success else 'FAILED'}")
    except Exception as e:
        print(f"   Exception: {e}")
    
    # Test 4: Synchronous execute_v2
    print("\n4. Testing synchronous execute_v2:")
    try:
        success = context.execute_v2(bindings)
        print(f"   Result: {'SUCCESS' if success else 'FAILED'}")
    except Exception as e:
        print(f"   Exception: {e}")
    
    # Test 5: Check if we have the new API methods
    print("\n5. Checking available context methods:")
    methods = [m for m in dir(context) if 'execute' in m]
    for method in methods:
        print(f"   - {method}")
    
    # Cleanup
    stream.synchronize()
    
    print("\nDiagnostic complete!")


if __name__ == "__main__":
    # Test RetinaFace engine
    test_trt_execution("D:/Projects/youquantipy/retinaface.trt")