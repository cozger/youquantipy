#!/usr/bin/env python3
"""Test script to diagnose TensorRT issues in GPU pipeline."""

import sys
import os
import logging
import numpy as np
import cupy as cp
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_trt_engine_directly():
    """Test TensorRT engine loading and execution directly."""
    
    # Path to TRT engine
    trt_path = "d:/Projects/youquantipy/retinaface.trt"
    
    if not os.path.exists(trt_path):
        logger.error(f"TRT engine not found at {trt_path}")
        return
    
    # Create runtime
    trt_logger = trt.Logger(trt.Logger.INFO)
    runtime = trt.Runtime(trt_logger)
    
    # Load engine
    with open(trt_path, 'rb') as f:
        engine_data = f.read()
    engine = runtime.deserialize_cuda_engine(engine_data)
    
    if engine is None:
        logger.error("Failed to deserialize engine")
        return
    
    logger.info(f"Successfully loaded engine with {engine.num_io_tensors} IO tensors")
    
    # Analyze engine
    print("\nEngine Analysis:")
    for i in range(engine.num_io_tensors):
        tensor_name = engine.get_tensor_name(i)
        shape = engine.get_tensor_shape(tensor_name)
        dtype = engine.get_tensor_dtype(tensor_name)
        mode = engine.get_tensor_mode(tensor_name)
        is_input = mode == trt.TensorIOMode.INPUT
        
        print(f"Tensor {i}: {tensor_name}")
        print(f"  Shape: {shape}")
        print(f"  Type: {dtype}")
        print(f"  Mode: {mode}")
        print(f"  Is Input: {is_input}")
    
    # Create context
    context = engine.create_execution_context()
    if context is None:
        logger.error("Failed to create context")
        return
    
    # Allocate buffers
    buffers = {}
    for i in range(engine.num_io_tensors):
        tensor_name = engine.get_tensor_name(i)
        shape = engine.get_tensor_shape(tensor_name)
        is_input = engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT
        
        # Replace dynamic dimensions
        shape_list = list(shape)
        for j, dim in enumerate(shape_list):
            if dim == -1:
                shape_list[j] = 1
        shape = tuple(shape_list)
        
        size = int(abs(np.prod(shape)) * 4)  # float32
        buffer = cuda.mem_alloc(size)
        
        buffers[tensor_name] = {
            'buffer': buffer,
            'shape': shape,
            'is_input': is_input
        }
        
        logger.info(f"Allocated buffer for {tensor_name}: shape={shape}, size={size}")
    
    # Create test input
    input_shape = (1, 608, 640, 3)  # NHWC format
    test_input = np.random.randint(0, 255, size=input_shape[1:], dtype=np.uint8)
    test_input_float = test_input.astype(np.float32)
    test_input_batched = test_input_float[np.newaxis, ...]
    
    logger.info(f"Test input shape: {test_input_batched.shape}, range: [{test_input_batched.min():.1f}, {test_input_batched.max():.1f}]")
    
    # Find input tensor
    input_name = None
    for name, buf in buffers.items():
        if buf['is_input']:
            input_name = name
            break
    
    if input_name is None:
        logger.error("No input tensor found")
        return
    
    # Copy input to GPU
    cuda.memcpy_htod(buffers[input_name]['buffer'], test_input_batched)
    
    # Test execution with new API
    if hasattr(context, 'set_tensor_address'):
        logger.info("Testing with new TensorRT API (execute_async_v3)")
        
        # Set input shape first
        context.set_input_shape(input_name, test_input_batched.shape)
        logger.info(f"Set input shape: {test_input_batched.shape}")
        
        # Set all tensor addresses
        for name, buf in buffers.items():
            context.set_tensor_address(name, int(buf['buffer']))
        
        # Execute
        try:
            stream = cuda.Stream()
            success = context.execute_async_v3(stream.handle)
            stream.synchronize()
            
            if success:
                logger.info("✓ execute_async_v3 succeeded!")
            else:
                logger.error("✗ execute_async_v3 returned False")
        except Exception as e:
            logger.error(f"✗ execute_async_v3 failed with exception: {e}")
    
    # Test with old API
    logger.info("\nTesting with old TensorRT API (execute_async_v2)")
    binding_list = []
    for name, buf in buffers.items():
        binding_list.append(int(buf['buffer']))
    
    try:
        stream = cuda.Stream()
        success = context.execute_async_v2(bindings=binding_list, stream_handle=stream.handle)
        stream.synchronize()
        
        if success:
            logger.info("✓ execute_async_v2 succeeded!")
        else:
            logger.error("✗ execute_async_v2 returned False")
    except Exception as e:
        logger.error(f"✗ execute_async_v2 failed with exception: {e}")
    
    # Test synchronous execution
    logger.info("\nTesting synchronous execution (execute_v2)")
    try:
        success = context.execute_v2(binding_list)
        if success:
            logger.info("✓ execute_v2 succeeded!")
        else:
            logger.error("✗ execute_v2 returned False")
    except Exception as e:
        logger.error(f"✗ execute_v2 failed with exception: {e}")

if __name__ == "__main__":
    print("TensorRT GPU Pipeline Test")
    print("=" * 50)
    test_trt_engine_directly()