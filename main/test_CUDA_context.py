#!/usr/bin/env python
"""
Test to verify CUDA context consistency between PyCUDA, CuPy, and TensorRT
"""

import numpy as np
import cupy as cp
import pycuda.driver as cuda
import tensorrt as trt

def test_context_consistency():
    """Test if PyCUDA and CuPy are using the same CUDA context"""
    print("Testing CUDA context consistency...\n")
    
    # Initialize CUDA with PyCUDA
    cuda.init()
    device = cuda.Device(0)
    ctx = device.make_context()
    
    print(f"PyCUDA context created: {ctx}")
    print(f"Current context (PyCUDA): {cuda.Context.get_current()}")
    
    # Create CuPy array
    try:
        cp_array = cp.zeros((10, 10), dtype=cp.float32)
        print(f"\nCuPy array created successfully")
        print(f"CuPy array ptr: {cp_array.data.ptr}")
        
        # Get CuPy's context handle
        cp_ctx = cp.cuda.runtime.getCurrentContext()
        print(f"CuPy context handle: {cp_ctx}")
        
    except Exception as e:
        print(f"CuPy error: {e}")
    
    # Try to use PyCUDA allocated memory with CuPy
    print("\nTest 1: PyCUDA memory with CuPy")
    try:
        # Allocate with PyCUDA
        pycuda_mem = cuda.mem_alloc(100 * 4)  # 100 floats
        print(f"PyCUDA memory allocated: {pycuda_mem}")
        
        # Try to wrap with CuPy
        cp_view = cp.ndarray(
            shape=(100,),
            dtype=cp.float32,
            memptr=cp.cuda.MemoryPointer(
                cp.cuda.UnownedMemory(int(pycuda_mem), 100 * 4, None),
                0
            )
        )
        cp_view[:] = 1.0
        print("SUCCESS: Can use PyCUDA memory with CuPy")
        
    except Exception as e:
        print(f"FAILED: {e}")
    
    # Test 2: CuPy memory with PyCUDA
    print("\nTest 2: CuPy memory with PyCUDA")
    try:
        # Create CuPy array
        cp_array2 = cp.ones((100,), dtype=cp.float32)
        
        # Try to use its pointer with PyCUDA
        cuda.memcpy_dtoh_async(
            np.zeros(100, dtype=np.float32),
            int(cp_array2.data.ptr),
            cuda.Stream()
        )
        print("SUCCESS: Can use CuPy memory with PyCUDA")
        
    except Exception as e:
        print(f"FAILED: {e}")
    
    # Test 3: TensorRT with both
    print("\nTest 3: TensorRT compatibility")
    try:
        # Create a simple TensorRT logger
        trt_logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(trt_logger)
        print("TensorRT runtime created successfully")
        
    except Exception as e:
        print(f"TensorRT error: {e}")
    
    # Cleanup
    ctx.pop()
    print("\nTest complete!")


def test_separate_contexts():
    """Test what happens with separate contexts"""
    print("\n" + "="*60)
    print("Testing separate contexts scenario...")
    
    # PyCUDA context
    cuda.init()
    device = cuda.Device(0)
    pycuda_ctx = device.make_context()
    print(f"PyCUDA context: {pycuda_ctx}")
    
    # Now create CuPy array (it might create its own context)
    cp_array = cp.zeros((10, 10), dtype=cp.float32)
    print(f"CuPy array created")
    
    # Check if contexts are different
    current_ctx = cuda.Context.get_current()
    print(f"Current context after CuPy: {current_ctx}")
    
    # Try to access PyCUDA memory after CuPy operations
    try:
        pycuda_mem = cuda.mem_alloc(100 * 4)
        print("PyCUDA allocation after CuPy: SUCCESS")
    except Exception as e:
        print(f"PyCUDA allocation after CuPy: FAILED - {e}")
    
    pycuda_ctx.pop()


if __name__ == "__main__":
    print("CUDA Context Consistency Test")
    print("=" * 60)
    
    # Test 1: With same context
    test_context_consistency()
    
    # Test 2: With potentially separate contexts
    test_separate_contexts()