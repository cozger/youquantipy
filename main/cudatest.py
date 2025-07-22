"""
Fixed CUDA/TensorRT Diagnostic Script for Windows
"""

import sys
import numpy as np
from multiprocessing import Process, Queue, set_start_method

# Global worker function for Windows compatibility
def cuda_worker(worker_id, result_queue):
    """Worker process that uses CUDA"""
    try:
        import pycuda.driver as cuda
        import pycuda.autoinit
        
        # Each process needs its own context
        device = cuda.Device(0)
        ctx = device.make_context()
        
        result_queue.put(f"Worker {worker_id}: CUDA context created successfully")
        
        # Cleanup
        ctx.pop()
        ctx.detach()
        
    except Exception as e:
        result_queue.put(f"Worker {worker_id} error: {e}")


def test_cuda_setup():
    """Test CUDA and TensorRT installation"""
    print("="*60)
    print("CUDA/TensorRT Diagnostic")
    print("="*60)
    
    # Test 1: PyCUDA
    try:
        import pycuda.driver as cuda
        import pycuda.autoinit
        print("✓ PyCUDA imported successfully")
        
        # Get device info
        device_count = cuda.Device.count()
        print(f"✓ CUDA devices found: {device_count}")
        
        for i in range(device_count):
            device = cuda.Device(i)
            print(f"\nDevice {i}: {device.name()}")
            print(f"  Compute capability: {device.compute_capability()}")
            print(f"  Total memory: {device.total_memory() / 1024**3:.1f} GB")
            
            # Test context creation
            ctx = device.make_context()
            print(f"  ✓ Context created successfully")
            ctx.pop()
            ctx.detach()
            print(f"  ✓ Context cleaned up successfully")
            
    except Exception as e:
        print(f"✗ PyCUDA error: {e}")
        return False
    
    # Test 2: TensorRT
    try:
        import tensorrt as trt
        print("\n✓ TensorRT imported successfully")
        print(f"  TensorRT version: {trt.__version__}")
        
        # Create logger
        logger = trt.Logger(trt.Logger.WARNING)
        print("  ✓ Logger created")
        
        # Create runtime
        runtime = trt.Runtime(logger)
        print("  ✓ Runtime created")
        
    except Exception as e:
        print(f"✗ TensorRT error: {e}")
        return False
    
    # Test 3: Multi-process CUDA (Windows-compatible)
    print("\n" + "="*60)
    print("Testing multi-process CUDA access...")
    print("="*60)
    
    try:
        # Use spawn method for Windows
        if sys.platform == 'win32':
            set_start_method('spawn', force=True)
        
        # Start multiple workers
        result_queue = Queue()
        workers = []
        
        for i in range(2):
            p = Process(target=cuda_worker, args=(i, result_queue))
            p.start()
            workers.append(p)
        
        # Wait for workers
        for p in workers:
            p.join()
        
        # Get results
        results = []
        while not result_queue.empty():
            result = result_queue.get()
            results.append(result)
            print(f"  {result}")
        
        # Check if all workers succeeded
        success_count = sum(1 for r in results if "successfully" in r)
        if success_count == len(workers):
            print("\n✓ Multi-process CUDA test passed!")
        else:
            print(f"\n✗ Only {success_count}/{len(workers)} workers succeeded")
            
    except Exception as e:
        print(f"✗ Multi-process test error: {e}")
        print("  Note: This is common on Windows. Single-process CUDA still works.")
    
    return True


def test_trt_engines():
    """Test loading TensorRT engines"""
    print("\n" + "="*60)
    print("Testing TensorRT Engine Loading")
    print("="*60)
    
    import os
    try:
        from confighandler import ConfigHandler
        config = ConfigHandler().config
        landmark_path = config['advanced_detection']['landmark_trt_path']
        blendshape_path = config['advanced_detection']['blendshape_trt_path']
        
        print(f"Landmark engine: {landmark_path}")
        print(f"  Exists: {os.path.exists(landmark_path)}")
        if os.path.exists(landmark_path):
            print(f"  Size: {os.path.getsize(landmark_path) / 1024**2:.1f} MB")
        
        print(f"\nBlendshape engine: {blendshape_path}")
        print(f"  Exists: {os.path.exists(blendshape_path)}")
        if os.path.exists(blendshape_path):
            print(f"  Size: {os.path.getsize(blendshape_path) / 1024**2:.1f} MB")
            
        # Try loading engines
        if os.path.exists(landmark_path):
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit
            
            logger = trt.Logger(trt.Logger.WARNING)
            runtime = trt.Runtime(logger)
            
            with open(landmark_path, 'rb') as f:
                engine = runtime.deserialize_cuda_engine(f.read())
                
            if engine:
                print("\n✓ Landmark engine loaded successfully")
                
                # Check API version
                try:
                    # New API
                    print(f"  Num IO tensors: {engine.num_io_tensors}")
                    for i in range(engine.num_io_tensors):
                        name = engine.get_tensor_name(i)
                        shape = engine.get_tensor_shape(name)
                        dtype = engine.get_tensor_dtype(name)
                        mode = engine.get_tensor_mode(name)
                        print(f"  Tensor '{name}': shape={shape}, dtype={dtype}, mode={mode}")
                        
                        # Calculate expected output size
                        if mode == trt.TensorIOMode.OUTPUT:
                            size = 1
                            for dim in shape:
                                if dim > 0:
                                    size *= dim
                            print(f"    Output size per batch: {size}")
                            if len(shape) > 1 and shape[-1] == 3:
                                landmarks = size // 3
                                print(f"    Landmarks: {landmarks}")
                except:
                    # Legacy API
                    print("  (Using legacy API)")
                    print(f"  Num bindings: {engine.num_bindings}")
                    for i in range(engine.num_bindings):
                        name = engine.get_binding_name(i)
                        shape = engine.get_binding_shape(i)
                        is_input = engine.binding_is_input(i)
                        print(f"  Binding '{name}': shape={shape}, is_input={is_input}")
                        
                        if not is_input and len(shape) > 1:
                            size = 1
                            for dim in shape:
                                if dim > 0:
                                    size *= dim
                            print(f"    Output size: {size}")
                            
                # Create context and check shapes
                ctx = engine.create_execution_context()
                print("\n  ✓ Execution context created")
                del ctx
                del engine
            else:
                print("✗ Failed to load landmark engine")
                
    except Exception as e:
        print(f"✗ Error testing engines: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("Starting CUDA/TensorRT diagnostics...\n")
    
    if test_cuda_setup():
        test_trt_engines()
    else:
        print("\n✗ Basic CUDA setup failed. Please check your installation.")
        print("\nRequired packages:")
        print("  - CUDA Toolkit (matching your GPU driver)")
        print("  - pip install pycuda")
        print("  - pip install tensorrt")