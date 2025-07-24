"""
GPU-optimized color conversion kernels.

This module provides CUDA kernels for fast color space conversions.
"""

import cupy as cp

# CUDA kernel for BGR to RGB conversion
bgr_to_rgb_kernel = cp.RawKernel(r'''
extern "C" __global__
void bgr_to_rgb(unsigned char* src, unsigned char* dst, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = (y * width + x) * 3;
    
    // Swap B and R channels
    dst[idx + 0] = src[idx + 2];  // R = B
    dst[idx + 1] = src[idx + 1];  // G = G
    dst[idx + 2] = src[idx + 0];  // B = R
}
''', 'bgr_to_rgb')

# CUDA kernel for in-place BGR to RGB conversion
bgr_to_rgb_inplace_kernel = cp.RawKernel(r'''
extern "C" __global__
void bgr_to_rgb_inplace(unsigned char* data, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = (y * width + x) * 3;
    
    // Swap B and R channels in-place
    unsigned char temp = data[idx + 0];
    data[idx + 0] = data[idx + 2];
    data[idx + 2] = temp;
}
''', 'bgr_to_rgb_inplace')


def bgr_to_rgb_gpu(src: cp.ndarray, dst: cp.ndarray = None, stream: cp.cuda.Stream = None) -> cp.ndarray:
    """
    Convert BGR to RGB on GPU using optimized CUDA kernel.
    
    Args:
        src: Source BGR image as CuPy array
        dst: Destination array (if None, creates new array)
        stream: CUDA stream for async execution
        
    Returns:
        RGB image as CuPy array
    """
    height, width = src.shape[:2]
    
    if dst is None:
        dst = cp.empty_like(src)
    
    # Configure kernel launch parameters
    threads_per_block = (32, 32)
    blocks_per_grid = (
        (width + threads_per_block[0] - 1) // threads_per_block[0],
        (height + threads_per_block[1] - 1) // threads_per_block[1]
    )
    
    # Launch kernel
    if stream is not None:
        bgr_to_rgb_kernel(
            blocks_per_grid, threads_per_block,
            (src, dst, width, height),
            stream=stream
        )
    else:
        bgr_to_rgb_kernel(
            blocks_per_grid, threads_per_block,
            (src, dst, width, height)
        )
    
    return dst


def bgr_to_rgb_inplace_gpu(data: cp.ndarray, stream: cp.cuda.Stream = None):
    """
    Convert BGR to RGB in-place on GPU using optimized CUDA kernel.
    
    Args:
        data: BGR image as CuPy array (modified in-place)
        stream: CUDA stream for async execution
    """
    height, width = data.shape[:2]
    
    # Configure kernel launch parameters
    threads_per_block = (32, 32)
    blocks_per_grid = (
        (width + threads_per_block[0] - 1) // threads_per_block[0],
        (height + threads_per_block[1] - 1) // threads_per_block[1]
    )
    
    # Launch kernel
    if stream is not None:
        bgr_to_rgb_inplace_kernel(
            blocks_per_grid, threads_per_block,
            (data, width, height),
            stream=stream
        )
    else:
        bgr_to_rgb_inplace_kernel(
            blocks_per_grid, threads_per_block,
            (data, width, height)
        )