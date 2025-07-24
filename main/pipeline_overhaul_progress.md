# Pipeline Overhaul Progress Tracker

## Overview
This document tracks the complete overhaul of the YouQuantiPy processing pipeline, transitioning from a broken multi-queue architecture to a true GPU-first ring buffer design.

## Session Information
- **Start Date**: 2025-07-23
- **Current Status**: Preparation Phase
- **Key Goal**: Rebuild pipeline with proper CUDA context management and minimal CPU-GPU transfers

## Progress Checklist

### Phase 1: Preparation and Analysis
- [ ] Create progress tracking documentation
- [ ] Identify and list all Python files
- [ ] Move deprecated/test files to legacy folder
- [ ] Analyze current broken pipeline architecture
- [ ] Compare with proposed new architecture
- [ ] Create detailed implementation plan
- [ ] Identify recyclable vs deprecated modules
- [ ] Document critical migration gotchas

### Phase 2: Implementation Planning
- [ ] Design new gpu_pipeline.py module
- [ ] Design camera_worker.py module
- [ ] Plan GUI integration changes
- [ ] Map existing functionality to new modules
- [ ] Identify shared memory requirements
- [ ] Plan TensorRT integration approach

### Phase 3: Core Implementation
- [ ] Implement GPUPipeline class
- [ ] Implement camera worker process
- [ ] Create shared memory management
- [ ] Integrate with existing GUI
- [ ] Implement ring buffer system
- [ ] Add CUDA stream management

### Phase 4: Migration and Testing
- [ ] Test single camera pipeline
- [ ] Test multi-camera setup
- [ ] Migrate LSL integration
- [ ] Migrate recording functionality
- [ ] Performance benchmarking
- [ ] Remove deprecated code

### Phase 5: Optimization and Cleanup
- [ ] Fine-tune ring buffer sizes
- [ ] Optimize CUDA stream usage
- [ ] Clean up legacy code
- [ ] Update documentation
- [ ] Final performance validation

## Key Decisions Made

### Architecture Decisions
1. **One CUDA context per camera** - No IPC attempts
2. **Ring buffer design** - Pre-allocated GPU memory
3. **Shared memory for IPC** - Zero-copy between processes
4. **Simplified queue structure** - One result queue per camera
5. **GPU persistence** - Data stays on GPU throughout pipeline

### Technology Stack
- **GPU Framework**: CuPy + PyCUDA
- **IPC Method**: multiprocessing.shared_memory
- **Inference**: TensorRT
- **Ring Buffer**: Custom implementation
- **GUI Integration**: Minimal changes to existing Tkinter

## Critical Issues Identified

### Current Architecture Flaws
1. **CUDA Context Sharing**: Attempting impossible IPC of CUDA contexts
2. **Queue Explosion**: 7+ queues per camera causing deadlocks
3. **CPU-GPU Ping-Pong**: Constant memory transfers destroying performance
4. **ThreadPool Misuse**: No per-thread CUDA context initialization

### Migration Risks
1. **Shared Memory Limits**: OS-dependent size restrictions
2. **Process Cleanup**: Must ensure proper cleanup of GPU resources
3. **Backward Compatibility**: Some features may need reimplementation
4. **TensorRT Version**: Must match CUDA/cuDNN versions

## Module Status

### To Be Deprecated
- gpu_memory_manager.py (flawed IPC design)
- gpu_frame_cache.py (incorrect caching approach)
- batch_roi_processor.py (will be integrated into pipeline)
- frame_distributor.py (replaced by ring buffer)
- sharedbuffer.py (replaced by multiprocessing.shared_memory)

### To Be Recycled/Modified
- retinaface_detector_gpu.py → Integrate into gpu_pipeline.py
- roi_processor.py → Adapt GPU kernels for new pipeline
- canvasdrawing.py → Keep with minor modifications
- LSLHelper.py → Keep as-is
- gui.py → Modify worker launching only

### New Modules
- gpu_pipeline.py (core GPU processing)
- camera_worker.py (simplified worker process)
- ring_buffer.py (optional, if needed separately)

## Implementation Notes

### Critical Implementation Details
1. **Pinned Memory**: Use cuda.pagelocked_empty for zero-copy transfers
2. **Stream Synchronization**: Careful ordering of CUDA operations
3. **Memory Layout**: Ensure proper alignment for TensorRT
4. **Error Handling**: GPU operations need careful exception handling
5. **Resource Cleanup**: CUDA contexts must be properly destroyed

### Performance Targets
- Capture → Display latency: < 50ms
- Face detection: 10-15ms per frame
- Landmark extraction: 5-10ms per face
- Total pipeline: 25-35ms (28-40 FPS)

## Next Steps
1. Complete file inventory and deprecated file identification
2. Create legacy folder structure
3. Perform detailed pipeline analysis
4. Generate implementation plan document

---
*This document will be updated throughout the migration process*