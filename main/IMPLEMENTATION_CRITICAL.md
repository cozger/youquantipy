# CRITICAL IMPLEMENTATION NOTES

## Architecture Changes

### What's Being Replaced
1. **frame_distributor.py** - Replaced with GPU-first version
2. **parallelworker_unified.py** - Face processing moved to GPU thread
3. **Multiple process queues** - Reduced to simple result queues
4. **CPU-based face pipeline** - Entirely on GPU now

### What Stays the Same
1. **GUI components** - Minimal changes needed
2. **Pose processing** - Still separate process with MediaPipe
3. **LSL streaming** - Consumes results as before
4. **Recording functionality** - Works with preview frames

## GPU Memory Management

### Critical Rules
1. **NEVER transfer unnecessarily** - Once on GPU, stay on GPU
2. **Pre-allocate everything** - No runtime allocation
3. **Use pinned memory** - 2-3x faster transfers
4. **Batch operations** - Process multiple faces together

### Memory Limits
- GPU memory usage: ~500MB for buffers
- Pinned memory: ~6MB per camera
- TensorRT engines: ~200MB each

## Thread vs Process

### GPU Components (Threads)
- Frame distributor
- Face detection
- Landmark extraction
- Must share CUDA context

### CPU Components (Processes)
- Pose detection
- GUI rendering
- LSL streaming
- File recording

## Performance Expectations

### Before (Current System)
- 10-15 CPU↔GPU transfers per frame
- ~50-100ms total latency
- Heavy PCIe bandwidth usage

### After (GPU Pipeline)
- 1 CPU→GPU transfer (capture)
- 1 GPU→CPU transfer (results)
- ~15-25ms total latency
- Minimal PCIe usage

## Common Pitfalls

1. **CUDA Context Issues**
   - Symptom: "invalid context" errors
   - Fix: Ensure GPU operations in same thread

2. **Memory Leaks**
   - Symptom: Growing GPU memory
   - Fix: Pre-allocate, reuse buffers

3. **Queue Deadlocks**
   - Symptom: Pipeline freezes
   - Fix: Use timeouts, non-blocking puts

4. **TensorRT Batch Size**
   - Symptom: Shape mismatch errors
   - Fix: Set context shape before inference

## Integration Checklist

- [ ] Replace frame distributor initialization
- [ ] Remove multiprocess face workers
- [ ] Connect GPU pipeline outputs
- [ ] Update GUI to expect new data format
- [ ] Test with single camera first
- [ ] Verify no CPU transfers in main loop
- [ ] Monitor GPU memory usage
- [ ] Check thread safety

## Debugging

### GPU Profiling
```python
# Add to GPU pipeline
import nvtx

with nvtx.annotate("Detection", color="red"):
    detections = self._run_detection(...)

# Monitor GPU memory
mempool = cp.get_default_memory_pool()
print(f"GPU memory: {mempool.used_bytes() / 1024**2:.1f} MB")