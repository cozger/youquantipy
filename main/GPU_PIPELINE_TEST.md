# GPU Pipeline Testing Guide

## Prerequisites

1. **Verify TensorRT Engines Exist**
   ```bash
   ls -la *.trt
   # Should see: retinaface.trt, landmark.trt, blendshape.trt
   ```

2. **Verify GPU and CUDA**
   ```bash
   nvidia-smi
   # Should show your GPU and CUDA version
   ```

3. **Required Python Packages**
   - cupy
   - tensorrt
   - pycuda
   - opencv-python
   - numpy

## Test Steps

### 1. Single Camera Test

Start the GUI with single camera:
```bash
python gui.py
```

Expected behavior:
- GUI should start normally
- Select single camera mode
- Click "Start Processing"
- Should see "[GPU WORKER] GPU pipeline started"
- Should see face detection working

### 2. Performance Monitoring

Look for these log messages:
```
[GPU Distributor] Performance Report:
  Actual FPS: 30.0
  GPU Transfer: <3ms (should be under 3ms)
  GPU Processing: <20ms (should be under 20ms)
  Total GPU time: <25ms

[GPU Face Processor] Performance Report:
  detection: ~5-10ms
  roi_extraction: ~2-5ms
  landmarks: ~5-10ms
  total: ~15-25ms
```

### 3. Memory Usage

Monitor GPU memory:
```bash
watch -n1 nvidia-smi
```

Expected:
- GPU memory usage should be stable (~500-800MB)
- No memory leaks (increasing usage over time)

### 4. Debug Mode

The pipeline runs with `DEBUG_GPU_PIPELINE = True` by default. You should see:
- Frame forwarding messages every 30 frames
- Performance reports every 5 seconds

### 5. Multi-Camera Test

If testing with multiple cameras:
1. Set camera count to 2+ in GUI
2. Each camera should have its own GPU pipeline
3. Performance should scale linearly

## Troubleshooting

### Issue: "TensorRT engines not found"
**Solution**: The TRT engines must exist. Check paths in config:
- retinaface.trt
- landmark.trt  
- blendshape.trt

### Issue: "No CUDA devices available"
**Solution**: 
- Check nvidia-smi works
- Verify CUDA installation
- Check GPU is not in use by other processes

### Issue: Low FPS or high latency
**Check**:
1. GPU transfer time (should be <3ms)
2. GPU processing time (should be <25ms)
3. Queue drops in stats
4. Camera actual FPS vs requested

### Issue: Import errors
**Solution**: Install required packages:
```bash
pip install cupy-cuda11x  # Use appropriate CUDA version
pip install tensorrt pycuda
pip install opencv-python numpy
```

## Validation Checklist

- [ ] GUI starts without errors
- [ ] Face detection works
- [ ] FPS meets target (within 80% of requested)
- [ ] GPU transfer time <3ms
- [ ] Total GPU processing <25ms
- [ ] No memory leaks over 5 minutes
- [ ] Pose detection works (if enabled)
- [ ] LSL streams output data
- [ ] Recording works (if enabled)

## Performance Expectations

### Before (CPU-GPU transfers)
- 10-15 transfers per frame
- 50-100ms total latency
- Heavy PCIe bandwidth usage

### After (GPU Pipeline)
- 1 CPU→GPU transfer
- 1 GPU→CPU transfer  
- 15-25ms total latency
- Minimal PCIe usage

## Next Steps

If all tests pass:
1. Test with your specific use case
2. Tune batch sizes if needed
3. Adjust detection confidence
4. Monitor long-term stability