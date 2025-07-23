# Deprecated Modules with GPU Pipeline

The following modules and functions are deprecated with the new GPU-first architecture:

## Replaced Modules

1. **frame_distributor.py** → **gpu_frame_distributor.py**
   - Old: Multi-process frame distribution with CPU memory
   - New: Thread-based distribution with GPU memory and pinned buffers

2. **Face Detection Process** (in parallelworker_unified_original.py)
   - Functions: `face_detection_process()`, `face_worker_process()`
   - Replaced by: GPU face processor running in thread

3. **Face Landmark Process** (in parallelworker_unified_original.py)
   - Function: `face_landmark_process()`
   - Replaced by: Integrated into GPU face processor

4. **ROI Processing** (separate process)
   - Now integrated into GPU face processor

5. **Multiple Worker Processes**
   - Old: Separate processes for detection, ROI, landmarks
   - New: Single GPU processing thread

## Deprecated Imports in parallelworker_unified.py

```python
# No longer needed:
from frame_distributor import FrameDistributor
from retinaface_detector_gpu import RetinaFaceDetectorGPU
from lightweight_tracker import LightweightTracker
from roi_processor import ROIProcessor
from face_recognition_process import FaceRecognitionProcess
from enrollment_manager import EnrollmentManager
from gpu_memory_manager import GPUMemoryHandle, GPUMemoryClient
```

## Deprecated Queues

- face_frame_queue (internal)
- detection_queue
- detection_result_queue
- roi_queue
- landmark_result_queue

## What Remains

1. **pose_worker_process** - Still uses MediaPipe on CPU
2. **LSL streaming** - Unchanged
3. **Recording** - Unchanged
4. **GUI components** - Minimal changes
5. **Participant management** - Unchanged

## Migration Notes

- The main entry point `parallel_participant_worker` now redirects to `parallel_participant_worker_gpu`
- All face processing happens on GPU in a single thread
- Only final results cross process boundaries
- Preview and pose frames still transfer to CPU but use optimized paths