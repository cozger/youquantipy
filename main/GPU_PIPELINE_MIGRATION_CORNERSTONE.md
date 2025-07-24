# GPU Pipeline Migration Cornerstone Instructions

## ✅ Introduction

This document serves as the comprehensive guide for migrating the existing GPU-based video processing pipeline to the newly proposed streamlined and performant architecture. It extracts and documents every crucial implementation detail from the existing codebase that isn't fully described in the new architecture instructions, providing a complete reference for the migration process.

**Migration Goals:**

- Single CUDA context per GPU/camera (never per thread/frame)
- Global preallocation of GPU memory & TensorRT buffers
- Minimal CPU-GPU data transfers
- Lock-free ring buffer using atomic indices
- Clear separation: Capture → GPU pipeline → GUI/results

---

## 📌 Module-by-Module Detailed Analysis

### Module: `gpu_pipeline.py` (NEW)

**Purpose:** Central GPU pipeline module that owns the single CUDA context and manages all GPU operations.

#### Key Requirements from Current Implementation

1. **CUDA Context Management**  
   _From `gpu_face_processor.py`, `parallelworker_unified.py`_  
   - Context must be created ONCE at initialization  
   - Never push/pop contexts in processing loops  
   - Store context reference for cleanup  

2. **Memory Management**  
   _From `gpu_memory_manager.py`, `gpu_frame_cache.py`_  
   - Pre-allocate all GPU buffers at initialization  
   - Use memory pools with size limits (2GB default)  
   - Implement proper cleanup with `atexit` handlers  

3. **TensorRT Engine Management**  
   _From `gpu_face_processor.py`, `retinaface_detector_gpu.py`_  
   - Load engines once at startup  
   - Pre-allocate input/output buffers for max batch size  
   - Use new TensorRT API (8.5+) with proper tensor addressing  

#### Implementation Specifics to Preserve

```python
# CUDA context creation (ONCE per camera)
cuda.init()
self.cuda_device = cuda.Device(gpu_device_id)
self.cuda_context = self.cuda_device.make_context()

# TensorRT buffer allocation pattern
for i in range(engine.num_io_tensors):
    name = engine.get_tensor_name(i)
    shape = engine.get_tensor_shape(name)
    size = int(abs(np.prod(shape)) * 4)  # float32
    buffer = cuda.mem_alloc(size)

# Ring buffer with atomic indexing (NEW requirement)
self.write_index = 0  # Atomic counter
self.read_index = 0   # Atomic counter
self.buffer_size = 32  # Power of 2 for fast modulo
````

---

### Module: `camera_worker.py` (NEW)

**Purpose:** Process that owns the GPU pipeline, captures frames, and communicates via IPC.

#### Key Requirements from Current Implementation

1. **Frame Capture**
   *From `gpu_frame_distributor.py`*

   * Use pinned memory for zero-copy transfers
   * Detect actual camera dimensions at runtime
   * Pre-compute standard formats (640x640 detection, 960x540 preview)

2. **IPC Mechanisms**
   *From `sharedbuffer.py`, `gpu_memory_manager.py`*

   * Shared memory for frame data using NumPy arrays
   * Non-blocking queues with timeout handling

#### Critical Implementation Details

```python
# Pinned memory allocation
buffer_size = self.actual_height * self.actual_width * 3
self.pinned_buffer = cp.cuda.alloc_pinned_memory(buffer_size)
self.pinned_frame = np.frombuffer(
    self.pinned_buffer, 
    dtype=np.uint8, 
    count=buffer_size
).reshape((self.actual_height, self.actual_width, 3))

# Shared memory creation
self.shm = shared_memory.SharedMemory(
    create=True, 
    size=self.size * 4 + 8  # float32 data + timestamp
)
```

---

### Module: `gui.py` (existing, needs modification)

**Purpose:** Main GUI process that reads shared memory and displays results.

#### Key Changes Required

1. Remove all direct GPU operations
2. Read from shared memory buffers only
3. Maintain and reuse existing coordinate transformation logic

#### Coordinate Transformation Logic to Preserve

* Detection happens at 640x640 with padding
* Landmarks are in ROI space (256x256)
* Must transform through: ROI → Detection → Original → Display

```python
# From gpu_face_processor.py:638-656
def _transform_landmarks_to_frame(self, landmarks, transform):
    transformed = landmarks.copy()
    transformed[:, 0] = landmarks[:, 0] * transform['scale_x'] + transform['x1']
    transformed[:, 1] = landmarks[:, 1] * transform['scale_y'] + transform['y1']
    return transformed
```

---

## 🚩 Cross-Module Interactions & Communication

### Shared Memory Layout

1. **NumpySharedBuffer**

   * Size: 104 floats (52 blendshapes × 2 faces)
   * Layout: `[8 bytes timestamp][float32 array data]`
   * Zero-copy access via NumPy views

2. **GPU Memory IPC**

   * *Currently broken — must be replaced in full*
   * Deprecated usage of CUDA IPC handles and cross-process context

### Queue Management

```python
# Thread-safe queues (within GPU process)
gpu_face_result_queue = queue.Queue(maxsize=10)

# Multiprocess queues (between processes)
face_result_queue = MPQueue(maxsize=10)
preview_queue = MPQueue(maxsize=2)
```

### Threading Architecture

* Capture thread runs in `GPUFrameDistributor`
* GPU processing runs in same thread (shared CUDA context)
* MediaPipe runs in separate processes
* Result forwarder thread bridges GPU → multiprocessing queues

---

## ⚠️ Pending Clarifications & Recommended Decisions

### 1. Ring Buffer Implementation

**Issue:** New architecture specifies lock-free atomic indexing, current system uses queues.
**Recommendation:** Use `multiprocessing.Value` for atomic indices.

```python
self.write_idx = multiprocessing.Value('L', 0)
self.read_idx = multiprocessing.Value('L', 0)

with self.write_idx.get_lock():
    idx = self.write_idx.value
    self.write_idx.value = (idx + 1) % self.buffer_size
```

---

### 2. Exact Frame Buffer Count

**Recommendation:** Use a buffer size of 32 frames (power of 2) to optimize modulo operations and match queue depth.

---

### 3. Error Recovery Strategy

**Recommendation:**

* On camera disconnection: Retry every 5 seconds
* On GPU errors: Log and skip frame (do not crash pipeline)
* On IPC errors: Reset shared memory and reinitialize

---

## 🔄 IPC Redesign Requirements

### ❗ Mandatory Redesign

The current IPC system is broken and must be replaced.

> **Do NOT reuse** legacy GPU IPC based on CUDA handles or shared GPU contexts across processes.

### ✅ New IPC Architecture Design

#### Shared Memory for Data Exchange

Use `multiprocessing.shared_memory` for:

* Preview frames (BGR, 960x540)
* Landmark outputs (float32 arrays)
* Optional metadata (frame\_id, timestamp, face count)

Each shared buffer must be:

* Created with defined shape and size
* Closed and unlinked at shutdown
* Documented in this file

#### Multiprocessing Queues for Signaling

Use `multiprocessing.Queue` for:

* Frame metadata (e.g., frame ready, ID, timestamp)
* Heartbeats and control messages (e.g., stop, ping)

**Constraints:**

* All `put()` and `get()` calls must use timeouts (e.g., `put(..., timeout=0.01)`)
* Wrap all IPC operations in try/except
* No blocking behavior allowed

#### Shared Memory Buffer Layout

Each shared memory region must include:

* **Data region**: e.g., `[540, 960, 3]` for preview, `[N, 478, 3]` for landmarks
* **Metadata region** (after data):

  * `frame_id: int32`
  * `timestamp_ms: int64`
  * `num_faces: int32`
  * `frame_ready: int8`

```python
# Example layout
preview_size = 960 * 540 * 3
metadata_size = 64
shm = shared_memory.SharedMemory(create=True, size=preview_size + metadata_size)
```

✅ Memory must be read **only when** `frame_ready == 1`.
Writer must reset it after write.
Reader must reset it after read.

### ❌ Deprecated IPC Patterns (Forbidden)

* CUDA IPC handles between processes
* Manual context push/pop
* Unsafe `np.frombuffer(multiprocessing.Array(...))`
* Blocking `queue.get()` or `put()` without timeouts
* Raw CuPy/NVIDIA device memory sharing between processes

If found, these must be **deleted and rewritten**.

---

## 🧭 Queue Overflow and Backpressure Strategy

### ✅ Requirements

* Use size-limited queues (e.g., `maxsize=10`)
* If full:

  * **Option A (preferred):** Replace oldest queue entry
  * **Option B:** Drop oldest frame in ring buffer (not the new one)

### 🚨 Diagnostics

* Log every dropped/overwritten frame
* Log frame ID gaps
* Every 5 seconds: print stats

  * Dropped frames
  * Processed frames
  * Pipeline latency
  * Shared memory health (write/read index)

---

## 🧠 Developer Notes

Claude or any automated tool must:

* Adhere to all IPC constraints above
* Flag any deviation or ambiguity in the “Pending Clarifications” section
* Refuse to preserve broken IPC logic
* Pause for clarification if architecture gaps are encountered

---

## ⚙️ TensorRT Dynamic Batching

**Recommendation:**

* Min batch: `1`
* Optimal batch: `4`
* Max batch: `8`
* Use runtime profiles if supported

---

## 🟡 Critical Implementation Checklist

### CUDA Context Management

* [ ] CUDA Context created **once per GPU-process**
* [ ] Stored as instance variable; never pushed/popped in loops
* [ ] Proper cleanup at shutdown
* [ ] All operations confirm current context is active

### Memory Allocation

* [ ] All buffers allocated at initialization
* [ ] Pinned memory used for CPU→GPU transfers
* [ ] Memory pool limits enforced
* [ ] No dynamic allocations during inference loop

### TensorRT Integration

* [ ] Input/output dimensions match engine expectations
* [ ] RetinaFace: `[1, 608, 640, 3]` NHWC
* [ ] Landmarks: `[N, 256, 256, 3]` NHWC
* [ ] Dynamic batching supported via profiles
* [ ] New API: `set_tensor_address()` used

### Data Flow

* [ ] Detection frame padded/resized to 640x640
* [ ] ROI extraction with 15% padding
* [ ] All transforms preserve spatial alignment
* [ ] ROIs batched before landmark stage

### IPC Mechanisms

* [ ] Shared memory buffers pre-sized and documented
* [ ] Queue operations non-blocking
* [ ] Proper lifecycle management for shared memory
* [ ] Metadata flags managed correctly

### Synchronization

* [ ] CUDA streams used for all async ops
* [ ] `stream.synchronize()` used before CPU access
* [ ] Frame IDs are strictly monotonic and preserved
* [ ] Timestamps consistent across components

---

## ✅ Final Logic & Correctness Check Summary

### Verified Correct

1. **Coordinate Transforms:** ❌ **STILL BROKEN** - Bounding boxes and face contours offset to top-left with jitter
2. **Memory Management:** GPU and pinned memory preallocated correctly
3. **TensorRT Integration:** Dynamic batching and correct IO shapes used
4. **Thread Safety:** Single context per GPU, proper synchronization

### Known Issues to Avoid

1. **GPU IPC (legacy):** Completely broken — must not be reused
2. **Memory Leaks:** Ensure cleanup via `atexit` or destructor
3. **Dropped Frames:** New ring buffer must explicitly handle overflow
4. **Silent Errors:** All errors should bubble up for UI feedback

---

## 🚀 Migration Strategy (Phased)

1. **Phase 1:** Implement `gpu_pipeline.py` with clean architecture
2. **Phase 2:** Build `camera_worker.py` with shared memory output
3. **Phase 3:** Modify `gui.py` to read from shared memory buffers
4. **Phase 4:** Integrate error handling, IPC diagnostics, watchdogs
5. **Phase 5:** Add GPU IPC (if needed) with new context-safe logic

---

## 🎯 Critical Success Factors

1. Never create multiple CUDA contexts per camera
2. Always preallocate memory at startup
3. Frame IDs must propagate across all modules
4. Always synchronize GPU before host access
5. IPC must follow this spec exactly

