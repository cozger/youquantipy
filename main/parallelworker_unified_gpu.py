# GPU-First Parallel Worker Implementation
# This replaces the multiprocess architecture with thread-based GPU pipeline

import time
import threading
import numpy as np
from multiprocessing import Process, Queue as MPQueue, Pipe
import queue
from sharedbuffer import NumpySharedBuffer
from tracker import UnifiedTracker

# Import new GPU components
from gpu_frame_distributor import GPUFrameDistributor
from gpu_face_processor import GPUFaceProcessor

# Import existing components that remain unchanged
from participantmanager_unified import GlobalParticipantManager
from confighandler import ConfigHandler

import warnings
import os
warnings.filterwarnings("ignore", category=UserWarning)

# Enable debugging for initial integration
DEBUG_GPU_PIPELINE = True


def pose_worker_process(frame_queue, result_queue, model_path: str, control_pipe):
    """Dedicated process for pose detection with MediaPipe - unchanged from original"""
    print("[POSE WORKER] Starting pose detection process")
    
    # Fix protobuf compatibility issue
    import os
    os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
    import sys

    # Remove any existing protobuf from modules
    modules_to_remove = []
    for module_name in sys.modules:
        if module_name.startswith('google.protobuf'):
            modules_to_remove.append(module_name)
    
    for module_name in modules_to_remove:
        del sys.modules[module_name]
    
    # Now import protobuf fresh
    try:
        import google.protobuf
        print(f"[POSE WORKER] Protobuf version: {google.protobuf.__version__}")
    except ImportError:
        print("[POSE WORKER] Protobuf not installed")
    
    # Try different MediaPipe import methods
    MEDIAPIPE_AVAILABLE = False
    pose_landmarker = None
    mp = None
    
    try:
        # Try the new import style first (MediaPipe >= 0.10.0)
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks import vision
        MEDIAPIPE_AVAILABLE = True
        print("[POSE WORKER] MediaPipe tasks API available")
    except ImportError as e:
        print(f"[POSE WORKER] MediaPipe tasks import failed: {e}")
        try:
            # Try basic MediaPipe import
            import mediapipe as mp
            # Check if we have the solutions API
            if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'pose'):
                MEDIAPIPE_AVAILABLE = True
                print("[POSE WORKER] Using MediaPipe solutions API (legacy mode)")
                python = None
                vision = None
            else:
                print("[POSE WORKER] MediaPipe solutions API not available")
        except ImportError as e2:
            print(f"[POSE WORKER] MediaPipe not installed: {e2}")
            MEDIAPIPE_AVAILABLE = False

    if not MEDIAPIPE_AVAILABLE:
        print("[POSE WORKER] ERROR: MediaPipe not properly installed")
        print("[POSE WORKER] Install with: pip install mediapipe==0.10.9 protobuf==3.20.3")
        print("[POSE WORKER] Running in fallback mode - no pose detection")
        
        # Fallback loop - just pass through empty results
        while True:
            if control_pipe.poll():
                msg = control_pipe.recv()
                if msg == 'stop':
                    break
            
            try:
                frame_data = frame_queue.get(timeout=0.1)
                result = {
                    'type': 'pose',
                    'data': [],
                    'timestamp': frame_data['timestamp']
                }
                result_queue.put_nowait(result)
            except:
                pass
        
        print("[POSE WORKER] Fallback worker stopped")
        return

    # Initialize pose detection based on available API
    if python and vision:
        # New API (MediaPipe >= 0.10.0)
        try:
            with open(model_path, 'rb') as f:
                model_buffer = f.read()
            base_opts = python.BaseOptions(model_asset_buffer=model_buffer)
            
            pose_options = vision.PoseLandmarkerOptions(
                base_options=base_opts,
                running_mode=vision.RunningMode.VIDEO,
                num_poses=5,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                output_segmentation_masks=False
            )
            pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)
            print("[POSE WORKER] Initialized with new MediaPipe API")
        except Exception as e:
            print(f"[POSE WORKER] Failed to initialize with new API: {e}")
            MEDIAPIPE_AVAILABLE = False
    else:
        # Legacy API (MediaPipe < 0.10.0)
        try:
            # Import with protobuf fix
            import mediapipe as mp
            mp_pose = mp.solutions.pose
            
            # Try to create pose instance with error handling
            try:
                pose_landmarker = mp_pose.Pose(
                    static_image_mode=False,
                    model_complexity=1,
                    enable_segmentation=False,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                print("[POSE WORKER] Initialized with legacy MediaPipe API")
            except Exception as init_error:
                print(f"[POSE WORKER] Pose initialization error: {init_error}")
                # Try simpler configuration
                pose_landmarker = mp_pose.Pose(
                    static_image_mode=False,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                print("[POSE WORKER] Initialized with minimal configuration")
                
        except Exception as e:
            print(f"[POSE WORKER] Failed to initialize with legacy API: {e}")
            import traceback
            traceback.print_exc()
            MEDIAPIPE_AVAILABLE = False

    if not MEDIAPIPE_AVAILABLE or pose_landmarker is None:
        print("[POSE WORKER] Failed to initialize pose detection, running in fallback mode")
        while True:
            if control_pipe.poll():
                msg = control_pipe.recv()
                if msg == 'stop':
                    break
            
            try:
                frame_data = frame_queue.get(timeout=0.1)
                result = {
                    'type': 'pose',
                    'data': [],
                    'timestamp': frame_data['timestamp']
                }
                result_queue.put_nowait(result)
            except:
                pass
        
        return

    # Monotonic timestamp generator
    class MonotonicTS:
        def __init__(self):
            self.ts = int(time.monotonic() * 1000)
            self.lock = threading.Lock()
        def next(self):
            with self.lock:
                now = int(time.monotonic() * 1000)
                if now <= self.ts:
                    self.ts += 1
                else:
                    self.ts = now
                return self.ts
    
    ts_gen = MonotonicTS()
    
    # Performance monitoring
    frame_count = 0
    detection_count = 0
    last_fps_report = time.time()
    
    # Main processing loop
    while True:
        # Check for control messages
        if control_pipe.poll():
            msg = control_pipe.recv()
            if msg == 'stop':
                break
            elif msg == 'get_stats':
                elapsed = time.time() - last_fps_report
                if elapsed > 0:
                    stats = {
                        'type': 'pose_stats',
                        'fps': frame_count / elapsed,
                        'detection_rate': detection_count / frame_count if frame_count > 0 else 0
                    }
                    control_pipe.send(stats)
        
        # Get frame
        try:
            frame_data = frame_queue.get(timeout=0.1)
            if frame_count == 0:
                print(f"[POSE WORKER] Successfully got first frame")
        except Exception as e:
            if frame_count == 0:
                print(f"[POSE WORKER] Error getting frame: {type(e).__name__}: {e}")
            continue
        
        rgb = frame_data['frame']  # GPU distributor sends 'frame' not 'rgb'
        timestamp = frame_data['timestamp']
        
        # Process based on API version
        pose_data = []
        
        try:
            if python and vision:
                # New API processing
                timestamp_ms = ts_gen.next()
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                pose_result = pose_landmarker.detect_for_video(mp_image, timestamp_ms)
                
                if pose_result.pose_landmarks:
                    for pose_landmarks in pose_result.pose_landmarks:
                        # Calculate centroid from key points
                        key_indices = [11, 12, 23, 24]  # shoulders and hips
                        key_points = [pose_landmarks[i] for i in key_indices if i < len(pose_landmarks)]
                        if key_points:
                            x_coords = [p.x for p in key_points]
                            y_coords = [p.y for p in key_points]
                            pose_centroid = (np.mean(x_coords), np.mean(y_coords))
                        else:
                            pose_centroid = (0.5, 0.5)
                        
                        # Create pose values array
                        pose_vals = []
                        for lm in pose_landmarks:
                            pose_vals.extend([lm.x, lm.y, lm.z, lm.visibility])
                        
                        pose_data.append({
                            'landmarks': [(lm.x, lm.y, lm.z) for lm in pose_landmarks],
                            'centroid': pose_centroid,
                            'values': pose_vals,
                            'timestamp': timestamp
                        })
            else:
                # Legacy API processing with error handling
                # Ensure frame is uint8
                if rgb.dtype != np.uint8:
                    rgb_uint8 = (rgb * 255).astype(np.uint8) if rgb.max() <= 1.0 else rgb.astype(np.uint8)
                else:
                    rgb_uint8 = rgb
                    
                results = pose_landmarker.process(rgb_uint8)
                
                if results and results.pose_landmarks:
                    # Legacy API returns single pose
                    pose_landmarks = results.pose_landmarks
                    
                    # Calculate centroid
                    key_indices = [11, 12, 23, 24]  # shoulders and hips
                    key_points = [pose_landmarks.landmark[i] for i in key_indices if i < len(pose_landmarks.landmark)]
                    if key_points:
                        x_coords = [p.x for p in key_points]
                        y_coords = [p.y for p in key_points]
                        pose_centroid = (np.mean(x_coords), np.mean(y_coords))
                    else:
                        pose_centroid = (0.5, 0.5)
                    
                    # Create pose values array
                    pose_vals = []
                    for lm in pose_landmarks.landmark:
                        pose_vals.extend([lm.x, lm.y, lm.z, lm.visibility if hasattr(lm, 'visibility') else 1.0])
                    
                    pose_data.append({
                        'landmarks': [(lm.x, lm.y, lm.z) for lm in pose_landmarks.landmark],
                        'centroid': pose_centroid,
                        'values': pose_vals,
                        'timestamp': timestamp
                    })
                    
        except Exception as e:
            if frame_count % 30 == 0:
                print(f"[POSE WORKER] Processing error: {e}")
        
        if pose_data:
            detection_count += len(pose_data)
        
        # Send result
        result = {
            'type': 'pose',
            'data': pose_data,
            'timestamp': timestamp,
            'frame_id': frame_data.get('frame_id', -1)
        }
        
        try:
            result_queue.put_nowait(result)
        except:
            pass
        
        frame_count += 1
        
        # Report FPS periodically
        if time.time() - last_fps_report > 5.0:
            elapsed = time.time() - last_fps_report
            fps = frame_count / elapsed
            det_rate = detection_count / frame_count if frame_count > 0 else 0
            print(f"[POSE WORKER] FPS: {fps:.1f}, Detection rate: {det_rate:.2%}")
            last_fps_report = time.time()
            frame_count = 0
            detection_count = 0
    
    # Cleanup
    if pose_landmarker:
        if hasattr(pose_landmarker, 'close'):
            pose_landmarker.close()
        del pose_landmarker
    
    print("[POSE WORKER] Stopped")


def fusion_process_gpu(face_result_queue, pose_result_queue, preview_frame_queue,
                      preview_queue, score_buffer,
                      result_pipe, recording_queue, lsl_queue, participant_update_queue,
                      worker_pipe, correlation_queue, cam_idx, enable_pose, resolution, max_participants):
    """
    Simplified fusion process for GPU pipeline.
    Receives face results from GPU pipeline and pose results from CPU process.
    """
    print(f"[FUSION GPU] Starting fusion for camera {cam_idx}")
    
    # Initialize participant manager
    participant_manager = GlobalParticipantManager(max_participants=max_participants)
    
    # Tracking state
    face_data = {}
    pose_data = {}
    latest_preview_frame = None
    frame_count = 0
    
    while True:
        try:
            # Check for control messages
            if worker_pipe and worker_pipe.poll():
                msg = worker_pipe.recv()
                if msg == 'stop':
                    break
            
            # Get latest preview frame
            try:
                frame_data = preview_frame_queue.get(timeout=0.001)
                latest_preview_frame = frame_data
            except queue.Empty:
                pass
            
            # Get face results from GPU pipeline
            try:
                face_result = face_result_queue.get(timeout=0.001)
                if face_result and face_result['type'] == 'face_data':
                    frame_id = face_result['frame_id']
                    face_data[frame_id] = face_result
                    
                    # Clean old face data
                    if len(face_data) > 10:
                        oldest_ids = sorted(face_data.keys())[:-10]
                        for old_id in oldest_ids:
                            del face_data[old_id]
            except queue.Empty:
                pass
            
            # Get pose results if enabled
            if enable_pose and pose_result_queue:
                try:
                    pose_result = pose_result_queue.get(timeout=0.001)
                    if pose_result:
                        frame_id = pose_result.get('frame_id', -1)
                        pose_data[frame_id] = pose_result
                        
                        # Clean old pose data
                        if len(pose_data) > 10:
                            oldest_ids = sorted(pose_data.keys())[:-10]
                            for old_id in oldest_ids:
                                del pose_data[old_id]
                except queue.Empty:
                    pass
            
            # Process and send combined results
            if latest_preview_frame:
                frame_id = latest_preview_frame['frame_id']
                timestamp = latest_preview_frame['timestamp']
                
                # Prepare combined result
                combined_result = {
                    'camera_index': cam_idx,
                    'frame_id': frame_id,
                    'timestamp': timestamp,
                    'faces': []
                }
                
                # Add face data
                if frame_id in face_data:
                    combined_result['faces'] = face_data[frame_id]['faces']
                
                # Add pose data if available
                if enable_pose and frame_id in pose_data:
                    combined_result['pose'] = pose_data[frame_id]
                
                # Send to preview queue with frame
                if not preview_queue.full() and latest_preview_frame:
                    preview_data = {
                        'frame': latest_preview_frame['frame'],
                        'data': combined_result
                    }
                    preview_queue.put(preview_data)
                
                # Send to LSL queue
                if not lsl_queue.full():
                    lsl_queue.put(combined_result)
                
                # Send to recording queue if needed
                if not recording_queue.full() and latest_preview_frame:
                    recording_queue.put({
                        'frame': latest_preview_frame['frame'],
                        'timestamp': timestamp,
                        'camera_index': cam_idx
                    })
                
                frame_count += 1
                
                if frame_count % 150 == 0:
                    print(f"[FUSION GPU] Camera {cam_idx}: Processed {frame_count} frames")
            
            time.sleep(0.001)  # Small sleep to prevent CPU spinning
            
        except Exception as e:
            print(f"[FUSION GPU] Error: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"[FUSION GPU] Stopped for camera {cam_idx}")


def parallel_participant_worker_gpu(cam_idx, face_model_path, pose_model_path,
                                   fps, enable_mesh, enable_pose,
                                   preview_queue, score_buffer_name, result_pipe,
                                   recording_queue, lsl_queue,
                                   participant_update_queue,
                                   worker_pipe, correlation_queue,
                                   max_participants,
                                   resolution,
                                   retinaface_model_path=None,
                                   arcface_model_path=None,
                                   enable_recognition=False,
                                   enable_gpu_ipc=False,
                                   gpu_device_id=0):
    """
    GPU-First parallel participant worker.
    Main changes:
    1. Uses GPUFrameDistributor instead of FrameDistributor
    2. Face processing happens in thread with GPUFaceProcessor
    3. Minimal CPU-GPU transfers
    """
    print(f"\n{'='*60}")
    print(f"[GPU WORKER] Camera {cam_idx} starting - GPU-First Architecture")
    print(f"[GPU WORKER] Resolution: {resolution}, FPS: {fps}")
    print(f"{'='*60}\n")
    
    # Get configuration
    config = ConfigHandler()
    
    # Get model paths
    retinaface_onnx_path = config.get('advanced_detection.retinaface_model', retinaface_model_path)
    retinaface_trt_path = '/mnt/d/Projects/youquantipy/retinaface.trt'
    landmark_trt_path = config.get('advanced_detection.landmark_trt_path',
                                  '/mnt/d/Projects/youquantipy/landmark.trt')
    
    # Check if TRT engines exist
    if not os.path.exists(retinaface_trt_path) or not os.path.exists(landmark_trt_path):
        print(f"[GPU WORKER] ERROR: TensorRT engines not found")
        print(f"[GPU WORKER] RetinaFace TRT: {retinaface_trt_path} - {'EXISTS' if os.path.exists(retinaface_trt_path) else 'MISSING'}")
        print(f"[GPU WORKER] Landmark TRT: {landmark_trt_path} - {'EXISTS' if os.path.exists(landmark_trt_path) else 'MISSING'}")
        print(f"[GPU WORKER] Please run modelconversion.py to create TRT engines from ONNX models")
        raise RuntimeError("TensorRT engines not found")
    
    # Initialize shared buffer
    score_buffer = NumpySharedBuffer(name=score_buffer_name)
    
    # Create queues
    preview_frame_queue = MPQueue(maxsize=2)  # For preview frames
    gpu_face_result_queue = queue.Queue(maxsize=10)  # Thread-safe queue from GPU pipeline
    face_result_queue = MPQueue(maxsize=10)   # Multiprocess queue to fusion process
    pose_frame_queue = MPQueue(maxsize=2) if enable_pose else None
    pose_result_queue = MPQueue(maxsize=2) if enable_pose else None
    
    # Initialize GPU components
    try:
        # Create GPU frame distributor
        gpu_distributor = GPUFrameDistributor(
            camera_index=cam_idx,
            resolution=resolution,
            fps=fps,
            gpu_device=gpu_device_id
        )
        
        # Create GPU face processor
        gpu_face_processor = GPUFaceProcessor(
            retinaface_engine=retinaface_trt_path,
            landmark_engine=landmark_trt_path,
            max_faces=max_participants,
            confidence_threshold=config.get('advanced_detection.detection_confidence', 0.9)
        )
        
        # Add GPU stream for face processing
        gpu_distributor.add_gpu_stream(
            name='face',
            gpu_processor=gpu_face_processor.process_frame,
            output_queue=gpu_face_result_queue
        )
        
        # Add CPU stream for preview
        gpu_distributor.add_cpu_stream(
            name='preview',
            output_queue=preview_frame_queue,
            resolution=(960, 540),
            format='bgr'
        )
        
        # Add CPU stream for pose if enabled
        if enable_pose:
            gpu_distributor.add_cpu_stream(
                name='pose',
                output_queue=pose_frame_queue,
                resolution=(640, 480),
                format='rgb'
            )
        
        # Start GPU pipeline
        gpu_distributor.start()
        print(f"[GPU WORKER] GPU pipeline started for camera {cam_idx}")
        
    except Exception as e:
        print(f"[GPU WORKER] Failed to initialize GPU pipeline: {e}")
        raise
    
    # Create result forwarder thread to bridge GPU thread results to multiprocess queues
    stop_forwarder = threading.Event()
    
    def forward_gpu_results():
        """Forward results from GPU pipeline to multiprocess queues"""
        while not stop_forwarder.is_set():
            try:
                # Get face results from GPU pipeline
                result = gpu_face_result_queue.get(timeout=0.1)
                if result:
                    # Format for fusion process
                    formatted_result = {
                        'type': 'face_data',
                        'camera_index': cam_idx,
                        'frame_id': result['frame_id'],
                        'timestamp': result['timestamp'],
                        'faces': result['faces'],
                        'processing_time_ms': result.get('processing_time_ms', 0)
                    }
                    
                    # Send to fusion process
                    face_result_queue.put(formatted_result)
                    
                    if DEBUG_GPU_PIPELINE and result['frame_id'] % 30 == 0:
                        print(f"[GPU WORKER] Forwarded {len(result['faces'])} faces for frame {result['frame_id']}")
                        
            except queue.Empty:
                continue
            except Exception as e:
                if 'stop' not in str(e):
                    print(f"[GPU WORKER] Error forwarding results: {e}")
    
    # Start result forwarder thread
    result_forwarder = threading.Thread(
        target=forward_gpu_results,
        name=f"GPUResultForwarder-{cam_idx}",
        daemon=True
    )
    result_forwarder.start()
    
    # Create control pipes for pose
    pose_control_parent, pose_control_child = Pipe() if enable_pose else (None, None)
    
    # Start pose worker process if enabled (remains unchanged)
    pose_proc = None
    if enable_pose:
        pose_proc = Process(
            target=pose_worker_process,
            args=(pose_frame_queue, pose_result_queue, pose_model_path,
                  pose_control_child)
        )
        pose_proc.start()
        print(f"[GPU WORKER] Pose process started for camera {cam_idx}")
    
    # Start fusion process
    fusion_proc = Process(
        target=fusion_process_gpu,
        args=(face_result_queue, pose_result_queue, preview_frame_queue,
              preview_queue, score_buffer,
              result_pipe, recording_queue, lsl_queue, participant_update_queue,
              worker_pipe, correlation_queue, cam_idx, enable_pose, resolution, max_participants)
    )
    fusion_proc.start()
    print(f"[GPU WORKER] Fusion process started for camera {cam_idx}")
    
    # Monitor loop
    last_stats_time = time.time()
    stats_interval = 5.0
    
    try:
        while True:
            # Check for control messages
            if result_pipe.poll():
                msg = result_pipe.recv()
                if msg == 'stop':
                    break
                elif isinstance(msg, tuple) and msg[0] == 'set_mesh':
                    # Handle mesh enable/disable
                    enable_mesh = msg[1]
                    print(f"[GPU WORKER] Mesh {'enabled' if enable_mesh else 'disabled'} for camera {cam_idx}")
            
            # Print statistics periodically
            current_time = time.time()
            if current_time - last_stats_time > stats_interval:
                stats = gpu_distributor.stats
                if stats['frames_captured'] > 0:
                    avg_transfer = np.mean(stats['gpu_transfer_ms'][-100:]) if stats['gpu_transfer_ms'] else 0
                    avg_process = np.mean(stats['gpu_processing_ms'][-100:]) if stats['gpu_processing_ms'] else 0
                    
                    print(f"\n[GPU WORKER] Camera {cam_idx} Performance:")
                    print(f"  Frames: {stats['frames_captured']}")
                    print(f"  GPU Transfer: {avg_transfer:.2f}ms")
                    print(f"  GPU Processing: {avg_process:.2f}ms")
                    print(f"  Queue drops: {stats['drops']}")
                
                last_stats_time = current_time
            
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        print(f"[GPU WORKER] Interrupted for camera {cam_idx}")
    except Exception as e:
        print(f"[GPU WORKER] Error in monitor loop: {e}")
        import traceback
        traceback.print_exc()
    
    # Cleanup
    print(f"[GPU WORKER] Stopping camera {cam_idx}...")
    
    # Stop result forwarder
    stop_forwarder.set()
    
    # Stop GPU pipeline
    gpu_distributor.stop()
    
    # Stop pose process
    if enable_pose and pose_control_parent:
        pose_control_parent.send('stop')
        if pose_proc:
            pose_proc.join(timeout=2.0)
    
    # Stop fusion
    if worker_pipe:
        try:
            worker_pipe.send('stop')
        except:
            pass
    fusion_proc.terminate()
    fusion_proc.join(timeout=2.0)
    
    print(f"[GPU WORKER] Camera {cam_idx} stopped")


# Export the GPU version as the main worker
parallel_participant_worker = parallel_participant_worker_gpu