# frame_server.py - Final fixed version
"""
Frame server for efficient multi-process camera capture with zero-copy shared memory.
Fixed for Windows multiprocessing compatibility with proper synchronization.
"""

import time
import struct
import os
import sys
from typing import Tuple, Optional, Dict, List
import atexit
import signal
import traceback
import threading

# Standard library imports that are safe at module level
import multiprocessing
from multiprocessing import Process, Queue, Value, Array

# NOTE: OpenCV and numpy imports happen inside processes to avoid pickling issues


def robust_initialize_camera_standalone(camera_index, fps, resolution, settle_time=1.0):
    """Standalone camera initialization for the server process"""
    import cv2  # Import inside function
    
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    cap = None

    for b in backends:
        test_cap = cv2.VideoCapture(camera_index, b)
        if test_cap.isOpened():
            ret, _ = test_cap.read()
            if ret:
                cap = test_cap
                print(f"[FrameServer] Using backend: {b}")
                break
            else:
                test_cap.release()
    
    if cap is None:
        raise RuntimeError(f"Cannot open camera {camera_index}")

    print(f"[FrameServer] Initializing camera {camera_index} for {resolution[0]}x{resolution[1]} @ {fps} FPS...")

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
    cap.set(cv2.CAP_PROP_FPS, fps)

    # Warm-up with empty frame handling
    warmup_start = time.time()
    valid_frame_count = 0
    
    while time.time() - warmup_start < settle_time:
        ret, frame = cap.read()
        if ret and frame is not None and frame.size > 0:
            valid_frame_count += 1
    
    print(f"[FrameServer] Camera warmup complete, {valid_frame_count} valid frames during warmup")

    return cap


def frame_server_process_main(camera_index: int, fps: int, width: int, height: int,
                             status_queue: multiprocessing.Queue, stop_flag: multiprocessing.Value,
                             ring_size: int = 3):
    """
    Main function for frame server process.
    All imports happen inside to avoid Windows multiprocessing pickling issues.
    """
    # Import everything inside the process
    import cv2
    import numpy as np
    from multiprocessing import shared_memory
    
    process_name = f"FrameServer-Cam{camera_index}"
    
    try:
        # Set process name for debugging
        multiprocessing.current_process().name = process_name
        
        print(f"[{process_name}] Starting for camera {camera_index}")
        print(f"[{process_name}] Resolution: {width}x{height}, FPS: {fps}")
        
        # Initialize state
        resolutions = {}
        shm_segments = {}
        capture_resolution = (width, height)
        
        # Send status
        status_queue.put(('status', 'Cleaning up existing shared memory...'))
        
        # Cleanup existing shared memory
        shm_prefix = f"yqp_cam{camera_index}_"
        possible_resolutions = ['native', 'high', 'medium', 'low', 'tiny', 'detection',
                               '720p', '480p', '360p', '640x640', 'reslist']
        
        for res in possible_resolutions:
            for suffix in ['frames', 'meta', '']:
                try:
                    name = shm_prefix + (f"{res}_{suffix}" if suffix else res)
                    shm = shared_memory.SharedMemory(name=name)
                    shm.close()
                    shm.unlink()
                except:
                    pass
        
        # Send status
        status_queue.put(('status', 'Initializing camera...'))
        
        # Initialize camera
        cap = robust_initialize_camera_standalone(camera_index, fps, capture_resolution)
        
        # Get actual resolution from camera
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"[{process_name}] Camera opened with resolution: {actual_w}x{actual_h}")
        
        # Send status
        status_queue.put(('status', 'Determining resolutions...'))
        
        # Determine resolutions
        resolutions['native'] = (actual_h, actual_w, 3)
        resolutions['detection'] = (640, 640, 3)
        resolutions['640x640'] = (640, 640, 3)  # Alias
        
        # Calculate other resolutions
        native_pixels = actual_w * actual_h
        resolution_tiers = [
            ('high', 0.5),      # ~50% of native pixels
            ('medium', 0.25),   # ~25% of native pixels  
            ('low', 0.11),      # ~11% of native pixels
            ('tiny', 0.04)      # ~4% of native pixels
        ]
        
        for tier_name, pixel_ratio in resolution_tiers:
            target_pixels = native_pixels * pixel_ratio
            aspect_ratio = actual_w / actual_h
            target_height = int(np.sqrt(target_pixels / aspect_ratio))
            target_width = int(target_height * aspect_ratio)
            
            # Round to nearest multiple of 8
            target_height = max(240, (target_height // 8) * 8)
            target_width = max(320, (target_width // 8) * 8)
            
            if target_height < actual_h and target_width < actual_w:
                resolutions[tier_name] = (target_height, target_width, 3)
        
        # Add compatibility resolutions
        compatibility_map = {
            '720p': (720, 1280, 3),
            '480p': (480, 640, 3),
            '360p': (360, 480, 3)
        }
        
        for compat_name, (h, w, c) in compatibility_map.items():
            if h <= actual_h and w <= actual_w:
                resolutions[compat_name] = (h, w, c)
        
        print(f"[{process_name}] Available resolutions:")
        for name, (h, w, c) in resolutions.items():
            print(f"  {name}: {w}x{h}")
        
        # Send status
        status_queue.put(('status', 'Creating shared memory buffers...'))
        
        # Create resolution list shared memory
        res_list = ','.join(resolutions.keys()) + f"|{actual_w}x{actual_h}"
        res_list_data = res_list.encode('utf-8')
        res_list_size = max(len(res_list_data) + 100, 1024)
        
        res_shm_name = shm_prefix + "reslist"
        res_shm = shared_memory.SharedMemory(name=res_shm_name, create=True, size=res_list_size)
        res_shm.buf[:len(res_list_data)] = res_list_data
        res_shm.buf[len(res_list_data)] = 0  # Null terminator
        
        print(f"[{process_name}] Created resolution list shared memory: {res_shm_name}")
        
        # Create buffers for each resolution
        for res_name, shape in resolutions.items():
            h, w, c = shape
            frame_size = h * w * c
            total_size = frame_size * ring_size
            
            # Frame buffer
            frame_shm_name = shm_prefix + f"{res_name}_frames"
            frame_shm = shared_memory.SharedMemory(name=frame_shm_name, create=True, size=total_size)
            
            # Initialize with zeros
            frame_array = np.ndarray((total_size,), dtype=np.uint8, buffer=frame_shm.buf)
            frame_array[:] = 0
            
            # Metadata buffer
            meta_size = 4 + (8 * ring_size) + (4 * ring_size) + 12 + ring_size + 4 + 4
            print(f"[{process_name}] Calculated meta_size: {meta_size} bytes (ring_size={ring_size})")
            meta_shm_name = shm_prefix + f"{res_name}_meta"
            meta_shm = shared_memory.SharedMemory(name=meta_shm_name, create=True, size=meta_size)
            
            # Debug: print layout
            print(f"[{process_name}] Metadata layout for {res_name}:")
            print(f"  Total size: {meta_size}")
            print(f"  Ring index: 0-3")
            print(f"  Timestamps: 4-{4 + 8*ring_size - 1}")
            print(f"  Frame IDs: {4 + 8*ring_size}-{4 + 8*ring_size + 4*ring_size - 1}")
            print(f"  Shape: {4 + 8*ring_size + 4*ring_size}-{4 + 8*ring_size + 4*ring_size + 11}")
            print(f"  Validity: {4 + 8*ring_size + 4*ring_size + 12}-{4 + 8*ring_size + 4*ring_size + 12 + ring_size - 1}")
            print(f"  Magic: {meta_size - 8}-{meta_size - 5}")
            print(f"  Frame count: {meta_size - 4}-{meta_size - 1}")
            
            # Initialize metadata
            meta_array = np.ndarray((meta_size,), dtype=np.uint8, buffer=meta_shm.buf)
            meta_array[:] = 0
            
            # Write shape info at the correct offset
            shape_offset = 4 + (8 * ring_size) + (4 * ring_size)  # After ring_index, timestamps, and frame_ids
            struct.pack_into('III', meta_shm.buf, shape_offset, h, w, c)
            
            # DON'T write magic number yet - wait until we have actual frames
            
            # Initialize frame count to 0
            frame_count_offset = meta_size - 4
            struct.pack_into('I', meta_shm.buf, frame_count_offset, 0)
            
            shm_segments[res_name] = {
                'frame_shm': frame_shm,
                'meta_shm': meta_shm,
                'shape': shape,
                'frame_size': frame_size
            }
            
            print(f"[{process_name}] Created shared memory for {res_name}: {w}x{h}")
        
        # Send status
        status_queue.put(('status', 'Waiting for valid frames...'))
        
        # Collect initial frames
        initial_frames = []
        while len(initial_frames) < 10:
            ret, frame_bgr = cap.read()
            if ret and frame_bgr is not None and frame_bgr.size > 0:
                initial_frames.append(frame_bgr.copy())
            time.sleep(0.01)
        
        print(f"[{process_name}] Got {len(initial_frames)} initial valid frames")
        
        # Write initial frames
        status_queue.put(('status', 'Writing initial frames to buffers...'))
        
        for i in range(min(ring_size, len(initial_frames))):
            frame_bgr = initial_frames[i]
            timestamp = time.time()
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            
            # Write to all resolutions
            for res_name, seg in shm_segments.items():
                h, w, c = seg['shape']
                
                # Resize frame
                if res_name == 'native':
                    frame_resized = frame_rgb
                elif res_name == 'detection' or res_name == '640x640':
                    # Special handling for square detection
                    if h == 640 and w == 640:
                        # Resize and pad to maintain aspect ratio
                        scale = min(640 / frame_rgb.shape[1], 640 / frame_rgb.shape[0])
                        new_w = int(frame_rgb.shape[1] * scale)
                        new_h = int(frame_rgb.shape[0] * scale)
                        
                        resized = cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                        
                        # Pad to 640x640
                        pad_h = (640 - new_h) // 2
                        pad_w = (640 - new_w) // 2
                        
                        frame_resized = np.zeros((640, 640, 3), dtype=np.uint8)
                        frame_resized[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized
                    else:
                        frame_resized = cv2.resize(frame_rgb, (w, h), interpolation=cv2.INTER_LINEAR)
                else:
                    # Direct resize for other resolutions
                    frame_resized = cv2.resize(frame_rgb, (w, h), interpolation=cv2.INTER_LINEAR)
                
                # Write frame data
                frame_size = seg['frame_size']
                offset = i * frame_size
                frame_buffer = np.ndarray(
                    seg['shape'],
                    dtype=np.uint8,
                    buffer=seg['frame_shm'].buf[offset:offset + frame_size]
                )
                frame_buffer[:] = frame_resized
                
                # Update metadata
                meta_buf = seg['meta_shm'].buf
                
                # Ring index
                struct.pack_into('I', meta_buf, 0, (i + 1) % ring_size)
                
                # Timestamp
                struct.pack_into('d', meta_buf, 4 + (i * 8), timestamp)
                
                # Frame ID
                struct.pack_into('I', meta_buf, 4 + (8 * ring_size) + (i * 4), i)
                
                # Validity
                validity_offset = 4 + (8 * ring_size) + (4 * ring_size) + 12
                struct.pack_into('B', meta_buf, validity_offset + i, 1)
                
                # Update frame count
                actual_meta_size = 4 + (8 * ring_size) + (4 * ring_size) + 12 + ring_size + 4 + 4
                frame_count_offset = actual_meta_size - 4
                struct.pack_into('I', meta_buf, frame_count_offset, i + 1)
                
                # NOW write magic number after we have actual data
                if i == 0:  # Only need to write once
                    # Use the calculated meta_size, not the actual buffer size
                    actual_meta_size = 4 + (8 * ring_size) + (4 * ring_size) + 12 + ring_size + 4 + 4
                    magic_offset = actual_meta_size - 8
                    print(f"[{process_name}] Writing magic for {res_name} at offset {magic_offset} (calculated size: {actual_meta_size}, buffer size: {len(meta_buf)})")
                    struct.pack_into('I', meta_buf, magic_offset, 0xDEADBEEF)
                    # Verify it was written
                    magic_check = struct.unpack_from('I', meta_buf, magic_offset)[0]
                    print(f"[{process_name}] Verified magic: {magic_check:#x}")
                    # Force memory sync
                    meta_buf[magic_offset:magic_offset+4] = meta_buf[magic_offset:magic_offset+4]
        
        print(f"[{process_name}] Initial frames written with magic number")
        
        # Signal ready AFTER frames are written
        time.sleep(0.1)
        status_queue.put(('ready', None))
        
        # Main capture loop
        print(f"[{process_name}] Starting capture loop...")
        frame_count = len(initial_frames)
        last_valid_frame = initial_frames[-1] if initial_frames else None
        empty_frame_count = 0
        
        while stop_flag.value == 0:
            ret, frame_bgr = cap.read()
            
            # Check if frame is valid
            is_valid_frame = (ret and frame_bgr is not None and 
                             frame_bgr.size > 0 and 
                             frame_bgr.shape[0] > 0 and 
                             frame_bgr.shape[1] > 0)
            
            if not is_valid_frame:
                empty_frame_count += 1
                
                # Use last valid frame if available
                if last_valid_frame is not None:
                    frame_bgr = last_valid_frame.copy()
                    is_valid_frame = True
                    
                    if empty_frame_count % 30 == 0:
                        print(f"[{process_name}] Using last valid frame (empty count: {empty_frame_count})")
                else:
                    # Skip this iteration
                    time.sleep(0.001)
                    continue
            else:
                # Valid frame received
                last_valid_frame = frame_bgr.copy()
                empty_frame_count = 0
            
            timestamp = time.time()
            ring_pos = frame_count % ring_size
            
            # Convert to RGB
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            
            # Write to all resolutions
            for res_name, seg in shm_segments.items():
                h, w, c = seg['shape']
                
                # Resize frame (same logic as initial frames)
                if res_name == 'native':
                    frame_resized = frame_rgb
                elif res_name == 'detection' or res_name == '640x640':
                    if h == 640 and w == 640:
                        scale = min(640 / frame_rgb.shape[1], 640 / frame_rgb.shape[0])
                        new_w = int(frame_rgb.shape[1] * scale)
                        new_h = int(frame_rgb.shape[0] * scale)
                        
                        resized = cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                        
                        pad_h = (640 - new_h) // 2
                        pad_w = (640 - new_w) // 2
                        
                        frame_resized = np.zeros((640, 640, 3), dtype=np.uint8)
                        frame_resized[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized
                    else:
                        frame_resized = cv2.resize(frame_rgb, (w, h), interpolation=cv2.INTER_LINEAR)
                else:
                    frame_resized = cv2.resize(frame_rgb, (w, h), interpolation=cv2.INTER_LINEAR)
                
                # Write frame data
                frame_size = seg['frame_size']
                offset = ring_pos * frame_size
                frame_buffer = np.ndarray(
                    seg['shape'],
                    dtype=np.uint8,
                    buffer=seg['frame_shm'].buf[offset:offset + frame_size]
                )
                frame_buffer[:] = frame_resized
                
                # Update metadata
                meta_buf = seg['meta_shm'].buf
                
                struct.pack_into('I', meta_buf, 0, (ring_pos + 1) % ring_size)
                struct.pack_into('d', meta_buf, 4 + (ring_pos * 8), timestamp)
                struct.pack_into('I', meta_buf, 4 + (8 * ring_size) + (ring_pos * 4), frame_count)
                
                validity_offset = 4 + (8 * ring_size) + (4 * ring_size) + 12
                struct.pack_into('B', meta_buf, validity_offset + ring_pos, 1 if is_valid_frame else 0)
                
                # Use calculated size, not buffer size
                actual_meta_size = 4 + (8 * ring_size) + (4 * ring_size) + 12 + ring_size + 4 + 4
                frame_count_offset = actual_meta_size - 4
                struct.pack_into('I', meta_buf, frame_count_offset, frame_count + 1)
            
            frame_count += 1
            
            if frame_count % 300 == 0:
                print(f"[{process_name}] Captured {frame_count} frames")
        
        print(f"[{process_name}] Stopping...")
        
    except Exception as e:
        print(f"[{process_name}] Error: {e}")
        traceback.print_exc()
        status_queue.put(('error', str(e)))
    
    finally:
        # Cleanup
        if 'cap' in locals() and cap is not None:
            cap.release()
        
        # Cleanup shared memory
        print(f"[{process_name}] Cleaning up shared memory...")
        
        if 'res_shm' in locals():
            try:
                res_shm.close()
                res_shm.unlink()
            except:
                pass
        
        for seg in shm_segments.values() if 'shm_segments' in locals() else []:
            try:
                seg['frame_shm'].close()
                seg['frame_shm'].unlink()
                seg['meta_shm'].close()
                seg['meta_shm'].unlink()
            except:
                pass
        
        print(f"[{process_name}] Stopped")


class FrameServer:
    """
    Frame server manager that spawns a truly separate process.
    Uses named shared memory for zero-copy frame access.
    Fixed for Windows multiprocessing compatibility.
    """
    
    def __init__(self, camera_index: int, fps: int, capture_resolution: Tuple[int, int], 
                 ring_size: int = 3):
        self.camera_index = camera_index
        self.fps = fps
        self.capture_resolution = capture_resolution
        self.ring_size = ring_size
        
        # Use Queue instead of Pipe for Windows compatibility
        self.status_queue = multiprocessing.Queue()
        self.stop_flag = multiprocessing.Value('i', 0)
        self.server_process = None
        
        # Client for local access
        self.client = None
        
        # Cleanup registration
        atexit.register(self.cleanup)
    
    def start(self):
        """Start the frame server process"""
        if self.server_process and self.server_process.is_alive():
            print(f"[FrameServer] Already running for camera {self.camera_index}")
            return
        
        # Reset stop flag
        self.stop_flag.value = 0
        
        # Start server process with simple arguments
        ctx = multiprocessing.get_context('spawn')
        
        self.server_process = ctx.Process(
            target=frame_server_process_main,
            args=(
                self.camera_index,
                self.fps,
                self.capture_resolution[0],  # width
                self.capture_resolution[1],  # height
                self.status_queue,
                self.stop_flag,
                self.ring_size
            ),
            daemon=False
        )
        self.server_process.start()
        
        # Wait for initialization
        print(f"[FrameServer] Waiting for server to initialize...")
        init_success = False
        timeout_count = 0
        max_timeout = 60  # 30 seconds total
        
        while timeout_count < max_timeout:
            try:
                msg_type, msg_data = self.status_queue.get(timeout=0.5)
                
                if msg_type == 'status':
                    print(f"[FrameServer] {msg_data}")
                    timeout_count = 0  # Reset timeout on status update
                elif msg_type == 'ready':
                    init_success = True
                    break
                elif msg_type == 'error':
                    print(f"[FrameServer] Server error: {msg_data}")
                    break
            except:
                timeout_count += 1
        
        if not init_success:
            print(f"[FrameServer] Server initialization timeout or failure")
            self.stop()
            raise RuntimeError("Failed to initialize frame server")
        
        # Give server additional time to ensure all buffers are ready
        print(f"[FrameServer] Waiting for buffers to stabilize...")
        time.sleep(2.0)
        
        # Create local client
        self.client = FrameServerClient(self.camera_index, self.ring_size)
        
        # Try to connect with more attempts
        if not self.client.connect(max_attempts=5):
            print(f"[FrameServer] Failed to connect to server for camera {self.camera_index}")
            self.stop()
            raise RuntimeError("Failed to connect to frame server")
        
        print(f"[FrameServer] Started successfully for camera {self.camera_index}")

    def stop(self):
        """Stop the frame server process"""
        # Signal stop
        self.stop_flag.value = 1
        
        # Wait for process to stop
        if self.server_process:
            self.server_process.join(timeout=2.0)
            if self.server_process.is_alive():
                self.server_process.terminate()
                self.server_process.join(timeout=1.0)
        
        # Disconnect client
        if self.client:
            self.client.disconnect()
        
        # Clear queue
        while not self.status_queue.empty():
            try:
                self.status_queue.get_nowait()
            except:
                break
        
        print(f"[FrameServer] Stopped for camera {self.camera_index}")
    
    def cleanup(self):
        """Cleanup on exit"""
        self.stop()
    
    # Delegate methods to client
    def get_latest_frame(self, resolution: str = 'native'):
        """Get latest frame at specified resolution"""
        if not self.client:
            return None, 0.0, -1
        return self.client.get_latest_frame(resolution)
    
    def get_frame_batch(self, resolution: str = 'medium'):
        """Get frame batch for compatibility"""
        if not self.client:
            return None
        return self.client.get_frame_batch(resolution)
    
    def extract_roi_smart(self, bbox: List[float], track_id: int, 
                         timestamp: float, frame_id: int,
                         original_resolution: Tuple[int, int] = None):
        """Extract ROI with smart resolution selection"""
        if not self.client:
            return None
        return self.client.extract_roi_smart(
            bbox, track_id, timestamp, frame_id, 
            original_resolution or self.capture_resolution
        )


class FrameServerClient:
    """
    Client for accessing frames from the frame server process.
    """
    
    def __init__(self, camera_index: int, ring_size: int = 3):
        self.camera_index = camera_index
        self.ring_size = ring_size
        
        self.resolutions = {}
        self.shm_handles = {}
        self._connected = False
        self._actual_resolution = None
        
        # Resolution aliases for backward compatibility
        self.resolution_aliases = {
            # Map old names to new generic names
            '720p': 'medium',
            '480p': 'low',
            '360p': 'tiny',
            '640x640': 'detection'
        }
    
    def _get_shm_name(self, suffix: str) -> str:
        """Generate shared memory name"""
        return f"yqp_cam{self.camera_index}_{suffix}"
    
    def connect(self, max_attempts: int = 2) -> bool:
        """Connect to frame server shared memory with reduced attempts"""
        # Import here to avoid module-level import issues
        from multiprocessing import shared_memory
        import numpy as np
        
        for attempt in range(max_attempts):
            try:
                # First, get resolution list
                res_list_name = self._get_shm_name("reslist")
                res_list_shm = shared_memory.SharedMemory(name=res_list_name)
                
                # Read resolution list and actual resolution
                res_data = bytes(res_list_shm.buf).split(b'\0')[0]
                
                # Split resolution list and actual resolution
                if b'|' in res_data:
                    res_list_part, actual_res_part = res_data.split(b'|')
                    res_list = res_list_part.decode('utf-8').split(',')
                    
                    # Parse actual resolution
                    actual_res_str = actual_res_part.decode('utf-8')
                    if 'x' in actual_res_str:
                        w, h = map(int, actual_res_str.split('x'))
                        self._actual_resolution = (w, h)
                    else:
                        self._actual_resolution = None
                else:
                    res_list = res_data.decode('utf-8').split(',')
                    self._actual_resolution = None
                
                res_list_shm.close()
                
                print(f"[FrameClient] Found resolutions: {res_list}")
                
                # Try to connect to all resolutions at once
                all_connected = True
                temp_handles = {}
                
                for res_name in res_list:
                    frame_shm_name = self._get_shm_name(f"{res_name}_frames")
                    meta_shm_name = self._get_shm_name(f"{res_name}_meta")
                    
                    try:
                        frame_shm = shared_memory.SharedMemory(name=frame_shm_name)
                        meta_shm = shared_memory.SharedMemory(name=meta_shm_name)
                        
                        # Read shape from metadata
                        meta_size = len(meta_shm.buf)
                        # Read shape from metadata at the correct offset
                        shape_offset = 4 + (8 * self.ring_size) + (4 * self.ring_size)
                        shape = struct.unpack_from('III', meta_shm.buf, shape_offset)

                        # Check for magic number
                        # Calculate the actual metadata size (not the buffer size)
                        actual_meta_size = 4 + (8 * self.ring_size) + (4 * self.ring_size) + 12 + self.ring_size + 4 + 4
                        magic_offset = actual_meta_size - 8
                        print(f"[FrameClient] Checking magic for {res_name} at offset {magic_offset} (calculated size: {actual_meta_size}, buffer size: {meta_size})")
                        magic = struct.unpack_from('I', meta_shm.buf, magic_offset)[0]
                        
                        # Also dump first and last few bytes
                        print(f"[FrameClient] Buffer start: {meta_shm.buf[:16].hex()}")
                        print(f"[FrameClient] Buffer end: {meta_shm.buf[-16:].hex()}")
                        
                        if magic != 0xDEADBEEF:
                            # Data not ready yet
                            frame_shm.close()
                            meta_shm.close()
                            all_connected = False
                            print(f"[FrameClient] No magic number for {res_name}, waiting...")
                            break
                        
                        # Check frame count using calculated size
                        actual_meta_size = 4 + (8 * self.ring_size) + (4 * self.ring_size) + 12 + self.ring_size + 4 + 4
                        frame_count_offset = actual_meta_size - 4
                        frame_count = struct.unpack_from('I', meta_shm.buf, frame_count_offset)[0]

                        if frame_count == 0:
                            # No frames written yet
                            frame_shm.close()
                            meta_shm.close()
                            all_connected = False
                            print(f"[FrameClient] No frames for {res_name}, waiting...")
                            break
                        
                        temp_handles[res_name] = {
                            'frame_shm': frame_shm,
                            'meta_shm': meta_shm,
                            'shape': shape,
                            'frame_size': int(np.prod(shape))
                        }
                        
                        self.resolutions[res_name] = shape
                        
                    except Exception as e:
                        all_connected = False
                        print(f"[FrameClient] Failed to connect to {res_name}: {e}")
                        break
                
                if not all_connected:
                    # Clean up partial connections
                    for handle in temp_handles.values():
                        handle['frame_shm'].close()
                        handle['meta_shm'].close()
                    temp_handles.clear()
                    self.resolutions.clear()
                    
                    if attempt < max_attempts - 1:
                        print(f"[FrameClient] Waiting for server to write data... (attempt {attempt + 1}/{max_attempts})")
                        time.sleep(2.0)  # Wait longer between attempts
                        continue
                else:
                    # All connected successfully
                    self.shm_handles = temp_handles
                    
                    # Fallback actual resolution detection if not provided
                    if self._actual_resolution is None:
                        if 'native' in self.resolutions:
                            shape = self.resolutions['native']
                            self._actual_resolution = (shape[1], shape[0])
                        else:
                            # Find highest resolution (excluding detection)
                            max_pixels = 0
                            for res_name, shape in self.resolutions.items():
                                if res_name != 'detection':
                                    pixels = shape[0] * shape[1]
                                    if pixels > max_pixels:
                                        max_pixels = pixels
                                        self._actual_resolution = (shape[1], shape[0])
                    
                    self._connected = True
                    print(f"[FrameClient] Connected to frame server for camera {self.camera_index}")
                    print(f"[FrameClient] Available resolutions: {list(self.resolutions.keys())}")
                    print(f"[FrameClient] Actual resolution: {self._actual_resolution}")
                    return True
                
            except FileNotFoundError as e:
                if attempt < max_attempts - 1:
                    print(f"[FrameClient] Waiting for server... (attempt {attempt + 1}/{max_attempts})")
                    time.sleep(2.0)
                    continue
                else:
                    print(f"[FrameClient] Failed to connect to frame server for camera {self.camera_index}: {e}")
                    return False
            except Exception as e:
                print(f"[FrameClient] Connection error: {e}")
                traceback.print_exc()
                return False
        
        return False
    
    def _resolve_resolution_name(self, resolution: str) -> str:
        """Resolve resolution name including aliases"""
        # Check if it's an alias
        if resolution in self.resolution_aliases:
            resolution = self.resolution_aliases[resolution]
        
        # Check if resolution exists
        if resolution in self.shm_handles:
            return resolution
        
        # Fallback logic
        if resolution == 'native' and 'native' not in self.shm_handles:
            # Use highest available resolution
            max_pixels = 0
            best_res = None
            for res_name in self.shm_handles:
                if res_name != 'detection':
                    shape = self.resolutions[res_name]
                    pixels = shape[0] * shape[1]
                    if pixels > max_pixels:
                        max_pixels = pixels
                        best_res = res_name
            return best_res or list(self.shm_handles.keys())[0]
        
        # Default fallbacks
        fallback_order = ['medium', 'low', 'high', 'tiny', 'native', 'detection']
        for fallback in fallback_order:
            if fallback in self.shm_handles:
                return fallback
        
        # Last resort - return any available
        return list(self.shm_handles.keys())[0] if self.shm_handles else None
    
    def get_latest_frame(self, resolution: str = None) -> Tuple[Optional['np.ndarray'], float, int]:
        """Get the latest frame at specified resolution"""
        if not self._connected:
            return None, 0.0, -1
        
        # Import here to avoid module-level import
        import numpy as np
        
        # Use native resolution if none specified
        if resolution is None:
            resolution = 'native'
        
        # Resolve resolution name
        resolution = self._resolve_resolution_name(resolution)
        if resolution is None:
            return None, 0.0, -1
        
        handle = self.shm_handles[resolution]
        
        # Read metadata
        meta_buf = handle['meta_shm'].buf
        
        # Get ring index
        ring_index = struct.unpack_from('I', meta_buf, 0)[0]
        if ring_index == 0:
            return None, 0.0, -1  # No frames yet
        
        latest_pos = (ring_index - 1) % self.ring_size
        
        # Check validity flag
        validity_offset = 4 + (self.ring_size * 8) + (self.ring_size * 4) + 12
        is_valid = struct.unpack_from('B', meta_buf, validity_offset + latest_pos)[0]
        
        if not is_valid:
            # Try to find a valid frame in the ring buffer
            for i in range(self.ring_size):
                check_pos = (ring_index - 2 - i) % self.ring_size
                if check_pos < 0:
                    break
                is_valid = struct.unpack_from('B', meta_buf, validity_offset + check_pos)[0]
                if is_valid:
                    latest_pos = check_pos
                    break
            
            if not is_valid:
                return None, 0.0, -1  # No valid frames
        
        # Get timestamp and frame_id
        timestamp = struct.unpack_from('d', meta_buf, 4 + (latest_pos * 8))[0]
        frame_id = struct.unpack_from('I', meta_buf, 4 + (self.ring_size * 8) + (latest_pos * 4))[0]
        
        # Get frame
        frame_size = handle['frame_size']
        offset = latest_pos * frame_size
        
        frame_view = np.ndarray(
            handle['shape'],
            dtype=np.uint8,
            buffer=handle['frame_shm'].buf[offset:offset + frame_size]
        )
        
        # Make a copy to avoid shared memory issues
        frame = frame_view.copy()
        
        # Validate frame has data
        if np.all(frame == 0):
            # Try previous frames
            for i in range(1, self.ring_size):
                prev_pos = (latest_pos - i) % self.ring_size
                if prev_pos < 0:
                    break
                    
                prev_valid = struct.unpack_from('B', meta_buf, validity_offset + prev_pos)[0]
                if prev_valid:
                    offset = prev_pos * frame_size
                    frame_view = np.ndarray(
                        handle['shape'],
                        dtype=np.uint8,
                        buffer=handle['frame_shm'].buf[offset:offset + frame_size]
                    )
                    if not np.all(frame_view == 0):
                        frame = frame_view.copy()
                        timestamp = struct.unpack_from('d', meta_buf, 4 + (prev_pos * 8))[0]
                        frame_id = struct.unpack_from('I', meta_buf, 4 + (self.ring_size * 8) + (prev_pos * 4))[0]
                        break
            
            # If still all zeros, return None
            if np.all(frame == 0):
                return None, 0.0, -1
        
        return frame, timestamp, frame_id
    
    def get_frame_batch(self, resolution: str = 'medium') -> Optional[Dict]:
        """Get frame data suitable for existing pipeline queues"""
        # Import here to avoid module-level import
        import cv2
        
        frame_rgb, timestamp, frame_id = self.get_latest_frame(resolution)
        
        if frame_rgb is None:
            return None
        
        # Convert to BGR for compatibility
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        
        return {
            'bgr': frame_bgr,
            'rgb': frame_rgb,
            'timestamp': timestamp,
            'frame_id': int(frame_id),
            'original_resolution': self._actual_resolution
        }
    
    def extract_roi_smart(self, bbox_norm: List[float], track_id: int, 
                         timestamp: float, frame_id: int,
                         original_resolution: Tuple[int, int] = None,
                         target_size: Tuple[int, int] = (256, 256),
                         padding_ratio: float = 0.3) -> Optional[Dict]:
        """Extract ROI using smart resolution selection"""
        if not self._connected:
            return None
        
        # Import here to avoid module-level import
        import cv2
        import numpy as np
            
        if original_resolution is None:
            original_resolution = self._actual_resolution
            
        if original_resolution is None:
            return None
        
        # Calculate bbox size in pixels
        bbox_width_pixels = (bbox_norm[2] - bbox_norm[0]) * original_resolution[0]
        bbox_height_pixels = (bbox_norm[3] - bbox_norm[1]) * original_resolution[1]
        bbox_size = max(bbox_width_pixels, bbox_height_pixels)
        
        # Smart resolution selection based on ROI size
        available_resolutions = [k for k in self.resolutions.keys() if k != 'detection']
        
        # Sort by resolution size
        sorted_resolutions = sorted(available_resolutions, 
                                  key=lambda k: self.resolutions[k][0] * self.resolutions[k][1],
                                  reverse=True)
        
        # Select resolution where ROI would be at least 80 pixels
        selected_resolution = sorted_resolutions[-1]  # Default to lowest
        for res in sorted_resolutions:
            res_height = self.resolutions[res][0]
            if bbox_size * (res_height / original_resolution[1]) >= 80:
                selected_resolution = res
                break
        
        # Get frame at selected resolution
        frame_rgb, _, _ = self.get_latest_frame(selected_resolution)
        if frame_rgb is None:
            return None
        
        # Get actual dimensions of selected resolution
        h, w = self.resolutions[selected_resolution][:2]
        
        # Convert normalized bbox to pixel coordinates for this resolution
        scale_x = w / original_resolution[0]
        scale_y = h / original_resolution[1]
        
        x1 = int(bbox_norm[0] * original_resolution[0] * scale_x)
        y1 = int(bbox_norm[1] * original_resolution[1] * scale_y)
        x2 = int(bbox_norm[2] * original_resolution[0] * scale_x)
        y2 = int(bbox_norm[3] * original_resolution[1] * scale_y)
        
        # Apply padding
        bbox_w = x2 - x1
        bbox_h = y2 - y1
        pad_w = int(bbox_w * padding_ratio)
        pad_h = int(bbox_h * padding_ratio)
        
        x1 = max(0, x1 - pad_w)
        y1 = max(0, y1 - pad_h)
        x2 = min(w, x2 + pad_w)
        y2 = min(h, y2 + pad_h)
        
        # Extract ROI
        roi = frame_rgb[y1:y2, x1:x2]
        
        if roi.size == 0:
            return None
        
        # Resize to target size
        roi_resized = cv2.resize(roi, target_size, interpolation=cv2.INTER_LINEAR)
        
        # Calculate quality score
        roi_area = (x2 - x1) * (y2 - y1)
        quality_score = min(1.0, roi_area / (w * h * 0.1))
        
        # Transform info for coordinate mapping back to original
        transform_scale = (x2 - x1) / target_size[0]
        transform_offset_x = x1 / scale_x  # Convert back to original resolution
        transform_offset_y = y1 / scale_y
        
        return {
            'roi': roi_resized,
            'transform': {
                'scale': transform_scale / scale_x,  # Scale relative to original
                'offset_x': transform_offset_x,
                'offset_y': transform_offset_y,
                'frame_width': original_resolution[0],
                'frame_height': original_resolution[1]
            },
            'quality_score': quality_score,
            'track_id': track_id,
            'timestamp': timestamp,
            'frame_id': frame_id,
            'resolution_used': selected_resolution
        }
    
    def disconnect(self):
        """Disconnect from shared memory"""
        for handle in self.shm_handles.values():
            try:
                handle['frame_shm'].close()
                handle['meta_shm'].close()
            except:
                pass
        self.shm_handles.clear()
        self._connected = False