import threading
import time
import numpy as np
from multiprocessing import shared_memory


class NumpySharedBuffer:
    """Zero-copy shared memory using numpy arrays"""
    def __init__(self, size=104, name=None):
        self.size = size
        
        if name is None:
            # Create new shared memory
            self.shm = shared_memory.SharedMemory(create=True, size=size * 4 + 8)  # float32 + timestamp
            self.name = self.shm.name
            self._created = True
        else:
            # Connect to existing shared memory
            self.shm = shared_memory.SharedMemory(name=name)
            self.name = name
            self._created = False
            
        # Create views into the shared memory
        self.arr = np.ndarray((size,), dtype=np.float32, buffer=self.shm.buf[8:])
        self.timestamp = np.ndarray((1,), dtype=np.float64, buffer=self.shm.buf[:8])
        
    def write(self, scores):
        """Write scores with zero copy"""
        self.timestamp[0] = time.time()
        scores_len = min(len(scores), self.size)
        self.arr[:scores_len] = scores[:scores_len]
        if scores_len < self.size:
            self.arr[scores_len:] = 0.0  # Clear remaining
        
    def read_latest(self):
        """Read with zero copy"""
        return self.arr.copy(), float(self.timestamp[0])
        
    def cleanup(self):
        """Cleanup shared memory"""
        try:
            self.shm.close()
            if self._created:
                self.shm.unlink()
        except:
            pass


class FastScoreReader:
    """High-performance score reader that runs in a tight loop"""
    def __init__(self, score_buffers, correlator, multi_face_mode, target_fps=30):
        self.score_buffers = score_buffers
        self.correlator = correlator
        self.multi_face_mode = multi_face_mode
        self.target_fps = target_fps
        self.running = False
        self.thread = None
        self.latest_correlation = None
        self.lock = threading.Lock()
        
    def start(self):
        """Start the reader thread"""
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        
    def stop(self):
        """Stop the reader thread"""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
            
    def get_latest_correlation(self):
        """Get the latest correlation data thread-safely"""
        with self.lock:
            return self.latest_correlation
            
    def _run(self):
        """Tight loop that constantly reads from buffers with FPS throttling"""
        last_timestamps = {}
        frame_count = 0
        fps_start = time.time()
        debug_counter = 0
        
        # Throttling based on desired FPS
        target_fps = self.target_fps
        frame_interval = 1.0 / target_fps  # ~33ms between frames at 30fps
        last_process_time = 0
        
        print(f"[READER] Fast score reader started - Multi-face: {self.multi_face_mode}, Buffers: {len(self.score_buffers)}")
        print(f"[READER] Target FPS: {target_fps}, Frame interval: {frame_interval*1000:.1f}ms")
        
        while self.running:
            start = time.time()
            debug_counter += 1
            
            # Throttle to target FPS
            time_since_last = start - last_process_time
            if time_since_last < frame_interval:
                sleep_time = frame_interval - time_since_last
                time.sleep(sleep_time)
                continue
            
            try:
                t1 = time.time()
                scores = []
                all_new = True
                
                if self.multi_face_mode and len(self.score_buffers) >= 1:
                    # Multi-face mode: single buffer with 104 values (52 per face)
                    buffer = self.score_buffers[0]
                    if buffer is not None:
                        data, ts = buffer.read_latest()
                        
                        # Check if actually new data
                        if ts == last_timestamps.get(0, 0):
                            all_new = False
                        else:
                            last_timestamps[0] = ts
                            if data is not None and len(data) >= 104:
                                scores.append(data[:52])    # Face 1
                                scores.append(data[52:104]) # Face 2
                else:
                    # Holistic mode: simplified approach with detailed debug
                    for i, buffer in enumerate(self.score_buffers[:2]):
                        if buffer is None:
                            if debug_counter % 1000 == 0:
                                print(f"[READER DEBUG] Buffer {i} is None")
                            continue
                        
                        data, ts = buffer.read_latest()
                        
                        # Only process if data is newer than what we've seen
                        if ts > last_timestamps.get(i, 0):
                            last_timestamps[i] = ts
                            if data is not None and len(data) >= 52:
                                scores.append(data[:52])
                        elif data is not None and len(data) >= 52:
                            # Still add the data even if timestamp is same (might be valid)
                            scores.append(data[:52])
                        elif debug_counter % 1000 == 0:
                            print(f"[READER DEBUG] Buffer {i}: Stale data, ts={ts}, last={last_timestamps.get(i, 0)}")
                    
                    # Process if we have data from both buffers
                    all_new = len(scores) >= 2
                
                t2 = time.time()
                queue_time = (t2 - t1) * 1000
                
                # Process correlation if we have data
                if len(scores) >= 2:
                    t3 = time.time()
                    corr = self.correlator.update(np.array(scores[0]), np.array(scores[1]))
                    t4 = time.time()
                    
                    collection_time = (t3 - t2) * 1000
                    correlate_time = (t4 - t3) * 1000
                    total_time = (t4 - t1) * 1000
                    
                    if corr is not None:
                        with self.lock:
                            self.latest_correlation = corr
                        
                        frame_count += 1
                        last_process_time = t4  # Update last process time after successful correlation
                        
                        if frame_count % 30 == 0:
                            fps = frame_count / (time.time() - fps_start)
                            print(f"[READER] Queue: {queue_time:.1f}ms, Collection: {collection_time:.1f}ms, "
                                  f"Correlate: {correlate_time:.1f}ms, Total: {total_time:.1f}ms, FPS: {fps:.1f}")
                else:
                    # Debug why we're not processing
                    if debug_counter % 1000 == 0:  # Every 1000 loops when not processing
                        print(f"[READER DEBUG] Skipping - scores length: {len(scores)}, multi_face: {self.multi_face_mode}, buffers: {len(self.score_buffers)}")
                        print(f"[READER DEBUG] Buffer states:")
                        for i, buffer in enumerate(self.score_buffers[:2]):
                            if buffer is not None:
                                data, ts = buffer.read_latest()
                                # Fix the array truth value error
                                data_len = len(data) if data is not None else 'None'
                                print(f"  Buffer {i}: data_len={data_len}, ts={ts}")
                            else:
                                print(f"  Buffer {i}: None")
                
                # Short sleep to prevent busy waiting
                time.sleep(0.001)  # 1ms
                
            except Exception as e:
                print(f"[READER] Error: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.01)
                
        print("[READER] Fast score reader stopped")