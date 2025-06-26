from multiprocessing import Array, Value
import numpy as np
import time
from multiprocessing import shared_memory


class SharedScoreBuffer:
    """Lock-free shared memory buffer for score data"""
    def __init__(self, size=104):  # 52*2 for multiface, or 52 for single
        self.data = Array('f', size)  # float array
        self.timestamp = Value('d', 0.0)  # double timestamp
        self.new_data = Value('i', 0)  # integer flag
        self.size = size
    
    def write(self, scores):
        """Write scores without locking"""
        # Update timestamp first
        self.timestamp.value = time.time()
        # Copy data
        for i in range(min(len(scores), self.size)):
            self.data[i] = scores[i]
        # Set flag last to indicate new data
        self.new_data.value = 1
    
    def read_if_new(self):
        """Read only if new data available"""
        if self.new_data.value:
            # Clear flag
            self.new_data.value = 0
            # Return copy of data
            return list(self.data), self.timestamp.value
        return None, None
    
    def read_latest(self):
        """Always read latest data"""
        self.new_data.value = 0
        return list(self.data), self.timestamp.value


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