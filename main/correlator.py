import time
import numpy as np
from collections import deque
import threading
from pylsl import StreamInfo, StreamOutlet

class ChannelCorrelator:
    """
    Detects per-channel expression changes based on deviations from a rolling baseline
    and computes comodulation magnitudes when both participants change similarly.
    Outputs a 52-channel float vector of comodulation magnitudes.
    """
    def __init__(self,
                 window_size=50,
                 fps=30,
                 delta_threshold=0.05,
                 similarity_threshold=0.6):
        # Exit threshold to release state latching
        self.exit_threshold = delta_threshold * 0.6
        
        self.w = window_size
        self.fps = fps
        self.delta_threshold = delta_threshold
        self.similarity_threshold = similarity_threshold
        # For state‐machine holding of sustained comodulation
        self.states     = np.zeros(52, dtype=bool)
        self.hold_vals  = np.zeros(52, dtype=float)
        # Buffers for rolling means
        self.buff1 = deque(maxlen=self.w)
        self.buff2 = deque(maxlen=self.w)
        self.sum1 = np.zeros(52, dtype=float)
        self.sum2 = np.zeros(52, dtype=float)
        self._lock = threading.Lock()
                # outlet will be created when you press Start
        self.outlet = None
        self.last_out = None

    def update(self, vec1: np.ndarray, vec2: np.ndarray):
        """
        Optimized version.
        Append new samples, compute deviation from rolling mean, detect changes,
        then compute comodulation magnitudes when both participants change in synchrony.
        Returns a length-52 float array of comodulation magnitudes.
        """
        with self._lock:
            # Use faster operations
            buf1 = vec1.copy()
            buf2 = vec2.copy()
            
            # Only modify latched channels (likely few)
            if self.states.any():
                if len(self.buff1) >= self.w:
                    baseline1 = self.sum1 / self.w
                    baseline2 = self.sum2 / self.w
                    buf1[self.states] = baseline1[self.states]
                    buf2[self.states] = baseline2[self.states]
            
            # Update buffers
            if len(self.buff1) == self.w:
                old1 = self.buff1[0]
                old2 = self.buff2[0]
                self.sum1 -= old1
                self.sum2 -= old2
            
            self.buff1.append(buf1)
            self.buff2.append(buf2)
            self.sum1 += buf1
            self.sum2 += buf2
            
            if len(self.buff1) < self.w:
                return None
            
            # Compute baselines only once
            baseline1 = self.sum1 / self.w
            baseline2 = self.sum2 / self.w
            delta1 = vec1 - baseline1
            delta2 = vec2 - baseline2
        
        # Vectorized operations outside lock
        abs_delta1 = np.abs(delta1)
        abs_delta2 = np.abs(delta2)
        
        # Vectorized comparisons
        change1 = abs_delta1 > self.delta_threshold
        change2 = abs_delta2 > self.delta_threshold
        similarity = np.abs(delta1 - delta2) < self.similarity_threshold
        
        # Compute all at once
        mask = change1 & change2 & similarity
        magnitudes = (abs_delta1 + abs_delta2) * 0.5
        comod = magnitudes * mask
        
        # Vectorized state machine
        out = self.hold_vals.copy()
        
        # Update states where entering
        entering = mask & (comod > self.delta_threshold) & ~self.states
        self.states[entering] = True
        self.hold_vals[entering] = comod[entering]
        
        # Update values where already in state
        in_state = self.states & mask
        update_mask = in_state & (comod > self.hold_vals)
        self.hold_vals[update_mask] = comod[update_mask]
        
        # Exit states
        exiting = self.states & (~mask | (comod < self.exit_threshold))
        self.states[exiting] = False
        self.hold_vals[exiting] = 0.0
        
        out = self.hold_vals
        
        # Push to LSL
        if self.outlet:
            try:
                self.outlet.push_sample(out.tolist())
            except Exception:
                pass
        
        return out

    def run(self, p1, p2, stop_evt):
        """
        Combined loop: read frames from two Participant instances,
        stream their samples, compute and stream comodulation magnitudes.
        """
        while not stop_evt.is_set():
            sample1 = p1.read_frame()
            sample2 = p2.read_frame()

            # Stream individual samples if outlets exist
            if sample1 is not None and getattr(p1, 'outlet', None):
                p1.outlet.push_sample(sample1)
            if sample2 is not None and getattr(p2, 'outlet', None):
                p2.outlet.push_sample(sample2)

            # Compute and stream comodulation magnitudes
            if sample1 is not None and sample2 is not None:
                vec1 = np.array(sample1[:52], dtype=float)
                vec2 = np.array(sample2[:52], dtype=float)
                self.update(vec1, vec2)

            stop_evt.wait(1.0 / self.fps)
    def setup_stream(self, fps=None,
                     name: str = "CoModulation",
                     source_id: str = "comodulator_uid"):
        """Call this once when Start is pressed."""
        info = StreamInfo(
            name=name,
            type="Comodulation",
            channel_count=52,
            nominal_srate=fps,
            channel_format="float32",
            source_id=source_id
        )
        self.outlet = StreamOutlet(info)

    def close(self):
        if self.outlet:
            try:
                self.outlet.close()
            except Exception:
                pass
            self.outlet = None