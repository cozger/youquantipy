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
        Append new samples, compute deviation from rolling mean, detect changes,
        then compute comodulation magnitudes when both participants change in synchrony.
        Returns a length-52 float array of comodulation magnitudes.
        """
        with self._lock:
            # 1) Compute baseline from incremental sums
            if len(self.buff1) >= self.w:
                baseline1 = self.sum1 / self.w
                baseline2 = self.sum2 / self.w
            else:
                baseline1 = np.zeros_like(vec1)
                baseline2 = np.zeros_like(vec2)

            # 2) Prepare new buffered values so latched channels keep their baseline
            buf1 = vec1.copy()
            buf2 = vec2.copy()
            buf1[self.states] = baseline1[self.states]
            buf2[self.states] = baseline2[self.states]

            # 3) Update running sums & deques
            if len(self.buff1) == self.w:
                # subtract oldest
                old1 = self.buff1[0]; old2 = self.buff2[0]
                self.sum1 -= old1
                self.sum2 -= old2
            self.buff1.append(buf1)
            self.buff2.append(buf2)
            self.sum1 += buf1
            self.sum2 += buf2

            # 4) Wait until buffer is full
            if len(self.buff1) < self.w:
                return None

            # 5) Compute deviations from baseline
            delta1 = vec1 - baseline1
            delta2 = vec2 - baseline2

        # Detect per-channel expression change
        change1 = np.abs(delta1) > self.delta_threshold
        change2 = np.abs(delta2) > self.delta_threshold

        # Detect similarity of changes
        similarity = np.abs(delta1 - delta2) < self.similarity_threshold

        # Compute comodulation magnitude: average of absolute deltas where both change and in sync
        mask = change1 & change2 & similarity
        magnitudes = (np.abs(delta1) + np.abs(delta2)) / 2.0
        comod = magnitudes  * mask.astype(float)
          # ── State‐machine: enter when above delta_threshold, exit when below exit_threshold ──
        out = np.zeros_like(comod)
        enter = self.delta_threshold
        exit_ = self.exit_threshold

        for i, val in enumerate(comod):
            if self.states[i]:
                # EXIT if either (a) the mask says they’re no longer co-moving, or
                # (b) the magnitude has truly fallen below the exit threshold.
                if (not mask[i]) or (val < exit_):
                    self.states[i]    = False
                    self.hold_vals[i] = 0.0
                else:
                    # still in a co-moving state, so keep or bump the peak
                    if val > self.hold_vals[i]:
                        self.hold_vals[i] = val

            else:
                # ENTER only if the mask says “both moved enough and are similar”
                if mask[i] and (val > enter):
                    self.states[i]    = True
                    self.hold_vals[i] = val

            out[i] = self.hold_vals[i]

        # Push only if we've called setup_stream()
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