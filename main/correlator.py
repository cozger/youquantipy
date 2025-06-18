import time
import numpy as np
from collections import deque
from pylsl import StreamInfo, StreamOutlet

class ChannelCorrelator:
    """
    Detects per-channel expression changes based on deviations from a rolling baseline
    and computes comodulation magnitudes when both participants change similarly.
    Outputs a 52-channel float vector of comodulation magnitudes.
    """
    def __init__(self,
                 window_size=5,
                 fps=60,
                 delta_threshold=0.05,
                 similarity_threshold=0.9):
        self.w = window_size
        self.fps = fps
        self.delta_threshold = delta_threshold
        self.similarity_threshold = similarity_threshold
        # For state‐machine holding of sustained comodulation
        self.states     = np.zeros(52, dtype=bool)
        self.hold_vals  = np.zeros(52, dtype=float)
        # Exit threshold set to half the enter threshold (adjust as you like)
        self.exit_threshold = delta_threshold * 0.98
        # Buffers for rolling means
        self.buff1 = deque(maxlen=self.w)
        self.buff2 = deque(maxlen=self.w)

        # Prepare LSL outlet for comodulation stream
        info = StreamInfo(
            name="CoModulation",
            type="Comodulation",
            channel_count=52,
            nominal_srate=self.fps,
            channel_format="float32",
            source_id="comodulator_uid"
        )
        self.outlet = StreamOutlet(info)

    def update(self, vec1: np.ndarray, vec2: np.ndarray):
        """
        Append new samples, compute deviation from rolling mean, detect changes,
        then compute comodulation magnitudes when both participants change in synchrony.
        Returns a length-52 float array of comodulation magnitudes.
        """
        # Add latest frames

                # Compute current baselines BEFORE appending
        if len(self.buff1) >= self.w:
            baseline1 = np.stack(self.buff1, axis=0).mean(axis=0)
            baseline2 = np.stack(self.buff2, axis=0).mean(axis=0)
        else:
            baseline1 = np.zeros_like(vec1)
            baseline2 = np.zeros_like(vec2)

        # Prepare buffered versions so active channels don’t shift their baseline
        buf1 = vec1.copy()
        buf2 = vec2.copy()
        # for any channel i that’s currently latched, keep its baseline value in the buffer
        buf1[self.states] = baseline1[self.states]
        buf2[self.states] = baseline2[self.states]

        # Now append into the rolling windows
        self.buff1.append(buf1)
        self.buff2.append(buf2)

        # Wait until buffer is full
        if len(self.buff1) < self.w:
            return None

        # Compute rolling baseline means
        arr1 = np.stack(self.buff1, axis=0)
        arr2 = np.stack(self.buff2, axis=0)
        baseline1 = arr1.mean(axis=0)
        baseline2 = arr2.mean(axis=0)

        # Compute deviations
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
        comod = magnitudes # * mask.astype(float) removing masking.
          # ── State‐machine: enter when above delta_threshold, exit when below exit_threshold ──
        out = np.zeros_like(comod)
        enter = self.delta_threshold
        exit_ = self.exit_threshold
        for i, val in enumerate(comod):
            if not self.states[i] and val > enter:
                # new smile‐sync starts
                self.states[i]    = True
                self.hold_vals[i] = val
            elif self.states[i] and val < exit_:
                # smile‐sync ends
                self.states[i]    = False
                self.hold_vals[i] = 0.0
            # if still in sync state and val spikes higher, update peak
            if self.states[i] and val > self.hold_vals[i]:
                self.hold_vals[i] = val
            out[i] = self.hold_vals[i]

        # Push comodulation magnitudes vector to LSL
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

            time.sleep(1.0 / self.fps)

    def close(self):
        """
        Close the comodulation LSL outlet cleanly.
        """
        try:
            self.outlet.close()
        except Exception:
            pass
