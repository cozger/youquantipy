import cv2
import time
import threading
import queue
import numpy as np
from datetime import datetime
from pathlib import Path
from pylsl import StreamInfo, StreamOutlet
from multiprocessing import Process, Queue as MPQueue, Value, Array
import ctypes
from tkinter import ttk



class VideoRecorder:
    """
    Records video from shared frames and outputs frame numbers via LSL.
    Designed to work alongside the existing YouQuantiPy system.
    """
    def __init__(self, output_dir="recordings", codec='MJPG', fps=30):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.codec = codec
        self.target_fps = fps
        self.recording = False
        self.writer = None
        self.frame_queue = queue.Queue(maxsize=30)
        self.record_thread = None
        self.frame_number = 0
        
        # LSL outlet for frame numbers
        self.lsl_outlet = None
        
    def setup_lsl_stream(self, participant_id=""):
        """Setup LSL stream for frame numbers"""
        stream_name = f"{participant_id}_frame_numbers" if participant_id else "frame_numbers"
        info = StreamInfo(
            name=stream_name,
            type="VideoSync",
            channel_count=2,  # frame_number, timestamp
            nominal_srate=self.target_fps,
            channel_format="float32",
            source_id=f"{stream_name}_uid"
        )
        self.lsl_outlet = StreamOutlet(info)
        print(f"[Recorder] LSL stream created: {stream_name}")
        
    def start_recording(self, width, height, participant_id=""):
        """Start recording video"""
        if self.recording:
            return
            
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{participant_id}_{timestamp}.avi" if participant_id else f"recording_{timestamp}.avi"
        filepath = self.output_dir / filename
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        self.writer = cv2.VideoWriter(
            str(filepath), fourcc, self.target_fps, (width, height)
        )
        
        if not self.writer.isOpened():
            print(f"[Recorder] Failed to open video writer for {filepath}")
            return False
            
        # Setup LSL if not already done
        if self.lsl_outlet is None:
            self.setup_lsl_stream(participant_id)
            
        self.recording = True
        self.frame_number = 0
        
        # Start recording thread
        self.record_thread = threading.Thread(target=self._record_loop, daemon=True)
        self.record_thread.start()
        
        print(f"[Recorder] Started recording to {filepath}")
        return True
        
    def stop_recording(self):
        """Stop recording and close video file"""
        if not self.recording:
            return
            
        self.recording = False
        
        # Wait for thread to finish
        if self.record_thread:
            self.record_thread.join(timeout=2.0)
            
        # Close video writer
        if self.writer:
            self.writer.release()
            self.writer = None
            
        print(f"[Recorder] Stopped recording. Total frames: {self.frame_number}")
        
    def add_frame(self, frame):
        """Add a frame to the recording queue"""
        if not self.recording:
            return False
            
        try:
            self.frame_queue.put_nowait(frame.copy())
            return True
        except queue.Full:
            # Drop frame if queue is full
            return False
            
    def _record_loop(self):
        """Recording thread loop"""
        while self.recording:
            try:
                frame = self.frame_queue.get(timeout=0.1)
                
                # Write frame
                if self.writer and self.writer.isOpened():
                    self.writer.write(frame)
                    self.frame_number += 1
                    
                    # Send frame number via LSL
                    if self.lsl_outlet:
                        timestamp = time.time()
                        self.lsl_outlet.push_sample([self.frame_number, timestamp])
                        
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[Recorder] Error in record loop: {e}")
                
    def close(self):
        """Clean up resources"""
        self.stop_recording()
        if self.lsl_outlet:
            self.lsl_outlet = None


class VideoRecorderProcess:
    """
    Multiprocess version that receives frames from shared memory.
    More efficient for high-performance recording.
    """
    def __init__(self, output_dir="recordings", codec='MJPG', fps=30):
        self.output_dir = output_dir
        self.codec = codec
        self.fps = fps
        self.process = None
        self.frame_queue = MPQueue(maxsize=30)
        self.stop_flag = Value(ctypes.c_bool, False)
        self.frame_count = Value(ctypes.c_int, 0)
        
    def start_recording(self, participant_id="", filename=None):
        """Start recording process - dimensions auto-detected from first frame"""
        if self.process and self.process.is_alive():
            return False
            
        self.stop_flag.value = False
        self.frame_count.value = 0
        
        self.process = Process(
            target=_recorder_worker,
            args=(self.frame_queue, self.stop_flag, self.frame_count,
                  self.output_dir, self.codec, self.fps, participant_id, filename),
            daemon=True
        )
        self.process.start()
        return True

        
    def add_frame(self, frame):
        """Add frame to recording queue"""
        if not self.process or not self.process.is_alive():
            return False
            
        try:
            self.frame_queue.put_nowait(frame)
            return True
        except:
            return False
                    
    def stop_recording(self):
        """Stop recording process"""
        if self.process and self.process.is_alive():
            print(f"[Recorder] Stopping recording, processing remaining frames...")
            self.stop_flag.value = True
            
            # DON'T clear the queue - let worker process remaining frames
            # Just wait for the process to finish
            self.process.join(timeout=5.0)
            
            if self.process.is_alive():
                print("[Recorder] Process didn't stop cleanly, terminating...")
                self.process.terminate()
                self.process.join(timeout=1.0)
                
        print(f"[Recorder] Total frames recorded: {self.frame_count.value}")

    def get_frame_count(self):
        """Get current frame count"""
        return self.frame_count.value


def _recorder_worker(frame_queue, stop_flag, frame_count, 
                    output_dir, codec, fps, participant_id, filename=None):
    """Worker process for video recording - auto-detects dimensions"""
    # Generate filename if not provided
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{participant_id}_{timestamp}.avi" if participant_id else f"recording_{timestamp}.avi"
    
    filepath = Path(output_dir) / filename
    filepath.parent.mkdir(exist_ok=True)

    # Setup LSL
    stream_name = f"{participant_id}_frame_numbers" if participant_id else "frame_numbers"
    info = StreamInfo(
        name=stream_name,
        type="VideoSync",
        channel_count=2,
        nominal_srate=fps,
        channel_format="float32",
        source_id=f"{stream_name}_uid"
    )
    outlet = StreamOutlet(info)

    writer = None  # Don't create writer until we know dimensions
    frames_written = 0
    expected_dimensions = None
    
    print(f"[Recorder] Waiting for first frame to auto-detect dimensions...")
    
    # Recording loop
    while not stop_flag.value or not frame_queue.empty():
        try:
            frame = frame_queue.get(timeout=0.1)
            
            # Auto-detect dimensions from first frame
            if writer is None:
                height, width = frame.shape[:2]
                channels = frame.shape[2] if len(frame.shape) > 2 else 1
                
                print(f"[Recorder] Auto-detected dimensions: {width}x{height} with {channels} channels")
                expected_dimensions = (height, width, channels) if channels == 3 else (height, width)
                
                # Create writer with detected dimensions
                fourcc = cv2.VideoWriter_fourcc(*codec)
                writer = cv2.VideoWriter(str(filepath), fourcc, fps, (width, height))
                
                if not writer.isOpened():
                    print(f"[Recorder] Failed to open video writer with {codec} codec")
                    # Try fallback codec
                    fourcc_fallback = cv2.VideoWriter_fourcc(*'XVID')
                    writer = cv2.VideoWriter(str(filepath), fourcc_fallback, fps, (width, height))
                    if writer.isOpened():
                        print(f"[Recorder] Fallback to XVID codec succeeded")
                    else:
                        print(f"[Recorder] All codecs failed for {width}x{height}!")
                        return
                else:
                    print(f"[Recorder] Writer opened successfully for {width}x{height} @ {fps}fps")
            
            # Verify frame matches expected dimensions
            current_shape = frame.shape[:2] if len(frame.shape) == 2 else frame.shape
            if expected_dimensions and current_shape != expected_dimensions:
                # Only warn once per dimension change
                if frames_written == 0 or frames_written % 100 == 0:
                    print(f"[Recorder] Warning: Frame dimensions changed from {expected_dimensions} to {current_shape}")
                # Resize to match initial dimensions
                height, width = expected_dimensions[:2]
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
            
            # Ensure proper format for OpenCV
            if len(frame.shape) == 2:  # Grayscale
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif len(frame.shape) == 3 and frame.shape[2] == 4:  # RGBA
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            
            # Ensure uint8 dtype
            if frame.dtype != np.uint8:
                frame = np.clip(frame, 0, 255).astype(np.uint8)
            
            # Write frame
            success = writer.write(frame)
            if success:
                frames_written += 1
                frame_count.value += 1
                
                # Send frame number via LSL
                outlet.push_sample([frame_count.value, time.time()])
                
                if frames_written % fps == 0:  # Log every second
                    print(f"[Recorder] Written {frames_written} frames")
            # Remove the else clause that prints the error message
                
        except queue.Empty:
            if stop_flag.value:
                break
            continue
        except Exception as e:
            print(f"[Recorder] Error: {e}")
            import traceback
            traceback.print_exc()
    
    # Cleanup
    print(f"[Recorder] Finalizing video with {frames_written} frames...")
    if writer is not None:
        writer.release()
    
    # Verify file
    if filepath.exists():
        file_size = filepath.stat().st_size
        print(f"[Recorder] Saved to {filepath} ({file_size:,} bytes, {frames_written} frames)")
        if file_size < 1000:
            print(f"[Recorder] WARNING: File size very small, recording may be corrupt")
    else:
        print(f"[Recorder] ERROR: Video file was not created")