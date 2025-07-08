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
        self.start_time = None
        
        # LSL outlet for frame numbers
        self.lsl_outlet = None
        
    def setup_lsl_stream(self, participant_id=""):
        """Setup LSL stream for frame numbers with proper metadata"""
        stream_name = f"{participant_id}_video_frames" if participant_id else "video_frames"
        info = StreamInfo(
            name=stream_name,
            type="VideoSync",
            channel_count=3,  # frame_number, lsl_timestamp, video_timestamp
            nominal_srate=0,  # Irregular rate
            channel_format="double64",
            source_id=f"{stream_name}_{participant_id}_uid"
        )
        
        # Add metadata
        desc = info.desc()
        desc.append_child_value("participant", participant_id)
        desc.append_child_value("fps", str(self.target_fps))
        
        # Channel descriptions
        channels = desc.append_child("channels")
        for label, unit in [("frame_number", "count"), ("lsl_timestamp", "seconds"), ("video_timestamp", "seconds")]:
            chan = channels.append_child("channel")
            chan.append_child_value("label", label)
            chan.append_child_value("unit", unit)
        
        self.lsl_outlet = StreamOutlet(info, chunk_size=1, max_buffered=360)
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
        self.start_time = time.time()
        
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
        last_lsl_push = 0
        push_interval = 1.0 / self.target_fps
        
        while self.recording:
            try:
                frame = self.frame_queue.get(timeout=0.1)
                
                # Write frame
                if self.writer and self.writer.isOpened():
                    self.writer.write(frame)
                    self.frame_number += 1
                    
                    # Send frame number via LSL with throttling
                    current_time = time.time()
                    if self.lsl_outlet and (current_time - last_lsl_push >= push_interval):
                        video_time = (self.frame_number - 1) / self.target_fps
                        sample = [float(self.frame_number), current_time, video_time]
                        self.lsl_outlet.push_sample(sample, current_time)
                        last_lsl_push = current_time
                        
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[Recorder] Error in record loop: {e}")
                
    def close(self):
        """Clean up resources"""
        self.stop_recording()
        if self.lsl_outlet:
            try:
                self.lsl_outlet.__del__()
            except:
                pass
            self.lsl_outlet = None

class VideoRecorderProcess:
    """
    Multiprocess version that receives frames from shared memory.
    More efficient for high-performance recording.
    """
    def __init__(self, output_dir="recordings", codec='MJPG', fps=30, camera_index=None):
        self.output_dir = output_dir
        self.codec = codec
        self.fps = fps
        self.camera_index = camera_index  # Store camera index
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
                  self.output_dir, self.codec, self.fps, participant_id, filename, 
                  self.camera_index),
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
                    output_dir, codec, fps, participant_id, filename=None, camera_index=None):
    """Worker process for video recording - auto-detects dimensions"""
    from pylsl import StreamInfo, StreamOutlet, local_clock
    
    # Generate filename if not provided
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{participant_id}_{timestamp}.avi" if participant_id else f"recording_{timestamp}.avi"
    
    filepath = Path(output_dir) / filename
    filepath.parent.mkdir(exist_ok=True)

    # Setup LSL with camera index in the name
    stream_name = f"camera_{camera_index}_video_frames" if camera_index is not None else "video_frames"
    
    info = StreamInfo(
        name=stream_name,
        type="VideoSync",
        channel_count=4,  # frame_number, unix_timestamp, lsl_timestamp, video_timestamp
        nominal_srate=0,  # Irregular rate
        channel_format="double64",
        source_id=f"{stream_name}_cam{camera_index}_uid"
    )
    
    # Add metadata
    desc = info.desc()
    desc.append_child_value("camera_index", str(camera_index) if camera_index is not None else "unknown")
    desc.append_child_value("participant", str(participant_id))
    desc.append_child_value("video_file", str(filename))
    desc.append_child_value("fps", str(fps))
    
    # Channel descriptions
    channels = desc.append_child("channels")
    for label, unit in [
        ("frame_number", "count"),
        ("unix_timestamp", "seconds"), 
        ("lsl_timestamp", "seconds"),
        ("video_timestamp", "seconds")
    ]:
        chan = channels.append_child("channel")
        chan.append_child_value("label", label)
        chan.append_child_value("unit", unit)
    
    outlet = StreamOutlet(info, chunk_size=1, max_buffered=360)
    print(f"[Recorder] Created LSL stream '{stream_name}' for camera {camera_index}")

    writer = None
    frames_written = 0
    lsl_samples_pushed = 0
    expected_dimensions = None
    start_time = None
    last_push_time = 0
    push_interval = 1.0 / fps
    
    print(f"[Recorder] Waiting for first frame...")
    
    # Recording loop
    while not stop_flag.value or not frame_queue.empty():
        try:
            frame = frame_queue.get(timeout=0.1)
            
            # Initialize on first frame
            if writer is None:
                height, width = frame.shape[:2]
                channels = frame.shape[2] if len(frame.shape) > 2 else 1
                
                print(f"[Recorder] Auto-detected dimensions: {width}x{height} with {channels} channels")
                expected_dimensions = (height, width, channels) if channels == 3 else (height, width)
                
                # Create writer
                fourcc = cv2.VideoWriter_fourcc(*codec)
                writer = cv2.VideoWriter(str(filepath), fourcc, fps, (width, height))
                
                if not writer.isOpened():
                    # Try fallback
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    writer = cv2.VideoWriter(str(filepath), fourcc, fps, (width, height))
                
                print(f"[Recorder] Video writer initialized")
                start_time = time.time()
            
            # Ensure proper dimensions
            current_shape = frame.shape[:2] if len(frame.shape) == 2 else frame.shape
            if expected_dimensions and current_shape != expected_dimensions:
                height, width = expected_dimensions[:2]
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
            
            # Ensure proper format
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif len(frame.shape) == 3 and frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            
            if frame.dtype != np.uint8:
                frame = np.clip(frame, 0, 255).astype(np.uint8)
            
            # Write frame (ignore return value since we know it works)
            if writer:
                writer.write(frame)
            
            frames_written += 1
            frame_count.value = frames_written
            
            # LSL push logic
            unix_time = time.time()
            video_time = (frames_written - 1) / fps
            
            # Throttle LSL pushes
            if unix_time - last_push_time >= push_interval:
                try:
                    lsl_time = local_clock()
                    sample = [
                        float(frames_written),
                        unix_time,
                        lsl_time,
                        video_time
                    ]
                    outlet.push_sample(sample, lsl_time)
                    last_push_time = unix_time
                    lsl_samples_pushed += 1
                    
                    if lsl_samples_pushed % 30 == 1:  # Log periodically
                        print(f"[Recorder] Frame {frames_written}, LSL samples: {lsl_samples_pushed}")
                        
                except Exception as e:
                    print(f"[Recorder] Error pushing to LSL: {e}")
                    
        except queue.Empty:
            if stop_flag.value:
                break
            continue
        except Exception as e:
            print(f"[Recorder] Error: {e}")
            import traceback
            traceback.print_exc()
    
    # Cleanup
    print(f"[Recorder] Finalizing - {frames_written} frames, {lsl_samples_pushed} LSL samples")
    
    if writer:
        writer.release()
    
    # Final LSL sample
    if outlet and frames_written > 0:
        try:
            sample = [float(frames_written), time.time(), local_clock(), frames_written/fps]
            outlet.push_sample(sample)
        except:
            pass
    
    # Wait before closing outlet
    time.sleep(0.5)
    
    # Verify file
    if filepath.exists():
        file_size = filepath.stat().st_size
        print(f"[Recorder] Saved {filepath.name} ({file_size/1024/1024:.1f} MB)")
    
    print(f"[Recorder] Done.")