import time
import numpy as np
from multiprocessing import Process, Queue as MPQueue, Value
from pylsl import StreamInfo, StreamOutlet
from collections import OrderedDict
import threading
from correlator import ChannelCorrelator
from multiprocessing import shared_memory

class LSLHelper:
    """
    Centralized LSL stream management and comodulation calculations.
    Runs in a separate process for optimal performance.
    """
    def __init__(self, fps=30):
        self.fps = fps
        self.streams = {}  # participant_id -> StreamOutlet
        self.pose_streams = {} 
        self.correlator = ChannelCorrelator(window_size=60, fps=fps)
        self.running = False
        
    def create_face_stream(self, participant_id, include_mesh=False):
        """Create or update LSL stream for a participant"""
        stream_name = f"{participant_id}_landmarks"
        
        # Calculate channel count
        blend_channels = 52
        mesh_channels = 478 * 3 if include_mesh else 0
        total_channels = blend_channels + mesh_channels
        
        # Close existing stream if it exists with different config
        if participant_id in self.streams:
            old_info = self.streams[participant_id]['info']
            if old_info.channel_count() != total_channels:
                self.streams[participant_id]['outlet'].close()
                del self.streams[participant_id]
        
        # Create new stream if needed
        if participant_id not in self.streams:
            info = StreamInfo(
                name=stream_name,
                type="Landmark",
                channel_count=total_channels,
                nominal_srate=self.fps,
                channel_format="float32",
                source_id=f"{participant_id}_uid"
            )
            outlet = StreamOutlet(info)
            
            self.streams[participant_id] = {
                'outlet': outlet,
                'info': info,
                'include_mesh': include_mesh,
                'last_sample': None,
                'frame_count': 0,
                'last_fps_time': time.time()
            }
            print(f"[LSL Helper] Created face stream for {participant_id} with {total_channels} channels")

    def push_face_data(self, participant_id, blend_scores, mesh_data=None, pose_data=None):
        """Push data for a participant"""
        if participant_id not in self.streams:
            return
            
        stream_info = self.streams[participant_id]
        
        # Build sample
        sample = blend_scores.copy()
        
        # Only include mesh data if the stream was created with mesh enabled AND mesh data is provided
        if stream_info['include_mesh']:
            if mesh_data is not None and len(mesh_data) > 0:
                sample.extend(mesh_data)
            else:
                # Pad with zeros if mesh is expected but not provided
                mesh_size = 478 * 3
                sample.extend([0.0] * mesh_size)
                print(f"[LSL Helper] Warning: Mesh expected but not provided for {participant_id}, padding with zeros")
        
        # Check for duplicate
        if stream_info['last_sample'] is not None and sample == stream_info['last_sample']:
            return
            
        # Verify sample size matches expected channel count
        expected_size = stream_info['info'].channel_count()
        if len(sample) != expected_size:
            print(f"[LSL Helper] ERROR: Sample size mismatch for {participant_id}: "
                f"got {len(sample)}, expected {expected_size}")
            return
            
        stream_info['last_sample'] = sample.copy()
        stream_info['outlet'].push_sample(sample)
        
        # Update frame count for FPS monitoring
        stream_info['frame_count'] += 1
        
    def get_fps_stats(self):
        """Get FPS statistics for all streams"""
        stats = {}
        current_time = time.time()
        
        for pid, stream_info in self.streams.items():
            elapsed = current_time - stream_info['last_fps_time']
            if elapsed > 0:
                fps = stream_info['frame_count'] / elapsed
                stats[pid] = fps
                # Reset counters
                stream_info['frame_count'] = 0
                stream_info['last_fps_time'] = current_time
                
        return stats
    
    def create_pose_stream(self, participant_id, fps=None):
        """Create a dedicated pose stream for participant"""
        if fps is None:
            fps = self.fps
        stream_name = f"{participant_id}_pose"
        channel_count = 33 * 4
        info = StreamInfo(
            name=stream_name,
            type="Pose",
            channel_count=channel_count,
            nominal_srate=fps,
            channel_format="float32",
            source_id=f"{participant_id}_pose_uid"
        )
        outlet = StreamOutlet(info)
        self.pose_streams[participant_id] = {
            'outlet': outlet,
            'info': info,
            'last_sample': None,
            'frame_count': 0,
            'last_fps_time': time.time()
        }
        print(f"[LSL Helper] Created POSE stream for {participant_id} ({channel_count} channels)")

    def close_pose_stream(self, participant_id):
        """Close a dedicated pose stream for participant"""
        if participant_id in self.pose_streams:
            try:
                self.pose_streams[participant_id]['outlet'].close()
            except:
                pass
            del self.pose_streams[participant_id]
            print(f"[LSL Helper] Closed POSE stream for {participant_id}")

    def push_pose_data(self, participant_id, pose_data):
        if participant_id not in self.pose_streams:
            return
        stream_info = self.pose_streams[participant_id]
        # To avoid flooding, optionally check for duplicate
        if stream_info['last_sample'] == pose_data:
            return
        stream_info['last_sample'] = pose_data.copy()
        stream_info['outlet'].push_sample(pose_data)
        stream_info['frame_count'] += 1

    def close_all_pose_streams(self):
        for pid in list(self.pose_streams.keys()):
            self.close_pose_stream(pid)
    
    def close_all_streams(self):
        """Close all LSL streams"""
        for pid, stream_info in self.streams.items():
            try:
                stream_info['outlet'].close()
            except:
                pass
        self.streams.clear()

def lsl_helper_process(command_queue: MPQueue, 
                       data_queue: MPQueue,
                       correlation_buffer_name: str,
                       fps: int = 30):
    """
    Process function for centralized LSL management with dynamic stream creation.
    
    Args:
        command_queue: Queue for receiving commands
        data_queue: Queue for receiving participant data
        correlation_buffer_name: Shared memory buffer name for correlation output
        fps: Target frame rate
    """
    helper = LSLHelper(fps)
    correlator = ChannelCorrelator(window_size=60, fps=fps)
    correlator_stream_active = False
    streaming_active = False
    max_participants = 6
    mesh_enabled_per_camera = {}  # Track mesh state per camera
    
    # Connect to correlation output buffer
    try:
        corr_buffer = shared_memory.SharedMemory(name=correlation_buffer_name)
        corr_array = np.ndarray((52,), dtype=np.float32, buffer=corr_buffer.buf)
        print("[LSL Process] Connected to correlation buffer")
    except Exception as e:
        print(f"[LSL Process] Failed to connect to correlation buffer: {e}")
        corr_buffer = None
        corr_array = None
    
    participant_scores = {}  # Store latest scores for correlation
    created_streams = set()  # Track which streams have been created
    running = True
    
    # Performance monitoring
    last_fps_report = time.time()
    fps_report_interval = 5.0
    data_counts = {}
    
    print("[LSL Process] Started")
    
    while running:
        # Check for commands
        try:
            while not command_queue.empty():
                cmd = command_queue.get_nowait()

                if cmd['type'] == 'streaming_started':
                    streaming_active = True
                    max_participants = cmd.get('max_participants', 6)
                    print(f"[LSL Process] Streaming started, max participants: {max_participants}")
                
                elif cmd['type'] == 'create_stream':
                    # Check if mesh is enabled for any camera (use the stored state)
                    include_mesh = any(mesh_enabled_per_camera.values())
                    helper.create_face_stream(
                        cmd['participant_id'],
                        include_mesh=include_mesh
                    )
                    if cmd.get('include_pose', False):
                        helper.create_pose_stream(cmd['participant_id'], cmd.get('fps', helper.fps))
                    print(f"[LSL Process] Created stream for {cmd['participant_id']} (mesh={'enabled' if include_mesh else 'disabled'})")

                elif cmd['type'] == 'start_comodulation':
                    if not correlator_stream_active:
                        correlator.setup_stream(fps=fps)
                        correlator_stream_active = True
                        print("[LSL Process] Started comodulation stream")
    
                elif cmd['type'] == 'stop':
                    running = False
                    streaming_active = False
                    break

                elif cmd['type'] == 'create_pose_stream':
                    helper.create_pose_stream(cmd['participant_id'], cmd.get('fps', fps))
                    
                elif cmd['type'] == 'close_stream':
                    pid = cmd['participant_id']
                    if pid in helper.streams:
                        try:
                            helper.streams[pid]['outlet'].close()
                            del helper.streams[pid]
                            print(f"[LSL Process] Closed stream for {pid}")
                        except:
                            pass
                    if pid in helper.pose_streams:
                        helper.close_pose_stream(pid)
                    if pid in created_streams:
                        created_streams.remove(pid)
                        
                elif cmd['type'] == 'force_recreate_streams':
                    mesh_enabled = cmd.get('mesh_enabled', False)
                    print(f"[LSL Process] Force recreating all streams with mesh={mesh_enabled}")
                    
                    # Update all camera states
                    for cam_idx in mesh_enabled_per_camera:
                        mesh_enabled_per_camera[cam_idx] = mesh_enabled
                    
                    # Recreate all active streams
                    for pid in list(helper.streams.keys()):
                        try:
                            helper.streams[pid]['outlet'].close()
                            del helper.streams[pid]
                        except:
                            pass
                    
                    # Clear created streams set to force recreation
                    created_streams.clear()
                    
                    print(f"[LSL Process] All streams cleared, will recreate on next data")
                    
        except:
            pass
        
        # Process data
        try:
            data_processed = False
            while not data_queue.empty():
                data = data_queue.get_nowait()
                data_processed = True
                
                if data['type'] == 'config_update':
                     # Handle mesh configuration updates
                    camera_index = data['camera_index']
                    mesh_enabled = data['mesh_enabled']
                    mesh_enabled_per_camera[camera_index] = mesh_enabled
                    print(f"[LSL Process] Camera {camera_index} mesh {'enabled' if mesh_enabled else 'disabled'}")
                    
                    # If streams exist and mesh state changed, recreate them
                    if streaming_active and created_streams:
                        include_mesh = any(mesh_enabled_per_camera.values())
                        
                        # Check each existing stream
                        for pid in list(created_streams):
                            if pid in helper.streams:
                                stream_info = helper.streams[pid]
                                current_include_mesh = stream_info.get('include_mesh', False)
                                
                                # Only recreate if mesh setting actually changed
                                if current_include_mesh != include_mesh:
                                    print(f"[LSL Process] Recreating stream for {pid} with mesh={'enabled' if include_mesh else 'disabled'}")
                                    
                                    # Close and delete the old stream
                                    try:
                                        stream_info['outlet'].close()
                                        del helper.streams[pid]
                                    except Exception as e:
                                        print(f"[LSL Process] Error closing stream: {e}")
                                    
                                    # Small delay to ensure LSL cleanup
                                    time.sleep(0.1)
                                    
                                    # Create new stream with updated mesh setting
                                    helper.create_face_stream(pid, include_mesh=include_mesh)
                                    
                                    # Also recreate pose stream if it exists
                                    if pid in helper.pose_streams:
                                        helper.create_pose_stream(pid, fps)
                
                elif data['type'] == 'participant_data':
                    pid = data['participant_id']
                    
                    # Dynamically create stream if it doesn't exist and we're streaming
                    if streaming_active and pid not in created_streams and len(created_streams) < max_participants:
                        # Check if mesh is enabled for any camera
                        include_mesh = any(mesh_enabled_per_camera.values())
                        print(f"[LSL Process] Dynamically creating stream for {pid} (mesh={'enabled' if include_mesh else 'disabled'})")
                        helper.create_face_stream(pid, include_mesh=include_mesh)
                        created_streams.add(pid)
                    
                    # Push to LSL if stream exists
                    if pid in helper.streams:
                        helper.push_face_data(
                            pid,
                            data['blend_scores'],
                            data.get('mesh_data'),
                        )
                        
                        # Store for correlation
                        participant_scores[pid] = np.array(data['blend_scores'])
                        
                        # Count data for performance monitoring
                        if pid not in data_counts:
                            data_counts[pid] = 0
                        data_counts[pid] += 1
                    
                    # Calculate correlation if we have 2+ participants
                    if correlator_stream_active and len(participant_scores) >= 2:
                        # Use the first two participants (sorted by name for consistency)
                        pids = sorted(participant_scores.keys())[:2]
                        corr = correlator.update(
                            participant_scores[pids[0]],
                            participant_scores[pids[1]]
                        )
                        if corr is not None:
                            if corr_array is not None:
                                corr_array[:] = corr
                            if correlator.outlet:
                                correlator.outlet.push_sample(corr.tolist())
                
                elif data['type'] == 'pose_data':
                    pid = data['participant_id']
                    
                    # Dynamically create pose stream if needed
                    if streaming_active and pid not in helper.pose_streams:
                        print(f"[LSL Process] Dynamically creating pose stream for {pid}")
                        helper.create_pose_stream(pid, fps)
                    
                    if pid in helper.pose_streams:
                        helper.push_pose_data(pid, data['pose_data'])

        except Exception as e:
            print(f"[LSL Process] Error processing data: {e}")
            import traceback
            traceback.print_exc()
        
        # Performance reporting
        current_time = time.time()
        if current_time - last_fps_report >= fps_report_interval:
            # Get stream FPS stats
            stream_fps = helper.get_fps_stats()
            
            # Report performance
            if data_counts or participant_scores:
                print("\n[LSL Process] Performance Report:")
                print(f"  Active participants: {list(participant_scores.keys())}")
                print(f"  Mesh enabled cameras: {[c for c, enabled in mesh_enabled_per_camera.items() if enabled]}")
                
                for pid, count in data_counts.items():
                    actual_fps = count / fps_report_interval
                    stream_fps_val = stream_fps.get(pid, 0)
                    print(f"  {pid}: {actual_fps:.1f} FPS received, {stream_fps_val:.1f} FPS pushed")
                
                if correlator_stream_active and len(participant_scores) >= 2:
                    pids = sorted(participant_scores.keys())[:2]
                    print(f"  Comodulation: Active for {pids}")
                
                # Reset counters
                data_counts.clear()
                last_fps_report = current_time
        
        # Small sleep to prevent CPU spinning
        if not data_processed:
            time.sleep(0.001)
    
    # Cleanup
    helper.close_all_streams()
    helper.close_all_pose_streams()
    correlator.close()
    if corr_buffer:
        corr_buffer.close()
        
    print("[LSL Process] Stopped")