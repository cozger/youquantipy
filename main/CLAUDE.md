# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

YouQuantiPy is a real-time multi-participant face and pose tracking application with GUI, designed for quantitative behavioral analysis. It uses MediaPipe for detection, implements sophisticated tracking algorithms with Procrustes shape analysis, and supports Lab Streaming Layer (LSL) for real-time data streaming.

The application supports two architecture modes:
1. **Standard Mode**: Basic MediaPipe detection for normal resolution video
2. **Enhanced Mode**: Advanced detection with face recognition for high-resolution video (1080p, 2K, 4K)

## Commands

### Running the Application
```bash
python gui.py
```

### Running Tests
```bash
# Test standard mode
python test_standard_mode.py

# Test enhanced mode
python test_enhanced_current.py

# Run specific tests in tests/ directory
python tests/test_enhanced.py
python tests/test_retinaface_model.py
```

### Installing Dependencies
```bash
pip install pillow opencv-python numpy mediapipe pygrabber pylsl scipy onnxruntime
```

### Code Organization
```bash
# Organize deprecated and experimental files
python cleanup_codebase.py
```

## Architecture Overview

### Architecture Selection
The system automatically selects the appropriate architecture based on:
1. `enable_face_recognition` setting in `youquantipy_config.json`
2. Availability of required model files (RetinaFace and ArcFace ONNX models)
3. Graceful fallback to standard mode if enhanced requirements aren't met

### Core Components (Both Modes)
- **gui.py**: Main entry point - Tkinter-based GUI that orchestrates the entire application
- **confighandler.py**: Configuration management system
- **tracker.py**: Unified tracking using Procrustes shape analysis and Hungarian assignment
- **participantmanager_unified.py**: Global participant state management across cameras
- **sharedbuffer.py**: Efficient frame sharing between processes using shared memory
- **canvasdrawing.py**: GUI canvas drawing and overlay rendering
- **LSLHelper.py**: Lab Streaming Layer integration for real-time data output
- **correlator.py**: Real-time correlation calculation between participants
- **guireliability.py**: GUI performance monitoring and recovery
- **videorecorder.py**: Video recording functionality
- **audiorecorder.py**: Audio recording functionality

### Standard Mode Components
- **parallelworker_unified.py**: Unified parallel processing (standard mode path)
- Uses direct MediaPipe detection on full frames
- Simple face/pose tracking without identity recognition

### Enhanced Mode Components
Enhanced mode uses a sophisticated multi-tier architecture:

#### Detection & Tracking Layer
- **retinaface_detector.py**: Async tiled detection for high-resolution video
- **lightweight_tracker.py**: Optical flow tracking with drift correction
- **frame_router.py**: Intelligent frame routing and buffering
- **roi_manager.py**: ROI extraction and coordinate transformation

#### Processing Layer
- **landmark_worker_pool_unified.py**: Adaptive multiprocessing pool for landmarks
- **face_recognition_process.py**: Separate process for ArcFace embeddings
- **enrollment_manager.py**: Progressive enrollment state machine
- **result_aggregator.py**: Aggregates results from multiple processing pipelines

#### Integration Layer
- **camera_worker_enhanced.py**: Main enhanced processing pipeline
- **parallelworker_unified.py**: Unified worker (enhanced mode path)
- **advanced_detection_integration.py**: Automatic mode selection and factory

### Data Flow

#### Standard Mode Flow
```
Camera → Frame Distribution → Face/Pose Workers → Fusion Process → GUI/LSL
```

#### Enhanced Mode Flow
```
Camera → Frame Router → RetinaFace Detector ↘
                      ↓                       → Result Aggregator → GUI/LSL
                   Tracker → ROI Manager → Landmark Pool ↗
```

### Key Design Patterns

1. **Multiprocessing Architecture**: Parallel processing with proper daemon handling
2. **Adaptive Worker Pools**: Fixed pool size with dynamic task distribution
3. **Queue Management**: Non-blocking operations with overflow protection
4. **Shared Memory Buffers**: Efficient frame sharing between processes
5. **Configuration-Driven**: All settings in `youquantipy_config.json`
6. **Automatic Mode Selection**: Seamless switching between architectures
7. **Graceful Degradation**: Falls back to standard mode if enhanced requirements not met

### Configuration

Main configuration file: `youquantipy_config.json`

Key settings:
- `startup_mode.enable_face_recognition`: Enables enhanced mode
- `advanced_detection.retinaface_model`: Path to RetinaFace model
- `advanced_detection.arcface_model`: Path to ArcFace model
- `advanced_detection.landmark_worker_count`: Number of landmark workers (default: 4)
- `cameras`: Camera configuration and participant limits
- `camera_settings`: Resolution and FPS targets
- `video_recording` / `audio_recording`: Recording configurations

Runtime state file: `recording_state.json`
- Tracks current recording status and active cameras

### Model Requirements

Standard Mode:
- Face: `D:\Projects\youquantipy\face_landmarker.task`
- Pose: `D:\Projects\youquantipy\pose_landmarker_heavy.task`

Enhanced Mode (additional):
- RetinaFace: `D:\Projects\youquantipy\retinaface.onnx`
- ArcFace: `D:\Projects\youquantipy\arcface.onnx`

### Performance Considerations

1. **Camera FPS**: System handles low FPS cameras with warnings
2. **Queue Management**: Automatic overflow handling for smooth operation
3. **Track Limiting**: Prevents track explosion (max 2x participant count)
4. **Frame Skipping**: Preview updates every other frame for performance
5. **Worker Pool Size**: 4 landmark workers by default (configurable)
6. **Detection Intervals**: Configurable intervals for face detection in enhanced mode

### Common Issues & Solutions

1. **No Face/Pose Overlays**
   - Check preview data format matches GUI expectations
   - Verify queue connections between workers and GUI
   - Ensure models are properly loaded

2. **Low FPS Warning**
   - Camera hardware limitation, not processing issue
   - Consider lower resolution or different camera

3. **"Daemonic processes" Error**
   - Enhanced worker processes are now non-daemon
   - Allows proper child process spawning

4. **Result Queue Full**
   - Increased timeout for result collection
   - Automatic queue clearing on overflow

5. **Enhanced Mode Not Working**
   - Verify model files exist at specified paths
   - Check `enable_face_recognition` is true in config
   - Review console for model loading errors

### Debugging Tips

1. Enable debug logging by checking frame_count modulo prints
2. Monitor worker stats in console output
3. Check participant ID assignment (-1 means not enrolled yet)
4. Verify model files exist at specified paths
5. Use `analyze_enhanced_architecture.py` for architecture analysis
6. Check `recording_state.json` for runtime state

### Module Status

**Active Modules**: All files listed in Core, Standard, and Enhanced components

**Deprecated** (`deprecated/` directory): 
- Previous implementations of parallel workers and participant managers
- `tileddetector.py` (replaced by retinaface_detector.py)

**Experimental** (`experimental_enhanced/` directory):
- Alternative enhanced architecture implementations
- Use `use_enhanced_architecture.py` to enable

**Test Files**:
- `test_standard_mode.py`: Standard mode testing
- `test_enhanced_current.py`: Enhanced mode testing
- `tests/test_enhanced.py`: Additional enhanced tests
- `tests/test_retinaface_model.py`: Model-specific tests