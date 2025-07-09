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

### Installing Dependencies
```bash
pip install pillow opencv-python numpy mediapipe pygrabber pylsl scipy onnxruntime
```

### Enabling Enhanced Architecture
```bash
python use_enhanced_architecture.py
```

## Architecture Overview

### Architecture Selection
The system automatically selects the appropriate architecture based on configuration:
- If face recognition is enabled in `youquantipy_config.json` → Enhanced Mode
- Otherwise → Standard Mode

### Core Components (Both Modes)
- **gui.py**: Main entry point - Tkinter-based GUI that orchestrates the entire application
- **confighandler.py**: Configuration management system
- **tracker.py**: Unified tracking using Procrustes shape analysis and Hungarian assignment
- **participantmanager.py**: Global participant state management across cameras
- **sharedbuffer.py**: Efficient frame sharing between processes using shared memory
- **canvasdrawing.py**: GUI canvas drawing and overlay rendering
- **LSLHelper.py**: Lab Streaming Layer integration for real-time data output
- **correlator.py**: Real-time correlation calculation between participants

### Standard Mode Components
- **parallelworker.py**: Basic parallel processing for each participant
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
- **landmark_worker_pool_adaptive.py**: Adaptive multiprocessing pool for landmarks
- **face_recognition_process.py**: Separate process for ArcFace embeddings
- **enrollment_manager.py**: Progressive enrollment state machine
- **result_aggregator.py**: Aggregates results from multiple processing pipelines

#### Integration Layer
- **camera_worker_enhanced.py**: Main enhanced processing pipeline
- **parallelworker_advanced_enhanced.py**: Enhanced face worker process
- **parallelworker_enhanced_integration.py**: Backward compatibility wrapper
- **advanced_detection_integration.py**: Automatic mode selection

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

#### Proposed Detailed Dataflow
- This should be the dataflow:  Camera (4K capable)
      ↓
  FrameDistributor (30Hz)
      ├─[Full]→ Face Detection (RetinaFace, every 7 frames)
      │              ↓
      │         ROI Locations → Face Landmarks (continuous)
      │                              ↓
      ├─[480p]→ Pose Detection ──────┤
      │                              ↓
      └─[Frame]────────────────→ Fusion → GUI

### Key Design Patterns

1. **Multiprocessing Architecture**: Parallel processing with proper daemon handling
2. **Adaptive Worker Pools**: Fixed pool size with dynamic task distribution
3. **Queue Management**: Non-blocking operations with overflow protection
4. **Shared Memory Buffers**: Efficient frame sharing between processes
5. **Configuration-Driven**: All settings in `youquantipy_config.json`
6. **Automatic Mode Selection**: Seamless switching between architectures

### Configuration

Main configuration file: `youquantipy_config.json`

Key settings:
- `startup_mode.enable_face_recognition`: Enables enhanced mode
- `advanced_detection.retinaface_model`: Path to RetinaFace model
- `advanced_detection.arcface_model`: Path to ArcFace model
- `cameras`: Camera configuration and participant limits

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
3. **Track Limiting**: Prevents track explosion in unstable conditions
4. **Frame Skipping**: Preview updates every other frame for performance
5. **Worker Pool Size**: 4 landmark workers by default (configurable)

### Common Issues & Solutions

1. **No Face/Pose Overlays**
   - Check preview data format matches GUI expectations
   - Verify queue connections between workers and GUI

2. **Low FPS Warning**
   - Camera hardware limitation, not processing issue
   - Consider lower resolution or different camera

3. **"Daemonic processes" Error**
   - Enhanced worker processes are now non-daemon
   - Allows proper child process spawning

4. **Result Queue Full**
   - Increased timeout for result collection
   - Automatic queue clearing on overflow

### Debugging Tips

1. Enable debug logging by checking frame_count modulo prints
2. Monitor worker stats in console output
3. Check participant ID assignment (-1 means not enrolled yet)
4. Verify model files exist at specified paths

### Module Status

**Active Modules**: All files listed in Core, Standard, and Enhanced components

**Deprecated**: 
- `tileddetector.py` (replaced by retinaface_detector.py)

**Test Files**:
- `test_enhanced.py`
- `test_retinaface_model.py`