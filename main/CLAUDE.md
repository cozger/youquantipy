# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

YouQuantiPy is a multimodal data capture and analysis system for real-time face detection, tracking, and analysis with Lab Streaming Layer (LSL) integration. It supports multiple cameras, multiple participants, and provides facial landmarks, blend shapes, and pose estimation data.

## Key Commands

### Running the Application
```bash
# Start the main GUI application
python gui.py

# Test RetinaFace detection standalone
python test_retinaface_standalone.py --camera 0 --confidence 0.3
python test_retinaface_standalone.py --video path/to/video.mp4
python test_retinaface_standalone.py --image path/to/image.jpg
```

### Development Tasks
Since the project lacks formal build tools, common tasks are performed manually:
- **Install Dependencies**: No requirements.txt exists. Key dependencies:
  - `opencv-python` (computer vision)
  - `numpy` (array operations)
  - `onnxruntime` (neural network inference)
  - `mediapipe` (facial landmarks and pose)
  - `pylsl` (Lab Streaming Layer)
  - `tkinter` (GUI framework)
  - `pillow` (image processing)
  - `pygrabber` (DirectShow camera enumeration)
- **Run Tests**: Execute test scripts directly (e.g., `python test_retinaface_standalone.py`)
- **No Linting/Formatting**: No automated tools configured

## Architecture Overview

The system uses RetinaFace for face detection and ArcFace for face recognition, with MediaPipe for facial landmarks and pose estimation. Files with `_unified.py` suffix are the current implementation (the suffix is historical).

### Multi-Process Architecture
1. **Main Process** (`gui.py`): Tkinter GUI managing the entire system
2. **Worker Processes** (`parallelworker_unified.py`): One per camera for video processing
3. **LSL Process** (`LSLHelper.py`): Manages all Lab Streaming Layer outputs
4. **Recording Processes**: Separate processes for video/audio recording
5. **Correlation Process**: Real-time correlation analysis between participants

### Communication Patterns
- **Multiprocessing Queues**: Frame data, scores, participant updates
- **Pipes**: Bidirectional control communication
- **Shared Memory** (`sharedbuffer.py`): Real-time data like blend scores
- **Global Participant Manager**: Maintains consistent IDs across cameras

### Core Module Responsibilities

#### Detection Pipeline
- `retinaface_detector.py`: Face detection using ONNX models (see FACEDETECTOR.md for detailed specs)
- `face_recognition_process.py`: ArcFace embeddings for face recognition
- `roi_processor.py`: Region of Interest processing for detection
- `lightweight_tracker.py`: Fast face tracking across frames
- `enrollment_manager.py`: Face enrollment for recognition

#### Worker System
- `parallelworker_unified.py`: Main worker handling camera capture and processing
- `landmark_worker_pool_unified.py`: Pool of workers for landmark processing
- `participantmanager_unified.py`: Global participant ID management

#### Data Output
- `LSLHelper.py`: All LSL stream management (scores, landmarks, tracking)
- `videorecorder.py`: Video recording with codec support
- `audiorecorder.py`: Audio recording with device selection
- `correlator.py`: Real-time correlation calculations

#### GUI Components
- `gui.py`: Main application window and control logic
- `canvasdrawing.py`: Optimized overlay rendering
- `guireliability.py`: System monitoring (memory, queues, performance)

### Configuration System
- Main config: `youquantipy_config.json`
- Key sections: video_recording, camera_settings, startup_mode, advanced_detection, audio_recording
- Model paths (typically in parent directory):
  - `face_landmarker.task` (MediaPipe facial landmarks)
  - `pose_landmarker_heavy.task` (MediaPipe pose estimation)
  - `retinaface.onnx` (RetinaFace face detection)
  - `arcface.onnx` (ArcFace face recognition)
- Configuration handled by `confighandler.py`

### Common Development Patterns

#### Adding New Features
1. **New Detection Method**: Create detector class similar to `retinaface_detector.py`
2. **New LSL Stream**: Add to `LSLHelper.py` with proper stream info
3. **New UI Control**: Add to `gui.py` control panel layout
4. **New Configuration**: Add to JSON config and `confighandler.py`

#### Debugging
- Many modules have `DEBUG_` flags for verbose output
- Use `diagnose_pipeline()` in GUI for system status
- Monitor reliability panel for performance metrics
- Check console output for module-specific debug info

#### Performance Tuning
- Adjust `max_detection_workers` and `landmark_worker_count` in config
- Tune `tile_size` for detection performance vs accuracy
- Configure `detection_confidence` and `tracking_confidence` thresholds
- Enable/disable features: mesh data, pose estimation, recognition

### Critical Implementation Notes

1. **Process Cleanup**: Always ensure proper cleanup of processes and shared memory
2. **Queue Management**: Use throttling to prevent queue overflow
3. **Camera Handling**: Robust error handling for camera disconnections
4. **Model Loading**: Verify model paths before starting workers
5. **Module Testing**: Ensure all detection and tracking features work correctly
6. **Memory Leaks**: Monitor with reliability system, implement proper cleanup

### Testing Approach
- No formal test framework; tests are standalone scripts
- Test with different camera counts and participant counts
- Verify LSL output with external LSL tools
- Check recording outputs for synchronization

### Known Issues
- No dependency management (requirements.txt)
- No automated testing or CI/CD
- Limited error recovery in some edge cases
- Performance varies with participant count and enabled features