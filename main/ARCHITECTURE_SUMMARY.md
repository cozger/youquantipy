# YouQuantiPy Architecture Summary

## Overview
YouQuantiPy uses a dual-architecture system that automatically selects the appropriate mode based on configuration:
- **Standard Mode**: For normal resolution video without face recognition
- **Enhanced Mode**: For high-resolution video with face recognition capabilities

## Active Components

### Core Infrastructure (Always Active)
| Module | Purpose |
|--------|---------|
| `gui.py` | Main application entry point and GUI management |
| `confighandler.py` | Configuration loading and management |
| `tracker.py` | Procrustes-based multi-participant tracking |
| `sharedbuffer.py` | Shared memory for efficient frame passing |
| `canvasdrawing.py` | GUI overlay rendering |
| `LSLHelper.py` | Lab Streaming Layer data output |
| `correlator.py` | Real-time correlation calculations |
| `videorecorder.py` | Video recording functionality |
| `audiorecorder.py` | Audio recording functionality |
| `guireliability.py` | GUI reliability monitoring |

### Standard Mode Components
| Module | Purpose |
|--------|---------|
| `parallelworker.py` | Basic parallel processing workers |
| `participantmanager.py` | Standard participant state management |

### Enhanced Mode Components (Active when face recognition enabled)
| Module | Purpose |
|--------|---------|
| `parallelworker_advanced.py` | Advanced worker with face recognition |
| `participantmanager_advanced.py` | Enhanced participant manager |
| `retinaface_detector.py` | High-resolution face detection |
| `lightweight_tracker.py` | Optical flow-based tracking |
| `roi_processor.py` | Region of interest extraction |
| `face_recognition_process.py` | ArcFace embedding generation |
| `enrollment_manager.py` | Progressive identity enrollment |
| `landmark_worker_pool.py` | Parallel landmark detection |

### Integration Components
| Module | Purpose |
|--------|---------|
| `advanced_detection_integration.py` | Automatic architecture selection |
| `parallelworker_integration.py` | Worker factory and compatibility layer |

## Data Flow

### Standard Mode
```
Camera → ParallelWorker → MediaPipe → Tracker → GUI/LSL
```

### Enhanced Mode
```
Camera → RetinaFace → Tracker → ROI → LandmarkPool → FaceRecognition → GUI/LSL
```

## Configuration

Key settings in `youquantipy_config.json`:
```json
{
  "startup_mode": {
    "enable_face_recognition": true/false  // Switches between modes
  },
  "advanced_detection": {
    "retinaface_model": "path/to/retinaface.onnx",
    "arcface_model": "path/to/arcface.onnx"
  }
}
```

## Performance Notes

1. **Worker Pool**: 4 landmark detection workers (fixed)
2. **Frame Processing**: Every other frame sent to preview (performance)
3. **Queue Management**: Automatic overflow handling
4. **Track Limiting**: Maximum 2x participant count
5. **Detection Interval**: Configurable (default every 7 frames)

## Common Operations

### Start Application
```bash
python gui.py
```

### Enable Face Recognition
1. Set `enable_face_recognition: true` in config
2. Ensure model paths are correct
3. Restart application

### Debug Issues
- Check console for worker stats
- Monitor participant IDs (-1 = not enrolled)
- Verify model files exist
- Check queue overflow warnings