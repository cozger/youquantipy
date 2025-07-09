# Unified Architecture Summary

This document describes the unified module architecture that merges the standard and enhanced/advanced versions of YouQuantiPy modules into single, mode-aware implementations.

## Overview

The unified architecture eliminates duplicate modules by creating single implementations that can operate in either "standard" or "enhanced" mode based on configuration. This simplifies maintenance and ensures consistency across modes.

## Unified Modules Created

### 1. `participantmanager_unified.py`
**Replaces:**
- `participantmanager.py` (standard)
- `participantmanager_advanced.py` (enhanced)

**Key Features:**
- Single `GlobalParticipantManager` class that adapts based on `enable_recognition` parameter
- Standard mode: Uses Procrustes shape analysis and position tracking
- Enhanced mode: Adds face recognition embeddings and enrollment management
- Backward compatible with both original APIs

**Usage:**
```python
from participantmanager_unified import GlobalParticipantManager

# Standard mode
manager = GlobalParticipantManager(max_participants=2, enable_recognition=False)

# Enhanced mode  
manager = GlobalParticipantManager(max_participants=10, enable_recognition=True)
```

### 2. `parallelworker_unified.py`
**Replaces:**
- `parallelworker.py` (standard)
- `parallelworker_advanced.py` (enhanced)
- Various wrapper and integration modules

**Key Features:**
- Unified `parallel_participant_worker` function with `enhanced_mode` parameter
- Standard mode: Direct MediaPipe detection on full frames
- Enhanced mode: RetinaFace tiled detection → tracking → ROI extraction → landmarks → face recognition
- Shared camera initialization and frame distribution logic
- Unified face/pose worker processes with mode-aware processing

**Usage:**
```python
from parallelworker_unified import parallel_participant_worker

# Standard mode
parallel_participant_worker(..., enhanced_mode=False)

# Enhanced mode
parallel_participant_worker(..., enhanced_mode=True, 
                          retinaface_model_path="path/to/model",
                          arcface_model_path="path/to/model")
```

### 3. `landmark_worker_pool_unified.py`
**Replaces:**
- `landmark_worker_pool.py` (standard)
- `experimental_enhanced/landmark_worker_pool_adaptive.py` (enhanced)

**Key Features:**
- Single `LandmarkWorkerPool` class with `adaptive_mode` parameter
- Standard mode: Basic worker pool with synchronous processing
- Adaptive mode: Enhanced features like callbacks, better idle management
- Unified `landmark_worker_process` that handles both modes

**Usage:**
```python
from landmark_worker_pool_unified import LandmarkWorkerPool

# Standard mode
pool = LandmarkWorkerPool(num_workers=4, adaptive_mode=False)

# Adaptive mode
pool = LandmarkWorkerPool(num_workers=4, adaptive_mode=True)
```

### 4. Updated `advanced_detection_integration.py`
**Changes:**
- Now imports and uses unified modules
- `create_participant_manager()` returns unified manager with appropriate mode
- `parallel_participant_worker_auto()` uses unified worker with configuration-based mode selection

## Architecture Benefits

1. **Single Source of Truth**: Each component has one implementation that handles both modes
2. **Easier Maintenance**: Bug fixes and improvements apply to both modes automatically
3. **Consistent APIs**: Same interfaces work for both standard and enhanced modes
4. **Reduced Code Duplication**: Shared logic is implemented once
5. **Seamless Mode Switching**: Applications can switch modes without code changes
6. **Backward Compatibility**: Existing code continues to work with minimal changes

## Mode Selection

The system automatically selects the appropriate mode based on:

1. **Configuration File** (`youquantipy_config.json`):
   - `startup_mode.enable_face_recognition`: Controls enhanced mode
   
2. **Model Availability**:
   - Enhanced mode requires RetinaFace and ArcFace models
   - Falls back to standard if models not found

3. **Environment Variables**:
   - `USE_ENHANCED_ARCHITECTURE`: Can override configuration

## Migration Guide

### For GUI/Application Code
No changes needed if using `advanced_detection_integration.py`. The integration layer automatically uses unified modules.

### For Direct Module Usage
Replace imports:
```python
# Old
from participantmanager import GlobalParticipantManager
from participantmanager_advanced import GlobalParticipantManagerAdvanced

# New
from participantmanager_unified import GlobalParticipantManager
# Use enable_recognition parameter to control mode
```

## Technical Details

### Mode-Aware Design Pattern
Each unified module follows this pattern:
1. Accept mode parameter in constructor/function
2. Initialize mode-specific components conditionally
3. Use mode checks in processing logic
4. Provide unified API regardless of mode

### Shared Components
- Camera initialization and frame capture
- Queue management and multiprocessing
- Basic data structures and interfaces
- Performance monitoring and statistics

### Mode-Specific Components
- **Standard**: MediaPipe-only processing
- **Enhanced**: RetinaFace, tracking, ROI extraction, face recognition

## Future Improvements

1. **Dynamic Mode Switching**: Allow mode changes without restart
2. **Hybrid Mode**: Use enhanced features selectively based on resolution
3. **Performance Profiling**: Automatic mode selection based on hardware
4. **Further Unification**: Merge remaining duplicate modules (ROI processor, etc.)

## Summary

The unified architecture successfully merges the dual-mode system into single, cohesive modules while maintaining all functionality. This provides a cleaner, more maintainable codebase that can seamlessly adapt to different use cases and hardware capabilities.