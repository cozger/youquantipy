# Bug Fixes TODO

This document tracks known issues and bugs that need to be addressed in the YouQuantiPy system.

## Issue 1: Face Landmarks Not Drawing
**Status**: Fixed (Pending Verification)  
**Priority**: High  
**Description**: The face landmarks are not drawn in the GUI overlay, despite the RetinaFace debug rectangles correctly being drawn around the target face.

**Symptoms**:
- RetinaFace successfully detects faces (debug rectangles visible)
- Face Worker receives detections
- ~~Face Worker reports "0 tracked objects" and "0 faces"~~ (Fixed - now tracking works)
- No face landmarks are drawin the GUI

**Root Cause**: 
Multiple issues in the detection-tracking-drawing pipeline:
1. ~~Color space mismatch in LightweightTracker~~ (Fixed)
2. ~~Detection timing issue~~ (Fixed) 
3. ~~Missing 'last_seen' key in drawing cache~~ (Fixed)

**Fixes Applied**:
1. Changed `cv2.COLOR_BGR2GRAY` to `cv2.COLOR_RGB2GRAY` in lightweight_tracker.py
2. Improved detection retrieval logic in parallelworker_unified.py:
   - Added check for `latest_detection_time > 0` to handle startup case
   - Uses most recent detections within 0.5 second window
   - Added debug logging for detection age tracking
3. Enhanced debug logging in lightweight_tracker.py:
   - Logs incoming detections to update() method
   - Logs matching results between detections and tracks
   - Logs track creation events
4. Fixed KeyError in canvasdrawing.py:
   - Added backward compatibility check for 'last_seen' cache key
   - Ensures cache['last_seen'] is initialized if missing

**Files Modified**:
- `/mnt/d/Projects/youquantipy/main/lightweight_tracker.py`
- `/mnt/d/Projects/youquantipy/main/parallelworker_unified.py`
- `/mnt/d/Projects/youquantipy/main/canvasdrawing.py`

**Verification**:
Latest logs show tracking is working:
- `[TRACKER] Created new track 0 at frame 4` - Tracks being created
- `[TRACKER] Matching: 1 matched, 0 unmatched dets` - Detections matching to tracks
- `[FUSION DEBUG] Sending preview: 1 faces, 1 poses` - Faces sent to GUI

## Issue 2: GUI Freeze After Brief Operation
**Status**: Fixed  
**Priority**: High  
**Description**: After a brief period of streaming video in the GUI, the entire GUI/overlays freeze. The terminal still stays responsive.

**Symptoms**:
- Camera preview visualizations freeze (GUI buttons remain responsive)
- Terminal continues to show activity
- Overlays stop updating

**Root Cause**: 
In enhanced mode, the GUI was receiving full resolution frames (1920x1080, 4K) instead of downsampled 640x480 frames. This caused:
1. Memory pressure from creating PhotoImage objects from high-res frames repeatedly
2. CPU overload from canvas operations on large images
3. GUI event loop blocking due to slow frame rendering

**Additional Issues Found**:
1. Typos in gui.py: Extra spaces in `self.drawingmanager .method()` calls
2. Incorrect comment claiming frames were "Already downsampled by distributor"

**Fixes Applied**:
1. Fixed typos in gui.py (removed extra spaces in drawingmanager method calls)
2. Added proper frame downsampling in fusion process before sending to GUI preview:
   - Frames are downsampled to max 640x480 while preserving aspect ratio
   - Uses cv2.INTER_LINEAR interpolation for smooth downsampling
   - Only downsamples if frame exceeds target dimensions

**Files Modified**:
- `/mnt/d/Projects/youquantipy/main/gui.py` (lines 1192, 1226, 1247, 1257, 1288)
- `/mnt/d/Projects/youquantipy/main/parallelworker_unified.py` (lines 1331-1355)

**Verification**:
- Enhanced mode now sends properly downsampled frames to GUI
- GUI receives consistent 640x480 max resolution frames regardless of mode
- Frame rendering should be fast enough to prevent GUI freezing
