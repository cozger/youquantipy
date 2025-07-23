# Landmark Coordinate Transformation Fix Summary

## Problem Identified
The landmarks were being returned in ROI (Region of Interest) space coordinates (256x256) instead of being transformed back to the original frame coordinates. This would cause landmarks to appear in the wrong positions when overlaid on the original video frame.

## Root Cause
In `gpu_face_processor.py`, the `_extract_rois_gpu` function was extracting and resizing face regions but not storing the transformation parameters needed to map landmarks back to frame coordinates.

## Fix Applied

### 1. Store ROI Transformation Parameters
Modified `_extract_rois_gpu` to store transformation info for each ROI:
```python
# Store transformation info for this ROI
roi_w = x2_padded - x1_padded
roi_h = y2_padded - y1_padded
self.roi_transforms.append({
    'x1': x1_padded,
    'y1': y1_padded,
    'scale_x': roi_w / 256.0,
    'scale_y': roi_h / 256.0
})
```

### 2. Add Transformation Method
Added `_transform_landmarks_to_frame` method to transform landmarks from ROI space to frame space:
```python
def _transform_landmarks_to_frame(self, landmarks: np.ndarray, transform: Dict) -> np.ndarray:
    """
    Transform landmarks from ROI space (256x256) to original frame space.
    """
    transformed = landmarks.copy()
    
    # Apply scale and translation to x and y coordinates
    transformed[:, 0] = landmarks[:, 0] * transform['scale_x'] + transform['x1']
    transformed[:, 1] = landmarks[:, 1] * transform['scale_y'] + transform['y1']
    
    # Z coordinate remains unchanged
    return transformed
```

### 3. Apply Transformation Before Output
Modified the result building section to transform landmarks before returning:
```python
# Transform landmarks from ROI space to frame space
transformed_landmarks = self._transform_landmarks_to_frame(
    landmarks[i], self.roi_transforms[i]
)

faces.append({
    'id': i + 1,
    'bbox': detections[i].bbox.tolist(),
    'confidence': float(detections[i].confidence),
    'landmarks': transformed_landmarks.tolist(),  # Now in frame coordinates
    'centroid': self._compute_centroid(transformed_landmarks)
})
```

## How the Transformation Works

1. **ROI Extraction**: When a face is detected, a padded bounding box is created (30% padding added)
2. **Resize**: The ROI is resized to 256x256 for the landmark model
3. **Landmark Detection**: The model outputs landmarks in 256x256 coordinate space
4. **Transformation**: Each landmark point is transformed back to frame space using:
   - `frame_x = roi_x * scale_x + offset_x`
   - `frame_y = roi_y * scale_y + offset_y`
   - Where `scale_x/y` is the ratio of padded bbox size to 256
   - And `offset_x/y` is the top-left corner of the padded bbox

## Testing
Created `test_landmark_transform.py` to verify the transformation logic with known test cases.

## Files Modified
- `/mnt/d/Projects/youquantipy/main/gpu_face_processor.py` - Main fix implementation

## Notes
- The `batch_roi_processor.py` already had similar transformation logic implemented correctly
- The fix ensures landmarks align properly with the original video frame for display and further processing
- Z-coordinates (depth) are preserved unchanged as they represent relative depth, not pixel positions