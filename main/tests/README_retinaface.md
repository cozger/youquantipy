# RetinaFace Detection Test Scripts

This directory contains test scripts for the RetinaFace detection system used in YouQuantiPy's enhanced mode.

## Test Scripts

### 1. `test_retinaface_standalone.py`
A comprehensive test script with multiple modes:

```bash
# Test on a single image
python test_retinaface_standalone.py --mode image --input path/to/image.jpg --show

# Test on video file
python test_retinaface_standalone.py --mode video --input path/to/video.mp4 --show

# Test with camera
python test_retinaface_standalone.py --mode camera --camera 0 --show

# Benchmark performance
python test_retinaface_standalone.py --mode benchmark --input path/to/image.jpg
```

**Features:**
- Multiple test modes (image, video, camera, benchmark)
- Visualization of bounding boxes and landmarks
- Performance metrics and FPS tracking
- Debug mode to show raw detections before NMS
- Configurable detection parameters

**Command-line options:**
- `--mode`: Test mode (image/video/camera/benchmark)
- `--input`: Input file path
- `--output`: Output file path
- `--camera`: Camera index (default: 0)
- `--model-path`: Path to RetinaFace ONNX model
- `--tile-size`: Tile size for detection (default: 640)
- `--overlap`: Tile overlap ratio (default: 0.2)
- `--confidence`: Detection confidence threshold (default: 0.9)
- `--nms`: NMS IoU threshold (default: 0.4)
- `--workers`: Number of parallel workers (default: 4)
- `--show`: Display results in window
- `--show-landmarks`: Show facial landmarks
- `--debug`: Enable debug mode
- `--skip`: Process every N frames (for video)

### 2. `test_retinaface_simple.py`
A minimal test script for quick verification:

```bash
python test_retinaface_simple.py
```

**Features:**
- Creates synthetic test image with drawn faces
- Tests basic detection functionality
- Minimal dependencies and configuration
- Automatic test with real images if available

**Output files:**
- `test_input.jpg`: Generated test image
- `test_output.jpg`: Detection results
- `*_detected.jpg`: Results for real images

### 3. `test_retinaface_direct.py`
Direct test of detector initialization:

```bash
python test_retinaface_direct.py
```

**Purpose:**
- Tests RetinaFace detector creation
- Verifies anchor initialization
- Checks model loading

### 4. `test_retinaface_model.py`
Model-specific test script:

```bash
python tests/test_retinaface_model.py
```

**Purpose:**
- Tests model initialization
- Verifies detection on dummy frames
- Checks async processing pipeline

## RetinaFace Detection Pipeline

### How It Works

1. **Model Loading**:
   - Loads ONNX model using onnxruntime
   - Initializes anchors for detection (15,960 anchors for 640x608 input)
   - Sets up async processing threads

2. **Detection Process**:
   - Converts image from BGR to RGB
   - Applies preprocessing (ImageNet mean subtraction)
   - Runs inference to get boxes, scores, and landmarks
   - Decodes predictions using anchors
   - Applies NMS to remove duplicates

3. **Key Parameters**:
   - **Confidence threshold**: Minimum score for valid detection (0.9 default)
   - **NMS threshold**: IoU threshold for duplicate removal (0.4 default)
   - **Tile size**: Size of processing tiles for high-res images (640x608)
   - **Overlap**: Overlap between tiles to avoid missing faces at boundaries

### Understanding the Output

Each detection contains:
- **bbox**: Bounding box coordinates [x1, y1, x2, y2]
- **confidence**: Detection confidence score (0-1)
- **landmarks**: Optional facial landmarks (5 points)

### Common Issues and Solutions

1. **No detections found**:
   - Lower confidence threshold (try 0.5-0.7)
   - Check if image contains faces
   - Verify model file exists and is valid

2. **Too many detections**:
   - Increase confidence threshold
   - Adjust NMS threshold
   - Check for duplicate detections from tiling

3. **Slow performance**:
   - Reduce number of workers
   - Increase detection skip interval
   - Use smaller input resolution

4. **Model loading errors**:
   - Install onnxruntime: `pip install onnxruntime`
   - Verify model path is correct
   - Check model file integrity

## Model Files

The RetinaFace detection requires:
- **RetinaFace model**: `D:/Projects/youquantipy/retinaface.onnx`
  - Input: 640x608x3 RGB image
  - Output: Face bounding boxes, scores, and landmarks
  
Model can be downloaded from:
- [InsightFace Model Zoo](https://github.com/deepinsight/insightface/tree/master/model_zoo)
- [ONNX Model Hub](https://github.com/onnx/models)

## Integration with YouQuantiPy

RetinaFace is used in enhanced mode for:
1. High-resolution video processing (1080p, 2K, 4K)
2. Multi-face detection with identity tracking
3. Face recognition pipeline (with ArcFace)

The detector runs in a separate process and communicates via queues:
- Input: RGB frames
- Output: Face detections with bounding boxes
- Processing: Asynchronous with configurable intervals

## Performance Tips

1. **For real-time processing**:
   - Use detection interval (process every N frames)
   - Enable lightweight tracking between detections
   - Adjust tile size based on video resolution

2. **For accuracy**:
   - Use full-frame processing for smaller images
   - Enable tiling for high-resolution images
   - Fine-tune confidence and NMS thresholds

3. **For debugging**:
   - Use debug mode to see raw detections
   - Monitor anchor generation logs
   - Check detection statistics in console output