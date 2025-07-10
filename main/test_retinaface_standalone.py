#!/usr/bin/env python3
"""
Standalone RetinaFace Test Script
Tests RetinaFace detection on 1080p camera feed with live visualization.
"""

import cv2
import numpy as np
import time
import os
import sys
from typing import List, Dict, Tuple, Optional
import argparse

try:
    import onnxruntime as ort
except ImportError:
    print("ERROR: onnxruntime is not installed!")
    print("Please install it using: pip install onnxruntime")
    sys.exit(1)


class SimpleRetinaFaceDetector:
    """Simplified RetinaFace detector for testing."""
    
    def __init__(self, model_path: str, confidence_threshold: float = 0.98):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = 0.4
        
        # Model dimensions (standard RetinaFace)
        self.model_width = 640
        self.model_height = 608
        
        # Anchor parameters
        self.anchors = None
        self.variances = [0.1, 0.2]
        
        # Initialize model
        self._init_model()
        
    def _init_model(self):
        """Initialize ONNX model."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
        print(f"Loading RetinaFace model from: {self.model_path}")
        self.model = ort.InferenceSession(self.model_path)
        self.input_name = self.model.get_inputs()[0].name
        self.output_names = [out.name for out in self.model.get_outputs()]
        
        # Print model info
        input_info = self.model.get_inputs()[0]
        print(f"Model input: {self.input_name}, shape: {input_info.shape}")
        print(f"Model outputs: {[(out.name, out.shape) for out in self.model.get_outputs()]}")
        
        # Generate anchors
        self._generate_anchors()
        print(f"Generated {len(self.anchors)} anchors")
        
    def _generate_anchors(self):
        """Generate anchors for RetinaFace (15960 configuration)."""
        min_sizes = [[64, 96, 128], [128, 192], [256, 384], [384, 512, 640]]
        steps = [8, 16, 32, 64]
        
        anchors = []
        
        for k, step in enumerate(steps):
            feature_h = self.model_height // step
            feature_w = self.model_width // step
            
            for i in range(feature_h):
                for j in range(feature_w):
                    for min_size in min_sizes[k]:
                        cx = (j + 0.5) * step / self.model_width
                        cy = (i + 0.5) * step / self.model_height
                        w = min_size / self.model_width
                        h = min_size / self.model_height
                        anchors.append([cx, cy, w, h])
        
        self.anchors = np.array(anchors, dtype=np.float32)
        
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for RetinaFace."""
        # Resize to model input size
        img = cv2.resize(image, (self.model_width, self.model_height))
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Convert to float32 and normalize
        img = img.astype(np.float32)
        img -= (104, 117, 123)  # ImageNet mean subtraction
        
        # Add batch dimension (NHWC format)
        img = np.expand_dims(img, axis=0)
        
        return img
        
    def detect(self, image: np.ndarray) -> List[Dict]:
        """Run detection on image."""
        h, w = image.shape[:2]
        
        # Preprocess
        input_tensor = self.preprocess(image)
        
        # Run inference
        outputs = self.model.run(self.output_names, {self.input_name: input_tensor})
        
        # Parse detections
        detections = self._parse_detections(outputs, w, h)
        
        # Apply NMS
        final_detections = self._apply_nms(detections)
        
        return final_detections
        
    def _parse_detections(self, outputs: List[np.ndarray], img_w: int, img_h: int) -> List[Dict]:
        """Parse model outputs."""
        detections = []
        
        if len(outputs) >= 2:
            boxes = outputs[0]
            scores = outputs[1]
            
            # Remove batch dimension if present
            if len(boxes.shape) == 3:
                boxes = boxes[0]
            if len(scores.shape) == 3:
                scores = scores[0]
                
            # Apply softmax to scores
            if scores.shape[1] == 2:
                scores_exp = np.exp(scores - np.max(scores, axis=1, keepdims=True))
                scores_softmax = scores_exp / np.sum(scores_exp, axis=1, keepdims=True)
                scores = scores_softmax[:, 1]  # Face class
                
            # Scale factors
            scale_x = img_w / self.model_width
            scale_y = img_h / self.model_height
            
            # Process high-confidence detections
            for i in range(len(scores)):
                if scores[i] > self.confidence_threshold:
                    if i < len(self.anchors):
                        anchor = self.anchors[i]
                        box = boxes[i]
                        
                        # Decode box
                        cx = anchor[0] + box[0] * self.variances[0] * anchor[2]
                        cy = anchor[1] + box[1] * self.variances[0] * anchor[3]
                        w = anchor[2] * np.exp(box[2] * self.variances[1])
                        h = anchor[3] * np.exp(box[3] * self.variances[1])
                        
                        # Convert to corner format and scale
                        x1 = (cx - w / 2) * self.model_width * scale_x
                        y1 = (cy - h / 2) * self.model_height * scale_y
                        x2 = (cx + w / 2) * self.model_width * scale_x
                        y2 = (cy + h / 2) * self.model_height * scale_y
                        
                        # Clamp to image bounds
                        x1 = max(0, min(x1, img_w - 1))
                        y1 = max(0, min(y1, img_h - 1))
                        x2 = max(0, min(x2, img_w - 1))
                        y2 = max(0, min(y2, img_h - 1))
                        
                        # Filter by size and aspect ratio
                        face_w = x2 - x1
                        face_h = y2 - y1
                        
                        if face_w < 30 or face_h < 30:  # Too small
                            continue
                            
                        aspect_ratio = face_w / face_h if face_h > 0 else 0
                        if aspect_ratio < 0.5 or aspect_ratio > 2.0:  # Bad aspect ratio
                            continue
                            
                        detections.append({
                            'bbox': [x1, y1, x2, y2],
                            'confidence': float(scores[i])
                        })
                        
        return detections
        
    def _apply_nms(self, detections: List[Dict]) -> List[Dict]:
        """Apply Non-Maximum Suppression."""
        if not detections:
            return []
            
        boxes = np.array([d['bbox'] for d in detections])
        scores = np.array([d['confidence'] for d in detections])
        
        # NMS
        indices = self._nms(boxes, scores, self.nms_threshold)
        
        return [detections[i] for i in indices]
        
    def _nms(self, boxes: np.ndarray, scores: np.ndarray, threshold: float) -> List[int]:
        """Non-Maximum Suppression."""
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            
            ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            
            inds = np.where(ovr <= threshold)[0]
            order = order[inds + 1]
            
        return keep


def draw_detections(image: np.ndarray, detections: List[Dict], fps: float = 0) -> np.ndarray:
    """Draw bounding boxes and info on image."""
    img_copy = image.copy()
    
    # Draw each detection
    for i, det in enumerate(detections):
        bbox = det['bbox']
        conf = det['confidence']
        
        # Draw bounding box
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw confidence
        label = f"Face {conf:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        
        # Background for text
        cv2.rectangle(img_copy, (x1, y1 - label_size[1] - 4), 
                     (x1 + label_size[0], y1), (0, 255, 0), -1)
        
        # Text
        cv2.putText(img_copy, label, (x1, y1 - 2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # Draw FPS
    if fps > 0:
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(img_copy, fps_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Draw detection count
    count_text = f"Faces: {len(detections)}"
    cv2.putText(img_copy, count_text, (10, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    return img_copy


def main():
    parser = argparse.ArgumentParser(description="Test RetinaFace detection")
    parser.add_argument('--model', type=str, 
                       default='D:/Projects/youquantipy/retinaface.onnx',
                       help='Path to RetinaFace ONNX model')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera index (default: 0)')
    parser.add_argument('--video', type=str, default=None,
                       help='Video file path (overrides camera)')
    parser.add_argument('--image', type=str, default=None,
                       help='Image file path (overrides camera/video)')
    parser.add_argument('--confidence', type=float, default=0.98,
                       help='Confidence threshold (default: 0.98)')
    parser.add_argument('--width', type=int, default=1920,
                       help='Camera width (default: 1920 for 1080p)')
    parser.add_argument('--height', type=int, default=1080,
                       help='Camera height (default: 1080 for 1080p)')
    parser.add_argument('--save', type=str, default=None,
                       help='Save output video to file')
    
    args = parser.parse_args()
    
    # Initialize detector
    try:
        detector = SimpleRetinaFaceDetector(args.model, args.confidence)
    except Exception as e:
        print(f"Failed to initialize detector: {e}")
        return
    
    # Process image if provided
    if args.image:
        print(f"Processing image: {args.image}")
        image = cv2.imread(args.image)
        if image is None:
            print(f"Failed to load image: {args.image}")
            return
            
        # Detect
        start = time.time()
        detections = detector.detect(image)
        detect_time = time.time() - start
        
        print(f"Detection took {detect_time*1000:.1f}ms")
        print(f"Found {len(detections)} faces")
        
        # Draw and show
        result = draw_detections(image, detections)
        cv2.imshow('RetinaFace Detection', result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Save if requested
        if args.save:
            cv2.imwrite(args.save, result)
            print(f"Saved result to: {args.save}")
        return
    
    # Initialize video capture
    if args.video:
        cap = cv2.VideoCapture(args.video)
        print(f"Using video file: {args.video}")
    else:
        cap = cv2.VideoCapture(args.camera)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        print(f"Using camera {args.camera} at {args.width}x{args.height}")
    
    if not cap.isOpened():
        print("Failed to open video source")
        return
    
    # Get actual capture properties
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Actual capture: {actual_width}x{actual_height} @ {actual_fps:.1f} FPS")
    
    # Video writer for saving
    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.save, fourcc, 30.0, 
                               (actual_width, actual_height))
    
    # FPS tracking
    fps_history = []
    frame_count = 0
    
    print("\nStarting detection loop. Press 'q' to quit, 's' to save screenshot")
    
    try:
        while True:
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                if args.video:
                    print("End of video")
                    break
                print("Failed to capture frame")
                continue
            
            # Run detection
            start_time = time.time()
            detections = detector.detect(frame)
            detect_time = time.time() - start_time
            
            # Calculate FPS
            fps = 1.0 / detect_time if detect_time > 0 else 0
            fps_history.append(fps)
            if len(fps_history) > 30:
                fps_history.pop(0)
            avg_fps = np.mean(fps_history) if fps_history else 0
            
            # Draw results
            display_frame = draw_detections(frame, detections, avg_fps)
            
            # Show frame
            cv2.imshow('RetinaFace Test - Press Q to quit', display_frame)
            
            # Save video if requested
            if writer:
                writer.write(display_frame)
            
            # Handle key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                screenshot_name = f"retinaface_screenshot_{frame_count}.jpg"
                cv2.imwrite(screenshot_name, display_frame)
                print(f"Saved screenshot: {screenshot_name}")
            
            # Print stats periodically
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Frame {frame_count}: {len(detections)} faces, "
                      f"{avg_fps:.1f} FPS, {detect_time*1000:.1f}ms/frame")
                
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        
        print(f"\nProcessed {frame_count} frames")
        if fps_history:
            print(f"Average FPS: {np.mean(fps_history):.1f}")


if __name__ == "__main__":
    main()