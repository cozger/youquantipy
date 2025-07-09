"""
Tiled Face Detection for 4K Video Processing
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
import time
from dataclasses import dataclass
from insightface.app import FaceAnalysis
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Face detection result"""
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2 in pixel coordinates
    confidence: float
    landmarks: Optional[np.ndarray] = None
    embedding: Optional[np.ndarray] = None


class TiledFaceDetector:
    """
    High-performance face detection for 4K video using tiled processing.
    """
    
    def __init__(self, 
                 tile_size: int = 640,
                 tile_overlap: float = 0.2,
                 detection_interval: int = 10,
                 detector_model: str = 'retinaface',
                 device: str = 'cpu',
                 max_workers: int = 4):
        
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self.detection_interval = detection_interval
        self.device = device
        self.max_workers = max_workers
        
        # Initialize face detector
        self.detector = self._init_detector(detector_model)
        
        # Thread pool for parallel tile processing
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Async detection thread
        self.detection_thread = None
        self.detection_queue = queue.Queue(maxsize=2)
        self.result_queue = queue.Queue(maxsize=2)
        self.running = False
        
        # Frame counter for interval detection
        self.frame_count = 0
        
        # Cache for last detection results
        self.last_detections = []
        
        logger.info(f"TiledFaceDetector initialized: tile_size={tile_size}, "
                   f"overlap={tile_overlap}, interval={detection_interval}")
    
    def _init_detector(self, model_name: str):
        """Initialize the face detection model"""
        try:
            app = FaceAnalysis(
                name='buffalo_l' if model_name == 'retinaface' else 'buffalo_sc',
                providers=['CUDAExecutionProvider'] if self.device == 'cuda' else ['CPUExecutionProvider']
            )
            app.prepare(ctx_id=0 if self.device == 'cuda' else -1, det_size=(640, 640))
            logger.info(f"Initialized {model_name} detector on {self.device}")
            return app
        except Exception as e:
            logger.error(f"Failed to initialize detector: {e}")
            return None
    
    def _generate_tiles(self, frame_shape: Tuple[int, int]) -> List[Tuple[int, int, int, int]]:
        """Generate overlapping tile coordinates for the frame."""
        height, width = frame_shape
        tiles = []
        
        stride = int(self.tile_size * (1 - self.tile_overlap))
        
        for y in range(0, height - self.tile_size + 1, stride):
            for x in range(0, width - self.tile_size + 1, stride):
                x2 = min(x + self.tile_size, width)
                y2 = min(y + self.tile_size, height)
                tiles.append((x, y, x2, y2))
        
        # Add edge tiles if needed
        if width % stride != 0:
            for y in range(0, height - self.tile_size + 1, stride):
                x1 = width - self.tile_size
                tiles.append((x1, y, width, min(y + self.tile_size, height)))
        
        if height % stride != 0:
            for x in range(0, width - self.tile_size + 1, stride):
                y1 = height - self.tile_size
                tiles.append((x, y1, min(x + self.tile_size, width), height))
        
        if width % stride != 0 and height % stride != 0:
            tiles.append((width - self.tile_size, height - self.tile_size, width, height))
        
        return tiles
    
    def _detect_faces_in_tile(self, frame: np.ndarray, tile_coords: Tuple[int, int, int, int]) -> List[Detection]:
        """Detect faces in a single tile."""
        x1, y1, x2, y2 = tile_coords
        tile = frame[y1:y2, x1:x2]
        
        detections = []
        
        if self.detector is not None:
            try:
                faces = self.detector.get(tile)
                
                for face in faces:
                    bbox = face.bbox
                    bbox_frame = (
                        bbox[0] + x1,
                        bbox[1] + y1,
                        bbox[2] + x1,
                        bbox[3] + y1
                    )
                    
                    landmarks = None
                    if hasattr(face, 'kps') and face.kps is not None:
                        landmarks = face.kps.copy()
                        landmarks[:, 0] += x1
                        landmarks[:, 1] += y1
                    
                    detection = Detection(
                        bbox=bbox_frame,
                        confidence=face.det_score,
                        landmarks=landmarks,
                        embedding=face.embedding if hasattr(face, 'embedding') else None
                    )
                    detections.append(detection)
                    
            except Exception as e:
                logger.debug(f"Detection failed on tile {tile_coords}: {e}")
        
        return detections
    
    def _global_nms(self, detections: List[Detection], iou_threshold: float = 0.5) -> List[Detection]:
        """Apply Non-Maximum Suppression globally to remove duplicates."""
        if not detections:
            return []
        
        boxes = np.array([d.bbox for d in detections])
        scores = np.array([d.confidence for d in detections])
        
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            if order.size == 1:
                break
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]
        
        return [detections[i] for i in keep]
    
    def detect_frame(self, frame: np.ndarray) -> List[Detection]:
        """Detect faces in a frame using tiled processing."""
        height, width = frame.shape[:2]
        
        tiles = self._generate_tiles((height, width))
        logger.debug(f"Processing {len(tiles)} tiles for {width}x{height} frame")
        
        all_detections = []
        futures = []
        
        for tile_coords in tiles:
            future = self.executor.submit(self._detect_faces_in_tile, frame, tile_coords)
            futures.append(future)
        
        for future in futures:
            try:
                tile_detections = future.result(timeout=1.0)
                all_detections.extend(tile_detections)
            except Exception as e:
                logger.error(f"Tile processing failed: {e}")
        
        filtered_detections = self._global_nms(all_detections)
        
        logger.debug(f"Detected {len(filtered_detections)} faces "
                    f"(from {len(all_detections)} before NMS)")
        
        return filtered_detections
    
    def _async_detection_loop(self):
        """Async detection thread main loop"""
        logger.info("Async detection thread started")
        
        while self.running:
            try:
                frame = self.detection_queue.get(timeout=0.1)
                
                start_time = time.time()
                detections = self.detect_frame(frame)
                detection_time = time.time() - start_time
                
                self.last_detections = detections
                
                try:
                    self.result_queue.put_nowait({
                        'detections': detections,
                        'timestamp': time.time(),
                        'detection_time': detection_time
                    })
                except queue.Full:
                    try:
                        self.result_queue.get_nowait()
                        self.result_queue.put_nowait({
                            'detections': detections,
                            'timestamp': time.time(),
                            'detection_time': detection_time
                        })
                    except:
                        pass
                
                logger.debug(f"Detection completed in {detection_time:.3f}s")
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Detection thread error: {e}")
        
        logger.info("Async detection thread stopped")
    
    def start_async(self):
        """Start async detection thread"""
        if self.running:
            return
        
        self.running = True
        self.detection_thread = threading.Thread(target=self._async_detection_loop, daemon=True)
        self.detection_thread.start()
        logger.info("Async detection started")
    
    def stop_async(self):
        """Stop async detection thread"""
        self.running = False
        if self.detection_thread:
            self.detection_thread.join(timeout=2.0)
        logger.info("Async detection stopped")
    
    def process_frame_async(self, frame: np.ndarray) -> Optional[List[Detection]]:
        """Process a frame asynchronously. Only runs detection every N frames."""
        self.frame_count += 1
        
        if self.frame_count % self.detection_interval == 0:
            try:
                self.detection_queue.put_nowait(frame)
                logger.debug(f"Queued frame {self.frame_count} for detection")
            except queue.Full:
                try:
                    self.detection_queue.get_nowait()
                    self.detection_queue.put_nowait(frame)
                except:
                    pass
        
        try:
            result = self.result_queue.get_nowait()
            return result['detections']
        except queue.Empty:
            return self.last_detections if self.last_detections else None
    
    def cleanup(self):
        """Clean up resources"""
        self.stop_async()
        self.executor.shutdown(wait=True)
        logger.info("TiledFaceDetector cleaned up")