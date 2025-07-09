import numpy as np
import multiprocessing as mp
from multiprocessing import shared_memory
import queue
import time
from typing import Dict, List, Optional, Tuple
import pickle
import os
import threading
from dataclasses import dataclass
import cv2
import multiprocessing

@dataclass
class FaceEmbedding:
    """Container for face embedding data."""
    participant_id: int
    embedding: np.ndarray
    quality_score: float
    timestamp: float
    frame_count: int

class FaceRecognitionProcess:
    def __init__(self,
                 model_path: str,
                 embedding_dim: int = 512,
                 max_embeddings_per_person: int = 50,
                 similarity_threshold: float = 0.5,
                 update_threshold: float = 0.7):
        """
        Face recognition process using ArcFace embeddings.
        
        Args:
            model_path: Path to ArcFace ONNX model
            embedding_dim: Dimension of face embeddings
            max_embeddings_per_person: Maximum embeddings to store per person
            similarity_threshold: Threshold for face matching
            update_threshold: Threshold for updating embeddings
        """
        self.model_path = model_path
        self.embedding_dim = embedding_dim
        self.max_embeddings_per_person = max_embeddings_per_person
        self.similarity_threshold = similarity_threshold
        self.update_threshold = update_threshold
        
        # Thread management (changed from process to avoid daemon issues)
        self.thread = None
        self.input_queue = queue.Queue(maxsize=100)
        self.output_queue = queue.Queue(maxsize=100)
        self.command_queue = queue.Queue()
        self.is_running = False
        
        # Shared memory for embeddings database
        self.embeddings_shm = None
        self.embeddings_shape = (100, max_embeddings_per_person, embedding_dim)
        self.embeddings_db = None
        
        # Local cache
        self.local_embeddings = {}
        self.participant_mapping = {}
        
    def start(self):
        """Start the face recognition thread."""
        if self.thread is None or not self.thread.is_alive():
            # Initialize shared memory
            self._init_shared_memory()
            
            # Start thread
            self.is_running = True
            self.thread = threading.Thread(
                target=self._recognition_loop_thread,
                args=()
            )
            self.thread.daemon = True
            self.thread.start()
            print("[FaceRecognition] Recognition thread started")
    
    def stop(self):
        """Stop the face recognition thread."""
        if self.thread and self.thread.is_alive():
            self.is_running = False
            self.command_queue.put(('stop', None))
            self.thread.join(timeout=2.0)
        
        # Clean up shared memory
        if self.embeddings_shm:
            self.embeddings_shm.close()
            self.embeddings_shm.unlink()
    
    def submit_face(self, face_roi: np.ndarray, track_id: int, frame_id: int, quality_score: float):
        """Submit a face for recognition."""
        try:
            self.input_queue.put_nowait({
                'face_roi': face_roi,
                'track_id': track_id,
                'frame_id': frame_id,
                'quality_score': quality_score
            })
        except queue.Full:
            pass
    
    def get_recognition_result(self, timeout: float = 0.001) -> Optional[Dict]:
        """Get recognition results."""
        try:
            return self.output_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def enroll_participant(self, participant_id: int, face_rois: List[np.ndarray]):
        """Enroll a new participant with face samples."""
        self.command_queue.put(('enroll', {
            'participant_id': participant_id,
            'face_rois': face_rois
        }))
    
    def remove_participant(self, participant_id: int):
        """Remove a participant from the database."""
        self.command_queue.put(('remove', participant_id))
    
    def update_participant(self, participant_id: int, new_embedding: np.ndarray, quality_score: float):
        """Update participant embeddings with new high-quality sample."""
        self.command_queue.put(('update', {
            'participant_id': participant_id,
            'embedding': new_embedding,
            'quality_score': quality_score
        }))
    
    def _init_shared_memory(self):
        """Initialize shared memory for embeddings database."""
        # Create shared memory
        size = np.zeros(self.embeddings_shape, dtype=np.float32).nbytes
        self.embeddings_shm = shared_memory.SharedMemory(create=True, size=size)
        
        # Initialize with zeros
        embeddings_array = np.ndarray(self.embeddings_shape, dtype=np.float32, 
                                     buffer=self.embeddings_shm.buf)
        embeddings_array[:] = 0
    
    def _recognition_loop_thread(self):
        """Main recognition loop running in thread."""
        # Initialize model
        model = self._init_arcface_model(self.model_path)
        
        # Use the shared memory we already initialized
        embeddings_db = np.ndarray(self.embeddings_shape, dtype=np.float32, 
                                  buffer=self.embeddings_shm.buf)
        
        # Local state
        participant_embeddings = {}  # participant_id -> list of embeddings
        participant_counts = {}      # participant_id -> count of valid embeddings
        next_participant_id = 0
        
        while self.is_running:
            # Check for commands
            try:
                cmd, data = self.command_queue.get_nowait()
                if cmd == 'stop':
                    self.is_running = False
                    continue
                elif cmd == 'enroll':
                    # Process enrollment
                    pid = data['participant_id']
                    face_rois = data['face_rois']
                    embeddings = []
                    
                    for roi in face_rois[:10]:  # Limit initial enrollment
                        emb = self._extract_embedding(model, roi)
                        if emb is not None:
                            embeddings.append(emb)
                    
                    if embeddings:
                        participant_embeddings[pid] = embeddings
                        participant_counts[pid] = len(embeddings)
                        # Update shared memory
                        embeddings_db[pid, :len(embeddings)] = np.array(embeddings)
                        
                elif cmd == 'remove':
                    pid = data
                    if pid in participant_embeddings:
                        del participant_embeddings[pid]
                        del participant_counts[pid]
                        embeddings_db[pid] = 0
                        
                elif cmd == 'update':
                    pid = data['participant_id']
                    new_emb = data['embedding']
                    quality = data['quality_score']
                    
                    if pid in participant_embeddings and quality > update_threshold:
                        # Add or replace embedding
                        if len(participant_embeddings[pid]) < shm_shape[1]:
                            participant_embeddings[pid].append(new_emb)
                        else:
                            # Replace lowest quality embedding
                            participant_embeddings[pid][-1] = new_emb
                        
                        # Update shared memory
                        count = len(participant_embeddings[pid])
                        embeddings_db[pid, :count] = np.array(participant_embeddings[pid])
                        
            except queue.Empty:
                pass
            
            # Process face recognition requests
            try:
                data = self.input_queue.get(timeout=0.01)
                face_roi = data['face_roi']
                track_id = data['track_id']
                frame_id = data['frame_id']
                quality_score = data['quality_score']
                
                # Extract embedding
                embedding = self._extract_embedding(model, face_roi)
                
                if embedding is not None:
                    # Find best match
                    best_match_id = -1
                    best_similarity = 0
                    
                    for pid, embeddings in participant_embeddings.items():
                        if embeddings:
                            # Calculate average similarity
                            similarities = [
                                self._cosine_similarity(embedding, emb)
                                for emb in embeddings
                            ]
                            avg_similarity = np.mean(similarities)
                            
                            if avg_similarity > best_similarity and avg_similarity > self.similarity_threshold:
                                best_similarity = avg_similarity
                                best_match_id = pid
                    
                    # Output result
                    result = {
                        'track_id': track_id,
                        'frame_id': frame_id,
                        'participant_id': best_match_id,
                        'similarity': best_similarity,
                        'embedding': embedding,
                        'quality_score': quality_score
                    }
                    
                    self.output_queue.put(result)
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Recognition error: {e}")
        
        print("[FaceRecognition] Recognition thread stopped")
    
    def _init_arcface_model(self, model_path: str):
        """Initialize ArcFace model."""
        try:
            import onnxruntime as ort
            model = ort.InferenceSession(model_path)
            return model
        except:
            # Return mock model for testing
            return None
    
    def _extract_embedding(self, model, face_roi: np.ndarray) -> Optional[np.ndarray]:
        """Extract face embedding using ArcFace model."""
        if model is None:
            # Mock embedding for testing
            return np.random.randn(512).astype(np.float32)
        
        try:
            # Preprocess face
            if face_roi.shape[:2] != (112, 112):
                face_roi = cv2.resize(face_roi, (112, 112))
            
            # Normalize
            face_roi = face_roi.astype(np.float32)
            face_roi = (face_roi - 127.5) / 128.0
            
            # Add batch dimension and transpose to CHW
            face_roi = face_roi.transpose(2, 0, 1)
            face_roi = np.expand_dims(face_roi, axis=0)
            
            # Run inference
            input_name = model.get_inputs()[0].name
            output_name = model.get_outputs()[0].name
            embedding = model.run([output_name], {input_name: face_roi})[0]
            
            # Normalize embedding
            embedding = embedding.flatten()
            embedding = embedding / np.linalg.norm(embedding)
            
            return embedding.astype(np.float32)
            
        except Exception as e:
            print(f"Embedding extraction error: {e}")
            return None
    
    def _cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calculate cosine similarity between embeddings."""
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    
    def save_embeddings(self, filepath: str):
        """Save embeddings database to file."""
        # Get current embeddings from shared memory
        embeddings_array = np.ndarray(self.embeddings_shape, dtype=np.float32,
                                     buffer=self.embeddings_shm.buf)
        
        data = {
            'embeddings': embeddings_array.copy(),
            'participant_mapping': self.participant_mapping,
            'metadata': {
                'embedding_dim': self.embedding_dim,
                'max_embeddings_per_person': self.max_embeddings_per_person
            }
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    def load_embeddings(self, filepath: str):
        """Load embeddings database from file."""
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            # Update shared memory
            embeddings_array = np.ndarray(self.embeddings_shape, dtype=np.float32,
                                         buffer=self.embeddings_shm.buf)
            loaded_embeddings = data['embeddings']
            
            # Copy what fits
            min_shape = tuple(min(a, b) for a, b in zip(self.embeddings_shape, loaded_embeddings.shape))
            embeddings_array[:min_shape[0], :min_shape[1], :min_shape[2]] = \
                loaded_embeddings[:min_shape[0], :min_shape[1], :min_shape[2]]
            
            self.participant_mapping = data.get('participant_mapping', {})