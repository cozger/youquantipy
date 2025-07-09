import numpy as np
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import time
import threading
from collections import deque

class EnrollmentState(Enum):
    """Enrollment state machine states."""
    UNKNOWN = auto()          # No enrollment data
    COLLECTING = auto()       # Collecting initial samples
    VALIDATING = auto()       # Validating consistency
    ENROLLED = auto()         # Successfully enrolled
    IMPROVING = auto()        # Continuously improving embeddings
    FAILED = auto()          # Enrollment failed

@dataclass
class ParticipantEnrollment:
    """Enrollment data for a participant."""
    participant_id: int
    state: EnrollmentState = EnrollmentState.UNKNOWN
    embeddings: List[np.ndarray] = field(default_factory=list)
    quality_scores: List[float] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)
    track_ids: List[int] = field(default_factory=list)
    
    # Validation metrics
    consistency_score: float = 0.0
    stability_score: float = 0.0
    confidence_score: float = 0.0
    
    # State timing
    state_start_time: float = field(default_factory=time.time)
    enrollment_start_time: float = field(default_factory=time.time)
    
    # Statistics
    total_samples: int = 0
    accepted_samples: int = 0
    rejected_samples: int = 0
    
    def add_sample(self, embedding: np.ndarray, quality_score: float, track_id: int):
        """Add a new sample to enrollment."""
        self.embeddings.append(embedding)
        self.quality_scores.append(quality_score)
        self.timestamps.append(time.time())
        self.track_ids.append(track_id)
        self.total_samples += 1
        
    def compute_consistency(self) -> float:
        """Compute consistency score across embeddings."""
        if len(self.embeddings) < 2:
            return 0.0
            
        # Calculate pairwise similarities
        similarities = []
        for i in range(len(self.embeddings)):
            for j in range(i + 1, len(self.embeddings)):
                sim = np.dot(self.embeddings[i], self.embeddings[j])
                similarities.append(sim)
        
        # Consistency is mean similarity
        return np.mean(similarities) if similarities else 0.0
    
    def compute_stability(self) -> float:
        """Compute temporal stability of embeddings."""
        if len(self.embeddings) < 3:
            return 0.0
            
        # Calculate similarity between consecutive embeddings
        consecutive_sims = []
        for i in range(1, len(self.embeddings)):
            sim = np.dot(self.embeddings[i-1], self.embeddings[i])
            consecutive_sims.append(sim)
        
        # Stability is based on variance of consecutive similarities
        if consecutive_sims:
            variance = np.var(consecutive_sims)
            return 1.0 / (1.0 + variance * 10)  # Lower variance = higher stability
        return 0.0

class EnrollmentManager:
    def __init__(self,
                 min_samples_for_enrollment: int = 10,
                 min_quality_score: float = 0.7,
                 min_consistency_score: float = 0.85,
                 min_stability_score: float = 0.8,
                 collection_timeout: float = 30.0,
                 improvement_window_size: int = 20):
        """
        Enrollment manager with progressive state machine.
        
        Args:
            min_samples_for_enrollment: Minimum samples needed for enrollment
            min_quality_score: Minimum quality for accepting samples
            min_consistency_score: Minimum consistency for validation
            min_stability_score: Minimum stability for validation
            collection_timeout: Timeout for collection phase
            improvement_window_size: Window size for continuous improvement
        """
        self.min_samples_for_enrollment = min_samples_for_enrollment
        self.min_quality_score = min_quality_score
        self.min_consistency_score = min_consistency_score
        self.min_stability_score = min_stability_score
        self.collection_timeout = collection_timeout
        self.improvement_window_size = improvement_window_size
        
        # Participant enrollments
        self.enrollments: Dict[int, ParticipantEnrollment] = {}
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Callbacks
        self.on_enrollment_complete = None
        self.on_enrollment_failed = None
        self.on_enrollment_improved = None
    
    def process_recognition_result(self, result: Dict) -> Optional[int]:
        """
        Process face recognition result and manage enrollment.
        
        Args:
            result: Recognition result from face_recognition_process
            
        Returns:
            Assigned participant_id or None
        """
        track_id = result['track_id']
        participant_id = result.get('participant_id', -1)
        embedding = result['embedding']
        quality_score = result['quality_score']
        similarity = result.get('similarity', 0)
        
        with self.lock:
            # If recognized with high confidence, return participant_id
            if participant_id >= 0 and similarity > 0.7:
                # Update enrollment if in IMPROVING state
                if participant_id in self.enrollments:
                    enrollment = self.enrollments[participant_id]
                    if enrollment.state == EnrollmentState.IMPROVING:
                        self._update_improvement(enrollment, embedding, quality_score, track_id)
                
                return participant_id
            
            # Check if track is already being enrolled
            for pid, enrollment in self.enrollments.items():
                if track_id in enrollment.track_ids:
                    return self._handle_existing_enrollment(
                        enrollment, embedding, quality_score, track_id
                    )
            
            # New track - check quality and start enrollment
            if quality_score >= self.min_quality_score:
                return self._start_new_enrollment(track_id, embedding, quality_score)
            
            return None
    
    def _start_new_enrollment(self, track_id: int, embedding: np.ndarray, 
                             quality_score: float) -> int:
        """Start enrollment for a new track."""
        # Find available participant_id
        participant_id = self._get_next_participant_id()
        
        # Create new enrollment
        enrollment = ParticipantEnrollment(participant_id=participant_id)
        enrollment.state = EnrollmentState.COLLECTING
        enrollment.add_sample(embedding, quality_score, track_id)
        
        self.enrollments[participant_id] = enrollment
        
        return participant_id
    
    def _handle_existing_enrollment(self, enrollment: ParticipantEnrollment,
                                   embedding: np.ndarray, quality_score: float,
                                   track_id: int) -> int:
        """Handle update to existing enrollment."""
        # Update based on current state
        if enrollment.state == EnrollmentState.COLLECTING:
            return self._update_collection(enrollment, embedding, quality_score, track_id)
        
        elif enrollment.state == EnrollmentState.VALIDATING:
            return self._update_validation(enrollment, embedding, quality_score, track_id)
        
        elif enrollment.state == EnrollmentState.ENROLLED:
            enrollment.state = EnrollmentState.IMPROVING
            return self._update_improvement(enrollment, embedding, quality_score, track_id)
        
        elif enrollment.state == EnrollmentState.IMPROVING:
            return self._update_improvement(enrollment, embedding, quality_score, track_id)
        
        elif enrollment.state == EnrollmentState.FAILED:
            # Restart enrollment if quality is good
            if quality_score >= self.min_quality_score:
                enrollment.state = EnrollmentState.COLLECTING
                enrollment.embeddings = [embedding]
                enrollment.quality_scores = [quality_score]
                enrollment.timestamps = [time.time()]
                enrollment.state_start_time = time.time()
                enrollment.accepted_samples = 1
                enrollment.rejected_samples = 0
                return enrollment.participant_id
        
        return enrollment.participant_id
    
    def _update_collection(self, enrollment: ParticipantEnrollment,
                          embedding: np.ndarray, quality_score: float,
                          track_id: int) -> int:
        """Update enrollment in collection phase."""
        # Check timeout
        if time.time() - enrollment.state_start_time > self.collection_timeout:
            self._transition_to_failed(enrollment, "Collection timeout")
            return enrollment.participant_id
        
        # Add sample if quality is sufficient
        if quality_score >= self.min_quality_score:
            # Check similarity to existing samples
            if self._is_sample_consistent(enrollment, embedding):
                enrollment.add_sample(embedding, quality_score, track_id)
                enrollment.accepted_samples += 1
                
                # Check if we have enough samples
                if len(enrollment.embeddings) >= self.min_samples_for_enrollment:
                    self._transition_to_validation(enrollment)
            else:
                enrollment.rejected_samples += 1
        
        return enrollment.participant_id
    
    def _update_validation(self, enrollment: ParticipantEnrollment,
                          embedding: np.ndarray, quality_score: float,
                          track_id: int) -> int:
        """Update enrollment in validation phase."""
        # Add high-quality samples during validation
        if quality_score >= self.min_quality_score and self._is_sample_consistent(enrollment, embedding):
            enrollment.add_sample(embedding, quality_score, track_id)
        
        # Compute validation metrics
        enrollment.consistency_score = enrollment.compute_consistency()
        enrollment.stability_score = enrollment.compute_stability()
        enrollment.confidence_score = np.mean(enrollment.quality_scores)
        
        # Check validation criteria
        if (enrollment.consistency_score >= self.min_consistency_score and
            enrollment.stability_score >= self.min_stability_score):
            self._transition_to_enrolled(enrollment)
        elif len(enrollment.embeddings) > self.min_samples_for_enrollment * 2:
            # Failed validation after many attempts
            self._transition_to_failed(enrollment, "Validation failed")
        
        return enrollment.participant_id
    
    def _update_improvement(self, enrollment: ParticipantEnrollment,
                           embedding: np.ndarray, quality_score: float,
                           track_id: int) -> int:
        """Update enrollment in continuous improvement phase."""
        # Only accept high-quality, consistent samples
        if (quality_score > np.mean(enrollment.quality_scores) and
            self._is_sample_consistent(enrollment, embedding, threshold=0.9)):
            
            # Add to rolling window
            if len(enrollment.embeddings) >= self.improvement_window_size:
                # Remove oldest, lowest quality sample
                min_idx = np.argmin(enrollment.quality_scores)
                enrollment.embeddings.pop(min_idx)
                enrollment.quality_scores.pop(min_idx)
                enrollment.timestamps.pop(min_idx)
            
            enrollment.add_sample(embedding, quality_score, track_id)
            
            # Notify improvement
            if self.on_enrollment_improved:
                self.on_enrollment_improved(enrollment.participant_id)
        
        return enrollment.participant_id
    
    def _is_sample_consistent(self, enrollment: ParticipantEnrollment,
                             embedding: np.ndarray, threshold: float = 0.8) -> bool:
        """Check if sample is consistent with existing embeddings."""
        if not enrollment.embeddings:
            return True
        
        # Calculate similarities to existing embeddings
        similarities = [np.dot(embedding, emb) for emb in enrollment.embeddings]
        mean_similarity = np.mean(similarities)
        
        return mean_similarity >= threshold
    
    def _transition_to_validation(self, enrollment: ParticipantEnrollment):
        """Transition to validation state."""
        enrollment.state = EnrollmentState.VALIDATING
        enrollment.state_start_time = time.time()
    
    def _transition_to_enrolled(self, enrollment: ParticipantEnrollment):
        """Transition to enrolled state."""
        enrollment.state = EnrollmentState.ENROLLED
        enrollment.state_start_time = time.time()
        
        if self.on_enrollment_complete:
            self.on_enrollment_complete(enrollment.participant_id, enrollment)
    
    def _transition_to_failed(self, enrollment: ParticipantEnrollment, reason: str):
        """Transition to failed state."""
        enrollment.state = EnrollmentState.FAILED
        enrollment.state_start_time = time.time()
        
        if self.on_enrollment_failed:
            self.on_enrollment_failed(enrollment.participant_id, reason)
    
    def _get_next_participant_id(self) -> int:
        """Get next available participant ID."""
        # Find first available ID starting from 0
        participant_id = 0
        while participant_id in self.enrollments:
            participant_id += 1
        return participant_id
    
    def get_enrollment_status(self, participant_id: int) -> Optional[Dict]:
        """Get enrollment status for a participant."""
        with self.lock:
            if participant_id in self.enrollments:
                enrollment = self.enrollments[participant_id]
                return {
                    'participant_id': participant_id,
                    'state': enrollment.state.name,
                    'samples_collected': len(enrollment.embeddings),
                    'consistency_score': enrollment.consistency_score,
                    'stability_score': enrollment.stability_score,
                    'confidence_score': enrollment.confidence_score,
                    'accepted_samples': enrollment.accepted_samples,
                    'rejected_samples': enrollment.rejected_samples,
                    'enrollment_duration': time.time() - enrollment.enrollment_start_time
                }
        return None
    
    def get_all_enrollments(self) -> Dict[int, Dict]:
        """Get status of all enrollments."""
        with self.lock:
            return {
                pid: self.get_enrollment_status(pid)
                for pid in self.enrollments.keys()
            }
    
    def remove_participant(self, participant_id: int):
        """Remove a participant from enrollment."""
        with self.lock:
            if participant_id in self.enrollments:
                del self.enrollments[participant_id]
    
    def get_enrolled_embeddings(self, participant_id: int) -> Optional[List[np.ndarray]]:
        """Get embeddings for enrolled participant."""
        with self.lock:
            if participant_id in self.enrollments:
                enrollment = self.enrollments[participant_id]
                if enrollment.state in [EnrollmentState.ENROLLED, EnrollmentState.IMPROVING]:
                    return enrollment.embeddings.copy()
        return None