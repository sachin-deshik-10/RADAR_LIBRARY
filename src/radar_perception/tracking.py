"""
Tracking algorithms for radar perception.

This module provides implementations of various tracking algorithms
for maintaining target trajectories over time in radar systems.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import uuid
from scipy.linalg import inv
from scipy.optimize import linear_sum_assignment

from .detection import Detection


class TrackState(Enum):
    """Track state enumeration."""
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    COASTED = "coasted"
    TERMINATED = "terminated"


@dataclass
class Track:
    """Represents a radar track."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    state: TrackState = TrackState.TENTATIVE
    position: np.ndarray = field(default_factory=lambda: np.zeros(2))  # [x, y]
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))  # [vx, vy]
    acceleration: np.ndarray = field(default_factory=lambda: np.zeros(2))  # [ax, ay]
    covariance: np.ndarray = field(default_factory=lambda: np.eye(4))
    last_update_time: float = 0.0
    creation_time: float = 0.0
    detection_count: int = 0
    missed_detection_count: int = 0
    last_detection: Optional[Detection] = None
    prediction: Optional[np.ndarray] = None
    
    def __post_init__(self):
        """Initialize track after creation."""
        if self.creation_time == 0.0:
            self.creation_time = self.last_update_time


class KalmanFilter:
    """
    Kalman filter for state estimation in tracking applications.
    
    Implements a constant velocity model for 2D tracking with optional
    acceleration estimation.
    """
    
    def __init__(self, 
                 process_noise_std: float = 1.0,
                 measurement_noise_std: float = 1.0,
                 use_acceleration: bool = False):
        """
        Initialize Kalman filter.
        
        Args:
            process_noise_std: Standard deviation of process noise
            measurement_noise_std: Standard deviation of measurement noise
            use_acceleration: Whether to include acceleration in state model
        """
        self.process_noise_std = process_noise_std
        self.measurement_noise_std = measurement_noise_std
        self.use_acceleration = use_acceleration
        
        # State dimension (position + velocity [+ acceleration])
        self.state_dim = 6 if use_acceleration else 4  # [x, y, vx, vy] or [x, y, vx, vy, ax, ay]
        self.measurement_dim = 2  # [x, y]
        
        # Initialize matrices
        self._setup_matrices()
    
    def _setup_matrices(self):
        """Setup Kalman filter matrices."""
        if self.use_acceleration:
            # State: [x, y, vx, vy, ax, ay]
            self.F = np.array([
                [1, 0, 1, 0, 0.5, 0],
                [0, 1, 0, 1, 0, 0.5],
                [0, 0, 1, 0, 1, 0],
                [0, 0, 0, 1, 0, 1],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1]
            ])
        else:
            # State: [x, y, vx, vy]
            self.F = np.array([
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
        
        # Measurement matrix (observe position only)
        self.H = np.zeros((self.measurement_dim, self.state_dim))
        self.H[0, 0] = 1  # x
        self.H[1, 1] = 1  # y
        
        # Process noise covariance
        q = self.process_noise_std ** 2
        if self.use_acceleration:
            self.Q = q * np.diag([0.25, 0.25, 1, 1, 1, 1])
        else:
            self.Q = q * np.diag([0.25, 0.25, 1, 1])
        
        # Measurement noise covariance
        r = self.measurement_noise_std ** 2
        self.R = r * np.eye(self.measurement_dim)
    
    def update_transition_matrix(self, dt: float):
        """Update state transition matrix with time step."""
        if self.use_acceleration:
            self.F[0, 2] = dt
            self.F[1, 3] = dt
            self.F[0, 4] = 0.5 * dt * dt
            self.F[1, 5] = 0.5 * dt * dt
            self.F[2, 4] = dt
            self.F[3, 5] = dt
        else:
            self.F[0, 2] = dt
            self.F[1, 3] = dt
    
    def predict(self, state: np.ndarray, covariance: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prediction step of Kalman filter.
        
        Args:
            state: Current state estimate
            covariance: Current covariance matrix
            dt: Time step
            
        Returns:
            Tuple of (predicted_state, predicted_covariance)
        """
        self.update_transition_matrix(dt)
        
        predicted_state = self.F @ state
        predicted_covariance = self.F @ covariance @ self.F.T + self.Q
        
        return predicted_state, predicted_covariance
    
    def update(self, state: np.ndarray, covariance: np.ndarray, 
               measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Update step of Kalman filter.
        
        Args:
            state: Predicted state
            covariance: Predicted covariance
            measurement: Measurement vector [x, y]
            
        Returns:
            Tuple of (updated_state, updated_covariance)
        """
        # Innovation
        innovation = measurement - self.H @ state
        
        # Innovation covariance
        S = self.H @ covariance @ self.H.T + self.R
        
        # Kalman gain
        K = covariance @ self.H.T @ inv(S)
        
        # Updated state and covariance
        updated_state = state + K @ innovation
        updated_covariance = (np.eye(self.state_dim) - K @ self.H) @ covariance
        
        return updated_state, updated_covariance
    
    def mahalanobis_distance(self, state: np.ndarray, covariance: np.ndarray, 
                           measurement: np.ndarray) -> float:
        """Calculate Mahalanobis distance between prediction and measurement."""
        predicted_measurement = self.H @ state
        innovation = measurement - predicted_measurement
        S = self.H @ covariance @ self.H.T + self.R
        
        try:
            distance = np.sqrt(innovation.T @ inv(S) @ innovation)
            return float(distance)
        except np.linalg.LinAlgError:
            return float('inf')


class MultiTargetTracker:
    """
    Multi-target tracker using Global Nearest Neighbor (GNN) association.
    
    Manages multiple tracks and performs data association between detections
    and existing tracks.
    """
    
    def __init__(self,
                 max_association_distance: float = 5.0,
                 min_detections_for_confirmation: int = 3,
                 max_missed_detections: int = 5,
                 process_noise_std: float = 1.0,
                 measurement_noise_std: float = 1.0):
        """
        Initialize multi-target tracker.
        
        Args:
            max_association_distance: Maximum distance for track-detection association
            min_detections_for_confirmation: Minimum detections needed to confirm track
            max_missed_detections: Maximum missed detections before track termination
            process_noise_std: Process noise standard deviation
            measurement_noise_std: Measurement noise standard deviation
        """
        self.max_association_distance = max_association_distance
        self.min_detections_for_confirmation = min_detections_for_confirmation
        self.max_missed_detections = max_missed_detections
        
        self.kalman_filter = KalmanFilter(process_noise_std, measurement_noise_std)
        self.tracks: Dict[str, Track] = {}
        self.track_counter = 0
    
    def predict_tracks(self, current_time: float):
        """Predict all active tracks to current time."""
        for track in self.tracks.values():
            if track.state in [TrackState.CONFIRMED, TrackState.TENTATIVE, TrackState.COASTED]:
                dt = current_time - track.last_update_time
                if dt > 0:
                    # Create state vector
                    state = np.array([
                        track.position[0], track.position[1],
                        track.velocity[0], track.velocity[1]
                    ])
                    
                    predicted_state, predicted_covariance = self.kalman_filter.predict(
                        state, track.covariance, dt
                    )
                    
                    # Store prediction
                    track.prediction = predicted_state
                    track.covariance = predicted_covariance
    
    def associate_detections(self, detections: List[Detection]) -> Tuple[Dict[str, Detection], List[Detection]]:
        """
        Associate detections with tracks using Hungarian algorithm.
        
        Args:
            detections: List of detections to associate
            
        Returns:
            Tuple of (track_detection_associations, unassociated_detections)
        """
        active_tracks = [track for track in self.tracks.values() 
                        if track.state in [TrackState.CONFIRMED, TrackState.TENTATIVE, TrackState.COASTED]]
        
        if not active_tracks or not detections:
            return {}, detections
        
        # Create cost matrix
        cost_matrix = np.full((len(active_tracks), len(detections)), float('inf'))
        
        for i, track in enumerate(active_tracks):
            if track.prediction is not None:
                track_pos = track.prediction[:2]
                
                for j, detection in enumerate(detections):
                    det_pos = np.array([detection.range_m, detection.velocity_mps])
                    distance = self.kalman_filter.mahalanobis_distance(
                        track.prediction, track.covariance, det_pos
                    )
                    
                    if distance <= self.max_association_distance:
                        cost_matrix[i, j] = distance
        
        # Solve assignment problem
        track_indices, detection_indices = linear_sum_assignment(cost_matrix)
        
        # Extract valid associations
        associations = {}
        used_detections = set()
        
        for track_idx, det_idx in zip(track_indices, detection_indices):
            if cost_matrix[track_idx, det_idx] < float('inf'):
                track = active_tracks[track_idx]
                detection = detections[det_idx]
                associations[track.id] = detection
                used_detections.add(det_idx)
        
        # Find unassociated detections
        unassociated_detections = [det for i, det in enumerate(detections) 
                                 if i not in used_detections]
        
        return associations, unassociated_detections
    
    def update_tracks(self, associations: Dict[str, Detection], current_time: float):
        """Update tracks with associated detections."""
        for track_id, detection in associations.items():
            track = self.tracks[track_id]
            
            # Update with measurement
            measurement = np.array([detection.range_m, detection.velocity_mps])
            updated_state, updated_covariance = self.kalman_filter.update(
                track.prediction, track.covariance, measurement
            )
            
            # Update track
            track.position = updated_state[:2]
            track.velocity = updated_state[2:4]
            track.covariance = updated_covariance
            track.last_update_time = current_time
            track.last_detection = detection
            track.detection_count += 1
            track.missed_detection_count = 0
            
            # Update track state
            if (track.state == TrackState.TENTATIVE and 
                track.detection_count >= self.min_detections_for_confirmation):
                track.state = TrackState.CONFIRMED
            elif track.state == TrackState.COASTED:
                track.state = TrackState.CONFIRMED
    
    def handle_missed_detections(self, current_time: float):
        """Handle tracks without associated detections."""
        tracks_to_remove = []
        
        for track_id, track in self.tracks.items():
            if track_id not in [t_id for t_id in self.tracks.keys()]:  # Not updated this cycle
                track.missed_detection_count += 1
                
                if track.state == TrackState.CONFIRMED:
                    track.state = TrackState.COASTED
                
                # Terminate tracks with too many missed detections
                if track.missed_detection_count >= self.max_missed_detections:
                    track.state = TrackState.TERMINATED
                    tracks_to_remove.append(track_id)
        
        # Remove terminated tracks
        for track_id in tracks_to_remove:
            del self.tracks[track_id]
    
    def initiate_new_tracks(self, unassociated_detections: List[Detection], current_time: float):
        """Create new tracks from unassociated detections."""
        for detection in unassociated_detections:
            track_id = f"track_{self.track_counter:04d}"
            self.track_counter += 1
            
            # Initialize track state
            initial_state = np.array([
                detection.range_m, detection.velocity_mps, 0.0, 0.0
            ])
            initial_covariance = np.diag([1.0, 1.0, 10.0, 10.0])  # High uncertainty in velocity
            
            track = Track(
                id=track_id,
                state=TrackState.TENTATIVE,
                position=initial_state[:2],
                velocity=initial_state[2:4],
                covariance=initial_covariance,
                last_update_time=current_time,
                creation_time=current_time,
                detection_count=1,
                last_detection=detection
            )
            
            self.tracks[track_id] = track
    
    def update(self, detections: List[Detection], current_time: float) -> List[Track]:
        """
        Main tracking update function.
        
        Args:
            detections: List of detections for current frame
            current_time: Current timestamp
            
        Returns:
            List of active tracks
        """
        # Predict all tracks
        self.predict_tracks(current_time)
        
        # Associate detections with tracks
        associations, unassociated_detections = self.associate_detections(detections)
        
        # Update tracks with associated detections
        self.update_tracks(associations, current_time)
        
        # Handle missed detections
        self.handle_missed_detections(current_time)
        
        # Create new tracks from unassociated detections
        self.initiate_new_tracks(unassociated_detections, current_time)
        
        # Return active tracks
        return [track for track in self.tracks.values() 
                if track.state in [TrackState.CONFIRMED, TrackState.TENTATIVE, TrackState.COASTED]]


class IMM_Tracker:
    """
    Interacting Multiple Model (IMM) tracker for maneuvering targets.
    
    Uses multiple motion models (e.g., constant velocity, constant acceleration)
    to handle different types of target motion.
    """
    
    def __init__(self, model_transition_probabilities: np.ndarray):
        """
        Initialize IMM tracker.
        
        Args:
            model_transition_probabilities: Model transition probability matrix
        """
        self.transition_probs = model_transition_probabilities
        self.num_models = len(model_transition_probabilities)
        
        # Create multiple Kalman filters for different models
        self.filters = [
            KalmanFilter(process_noise_std=1.0, use_acceleration=False),  # CV model
            KalmanFilter(process_noise_std=2.0, use_acceleration=True),   # CA model
        ]
        
        self.model_probabilities = np.ones(self.num_models) / self.num_models
    
    def update(self, state: np.ndarray, covariance: np.ndarray, 
               measurement: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        IMM update step.
        
        Args:
            state: Current mixed state estimate
            covariance: Current mixed covariance
            measurement: Current measurement
            dt: Time step
            
        Returns:
            Tuple of (updated_state, updated_covariance, model_probabilities)
        """
        # Implementation would involve:
        # 1. Model mixing
        # 2. Model-specific filtering
        # 3. Model probability update
        # 4. Estimate and covariance mixing
        
        # Simplified version - use primary model
        updated_state, updated_covariance = self.filters[0].update(state, covariance, measurement)
        return updated_state, updated_covariance, self.model_probabilities


def smooth_track_trajectory(track_history: List[Track], 
                          smoothing_window: int = 5) -> List[Track]:
    """
    Apply smoothing to track trajectory using moving average.
    
    Args:
        track_history: List of track states over time
        smoothing_window: Size of smoothing window
        
    Returns:
        List of smoothed track states
    """
    if len(track_history) < smoothing_window:
        return track_history
    
    smoothed_tracks = []
    
    for i in range(len(track_history)):
        start_idx = max(0, i - smoothing_window // 2)
        end_idx = min(len(track_history), i + smoothing_window // 2 + 1)
        
        window_tracks = track_history[start_idx:end_idx]
        
        # Calculate smoothed position and velocity
        positions = np.array([track.position for track in window_tracks])
        velocities = np.array([track.velocity for track in window_tracks])
        
        smoothed_position = np.mean(positions, axis=0)
        smoothed_velocity = np.mean(velocities, axis=0)
        
        # Create smoothed track
        smoothed_track = Track(
            id=track_history[i].id,
            state=track_history[i].state,
            position=smoothed_position,
            velocity=smoothed_velocity,
            last_update_time=track_history[i].last_update_time,
            creation_time=track_history[i].creation_time,
            detection_count=track_history[i].detection_count
        )
        
        smoothed_tracks.append(smoothed_track)
    
    return smoothed_tracks
