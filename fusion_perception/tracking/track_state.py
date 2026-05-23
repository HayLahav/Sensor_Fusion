"""Per-track Kalman state and status for KalmanCoWTracker."""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional
import numpy as np
from fusion_perception.utils.dataclasses import Track


class TrackStatus(Enum):
    TENTATIVE = auto()   # needs confirm_age matches before activation
    CONFIRMED = auto()   # actively tracked
    LOST = auto()        # no detection; propagating via KF prediction only


@dataclass
class TrackState:
    track_id: int
    class_name: str
    kf: object                        # filterpy.KalmanFilter
    status: TrackStatus
    age: int = 0
    miss_count: int = 0
    confirm_hits: int = 0             # consecutive matched frames in TENTATIVE
    first_seen: int = 0
    last_seen: int = 0
    last_box3d: Optional[np.ndarray] = None   # [cx,cy,cz,θ,l,w,h]
    velocity_estimate: np.ndarray = field(default_factory=lambda: np.zeros(3))  # smoothed estimate for LOST-track decay, separate from kf.x[7:10]
    cow_points_abs: Optional[np.ndarray] = None   # [N,2] pixel coords
    cow_points_rel: Optional[np.ndarray] = None   # [N,2] normalised to bbox
    last_bbox2d: Optional[list[float]] = None            # [x1,y1,x2,y2] for re-spawn
    centroid_history: list[list[float]] = field(default_factory=list)
    position_3d_history: list[list[float]] = field(default_factory=list)
    innovation_norm: float = 0.0      # ||z - Hx||, used for lazy CoW gate

    def to_track(self) -> Track:
        """Convert to the shared Track dataclass used by downstream stages."""
        if self.kf is None:
            raise ValueError(
                f"TrackState(track_id={self.track_id}) has kf=None; "
                "cannot convert to Track before Kalman Filter is initialized."
            )
        if self.cow_points_abs is not None:
            cow_qp = [float(self.cow_points_abs[0][0]), float(self.cow_points_abs[0][1])]
        else:
            cow_qp = [0., 0.]
        return Track(
            track_id=self.track_id,
            class_name=self.class_name,
            first_seen=self.first_seen,
            last_seen=self.last_seen,
            centroid_history=list(self.centroid_history),
            position_3d_history=list(self.position_3d_history),
            cow_query_point=cow_qp,
            is_active=(self.status != TrackStatus.LOST),
            occlusion_count=self.miss_count,
        )
