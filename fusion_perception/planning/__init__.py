"""Planning support: ego state from poses and a BEV safety check for planned paths.

Everything here works without a planner model, so it can be tested on
KITTI-360 now and used to check Qwen-Drive's paths later.

Ego frame used throughout (Qwen-Drive convention): x forward, y left,
heading in radians, positive for a left turn, origin at the camera.
"""
from fusion_perception.planning.ego_state import EgoState, EgoStateExtractor
from fusion_perception.planning.bev_check import (
    BEVCheckConfig, BEVCheckResult, BEVPlanContext, build_plan_context, check_trajectory,
)
from fusion_perception.planning.metrics import displacement_errors

__all__ = [
    "EgoState", "EgoStateExtractor",
    "BEVCheckConfig", "BEVCheckResult", "BEVPlanContext",
    "build_plan_context", "check_trajectory", "displacement_errors",
]
