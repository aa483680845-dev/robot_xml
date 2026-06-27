"""High-level motion pipeline entrypoints."""

from robot_arm.pipeline.factory import build_motion_service
from robot_arm.pipeline.motion_service import MotionService

__all__ = ["MotionService", "build_motion_service"]
