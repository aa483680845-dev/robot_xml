from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True, slots=True)
class RobotDescription:
    """All robot-specific data needed to assemble the motion stack."""

    scene_path: Path
    pin_model_path: Path
    joint_names: tuple[str, ...]
    actuator_names: tuple[str, ...]
    pin_end_effector_frame: str
    mujoco_end_effector_site: str
    velocity_limits: np.ndarray
    acceleration_limits: np.ndarray
    jerk_limits: np.ndarray


@dataclass(frozen=True, slots=True)
class ControlConfig:
    """Framework-level knobs that are independent of which robot is loaded."""

    control_dt: float = 0.01
    ik_max_iterations: int = 200
    ik_tolerance: float = 1e-4
    ik_damping: float = 1e-4
    ik_step_size: float = 0.4


A02L_JOINT_NAMES = (
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "joint_6",
    "joint_7",
)

A02L_ACTUATOR_NAMES = tuple(f"{name}_pd" for name in A02L_JOINT_NAMES)


A02L_DESCRIPTION = RobotDescription(
    scene_path=PROJECT_ROOT / "scene.xml",
    pin_model_path=PROJECT_ROOT / "A02L-MP4-HT_defeature.xml",
    joint_names=A02L_JOINT_NAMES,
    actuator_names=A02L_ACTUATOR_NAMES,
    pin_end_effector_frame="end_effector",
    mujoco_end_effector_site="end_effector",
    velocity_limits=np.full(len(A02L_JOINT_NAMES), 3.0, dtype=float),
    acceleration_limits=np.array([6.0, 6.0, 6.0, 6.0, 8.0, 8.0, 8.0], dtype=float),
    jerk_limits=np.array([40.0, 40.0, 40.0, 40.0, 60.0, 60.0, 60.0], dtype=float),
)
