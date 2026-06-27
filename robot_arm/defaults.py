from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCENE_PATH = PROJECT_ROOT / "scene.xml"
ROBOT_MJCF_PATH = PROJECT_ROOT / "A02L-MP4-HT_defeature.xml"

JOINT_NAMES = (
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "joint_6",
    "joint_7",
)

ACTUATOR_NAMES = (
    "joint_1_pd",
    "joint_2_pd",
    "joint_3_pd",
    "joint_4_pd",
    "joint_5_pd",
    "joint_6_pd",
    "joint_7_pd",
)


@dataclass(frozen=True, slots=True)
class RobotControlConfig:
    scene_path: Path = SCENE_PATH
    pin_model_path: Path = ROBOT_MJCF_PATH
    joint_names: tuple[str, ...] = JOINT_NAMES
    actuator_names: tuple[str, ...] = ACTUATOR_NAMES
    pin_end_effector_frame: str = "end_effector"
    mujoco_end_effector_site: str = "end_effector"
    control_dt: float = 0.01
    ik_max_iterations: int = 200
    ik_tolerance: float = 1e-4
    ik_damping: float = 1e-4
    ik_step_size: float = 0.4


DEFAULT_VELOCITY_LIMITS = np.full(len(JOINT_NAMES), 3.0, dtype=float)
DEFAULT_ACCELERATION_LIMITS = np.array([6.0, 6.0, 6.0, 6.0, 8.0, 8.0, 8.0], dtype=float)
DEFAULT_JERK_LIMITS = np.array([40.0, 40.0, 40.0, 40.0, 60.0, 60.0, 60.0], dtype=float)
