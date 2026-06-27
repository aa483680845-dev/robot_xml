from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _as_vector(values: np.ndarray | list[float] | tuple[float, ...], size: int) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape != (size,):
        raise ValueError(f"Expected shape {(size,)}, got {array.shape}.")
    return array


def _as_matrix(values: np.ndarray | list[list[float]], shape: tuple[int, int]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape != shape:
        raise ValueError(f"Expected shape {shape}, got {array.shape}.")
    return array


@dataclass(slots=True)
class Pose:
    position: np.ndarray
    rotation: np.ndarray

    def __post_init__(self) -> None:
        self.position = _as_vector(self.position, 3)
        self.rotation = _as_matrix(self.rotation, (3, 3))


@dataclass(slots=True)
class RobotState:
    q: np.ndarray
    dq: np.ndarray
    ddq: np.ndarray

    def __post_init__(self) -> None:
        self.q = np.asarray(self.q, dtype=float).copy()
        self.dq = np.asarray(self.dq, dtype=float).copy()
        self.ddq = np.asarray(self.ddq, dtype=float).copy()


@dataclass(slots=True)
class JointLimits:
    lower: np.ndarray
    upper: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    jerk: np.ndarray

    def __post_init__(self) -> None:
        size = len(self.lower)
        self.lower = _as_vector(self.lower, size)
        self.upper = _as_vector(self.upper, size)
        self.velocity = _as_vector(self.velocity, size)
        self.acceleration = _as_vector(self.acceleration, size)
        self.jerk = _as_vector(self.jerk, size)


@dataclass(slots=True)
class IKResult:
    q: np.ndarray
    success: bool
    iterations: int
    error_norm: float

    def __post_init__(self) -> None:
        self.q = np.asarray(self.q, dtype=float).copy()


@dataclass(slots=True)
class TrajectoryPoint:
    time_from_start: float
    q: np.ndarray
    dq: np.ndarray
    ddq: np.ndarray

    def __post_init__(self) -> None:
        self.q = np.asarray(self.q, dtype=float).copy()
        self.dq = np.asarray(self.dq, dtype=float).copy()
        self.ddq = np.asarray(self.ddq, dtype=float).copy()


@dataclass(slots=True)
class MotionPlan:
    trajectory: list[TrajectoryPoint]
    ik_result: IKResult | None = None
