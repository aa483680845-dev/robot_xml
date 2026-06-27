from __future__ import annotations

from typing import Protocol

from robot_arm.core.types import IKResult, JointLimits, MotionPlan, Pose, RobotState


class KinematicsSolver(Protocol):
    joint_names: tuple[str, ...]
    joint_limits: JointLimits

    def forward_kinematics(self, q: object) -> Pose:
        ...

    def solve_ik(self, target_pose: Pose, q0: object) -> IKResult:
        ...


class TrajectoryPlanner(Protocol):
    joint_limits: JointLimits

    def plan(self, start: RobotState, goal_q: object) -> list:
        ...


class RobotBackend(Protocol):
    joint_names: tuple[str, ...]
    actuator_names: tuple[str, ...]
    joint_limits: JointLimits

    def get_state(self) -> RobotState:
        ...

    def set_position_target(self, q_target: object) -> None:
        ...

    def set_state(self, q: object, dq: object | None = None) -> None:
        ...

    def step(self, count: int = 1) -> None:
        ...


class MotionPlanner(Protocol):
    def plan_to_pose(self, target_pose: Pose) -> MotionPlan:
        ...

    def plan_to_joints(self, goal_q: object) -> MotionPlan:
        ...
