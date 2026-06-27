from __future__ import annotations

import numpy as np

from robot_arm.core.types import MotionPlan, Pose, RobotState


class MotionService:
    def __init__(self, backend, kinematics_solver, trajectory_planner) -> None:
        self.backend = backend
        self.kinematics_solver = kinematics_solver
        self.trajectory_planner = trajectory_planner

    def _planning_start(self) -> RobotState:
        """Snapshot the backend state, but zero acceleration.

        MuJoCo's ``data.qacc`` reflects physical acceleration (gravity, contact,
        residual dynamics) rather than the control command. Feeding that into a
        jerk-limited planner treats the robot as already accelerating and yields
        wildly overshooting trajectories. Replanning always starts from a
        commanded-zero-acceleration state.
        """
        snapshot = self.backend.get_state()
        return RobotState(q=snapshot.q, dq=snapshot.dq, ddq=np.zeros_like(snapshot.q))

    def plan_to_joints(self, goal_q: np.ndarray | list[float]) -> MotionPlan:
        start_state = self._planning_start()
        trajectory = self.trajectory_planner.plan(start_state, goal_q)
        return MotionPlan(trajectory=trajectory)

    def plan_to_pose(self, target_pose: Pose) -> MotionPlan:
        start_state = self._planning_start()
        ik_result = self.kinematics_solver.solve_ik(target_pose, start_state.q)
        if not ik_result.success:
            raise RuntimeError(
                f"IK did not converge after {ik_result.iterations} iterations. Final error: {ik_result.error_norm:.6f}"
            )

        trajectory = self.trajectory_planner.plan(start_state, ik_result.q)
        return MotionPlan(trajectory=trajectory, ik_result=ik_result)
