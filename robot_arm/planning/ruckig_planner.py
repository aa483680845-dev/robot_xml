from __future__ import annotations

import numpy as np
from ruckig import InputParameter, OutputParameter, Result, Ruckig

from robot_arm.core.types import JointLimits, RobotState, TrajectoryPoint


class RuckigTrajectoryPlanner:
    def __init__(self, joint_limits: JointLimits, control_dt: float) -> None:
        self.joint_limits = joint_limits
        self.control_dt = control_dt
        self.degrees_of_freedom = len(joint_limits.lower)

    def plan(self, start: RobotState, goal_q: np.ndarray | list[float]) -> list[TrajectoryPoint]:
        goal = np.asarray(goal_q, dtype=float)
        if goal.shape != start.q.shape:
            raise ValueError(f"Goal shape {goal.shape} does not match current joint state {start.q.shape}.")

        if np.allclose(goal, start.q, atol=1e-6) and np.allclose(start.dq, 0.0, atol=1e-6):
            return [TrajectoryPoint(0.0, start.q, start.dq, start.ddq)]

        otg = Ruckig(self.degrees_of_freedom, self.control_dt)
        input_parameter = InputParameter(self.degrees_of_freedom)
        output_parameter = OutputParameter(self.degrees_of_freedom)

        input_parameter.current_position = start.q.tolist()
        input_parameter.current_velocity = start.dq.tolist()
        input_parameter.current_acceleration = start.ddq.tolist()
        input_parameter.target_position = goal.tolist()
        input_parameter.target_velocity = [0.0] * self.degrees_of_freedom
        input_parameter.target_acceleration = [0.0] * self.degrees_of_freedom
        input_parameter.max_velocity = self.joint_limits.velocity.tolist()
        input_parameter.max_acceleration = self.joint_limits.acceleration.tolist()
        input_parameter.max_jerk = self.joint_limits.jerk.tolist()

        samples: list[TrajectoryPoint] = [
            TrajectoryPoint(
                time_from_start=0.0,
                q=np.asarray(start.q, dtype=float).copy(),
                dq=np.asarray(start.dq, dtype=float).copy(),
                ddq=np.asarray(start.ddq, dtype=float).copy(),
            )
        ]
        result = Result.Working

        while result == Result.Working:
            result = otg.update(input_parameter, output_parameter)
            if result not in (Result.Working, Result.Finished):
                raise RuntimeError(f"Ruckig failed with result code {result}.")

            samples.append(
                TrajectoryPoint(
                    time_from_start=len(samples) * self.control_dt,
                    q=np.array(output_parameter.new_position),
                    dq=np.array(output_parameter.new_velocity),
                    ddq=np.array(output_parameter.new_acceleration),
                )
            )
            output_parameter.pass_to_input(input_parameter)

        return samples
