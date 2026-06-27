from __future__ import annotations

from robot_arm.backends.mujoco_backend import MujocoBackend
from robot_arm.defaults import ControlConfig, RobotDescription
from robot_arm.kinematics.pin_solver import PinKinematicsSolver
from robot_arm.pipeline.motion_service import MotionService
from robot_arm.planning.ruckig_planner import RuckigTrajectoryPlanner


def build_motion_service(
    robot: RobotDescription,
    control: ControlConfig | None = None,
) -> MotionService:
    control = control or ControlConfig()

    backend = MujocoBackend(
        scene_path=robot.scene_path,
        joint_names=robot.joint_names,
        actuator_names=robot.actuator_names,
        velocity_limits=robot.velocity_limits,
        acceleration_limits=robot.acceleration_limits,
        jerk_limits=robot.jerk_limits,
    )
    solver = PinKinematicsSolver(
        model_path=robot.pin_model_path,
        joint_names=robot.joint_names,
        end_effector_frame=robot.pin_end_effector_frame,
        acceleration_limits=robot.acceleration_limits,
        jerk_limits=robot.jerk_limits,
        max_iterations=control.ik_max_iterations,
        tolerance=control.ik_tolerance,
        damping=control.ik_damping,
        step_size=control.ik_step_size,
    )
    planner = RuckigTrajectoryPlanner(
        joint_limits=backend.joint_limits,
        control_dt=control.control_dt,
    )
    return MotionService(backend=backend, kinematics_solver=solver, trajectory_planner=planner)
