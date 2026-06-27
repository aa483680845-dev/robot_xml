import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import robot_arm
import mujoco
import mujoco.viewer
import numpy as np

from robot_arm.backends.mujoco_backend import MujocoBackend
from robot_arm.core.types import Pose
from robot_arm.defaults import (
    DEFAULT_ACCELERATION_LIMITS,
    DEFAULT_JERK_LIMITS,
    DEFAULT_VELOCITY_LIMITS,
    RobotControlConfig,
)
from robot_arm.kinematics.pin_solver import PinKinematicsSolver
from robot_arm.pipeline.motion_service import MotionService
from robot_arm.planning.ruckig_planner import RuckigTrajectoryPlanner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="View the MuJoCo model and optionally replay a pose target.")
    parser.add_argument(
        "--target-position",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        help="Optional end-effector target in world coordinates. Orientation stays at the current pose.",
    )
    return parser.parse_args()


def build_motion_plan(backend: MujocoBackend, config: RobotControlConfig, target_position: list[float] | None):
    if target_position is None:
        return None

    solver = PinKinematicsSolver(
        model_path=config.pin_model_path,
        joint_names=config.joint_names,
        end_effector_frame=config.pin_end_effector_frame,
        acceleration_limits=DEFAULT_ACCELERATION_LIMITS,
        jerk_limits=DEFAULT_JERK_LIMITS,
        max_iterations=config.ik_max_iterations,
        tolerance=config.ik_tolerance,
        damping=config.ik_damping,
        step_size=config.ik_step_size,
    )
    planner = RuckigTrajectoryPlanner(joint_limits=backend.joint_limits, control_dt=config.control_dt)
    motion_service = MotionService(backend=backend, kinematics_solver=solver, trajectory_planner=planner)
    current_pose = solver.forward_kinematics(backend.get_state().q)
    target_pose = Pose(position=np.array(target_position, dtype=float), rotation=current_pose.rotation)
    return motion_service.plan_to_pose(target_pose)


def main() -> None:
    args = parse_args()
    config = RobotControlConfig()
    backend = MujocoBackend(
        scene_path=config.scene_path,
        joint_names=config.joint_names,
        actuator_names=config.actuator_names,
        velocity_limits=DEFAULT_VELOCITY_LIMITS,
        acceleration_limits=DEFAULT_ACCELERATION_LIMITS,
        jerk_limits=DEFAULT_JERK_LIMITS,
    )
    plan = build_motion_plan(backend, config, args.target_position)
    plan_index = 0

    with mujoco.viewer.launch_passive(backend.model, backend.data) as viewer:
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_COM] = True
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_BODY

        real_start = time.time()
        while viewer.is_running():
            if plan is not None and plan_index < len(plan.trajectory):
                backend.set_position_target(plan.trajectory[plan_index].q)
                plan_index += 1

            backend.step()
            viewer.sync()

            elapsed = time.time() - real_start
            if backend.data.time > elapsed:
                time.sleep(backend.data.time - elapsed)


if __name__ == "__main__":
    main()
