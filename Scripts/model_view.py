import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import robot_arm  # noqa: F401  -- runs cmeel/pinocchio bootstrap
import mujoco
import mujoco.viewer
import numpy as np

from robot_arm.core.types import MotionPlan, Pose
from robot_arm.defaults import A02L_DESCRIPTION, ControlConfig, RobotDescription
from robot_arm.pipeline import MotionService, build_motion_service


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


def build_motion_plan(
    service: MotionService,
    robot: RobotDescription,
    target_position: list[float] | None,
) -> MotionPlan | None:
    if target_position is None:
        return None
    current_pose = service.kinematics_solver.forward_kinematics(service.backend.get_state().q)
    target_pose = Pose(position=np.array(target_position, dtype=float), rotation=current_pose.rotation)
    return service.plan_to_pose(target_pose)


def main() -> None:
    args = parse_args()
    robot = A02L_DESCRIPTION
    service = build_motion_service(robot, ControlConfig())
    backend = service.backend
    plan = build_motion_plan(service, robot, args.target_position)
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
