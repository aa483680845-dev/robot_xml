from __future__ import annotations

from pathlib import Path

import numpy as np

from robot_arm.core.types import IKResult, JointLimits, Pose

try:
    import pinocchio as pin
except Exception as exc:  # pragma: no cover - import failure depends on local environment.
    pin = None
    PIN_IMPORT_ERROR = exc
else:
    PIN_IMPORT_ERROR = None


class PinKinematicsSolver:
    def __init__(
        self,
        model_path: str | Path,
        joint_names: tuple[str, ...],
        end_effector_frame: str,
        acceleration_limits: np.ndarray | list[float] | tuple[float, ...],
        jerk_limits: np.ndarray | list[float] | tuple[float, ...],
        max_iterations: int = 200,
        tolerance: float = 1e-4,
        damping: float = 1e-4,
        step_size: float = 0.4,
    ) -> None:
        if pin is None:
            raise RuntimeError(
                "Pinocchio import failed. In this workspace, keep the local virtualenv on NumPy < 2 before using IK."
            ) from PIN_IMPORT_ERROR

        model_path = Path(model_path)
        suffix = model_path.suffix.lower()
        if suffix == ".urdf":
            self.model = pin.buildModelFromUrdf(str(model_path))
        elif suffix == ".xml":
            self.model = pin.buildModelFromMJCF(str(model_path))
        else:
            raise ValueError(
                f"Unsupported model file {model_path.name!r}: expected .urdf or .xml (MJCF)."
            )
        self.data = self.model.createData()
        self.joint_names = joint_names
        self.end_effector_frame = end_effector_frame
        self.frame_id = self.model.getFrameId(end_effector_frame)
        if self.frame_id >= len(self.model.frames):
            raise ValueError(f"Could not find frame {end_effector_frame!r} in URDF model.")

        lower = np.asarray(self.model.lowerPositionLimit, dtype=float).copy()
        upper = np.asarray(self.model.upperPositionLimit, dtype=float).copy()
        finite_lower = np.where(np.isfinite(lower), lower, -np.pi)
        finite_upper = np.where(np.isfinite(upper), upper, np.pi)
        velocity = np.asarray(self.model.velocityLimit, dtype=float).copy()
        velocity = np.where(np.isfinite(velocity) & (velocity > 0.0), velocity, np.pi)
        self.joint_limits = JointLimits(
            lower=finite_lower,
            upper=finite_upper,
            velocity=velocity,
            acceleration=np.asarray(acceleration_limits, dtype=float),
            jerk=np.asarray(jerk_limits, dtype=float),
        )
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.damping = damping
        self.step_size = step_size

        for name in joint_names:
            if not self.model.existJointName(name):
                raise ValueError(f"Could not find joint {name!r} in URDF model.")

    def forward_kinematics(self, q: np.ndarray | list[float]) -> Pose:
        q_array = np.asarray(q, dtype=float)
        pin.forwardKinematics(self.model, self.data, q_array)
        pin.updateFramePlacements(self.model, self.data)
        placement = self.data.oMf[self.frame_id]
        return Pose(position=np.array(placement.translation), rotation=np.array(placement.rotation))

    def solve_ik(self, target_pose: Pose, q0: np.ndarray | list[float]) -> IKResult:
        target = pin.SE3(target_pose.rotation, target_pose.position)
        q = np.asarray(q0, dtype=float).copy()
        last_error_norm = np.inf

        for iteration in range(1, self.max_iterations + 1):
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            current = self.data.oMf[self.frame_id]
            error = pin.log6(current.actInv(target)).vector
            last_error_norm = float(np.linalg.norm(error))
            if last_error_norm < self.tolerance:
                return IKResult(q=q, success=True, iterations=iteration, error_norm=last_error_norm)

            jacobian = pin.computeFrameJacobian(
                self.model,
                self.data,
                q,
                self.frame_id,
                pin.ReferenceFrame.LOCAL,
            )
            lhs = jacobian @ jacobian.T + self.damping * np.eye(6)
            delta_q = jacobian.T @ np.linalg.solve(lhs, error)
            q = pin.integrate(self.model, q, delta_q * self.step_size)
            q = np.clip(q, self.joint_limits.lower, self.joint_limits.upper)

        return IKResult(q=q, success=False, iterations=self.max_iterations, error_norm=last_error_norm)
