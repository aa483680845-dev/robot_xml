from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from robot_arm.core.types import JointLimits, Pose, RobotState


class MujocoBackend:
    def __init__(
        self,
        scene_path: str | Path,
        joint_names: tuple[str, ...],
        actuator_names: tuple[str, ...],
        velocity_limits: np.ndarray | list[float] | tuple[float, ...],
        acceleration_limits: np.ndarray | list[float] | tuple[float, ...],
        jerk_limits: np.ndarray | list[float] | tuple[float, ...],
    ) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(scene_path))
        self.data = mujoco.MjData(self.model)
        self.joint_names = joint_names
        self.actuator_names = actuator_names
        self._joint_ids = tuple(self._id_for_name(mujoco.mjtObj.mjOBJ_JOINT, name) for name in joint_names)
        self._actuator_ids = tuple(self._id_for_name(mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in actuator_names)
        self._qpos_ids = np.array([self.model.jnt_qposadr[jid] for jid in self._joint_ids], dtype=int)
        self._qvel_ids = np.array([self.model.jnt_dofadr[jid] for jid in self._joint_ids], dtype=int)
        lower = np.array([self.model.jnt_range[jid][0] for jid in self._joint_ids], dtype=float)
        upper = np.array([self.model.jnt_range[jid][1] for jid in self._joint_ids], dtype=float)
        self.joint_limits = JointLimits(
            lower=lower,
            upper=upper,
            velocity=np.asarray(velocity_limits, dtype=float),
            acceleration=np.asarray(acceleration_limits, dtype=float),
            jerk=np.asarray(jerk_limits, dtype=float),
        )
        self._validate_joint_alignment()
        self._validate_actuator_kind()
        mujoco.mj_forward(self.model, self.data)

    def _id_for_name(self, object_type: mujoco._enums.mjtObj, name: str) -> int:
        object_id = mujoco.mj_name2id(self.model, object_type, name)
        if object_id < 0:
            raise ValueError(f"Could not find {name!r} in MuJoCo model.")
        return object_id

    def _validate_joint_alignment(self) -> None:
        for index, (actuator_id, joint_id, expected_joint) in enumerate(
            zip(self._actuator_ids, self._joint_ids, self.joint_names, strict=True)
        ):
            actuator_joint_id = int(self.model.actuator_trnid[actuator_id][0])
            actuator_joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, actuator_joint_id)
            if actuator_joint_id != joint_id:
                raise ValueError(
                    f"Actuator {self.actuator_names[index]!r} drives {actuator_joint_name!r}, "
                    f"expected {expected_joint!r}."
                )

    def _validate_actuator_kind(self) -> None:
        # set_position_target() writes the joint target into data.ctrl, which is
        # only the correct semantics for MJCF <position> actuators (affine bias
        # with a non-zero position gain). Reject <motor>/<velocity>/general so a
        # mis-modelled scene fails loudly instead of running with wrong control.
        for index, actuator_id in enumerate(self._actuator_ids):
            biastype = int(self.model.actuator_biastype[actuator_id])
            position_gain = float(self.model.actuator_biasprm[actuator_id, 1])
            if biastype != int(mujoco.mjtBias.mjBIAS_AFFINE) or position_gain == 0.0:
                raise ValueError(
                    f"Actuator {self.actuator_names[index]!r} is not a <position> actuator "
                    f"(biastype={biastype}, position_gain={position_gain}). "
                    "MujocoBackend only supports position-controlled actuators."
                )

    def get_state(self) -> RobotState:
        q = self.data.qpos[self._qpos_ids]
        dq = self.data.qvel[self._qvel_ids]
        ddq = self.data.qacc[self._qvel_ids]
        return RobotState(q=q, dq=dq, ddq=ddq)

    def set_state(self, q: np.ndarray | list[float], dq: np.ndarray | list[float] | None = None) -> None:
        q_array = np.asarray(q, dtype=float)
        dq_array = np.zeros_like(q_array) if dq is None else np.asarray(dq, dtype=float)
        self.data.qpos[self._qpos_ids] = q_array
        self.data.qvel[self._qvel_ids] = dq_array
        self.data.ctrl[np.array(self._actuator_ids, dtype=int)] = q_array
        mujoco.mj_forward(self.model, self.data)

    def set_position_target(self, q_target: np.ndarray | list[float]) -> None:
        self.data.ctrl[np.array(self._actuator_ids, dtype=int)] = np.asarray(q_target, dtype=float)

    def step(self, count: int = 1) -> None:
        for _ in range(count):
            mujoco.mj_step(self.model, self.data)

    def site_pose(self, site_name: str) -> Pose:
        site_id = self._id_for_name(mujoco.mjtObj.mjOBJ_SITE, site_name)
        rotation = self.data.site_xmat[site_id].reshape(3, 3).copy()
        position = self.data.site_xpos[site_id].copy()
        return Pose(position=position, rotation=rotation)
