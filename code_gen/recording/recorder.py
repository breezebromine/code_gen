# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Adapt task observations to the public MGG recorder without owning serialization."""

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from isaaclab_arena.grasping.robot_layout import RobotLayout
from isaaclab_arena.server.robo_challenge.observations import object_observations, relative_object_pose
from isaaclab_arena.server.robo_challenge.recorder import RoboChallengeRecorder
from isaaclab_arena.utils.path_tracing_replay import SceneStateBuffer


def pose_matrix(position, quaternion_wxyz):
    """World-from-frame transform; camera poses supplied in ROS convention."""
    matrix = np.eye(4)
    q = np.asarray(quaternion_wxyz)
    matrix[:3, :3] = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
    matrix[:3, 3] = np.asarray(position)
    return matrix


class CodegenRecorder:
    """Supply measured observations; MGG owns the recording and replay formats."""

    def __init__(self, backend):
        self.backend = backend
        self.env = backend.env.unwrapped
        self.policy = backend.policy
        self.arms = ("left", "right") if backend.layout.robot == "aloha" else ("left",)
        self.base_ids = {
            arm: self.policy.robot.find_bodies(RobotLayout(backend.layout.robot, arm).base_link)[0][0]
            for arm in self.arms
        }
        self.objects = {name: name for name in (backend.scene_spec or {}).get("objects", {})}
        self.tracked_object = None
        directory = backend.output / "demo_0000"
        directory.mkdir(exist_ok=False)
        aloha = backend.layout.robot == "aloha"
        self.writer = RoboChallengeRecorder(
            directory,
            "aloha" if aloha else "ARX5",
            backend.task.name,
            env=self.env if aloha else None,
            env_id=backend.env_id,
            record_scene_states=aloha,
        )
        self.writer.save_meta["clock"] = "session_t0_epoch + control_step * step_dt" + (
            "" if aloha else "; simulation clock shared by all streams"
        )
        self.writer.save_meta["arms_order"] = list(self.arms)
        if not aloha:
            self._configure_arx5()
        camera_cfg = self.env.cfg.isaaclab_arena_env.embodiment.get_camera_cfg()
        for sensor_name in self.writer.camera_record_to_sim_remap.values():
            getattr(camera_cfg, sensor_name).update_period = 0.0
            if backend.args.enable_cameras:
                self.env.scene[sensor_name].cfg.update_period = 0.0
                getattr(self.env.cfg.scene, sensor_name).update_period = 0.0

    def frames(self):
        """Read current poses with link indices resolved once per episode."""
        data = self.policy.robot.data
        return {
            arm: {
                "base_p": data.body_link_pos_w[self.backend.env_id : self.backend.env_id + 1, body_id],
                "base_q": data.body_link_quat_w[self.backend.env_id : self.backend.env_id + 1, body_id],
            }
            for arm, body_id in self.base_ids.items()
        }

    def _configure_arx5(self):
        """Keep the legacy single-arm camera and TCP contract."""
        writer = self.writer
        writer.env = self.env
        writer.scene_state_buffer = SceneStateBuffer(self.env)
        writer.camera_record_to_sim_remap = {"head": "high", "wrist_left": "left_hand", "side": "right_hand"}
        writer.save_meta["videos"]["side"] = writer.save_meta["videos"].pop("wrist_right")
        self.arm_ids = self.policy.robot.find_joints([f"joint{i}" for i in range(1, 7)], preserve_order=True)[0]
        for field in ("target_object_pose", "container_object_pose"):
            schema = writer.save_trajectory["fields"][field]
            for key in ("right", "right_quat"):
                schema.pop(key, None)
            schema.update(left="xyz[m] in ARX5 base_link", left_quat="wxyz in ARX5 base_link")
        frame = self.frames()["left"]
        base_matrix = pose_matrix(frame["base_p"][0].cpu(), frame["base_q"][0].cpu())
        tcp_p, tcp_q = self.policy._tcp()
        tcp_matrix = pose_matrix(tcp_p[0].cpu(), tcp_q[0].cpu())
        for name, sensor_name in writer.camera_record_to_sim_remap.items():
            sensor = self.env.scene[sensor_name]
            data = sensor.data
            camera_matrix = pose_matrix(data.pos_w[0].cpu(), data.quat_w_ros[0].cpu())
            reference = tcp_matrix if name == "wrist_left" else base_matrix
            extrinsics = (np.linalg.inv(camera_matrix) @ reference).tolist()
            writer.save_meta["videos"][name].update(
                file=f"videos/{name}.mp4",
                pack_name=name,
                label=f"Isaac Sim {sensor_name}",
                info={"head": "固定全局相机", "wrist_left": "单臂手腕相机", "side": "固定侧视相机"}[name],
                width=sensor.cfg.width,
                height=sensor.cfg.height,
                intrinsics=data.intrinsic_matrices[0].cpu().tolist(),
                extrinsics=extrinsics if name == "wrist_left" else {"left": extrinsics},
                extrinsics_reference="left_tcp" if name == "wrist_left" else "left_base_link",
                sim_camera=sensor_name,
            )

    def record_step(self, action):
        step = len(self.backend.actions)
        frames = self.frames()
        if self.backend.layout.robot == "aloha":
            skills = self.policy.skills
            active = skills.object_id or skills.holding or self.tracked_object
            active = active or self.backend.task.goals[0].object_id
            goal = next((g for g in self.backend.task.goals if g.object_id == active), None)
            target = skills.target_id
            if target is None:
                target = getattr(goal, "target_id", None)
            poses = object_observations(
                self.env, self.objects, frames, target_id=active, container_id=target, env_id=self.backend.env_id
            )
            self.tracked_object = active
            obs = {arm + "_arm": self.backend.obs[arm + "_arm"] for arm in self.arms}
        else:
            import isaaclab.utils.math as math_utils

            frame = frames["left"]
            ee_p, ee_q = math_utils.subtract_frame_transforms(frame["base_p"], frame["base_q"], *self.policy._tcp())
            obs = {
                "left_arm": {
                    "arm_joint_pos": self.policy.robot.data.joint_pos[:, self.arm_ids],
                    "gripper_pos": self.policy.robot.data.joint_pos[:, self.policy.finger_ids].sum(dim=-1),
                    "ee_frame_pose_xyzw": torch.cat((ee_p, ee_q[:, [1, 2, 3, 0]]), dim=-1),
                },
                "control_action": action.detach().cpu(),
                "control_step": step,
            }
            poses = {"target_object_pose": relative_object_pose(self.env, self.backend.task.goals[0].object_id, frames)}
        self.writer.record_step(
            obs,
            epoch=self.backend.recording_epoch,
            duration=step * self.backend.dt,
            object_poses=poses,
            capture_cameras=self.backend.args.enable_cameras,
        )
