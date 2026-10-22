# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Measured-state task adapters using the registered robot action/frame conventions."""

import base64
import io
import math
import torch
from pathlib import Path

import isaaclab.utils.math as math_utils
from PIL import Image

from code_gen.runtime.skills import TaskSkills
from isaaclab_arena.environments.tabletop_environment import CUBE_SIZE, OBJECTS, TARGETS, TCP_OFFSET, default_scene
from isaaclab_arena.grasping.robot_layout import RobotLayout
from isaaclab_arena.grasping.skills import TabletopSkills
from isaaclab_arena.policy.policy_base import PolicyBase


class RobotPolicy(PolicyBase):
    """Shared observation services for independently implemented robot policies."""

    def _scene_snapshot(self):
        """Observe configured bodies, object contacts and live relative targets."""
        reader = getattr(self.base, "scene_state_reader", None)
        if reader is not None and not getattr(self, "_scene_reader_registered", False):
            reader.register_geometry(getattr(self, "env_id", 0), self.scene_spec["objects"])
            self._scene_reader_registered = True
        measured = reader.read() if reader is not None else None
        pos, quat = reader.tcp(self) if reader is not None else self._tcp()
        env_id = getattr(self, "env_id", 0)
        objects = {}
        for name, info in self.scene_spec["objects"].items():
            body = self.base.scene[name]
            if measured is None:
                root = body.data.root_state_w[env_id]
                p, q = root[:3], root[3:7]
                corners = torch.tensor(info["bbox_corners"], device=self.base.device, dtype=p.dtype)
                lowest_z = float((math_utils.quat_apply(q.expand(len(corners), -1), corners) + p)[:, 2].min())
                initial_p, initial_q = self.initial_poses[name]
            else:
                index = reader.names.index(name)
                root = measured["roots"][env_id, index]
                p, q = root[:3], root[3:7]
                lowest_z = float(measured["lowest_z"][env_id, index])
                if not hasattr(self, "_cpu_initial_poses"):
                    self._cpu_initial_poses = {n: (a.cpu(), b.cpu()) for n, (a, b) in self.initial_poses.items()}
                initial_p, initial_q = self._cpu_initial_poses[name]
            dot = max(0.0, min(1.0, abs(float(torch.dot(q, initial_q)))))
            obj = {
                "label": info["label"],
                "size": info["size"],
                "dimensions_m": info.get("dimensions_m"),
                "grasp_note": info.get("grasp_note", ""),
                "position": p.cpu().tolist(),
                "quaternion": q.cpu().tolist(),
                "lowest_z": lowest_z,
                "linear_velocity": root[7:10].cpu().tolist(),
                "angular_velocity": root[10:13].cpu().tolist(),
                "initial_position": initial_p.cpu().tolist(),
                "initial_rotation_error_rad": 2 * math.acos(dot),
                "grasp_offset_m": info.get("grasp_offset_m", [0, 0, 0]),
                "grasp_close_width_m": info.get("grasp_close_width_m", 0.0),
            }
            targets = self.scene_spec.get("contact_object_ids", [self.scene_spec.get("contact_object_id")])
            if name in targets:
                target_index = targets.index(name)
                forces = []
                for sensor_name in getattr(self, "contact_sensor_names", ("grasp_left_contact", "grasp_right_contact")):
                    if measured is not None:
                        contacts = measured.get("contact:" + sensor_name)
                        if contacts is None or contacts.shape[1] != len(targets):
                            raise ValueError("Object-specific contact sensor filters do not match scene metadata")
                        forces.append(float(contacts[env_id, target_index]))
                    else:
                        matrix = self.base.scene[sensor_name].data.force_matrix_w
                        if matrix is None or matrix.shape[2] != len(targets):
                            raise ValueError("Object-specific contact sensor filters do not match scene metadata")
                        forces.append(float(torch.linalg.vector_norm(matrix[env_id, :, target_index, :].sum(dim=0))))
                obj["finger_contact_forces_n"] = forces
            objects[name] = obj
        from isaaclab_arena.grasping.profiles import compose_pose

        targets = {}
        for name, target in self.scene_spec.get("targets", {}).items():
            target = dict(target)
            reference = target.get("reference_object_id")
            if not reference:
                target["position"] = [
                    float(value) + float(origin)
                    for value, origin in zip(
                        target["position"],
                        measured["origins"][env_id] if measured is not None else self.base.scene.env_origins[env_id],
                    )
                ]
            if reference:
                base = objects[reference]
                goal = compose_pose(
                    [*base["position"], *base["quaternion"]],
                    [*target["reference_position_m"], *target.get("reference_quaternion_wxyz", [1, 0, 0, 0])],
                )
                target.update(position=goal[:3], quaternion=goal[3:])
            targets[name] = target
        return {
            "geometry_source": "isaac_sim_ground_truth",
            "coordinate_frame": "world",
            "units": "metres",
            "frame_id": getattr(self, "frame_id", 0),
            "simulation_time": float(self.base.common_step_counter * self.base.step_dt),
            "tcp_pose": torch.cat((pos, quat), dim=-1)[0].cpu().tolist(),
            "quaternion_order": "wxyz",
            "joint_positions": (
                (measured["joints"][env_id] if measured is not None else self.robot.data.joint_pos[env_id])
                .cpu()
                .tolist()
            ),
            "joint_names": list(self.robot.data.joint_names),
            "gripper_width": self.gripper_width(),
            "objects": objects,
            "targets": targets,
            "holding": self.skills.holding if hasattr(self, "skills") else None,
            "pending_grasp": getattr(getattr(self, "skills", None), "pending_grasp", None),
            "grasp_uncertain": self.skills.grasp_uncertain if hasattr(self, "skills") else False,
            "phase": self.skills.phase if hasattr(self, "skills") else "idle",
            "scene_description": self.scene_spec.get("description", ""),
            "task_selection": self.scene_spec.get("task_selection"),
        }

    def images(self, observation, output_dir: str | None = None):
        result = {}
        for alias, key in (("head_global", "high_rgb"), ("wrist", "left_hand_rgb")):
            pixels = observation.get("camera_obs", {}).get(key)
            if pixels is None:
                continue
            image = Image.fromarray(pixels[0, ..., :3].detach().cpu().numpy().astype("uint8"))
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=80)
            result[alias] = "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
            if output_dir:
                directory = Path(output_dir)
                directory.mkdir(parents=True, exist_ok=True)
                image.save(directory / f"{alias}.png")
        return result

    @staticmethod
    def add_args_to_parser(parser):
        return parser

    @staticmethod
    def from_args(args):
        raise RuntimeError("Use code_gen; its runtime constructs the policy with the simulation environment.")


class AlohaPolicy(RobotPolicy):
    """Dual-arm adapter with Aloha-specific TCP, jaw and camera conventions."""

    def __init__(self, env, scene_spec, arm, *, env_id=0):
        self.base = env.unwrapped
        self.env_id = env_id
        self.layout = RobotLayout("aloha", arm)
        self.contact_sensor_names = tuple(f"grasp_{arm}_{index}_contact" for index in (0, 1))
        if not 0 <= env_id < self.base.num_envs or self.base.action_manager.total_action_dim != 14:
            raise ValueError("Aloha codegen requires a valid environment index and 14D joint actions")
        self.robot = self.base.scene["robot"]
        self.body_id = self.robot.find_bodies(self.layout.ee_link)[0][0]
        self.finger_ids = self.robot.find_joints(self.layout.finger_names, preserve_order=True)[0]
        self.offset = torch.tensor([self.layout.tcp_offset], device=self.base.device)
        self.tool_rotation = torch.tensor([self.layout.tcp_rotation], device=self.base.device)
        self.scene_spec = scene_spec or default_scene()
        self.initial_poses = {
            name: (
                self.base.scene[name].data.root_pos_w[env_id].clone(),
                self.base.scene[name].data.root_quat_w[env_id].clone(),
            )
            for name in self.scene_spec["objects"]
        }
        self.frame_id = 0
        table_z = self.scene_spec.get("aloha_layout", {}).get("table_top_z", 0.0)
        self.skills = TaskSkills(
            self.snapshot(),
            # The TCP is a frame origin, not the bottom of the gripper. Thin
            # parts can have valid grasp origins less than 5 mm above the table.
            # Keep the coarse table bound; robot geometry is checked by cuRobo.
            workspace_bounds=tuple(
                (lo + float(origin), hi + float(origin))
                for (lo, hi), origin in zip(
                    ((-0.8, 0.6), (-0.8, 0.8), (table_z, table_z + 0.85)), self.base.scene.env_origins[env_id]
                )
            ),
            max_gripper_width=0.08,
            home_joint_names=self.layout.arm_names,
        )

    def gripper_width(self):
        reader = getattr(self.base, "scene_state_reader", None)
        values = reader.read()["joints"] if reader is not None else self.robot.data.joint_pos
        joints = values[self.env_id, self.finger_ids]
        return float(joints[0] - joints[1])

    def _tcp(self):
        position = self.robot.data.body_pos_w[self.env_id : self.env_id + 1, self.body_id]
        quaternion = self.robot.data.body_quat_w[self.env_id : self.env_id + 1, self.body_id]
        return (
            position + math_utils.quat_apply(quaternion, self.offset),
            math_utils.quat_mul(quaternion, self.tool_rotation),
        )

    def snapshot(self):
        state = self._scene_snapshot()
        state["robot"] = "aloha"
        state["active_arm"] = self.layout.arm
        reader = getattr(self.base, "scene_state_reader", None)
        state["robot_root_pose"] = (
            (
                reader.read()["robot_root"][self.env_id]
                if reader is not None
                else self.robot.data.root_pose_w[self.env_id]
            )
            .cpu()
            .tolist()
        )
        state["table_layout"] = self.scene_spec.get("aloha_layout", {})
        for name, obj in state["objects"].items():
            calibrated = self.scene_spec["objects"][name].get("grasps_by_robot", {}).get("aloha", {})
            obj["grasp_frames"] = calibrated.get("grasp_frames", [])
            obj["default_grasp_id"] = calibrated.get("default_grasp_id")
            obj["grasp_offset_m"] = calibrated.get("grasp_offset_m", [0.0, 0.0, 0.0])
            obj["grasp_yaw_candidates_deg"] = calibrated.get("yaw_candidates_deg", [0])
            obj["grasp_tilt_candidates_deg"] = calibrated.get("tilt_candidates_deg", [0])
            obj["grasp_prefer_level_fingers"] = calibrated.get("prefer_level_fingers", False)
            obj["grasp_close_width_m"] = calibrated.get("close_width_m", 0.0)
            obj["grasp_contact_preload_m"] = calibrated.get("contact_preload_m", 0.004)
            obj["grasp_table_clearance_m"] = calibrated.get("table_clearance_m", 0.003)
            obj["grasp_follow_anchor_object_m"] = calibrated.get("follow_anchor_object_m")
            obj["grasp_lift_height_m"] = calibrated.get("lift_height_m", 0.15)
            obj["grasp_note"] = calibrated.get("note") or (
                "Aloha jaw-centre frame. ARX5 offsets are not transferable; "
                "default centred grasp must be validated in simulation."
            )
        return state

    def images(self, observation, output_dir=None):
        cameras = observation.get("camera_obs", {})
        selected = "left_camera_rgb" if self.layout.arm == "left" else "right_camera_rgb"
        return super().images(
            {"camera_obs": {"high_rgb": cameras.get("head_camera_rgb"), "left_hand_rgb": cameras.get(selected)}},
            output_dir,
        )

    def get_action(self, env, observation):
        raise RuntimeError("Aloha codegen executes only planned joint trajectories")


class ARX5Policy(RobotPolicy):
    """Single-arm tabletop control using bounded relative IK commands."""

    def __init__(self, env):
        super().__init__(config=None)
        self.base = env.unwrapped
        if self.base.num_envs != 1 or self.base.action_manager.total_action_dim != 7:
            raise ValueError("ARX5 control requires one environment and the 7D relative-EEF action.")
        self.robot = self.base.scene["robot"]
        self.body_id = self.robot.find_bodies("link6")[0][0]
        self.finger_ids = self.robot.find_joints(["joint7", "joint8"])[0]
        self.offset = torch.tensor([TCP_OFFSET], device=self.base.device)
        self.skills = TabletopSkills(self.snapshot())
        self.frame_id = 0
        self.command_pos, self.command_quat = (value.clone() for value in self._tcp())

    def _tcp(self):
        pos = self.robot.data.body_pos_w[:, self.body_id]
        quat = self.robot.data.body_quat_w[:, self.body_id]
        return pos + math_utils.quat_apply(quat, self.offset), quat

    def snapshot(self):
        pos, quat = self._tcp()
        objects = {}
        for name, info in OBJECTS.items():
            obj = self.base.scene[name]
            objects[name] = {
                "label": info["label"],
                "size": CUBE_SIZE,
                "position": obj.data.root_pos_w[0].detach().cpu().tolist(),
                "linear_velocity": obj.data.root_lin_vel_w[0].detach().cpu().tolist(),
            }
        return {
            "geometry_source": "isaac_sim_ground_truth",
            "coordinate_frame": "world",
            "units": "metres",
            "frame_id": getattr(self, "frame_id", 0),
            "simulation_time": float(self.base.common_step_counter * self.base.step_dt),
            "tcp_pose": torch.cat((pos, quat), dim=-1)[0].detach().cpu().tolist(),
            "quaternion_order": "wxyz",
            "joint_positions": self.robot.data.joint_pos[0].detach().cpu().tolist(),
            "gripper_width": float(self.robot.data.joint_pos[0, self.finger_ids].sum()),
            "objects": objects,
            "targets": {
                name: {"label": info["label"], "position": list(info["position"])} for name, info in TARGETS.items()
            },
            "holding": self.skills.holding if hasattr(self, "skills") else None,
            "grasp_uncertain": self.skills.grasp_uncertain if hasattr(self, "skills") else False,
            "phase": self.skills.phase if hasattr(self, "skills") else "idle",
        }

    def get_action(self, env, observation):
        self.frame_id += 1
        target = self.skills.tick(self.snapshot(), self.base.step_dt)
        pos, quat = self._tcp()
        desired_pos = torch.tensor([target.position], device=self.base.device, dtype=pos.dtype)
        desired_quat = torch.tensor([target.quaternion], device=self.base.device, dtype=quat.dtype)
        if self.skills.phase == "idle":
            self.command_pos, self.command_quat = pos.clone(), quat.clone()
        dp, dr = math_utils.compute_pose_error(
            self.command_pos, self.command_quat, desired_pos, desired_quat, rot_error_type="axis_angle"
        )
        # Cartesian speed limits apply to the TCP, including during reorientation.
        dp *= torch.clamp(0.08 * self.base.step_dt / dp.norm(dim=-1, keepdim=True).clamp_min(1e-8), max=1.0)
        dr *= torch.clamp(0.65 * self.base.step_dt / dr.norm(dim=-1, keepdim=True).clamp_min(1e-8), max=1.0)
        next_pos, next_quat = math_utils.apply_delta_pose(
            self.command_pos, self.command_quat, torch.cat((dp, dr), dim=-1)
        )
        self.command_pos, self.command_quat = next_pos, next_quat
        flange_pos = next_pos - math_utils.quat_apply(next_quat, self.offset)
        root_pos, root_quat = self.robot.data.root_pos_w, self.robot.data.root_quat_w
        desired_pos_b, desired_quat_b = math_utils.subtract_frame_transforms(root_pos, root_quat, flange_pos, next_quat)
        current_pos_b, current_quat_b = math_utils.subtract_frame_transforms(
            root_pos,
            root_quat,
            self.robot.data.body_pos_w[:, self.body_id],
            self.robot.data.body_quat_w[:, self.body_id],
        )
        delta_pos, delta_rot = math_utils.compute_pose_error(
            current_pos_b, current_quat_b, desired_pos_b, desired_quat_b, rot_error_type="axis_angle"
        )
        action = torch.zeros((1, 7), device=self.base.device)
        action[:, :6] = torch.cat((delta_pos, delta_rot), dim=-1) / 0.5  # embodiment action scale
        action[:, 6] = max(0.0, min(0.088, target.gripper_width))
        if not torch.isfinite(action).all():
            raise RuntimeError("Non-finite robot state; simulation control stopped.")
        return action


class ConfiguredARX5Policy(ARX5Policy):
    """ARX5 control with the same configured scene schema as other robots."""

    def __init__(self, env, scene_spec):
        self.scene_spec = scene_spec
        self.initial_poses = {
            name: (
                env.unwrapped.scene[name].data.root_pos_w[0].clone(),
                env.unwrapped.scene[name].data.root_quat_w[0].clone(),
            )
            for name in scene_spec["objects"]
        }
        super().__init__(env)

    def gripper_width(self):
        return float(self.robot.data.joint_pos[0, self.finger_ids].sum())

    def snapshot(self):
        return self._scene_snapshot()
