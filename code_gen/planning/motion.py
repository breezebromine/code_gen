# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Joint trajectory execution through MGG ActionPlanner and native cuRobo."""

import hashlib
import numpy as np
import time
import torch
from contextlib import nullcontext
from pathlib import Path

from code_gen._io import write_json
from code_gen.planning.trajectory.collision import CollisionChecks
from isaaclab_arena.grasping.action_planner import ActionPlanner
from isaaclab_arena.grasping.batch_planner import PosePlanRequest, resolve_planning_steps
from isaaclab_arena.grasping.joint_actions import write_arm_action
from isaaclab_arena.grasping.robot_config import prepare_aloha_config, prepare_robot_config
from isaaclab_arena.grasping.solver import alignment_precision, world_tcp_to_planner_pose


class CuroboMotion:
    def __init__(self, backend, planner=None):
        self.backend = backend
        self.env = backend.env.unwrapped
        self.env_id = backend.env_id
        self.policy = backend.policy
        self.robot = self.policy.robot
        self.output = backend.output / "planning"
        self.output.mkdir(exist_ok=False)
        self.layout = backend.layout
        if self.layout.robot == "aloha":
            cfg = prepare_aloha_config(
                self.layout,
                self.output,
                dt=backend.dt,
                attachment_collision=(backend.scene_spec or {}).get("attachment_collision"),
            )
        else:
            if not backend.args.robot_urdf:
                raise ValueError("The cuRobo ARX5 adapter requires --robot-urdf and its collision config files")
            cfg = prepare_robot_config(backend.args.robot_urdf, self.output, self.robot, backend.dt)
        self.planner = planner or ActionPlanner(
            self.env,
            self.env.cfg.isaaclab_arena_env.embodiment,
            backend.args.seed,
            planner_configs={self.layout.arm: cfg},
            batch_mode=True,
            env_id=self.env_id,
        )
        self.collision_checks = CollisionChecks(self.planner, self.layout, self.output)
        self.arm_names = self.layout.arm_names
        self.arm_ids = self.robot.find_joints(self.arm_names, preserve_order=True)[0]
        # Preserve a preceding local servo's compensated command during planner handoff.
        self.hold = backend.joint_hold[:, self.layout.action_start : self.layout.action_start + 6].clone()
        self.stage = None
        self.local = None
        self.positions = None
        self.index = 0
        self.records = []
        self.grasp_selections = []
        self.grasp_quaternion = None
        self.preplanned_approach = None
        self.release_width_bias = 0.0
        lower = self.planner._get_planner(self.layout.arm)
        write_json(
            self.output / "world_geometry.json",
            {
                "rigid_object_paths": {
                    name: list(lower._get_object_paths(name)) for name in lower._get_object_mappings()
                },
                "all_obstacle_paths": [obj.name for obj in lower.motion_gen.world_model.objects],
                "mesh_bounds": {
                    obj.name: {
                        "vertices": len(obj.vertices),
                        "triangles": len(obj.faces),
                        "minimum": np.asarray(obj.vertices).min(axis=0).tolist(),
                        "maximum": np.asarray(obj.vertices).max(axis=0).tolist(),
                    }
                    for obj in lower.motion_gen.world_model.objects
                    if getattr(obj, "vertices", None) is not None and len(obj.vertices)
                },
            },
        )
        write_json(
            self.output / "config.json",
            {
                "backend": "ActionPlanner/CuroboPlanner/MotionGen",
                "physics_device": str(self.env.device),
                "planning_device": str(self.planner._get_planner(self.layout.arm).tensor_args.device),
                "robot_config_sha256": hashlib.sha256(Path(cfg.robot_config_file).read_bytes()).hexdigest(),
                "action_space": "joint_position[6], gripper_width_m per arm",
                "robot": self.layout.robot,
                "active_arm": self.layout.arm,
                "action_dim": self.layout.action_dim,
                "fallback": None,
                "interpolation_dt": backend.dt,
                "collision_cache": cfg.collision_cache_size,
                "trajectory_seeds": cfg.num_trajopt_seeds,
                "graph_seeds": cfg.num_graph_seeds,
                "geometry_policy": "merge all enabled collision Mesh/Cube/Cylinder geometry per rigid body",
                "world_ignore_substrings": cfg.world_ignore_substrings,
            },
        )

    def get_action(self):
        return resolve_planning_steps(self.action_steps())

    def action_steps(self):
        self.policy.frame_id += 1
        snapshot = self.policy.snapshot()
        skills = self.policy.skills
        if skills.skill == "align_object":
            complete = (
                bool(skills.stages)
                and self.stage is skills.stages[0]
                and self.positions is not None
                and self.index >= len(self.positions)
            )
            target = skills.tick(snapshot, self.backend.dt, trajectory_complete=complete)
        else:
            target = skills.tick(
                snapshot,
                self.backend.dt,
                trajectory_complete=(
                    self.local.done
                    if self.local is not None
                    else self.positions is None or self.index >= len(self.positions)
                ),
            )
        stage = skills.stages[0] if skills.stages else None
        if stage is not self.stage:
            if self.local is not None:
                self.local.close()
                self.backend.joint_hold[:, self.layout.action_start : self.layout.action_start + 6] = self.hold
                self.local = None
            self.stage = stage
            self.positions, self.index = None, 0
            if stage is not None and stage.name == "release" and getattr(skills, "insertion_release_object", None):
                self.collision_checks.prepare_release(stage, snapshot, skills, len(self.backend.events))
            if stage is not None and stage.name == "reopen":
                self.collision_checks.prepare_reopen(stage, snapshot, len(self.backend.events))
            if stage is not None and stage.check_pose:
                if skills.skill == "pick" and stage.name == "approach":
                    self.grasp_quaternion = None
                    obj = snapshot["objects"][skills.object_id]
                    if (
                        getattr(skills, "auto_grasp_library", False)
                        or getattr(skills, "grasp_yaws", obj.get("grasp_yaw_candidates_deg", [0])) != [0]
                        or getattr(skills, "grasp_tilts", obj.get("grasp_tilt_candidates_deg", [0])) != [0]
                    ):
                        from code_gen.planning.grasp_selector import GraspSelector

                        self.grasp_quaternion = yield from GraspSelector(self, snapshot).selection_steps()
                if skills.skill == "place" and self.grasp_quaternion is not None:
                    for pending in skills.stages:
                        pending.target.quaternion = tuple(snapshot["tcp_pose"][3:])
                local = (
                    (skills.skill == "pick" and stage.name in ("descend", "lift"))
                    or (skills.skill == "place" and stage.name in ("raise", "lower", "retreat"))
                    or getattr(skills, "require_vertical", False)
                )
                if local:
                    from code_gen.runtime.cartesian import LocalSegment

                    self.local = LocalSegment(self, stage, snapshot)
                else:
                    yield from self.plan_steps(stage, snapshot)
        if self.local is not None:
            self.hold = yield from self.local.advance_steps(snapshot)
        if self.positions is not None:
            self.hold = self.positions[min(self.index, len(self.positions) - 1)].unsqueeze(0)
            self.index += 1
        action = self.backend.joint_hold.clone()
        write_arm_action(
            action,
            0,
            arm_joint_names=self.arm_names,
            offset=self.layout.action_start,
            gripper_width=self.gripper_command(target.gripper_width, stage, snapshot),
            positions=self.hold[0],
            joint_names=self.arm_names,
        )
        return action

    def gripper_command(self, width, stage, snapshot):
        """Remove static jaw tracking error within the checked release reserve."""
        release = stage is not None and (
            stage.name == "reopen"
            or (stage.name == "release" and getattr(self.policy.skills, "insertion_release_object", None))
        )
        if not release:
            self.release_width_bias = 0.0
            return width
        # prepare_release checks 0.5 mm around the robot. Limit extra jaw
        # travel to 0.2 mm each; acceptance still uses the original target.
        error = width - snapshot["gripper_width"]
        self.release_width_bias = float(
            np.clip(self.release_width_bias + 2.0 * self.backend.dt * error, -0.0004, 0.0004)
        )
        return float(np.clip(width + self.release_width_bias, 0.0, self.policy.skills.max_gripper_width))

    def plan(self, stage, snapshot):
        return resolve_planning_steps(self.plan_steps(stage, snapshot))

    def plan_steps(self, stage, snapshot):
        """Plan global motion; vertical stages use the shared local controller."""
        position = torch.tensor(stage.target.position, device=self.env.device)
        quaternion = torch.tensor(stage.target.quaternion, device=self.env.device)
        pose = world_tcp_to_planner_pose(
            position,
            quaternion,
            tcp_offset=self.policy.offset[0],
            env_origin=self.env.scene.env_origins[self.env_id],
            tool_rotation=self.policy.tool_rotation[0] if self.layout.robot == "aloha" else None,
        )
        skills = self.policy.skills
        attached = skills.holding or getattr(skills, "pending_grasp", None)
        if attached:
            from isaaclab_arena.grasping.geometry import contact_evidence

            evidence = contact_evidence(snapshot, attached)
            proven_recent = (
                getattr(skills, "grasp_reference", None) is not None
                and (skills.holding or getattr(skills, "pending_grasp", None)) == attached
                and skills.check_follow(snapshot)
            )
            if not evidence["passed"] and not proven_recent:
                raise RuntimeError(f"GRASP_NOT_ESTABLISHED: refusing carried-object planning: {evidence}")
        lower = self.planner._get_planner(self.layout.arm)
        actual_fingers = self.robot.data.joint_pos[self.env_id, self.policy.finger_ids].tolist()
        lower.config.gripper_closed_positions = dict(zip(self.layout.finger_names, actual_fingers))
        lower.config.gripper_open_positions = {
            name: sign * stage.target.gripper_width / 2
            for name, sign in zip(self.layout.finger_names, self.layout.finger_signs)
        }
        record = {
            "phase": stage.name,
            "action": skills.skill,
            "start_joints": snapshot["joint_positions"],
            "goal_link6_pose": pose.cpu().tolist(),
            "attached_object": attached,
            "success": False,
        }
        self.records.append(record)
        write_json(self.output / "plans.json", self.records)
        started = time.monotonic()
        cached = getattr(self, "preplanned_approach", None) if stage.name == "approach" else None
        if cached is not None and torch.allclose(
            self.robot.data.joint_pos[self.env_id : self.env_id + 1, self.arm_ids], cached["start"], atol=1e-5, rtol=0
        ):
            trajectory = cached["trajectory"]
            record["source"] = cached.get("source", "MGG_batch_candidate")
        elif stage.name == "home" and skills.home_joint_positions:
            home_positions = torch.tensor(
                [skills.home_joint_positions[name] for name in self.arm_names], device=self.env.device
            )
            # Physics can settle a joint a few microradians beyond its stop.
            # Match cuRobo's start-state drift tolerance without relaxing limits.
            indices = [lower.motion_gen.kinematics.joint_names.index(name) for name in self.arm_names]
            limits = lower.motion_gen.kinematics.get_joint_limits().position[:, indices].to(home_positions.device)
            bounded = home_positions.clamp(min=limits[0], max=limits[1])
            drift = float((bounded - home_positions).abs().max())
            if drift > 1e-3:
                raise ValueError(f"Initial arm configuration exceeds joint limits by {drift:.6f} rad")
            home_positions = bounded
            record["goal_joint_positions"] = dict(zip(self.arm_names, home_positions.cpu().tolist()))
            record["initial_joint_limit_drift_rad"] = drift
            trajectory = self.planner.plan_to_joint_positions(self.env_id, self.layout.arm, home_positions)
        else:
            measured_correction = skills.skill == "move_object_to_pose" and getattr(skills, "transport_attempt", 0) > 0
            precision_required = skills.skill == "align_object" or measured_correction
            precision = alignment_precision(lower) if precision_required else nullcontext()
            if precision_required:
                record["alignment_precision"] = {
                    "position_threshold_m": 0.0002,
                    "rotation_threshold": 0.002,
                    "max_collision_cost_activation_m": 0.01,
                }
                if measured_correction:
                    record["transport_correction_attempt"] = skills.transport_attempt
            with precision:

                def refresh_grasp():
                    if attached and not skills.check_follow(self.policy.snapshot()):
                        raise RuntimeError("GRASP_NOT_ESTABLISHED: grasp changed while waiting for planning")

                rows = yield PosePlanRequest(
                    self.planner,
                    self.env_id,
                    self.layout.arm,
                    [pose],
                    attached_object=attached,
                    precision=precision_required,
                    refresh=refresh_grasp if attached else None,
                )
                result = rows[0]
                record["source"] = result.get("source", "single_env")
                if "payload_validation" in result:
                    record["payload_validation"] = result["payload_validation"]
                trajectory = (
                    (result["positions"], result["velocities"], result["joint_names"]) if result["success"] else None
                )
        if stage.name == "approach":
            self.preplanned_approach = None
        record["elapsed_seconds"] = time.monotonic() - started
        record["status"] = "Success" if trajectory is not None else str(getattr(lower, "last_plan_status", "unknown"))
        record["world_objects"] = list(lower._get_object_mappings())
        if trajectory is None:
            if "WORLD_COLLISION" in record["status"]:
                try:
                    self.collision_checks.diagnose_start(attached)
                except Exception as exc:
                    record["collision_diagnostic_error"] = str(exc)
            write_json(self.output / "plans.json", self.records)
            raise RuntimeError(f"cuRobo planning failed in {stage.name}: {record['status']}; no IK fallback")
        positions, _, names = trajectory
        self.index = 0
        indices = [list(names).index(name) for name in self.arm_names]
        self.positions = positions[:, indices].to(self.env.device)
        if not len(self.positions) or not torch.isfinite(self.positions).all():
            raise RuntimeError("cuRobo returned an empty or non-finite joint trajectory")
        record["success"] = True
        record["waypoints"] = len(positions)
        record["joint_names"] = list(names)
        write_json(self.output / "plans.json", self.records)
        print(f"CUROBO phase={stage.name} waypoints={len(positions)} status={record['status']}", flush=True)
