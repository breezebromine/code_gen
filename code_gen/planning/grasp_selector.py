# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Task grasp-frame adapter for MGG's bounded, first-success batch search."""

import math
import time

from code_gen._io import write_json
from isaaclab_arena.grasping.batch_planner import PosePlanRequest, resolve_planning_steps
from isaaclab_arena.grasping.geometry import compose, inverse, workspace_rejection
from isaaclab_arena.grasping.library.orientations import grasp_orientations, grasp_ranking_key


class GraspSelector:
    MAX_CANDIDATES = 64
    BATCH_SIZE = 8

    def __init__(self, motion, snapshot, *, object_id=None):
        self.m, self.snapshot = motion, snapshot
        self.lower = motion.planner._get_planner(motion.layout.arm)
        self.skills = motion.policy.skills
        self.object_id = object_id if object_id is not None else self.skills.object_id
        self.stages = {stage.name: stage for stage in self.skills.stages}
        self.path = motion.output / f"grasp_candidates_{len(motion.grasp_selections):02d}.json"
        self.report = {
            "object_id": self.object_id,
            "search_policy": "mgg",
            "search_method": "ActionPlanner.iter_grasp_candidate_batches/cuRobo.plan_batch",
            "selected_yaw_deg": None,
            "prediction_only": True,
            "preflight_scope": ["approach"],
            "later_phases": "plan_and_validate_during_execution",
            "candidates": [],
            "exhausted": False,
            "search_round_limit": 1,
            "search_budget": {
                "max_candidates": self.MAX_CANDIDATES,
                "batch_size": self.BATCH_SIZE,
                "batch_planning_attempts": 2,
                "approach_rounds_per_candidate": 1,
            },
        }

    def world_pose(self, position, quaternion):
        """Convert the task's world TCP pose into ActionPlanner's env-local flange pose."""
        import torch

        from isaaclab_arena.grasping.solver import world_tcp_to_planner_pose

        return world_tcp_to_planner_pose(
            position,
            quaternion,
            tcp_offset=self.m.policy.offset[0],
            env_origin=self.m.env.scene.env_origins[self.m.env_id],
            tool_rotation=self.m.policy.tool_rotation[0] if self.m.layout.robot == "aloha" else None,
            dtype=torch.float32,
        )

    def ranking_key(self, row, prefer_level=False):
        """Cheap geometric preference; future task paths are not evaluated here."""
        preferred = self.preferred_candidates
        candidate_id = row["candidate_id"]
        rank = preferred.index(candidate_id) if candidate_id in preferred else len(preferred)
        w, x, y, z = row["quaternion"]
        down_alignment = max(-1.0, min(1.0, -2 * (x * z - w * y)))
        # Slanted tabletop grasps are useful too. Keep the supported 45-degree
        # approach cone ahead of horizontal grasps within the candidate budget.
        vertical_tier = 0 if down_alignment >= math.cos(math.radians(45)) else 1
        return (rank, vertical_tier, *grasp_ranking_key(row, prefer_level))

    def bounded_candidates(self, candidates):
        """Apply cheap workspace checks, then MGG's preferred/fallback sampling."""
        obj = self.snapshot["objects"][self.object_id]
        admissible = []
        for candidate in candidates:
            row = dict(candidate)
            row.setdefault("position", list(self.stages["descend"].target.position))
            row.setdefault("approach_position", list(self.stages["approach"].target.position))
            rejection = workspace_rejection(row, self.skills.workspace_bounds)
            if rejection:
                self.report["candidates"].append({**row, "rejection": rejection, "preflight_success": False})
                continue
            row["ranking_score"] = min(row["yaw_deg"], 360 - row["yaw_deg"]) + abs(row["tilt_deg"])
            admissible.append(row)
        admissible.sort(key=lambda row: self.ranking_key(row, obj.get("grasp_prefer_level_fingers", False)))
        preferred, fallback = [], []
        for row in admissible:
            rank, vertical_tier, *_ = self.ranking_key(row, obj.get("grasp_prefer_level_fingers", False))
            target = preferred if rank < len(self.preferred_candidates) or vertical_tier == 0 else fallback
            target.append(row)
        selected, _ = self.m.planner.prioritize_grasp_candidates(
            preferred,
            list(range(len(preferred))),
            fallback,
            list(range(len(preferred), len(admissible))),
            max_candidates=self.MAX_CANDIDATES,
        )
        self.report.update(
            admissible_candidates=len(admissible),
            planned_candidate_pool=len(selected),
            planned_candidate_ids=[row["candidate_id"] for row in selected],
        )
        return selected

    def endpoint_batches(self, candidates):
        """Screen small batches against the declared live insertion destination."""
        declaration = (self.m.backend.scene_spec or {}).get("grasp_insertion_goals", {}).get(self.object_id)
        if not declaration:
            yield candidates
            return
        target = self.snapshot["targets"][declaration["target_id"]]
        goal = [*target["position"], *target["quaternion"]]
        obj = self.snapshot["objects"][self.object_id]
        object_pose = [*obj["position"], *obj["quaternion"]]
        phases = ["above", "insert"]
        if declaration.get("retreat_height_m"):
            phases.append("retreat")
        if declaration.get("transport_preserves_orientation"):
            phases.insert(0, "transport_above")
        phases += ["grasp", "approach"]
        self.report.update(preflight_scope=phases, endpoint_evaluations=0, endpoint_rejected=0)
        self.lower.update_world()
        # Similar aperture candidates can share one collision model in batched
        # IK. Use the available contact aperture, rounded up by at most 1 mm.
        buckets = {}
        for row in candidates:
            width = row.get("predicted_held_width_m", row.get("close_width_m", 0.0))
            bucket = math.ceil(width * 1000)
            buckets.setdefault(bucket, []).append(row)
        groups = (
            (bucket / 1000, rows[start : start + self.BATCH_SIZE])
            for bucket, rows in buckets.items()
            for start in range(0, len(rows), self.BATCH_SIZE)
        )
        for nominal_width, group in groups:
            endpoints = []
            for row in group:
                grasp = [*row["position"], *row["quaternion"]]
                # world_T_goal_object @ object_T_tcp; never use the source yaw
                # directly as the destination wrist orientation.
                final = compose(goal, compose(inverse(object_pose), grasp))
                above = list(final)
                above[2] += declaration.get("approach_height_m", 0.02)
                retreat = list(final)
                retreat[2] += declaration.get("retreat_height_m", 0.0)
                transport = compose([*goal[:3], *object_pose[3:]], compose(inverse(object_pose), grasp))
                transport[2] += declaration.get("approach_height_m", 0.02)
                endpoints.append({
                    "approach": [*row["approach_position"], *row["quaternion"]],
                    "grasp": grasp,
                    "above": above,
                    "transport_above": transport,
                    "insert": final,
                    "retreat": retreat,
                })
                row["endpoint_checks"] = []
            for phase in phases:
                if not group:
                    break
                open_jaws = phase in ("approach", "grasp")
                width = nominal_width + (0.00025 if phase == "retreat" else 0.0)
                if open_jaws:
                    width = max(row.get("open_width_m", self.stages["approach"].target.gripper_width) for row in group)
                fingers = {
                    name: sign * width / 2 for name, sign in zip(self.m.layout.finger_names, self.m.layout.finger_signs)
                }
                poses = [self.world_pose(e[phase][:3], e[phase][3:]) for e in endpoints]
                checks = self.m.planner.evaluate_ik_candidates(
                    self.m.env_id,
                    poses,
                    self.m.layout.arm,
                    batch_size=self.BATCH_SIZE,
                    excluded_object=self.object_id if phase != "approach" else None,
                    gripper_positions=fingers,
                )
                self.report["endpoint_evaluations"] += len(group)
                keep, next_endpoints = [], []
                for row, endpoint, check in zip(group, endpoints, checks):
                    row["endpoint_checks"].append({"phase": phase, "nominal_gripper_positions": dict(fingers), **check})
                    if check["success"]:
                        keep.append(row)
                        next_endpoints.append(endpoint)
                    else:
                        self.report["endpoint_rejected"] += 1
                        self.report["candidates"].append(
                            {**row, "preflight_success": False, "rejection": {"phase": phase, **check}}
                        )
                group, endpoints = keep, next_endpoints
            write_json(self.path, self.report)
            if group:
                yield group

    def candidate_batches(self, candidates):
        for group in self.endpoint_batches(candidates):
            # Ordinary pick/place tasks have no insertion endpoint screen and
            # may return the whole library here. Preserve MGG's fixed batches.
            for start in range(0, len(group), self.BATCH_SIZE):
                chunk = group[start : start + self.BATCH_SIZE]
                width = max(row.get("open_width_m", self.stages["approach"].target.gripper_width) for row in chunk)
                self.lower.config.gripper_open_positions = {
                    name: sign * width / 2 for name, sign in zip(self.m.layout.finger_names, self.m.layout.finger_signs)
                }
                self.lower._set_gripper_state(False)
                poses = [self.world_pose(row["approach_position"], row["quaternion"]) for row in chunk]
                yield chunk, poses

    def select_batched_steps(self, candidates):
        """Use the same first-success search loop and selection rule as MGG."""
        self.report.update(approach_batches=0, approach_evaluations=0, approach_candidates_evaluated=0)
        for group, poses in self.candidate_batches(candidates):
            evaluations = yield PosePlanRequest(
                self.m.planner, self.m.env_id, self.m.layout.arm, poses, candidates=True, batch_size=self.BATCH_SIZE
            )
            if len(evaluations) != len(group):
                raise ValueError("Batch planner returned the wrong number of candidate results")
            self.report["approach_batches"] += 1
            self.report["approach_evaluations"] += len(group)
            self.report["approach_candidates_evaluated"] += len(group)
            viable = []
            for candidate, evaluation in zip(group, evaluations):
                row = {
                    **candidate,
                    "search_round": 0,
                    "approach_batch": self.report["approach_batches"],
                    "approach_plan_error_type": evaluation.get("error_type"),
                    "approach_plan_success": evaluation["success"],
                    "approach_plan_status": evaluation["status"],
                    "approach_plan_source": evaluation.get("source", "MGG_batch_candidate"),
                    "preflight_success": bool(evaluation["success"]),
                    "preflight_scope": self.report["preflight_scope"],
                }
                self.report["candidates"].append(row)
                if evaluation.get("error_type"):
                    self.report["stopping_reason"] = "infrastructure_error"
                    raise ValueError("Batch planner infrastructure error: " + evaluation["status"])
                if evaluation["success"]:
                    viable.append((row, evaluation))
            write_json(self.path, self.report)
            if not viable:
                continue
            row, evaluation = self.m.planner.choose_grasp_candidate(viable)
            if "grasp_id" in row:
                self.skills.apply_grasp_candidate(row)
            else:
                for stage in self.skills.stages:
                    if stage.name in ("approach", "descend", "close", "lift"):
                        stage.target.quaternion = tuple(row["quaternion"])
            self.m.preplanned_approach = {
                "trajectory": (evaluation["positions"], evaluation["velocities"], evaluation["joint_names"]),
                "source": evaluation.get("source", "MGG_batch_candidate"),
                "start": self.m.robot.data.joint_pos[self.m.env_id : self.m.env_id + 1, self.m.arm_ids].clone(),
            }
            self.report.update(
                stopping_reason="selected",
                selected_search_round=0,
                selected_yaw_deg=row["yaw_deg"],
                selected_tilt_deg=row["tilt_deg"],
                selected_candidate_id=row["candidate_id"],
                selected_plan_source=row["approach_plan_source"],
                selected_quaternion=row["quaternion"],
                selected_grasp_id=row.get("grasp_id"),
                selected_position=row["position"],
            )
            return row["quaternion"]
        self.report.update(
            stopping_reason="candidate_budget_exhausted",
            exhausted=True,
            failure_reason="search_budget_exhausted_not_proven_unreachable",
        )
        raise RuntimeError("Grasp search budget exhausted; reachability is undetermined")

    def select(self):
        return resolve_planning_steps(self.selection_steps())

    def selection_steps(self):
        lower = self.lower
        self.m.grasp_selections.append(self.report)
        obj = self.snapshot["objects"][self.object_id]
        if getattr(self.skills, "auto_grasp_library", False):
            candidates = self.skills.library_candidates(self.snapshot)
            self.report["candidate_source"] = "object_local_grasp_library"
        else:
            candidates = grasp_orientations(
                self.stages["approach"].target.quaternion,
                getattr(self.skills, "grasp_yaws", obj.get("grasp_yaw_candidates_deg", [0])),
                getattr(self.skills, "grasp_tilts", obj.get("grasp_tilt_candidates_deg", [0])),
            )
        preferences = getattr(self.m.backend.args, "preferred_grasp_candidates", {})
        self.preferred_candidates = list(preferences.get(self.object_id, []))
        excluded = getattr(self.m.backend.args, "excluded_grasp_candidates", {}).get(self.object_id, [])
        candidates = [row for row in candidates if row["candidate_id"] not in excluded]
        declaration = (self.m.backend.scene_spec or {}).get("grasp_insertion_goals", {}).get(self.object_id, {})
        self.report.update(
            preferred_candidate_ids=self.preferred_candidates,
            excluded_candidate_ids=list(excluded),
            available_candidate_ids=[row["candidate_id"] for row in candidates],
            available_yaws=sorted({row["yaw_deg"] for row in candidates}),
            # Retained only for the existing physical retry/retreat evidence contract.
            declared_retreat_height_m=declaration.get("retreat_height_m"),
        )
        started = time.monotonic()
        original_open = dict(lower.config.gripper_open_positions)
        original_closed = dict(lower.config.gripper_closed_positions)
        try:
            candidates = self.bounded_candidates(candidates)
            if not candidates:
                self.report.update(exhausted=True, failure_reason="search_budget_exhausted_not_proven_unreachable")
                raise RuntimeError("No remaining grasp candidates")
            lower.config.gripper_open_positions = {
                name: sign * self.stages["approach"].target.gripper_width / 2
                for name, sign in zip(self.m.layout.finger_names, self.m.layout.finger_signs)
            }
            lower._set_gripper_state(False)
            return (yield from self.select_batched_steps(candidates))
        finally:
            lower.config.gripper_open_positions = original_open
            lower.config.gripper_closed_positions = original_closed
            lower._set_gripper_state(False)
            self.report["search_seconds"] = time.monotonic() - started
            write_json(self.path, self.report)
            print(
                f"CODEGEN grasp_search arm={self.m.layout.arm} object={self.object_id} "
                f"endpoint_checks={self.report.get('endpoint_evaluations', 0)} "
                f"endpoint_rejected={self.report.get('endpoint_rejected', 0)} "
                f"approach_candidates={self.report.get('approach_candidates_evaluated', 0)} "
                f"stop={self.report.get('stopping_reason')} seconds={self.report['search_seconds']:.2f}",
                flush=True,
            )
