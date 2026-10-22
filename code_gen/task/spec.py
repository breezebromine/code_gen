# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Frozen task specification, model contract, and simulator-independent goals."""

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class PlacementGoal:
    object_id: str
    target_id: str
    xy_tolerance_m: float = 0.025
    z_tolerance_m: float = 0.015
    max_speed_m_s: float = 0.02

    def __post_init__(self):
        if not all(isinstance(x, str) and x for x in (self.object_id, self.target_id)):
            raise ValueError("Goal object and target IDs are required")
        for value in (self.xy_tolerance_m, self.z_tolerance_m, self.max_speed_m_s):
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 0.1:
                raise ValueError("Goal tolerances must be finite, positive and at most 0.1")


@dataclass(frozen=True)
class TablePlacementGoal(PlacementGoal):
    """Bare-table placement uses measured geometry, not a cube-center assumption."""

    max_angular_speed_rad_s: float = 0.1
    max_other_translation_m: float = 0.004
    max_other_rotation_rad: float = math.radians(2)
    kind: str = "table_placement"

    def __post_init__(self):
        super().__post_init__()
        if self.kind != "table_placement":
            raise ValueError("Invalid table placement kind")
        for value in (self.max_angular_speed_rad_s, self.max_other_translation_m, self.max_other_rotation_rad):
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError("Table placement limits must be finite and positive")


@dataclass(frozen=True)
class LiftGoal:
    object_id: str
    reference_id: str
    reference_point_m: tuple[float, float, float]
    minimum_clearance_m: float = 0.015
    minimum_lift_m: float = 0.035
    max_speed_m_s: float = 0.01
    max_angular_speed_rad_s: float = 0.1
    minimum_finger_force_n: float = 0.005
    max_other_translation_m: float = 0.004
    max_other_rotation_rad: float = math.radians(2)
    kind: str = "lift"

    def __post_init__(self):
        if not all(isinstance(x, str) and x for x in (self.object_id, self.reference_id)) or self.kind != "lift":
            raise ValueError("Lift goal requires valid object/reference IDs")
        if len(self.reference_point_m) != 3 or not all(
            type(x) in (int, float) and math.isfinite(x) for x in self.reference_point_m
        ):
            raise ValueError("Reference point must be a finite XYZ vector")
        object.__setattr__(self, "reference_point_m", tuple(self.reference_point_m))
        for value in (
            self.minimum_clearance_m,
            self.minimum_lift_m,
            self.max_speed_m_s,
            self.max_angular_speed_rad_s,
            self.minimum_finger_force_n,
            self.max_other_translation_m,
            self.max_other_rotation_rad,
        ):
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError("Lift goal limits must be finite and positive")

    def evaluate(self, snapshot, *, movable_object_ids=()):
        obj = snapshot["objects"][self.object_id]
        reference = snapshot["objects"][self.reference_id]
        w, x, y, z = reference["quaternion"]
        # Third row of the reference quaternion rotation, applied to the local peg cap.
        row = (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y))
        cap_z = reference["position"][2] + sum(a * b for a, b in zip(row, self.reference_point_m))
        clearance = obj["lowest_z"] - cap_z
        lift = obj["position"][2] - obj["initial_position"][2]
        speed = math.sqrt(sum(v * v for v in obj["linear_velocity"]))
        angular_speed = math.sqrt(sum(v * v for v in obj["angular_velocity"]))
        forces = obj.get("finger_contact_forces_n", [])
        contact = len(forces) == 2 and all(math.isfinite(v) and v >= self.minimum_finger_force_n for v in forces)
        movable = {self.object_id, *movable_object_ids}
        others = [v for k, v in snapshot["objects"].items() if k not in movable]
        other_motion = max((math.dist(v["position"], v["initial_position"]) for v in others), default=0)
        other_rotation = max((v["initial_rotation_error_rad"] for v in others), default=0)
        finite = all(math.isfinite(v) for v in (clearance, lift, speed, angular_speed, other_motion, other_rotation))
        return {
            "object_id": self.object_id,
            "clearance_m": clearance,
            "lift_m": lift,
            "speed_m_s": speed,
            "angular_speed_rad_s": angular_speed,
            "finger_contact_forces_n": forces,
            "both_fingers_contact": contact,
            "other_max_translation_m": other_motion,
            "other_max_rotation_rad": other_rotation,
            "passed": (
                finite
                and contact
                and clearance >= self.minimum_clearance_m
                and lift >= self.minimum_lift_m
                and speed <= self.max_speed_m_s
                and angular_speed <= self.max_angular_speed_rad_s
                and other_motion <= self.max_other_translation_m
                and other_rotation <= self.max_other_rotation_rad
            ),
        }


@dataclass(frozen=True)
class TaskSpec:
    name: str
    instruction: str
    goals: tuple[PlacementGoal | LiftGoal, ...]
    hold_seconds: float = 1.0
    scene_file: str | None = None
    milestones: tuple["TaskSpec", ...] = ()

    def __post_init__(self):
        if (
            not isinstance(self.name, str)
            or not self.name
            or not isinstance(self.instruction, str)
            or not self.instruction
            or not isinstance(self.goals, tuple)
            or not 1 <= len(self.goals) <= 32
            or not all(isinstance(g, (PlacementGoal, LiftGoal)) for g in self.goals)
        ):
            raise ValueError("Task requires a name, instruction and one to 32 placement goals")
        if (
            type(self.hold_seconds) not in (int, float)
            or not math.isfinite(self.hold_seconds)
            or not 0.2 <= self.hold_seconds <= 10
        ):
            raise ValueError("Goal hold duration must be between 0.2 and 10 seconds")
        if (
            not isinstance(self.milestones, tuple)
            or len(self.milestones) > 32
            or any(not isinstance(m, TaskSpec) or m.milestones for m in self.milestones)
        ):
            raise ValueError("Milestones must be at most 32 non-nested task goals")
        placements = [g for g in self.goals if isinstance(g, PlacementGoal)]
        if len({g.target_id for g in placements}) != len(placements):
            raise ValueError("Tabletop goals cannot share an occupied target pad")
        if len({g.object_id for g in self.goals}) != len(self.goals):
            raise ValueError("Each object can have only one final goal")

    @property
    def all_goals(self):
        return tuple(g for milestone in self.milestones for g in milestone.goals) + self.goals

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        types = {"lift": LiftGoal, "table_placement": TablePlacementGoal, None: PlacementGoal}
        goals = tuple(types[g.get("kind")](**g) for g in value["goals"])
        milestones = tuple(
            cls.from_dict({"instruction": "Ordered task milestone", **m}) for m in value.get("milestones", ())
        )
        return cls(**{**value, "goals": goals, "milestones": milestones})

    def validate_scene(self, snapshot):
        for milestone in self.milestones:
            milestone.validate_scene(snapshot)
        for goal in self.goals:
            target_exists = (
                goal.reference_id in snapshot["objects"]
                if isinstance(goal, LiftGoal)
                else goal.target_id in snapshot["targets"]
            )
            if goal.object_id not in snapshot["objects"] or not target_exists:
                raise ValueError("Task goal references an absent object or target")

    def evaluate(self, snapshot, *, movable_object_ids=()):
        movable = {g.object_id for g in self.all_goals} | set(movable_object_ids)
        checks = []
        for goal in self.goals:
            if isinstance(goal, LiftGoal):
                checks.append(goal.evaluate(snapshot, movable_object_ids=movable))
                continue
            obj = snapshot["objects"][goal.object_id]
            target = snapshot["targets"][goal.target_id]["position"]
            position = obj["position"]
            xy_error = math.dist(position[:2], target[:2])
            z_error = abs(position[2] - target[2] - obj["size"] / 2)
            if isinstance(goal, TablePlacementGoal):
                z_error = abs(obj["lowest_z"] - target[2])
            speed = math.sqrt(sum(v * v for v in obj["linear_velocity"]))
            released = snapshot.get("holding") is None and not snapshot.get("grasp_uncertain", True)
            clear = math.dist(position, snapshot["tcp_pose"][:3]) > 0.06
            opened = snapshot["gripper_width"] > 0.05
            passed = all(math.isfinite(v) for v in (xy_error, z_error, speed)) and (
                xy_error <= goal.xy_tolerance_m
                and z_error <= goal.z_tolerance_m
                and speed <= goal.max_speed_m_s
                and released
                and clear
                and opened
            )
            extra = {}
            if isinstance(goal, TablePlacementGoal):
                angular_speed = math.sqrt(sum(v * v for v in obj["angular_velocity"]))
                others = [v for k, v in snapshot["objects"].items() if k not in movable]
                translation = max((math.dist(v["position"], v["initial_position"]) for v in others), default=0)
                rotation = max((v["initial_rotation_error_rad"] for v in others), default=0)
                passed = (
                    passed
                    and all(math.isfinite(v) for v in (angular_speed, translation, rotation))
                    and (
                        angular_speed <= goal.max_angular_speed_rad_s
                        and translation <= goal.max_other_translation_m
                        and rotation <= goal.max_other_rotation_rad
                    )
                )
                extra = {
                    "angular_speed_rad_s": angular_speed,
                    "other_max_translation_m": translation,
                    "other_max_rotation_rad": rotation,
                }
            checks.append({
                "object_id": goal.object_id,
                "target_id": goal.target_id,
                "xy_error_m": xy_error,
                "z_error_m": z_error,
                "speed_m_s": speed,
                "released": released,
                "gripper_clear": clear,
                "gripper_open": opened,
                **extra,
                "passed": passed,
            })
        return {"success": all(c["passed"] for c in checks), "checks": checks}
