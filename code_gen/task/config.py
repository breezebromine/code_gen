# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Declarative taskinfo and seed preparation for reusable task programs."""

import copy
import json
import math
import random
from dataclasses import replace
from pathlib import Path

from code_gen._io import digest_file, resolve_path, write_json
from code_gen.task.catalog import merge, read_document, resolve_taskinfo
from code_gen.task.spec import TaskSpec


def development_seeds(development):
    """Preserve legacy fixed seeds; optional retries use a deterministic disjoint pool."""
    seeds = [development["generation_seed"], *development["validation_seeds"]]
    budget = development.get("max_layout_attempts", len(seeds))
    return seeds + list(range(max(seeds) + 1, max(seeds) + 1 + budget - len(seeds)))


class BatchTaskInfo:
    """Paths in taskinfo are relative to the repository, or absolute container paths."""

    def __init__(self, document, *, catalog=None):
        self.source_document = copy.deepcopy(document)
        document, self.catalog_files = resolve_taskinfo(document, catalog)
        self.document = copy.deepcopy(document)
        if document.get("schema_version") != 1:
            raise ValueError("Expected taskinfo schema_version=1")
        allowed = {
            "schema_version",
            "task",
            "scene_file",
            "scene_overrides",
            "robot",
            "randomization",
            "development",
            "collection",
        }
        if set(document) - allowed:
            raise ValueError("Unknown taskinfo fields: " + str(set(document) - allowed))
        try:
            self.task = TaskSpec.from_dict(document["task"])
        except (KeyError, TypeError) as exc:
            raise ValueError(f"Invalid task success specification: {exc}") from exc
        if self.task.scene_file is not None:
            raise ValueError("Use top-level scene_file, not task.scene_file")
        self.robot = document["robot"]
        if set(self.robot) - {"name", "arm", "assets_root"}:
            raise ValueError("Unknown robot settings")
        if self.robot["name"] != "aloha" or self.robot["arm"] not in ("left", "right", "auto"):
            raise ValueError("Batch taskinfo supports Aloha left/right/auto with cuRobo")
        self.assets = resolve_path(self.robot["assets_root"])
        self.scene_path = resolve_path(document["scene_file"])
        self.scene = merge(json.loads(self.scene_path.read_text()), document.get("scene_overrides", {}))
        self.task.validate_scene(self.scene)
        self._validate_scene_metadata()
        from isaaclab_arena.grasping.robot_layout import aloha_robot_path

        if not aloha_robot_path(self.assets).is_file():
            raise FileNotFoundError(aloha_robot_path(self.assets))
        self.randomization = document["randomization"]
        self._validate_randomization()
        self._prepare_visuals()
        self.development, self.collection = document["development"], document["collection"]
        self._validate_budgets()

    def _validate_scene_metadata(self):
        """Validate support/contact declarations before constructing a simulator."""
        for obj in self.scene["objects"].values():
            anchor = obj.get("grasps_by_robot", {}).get(self.robot["name"], {}).get("follow_anchor_object_m")
            if anchor is not None:
                corners = obj.get("bbox_corners", [])
                if (
                    not isinstance(anchor, list)
                    or len(anchor) != 3
                    or not corners
                    or any(type(v) not in (int, float) or not math.isfinite(v) for v in anchor)
                    or any(
                        not min(p[i] for p in corners) - 1e-6 <= value <= max(p[i] for p in corners) + 1e-6
                        for i, value in enumerate(anchor)
                    )
                ):
                    raise ValueError("Grasp follow anchor must lie inside the object's authored local bounds")
        for object_id, contact in self.scene.get("support_contacts", {}).items():
            if (
                object_id not in self.scene["objects"]
                or contact.get("support_id") not in self.scene["objects"]
                or contact["support_id"] == object_id
                or not 0.008 <= contact.get("distance_m", 0) <= 0.015
            ):
                raise ValueError("Support contacts require distinct scene objects and an 8–15 mm departure")
        for contact in self.scene.get("contact_surfaces", []):
            if (
                contact.get("object_id") not in self.scene["objects"]
                or contact.get("target_id") not in self.scene["targets"]
            ):
                raise ValueError("Contact surfaces require existing object and target IDs")
            if ("support_id" in contact) == ("obstacle_paths" in contact):
                raise ValueError("Declare either a support object or exact static obstacle paths")
            if "support_id" in contact and (
                contact["support_id"] not in self.scene["objects"] or contact["support_id"] == contact["object_id"]
            ):
                raise ValueError("Contact support must be a different scene object")
            if "obstacle_paths" in contact and (
                not contact["obstacle_paths"]
                or not all(isinstance(p, str) and p.startswith("/World/") for p in contact["obstacle_paths"])
            ):
                raise ValueError("Static contact supports require explicit absolute prim paths")
            if not 0 < contact.get("xy_radius_m", 0.01) <= 0.03:
                raise ValueError("Contact region radius must be within 30 mm")
        for object_id, goal in self.scene.get("grasp_insertion_goals", {}).items():
            if object_id not in self.scene["objects"] or goal.get("target_id") not in self.scene["targets"]:
                raise ValueError("Insertion grasp previews require existing object and target IDs")
            height = goal.get("approach_height_m", 0.02)
            if type(height) not in (int, float) or not 0.005 <= height <= 0.06:
                raise ValueError("Insertion grasp preview height must be 5..60 mm")
            retreat = goal.get("retreat_height_m")
            if retreat is not None and (type(retreat) not in (int, float) or not 0.01 <= retreat <= 0.15):
                raise ValueError("Insertion grasp preview retreat must be 10..150 mm")

    def _prepare_visuals(self):
        """Resolve optional appearance pools once, before sampling layouts."""
        visuals = self.randomization.get("visuals")
        self.hdrs, self.cloths = [], []
        if visuals:
            if set(visuals) != {"hdr_dir", "tablecloth_dir", "hdr_intensity"}:
                raise ValueError("Visuals require hdr_dir, tablecloth_dir, hdr_intensity")
            self.hdrs = sorted(resolve_path(visuals["hdr_dir"]).glob("*.hdr"))
            self.cloths = sorted(resolve_path(visuals["tablecloth_dir"]).glob("*.png"))
            if len(self.hdrs) < 2 or len(self.cloths) < 2:
                raise ValueError("Random visuals require at least two HDRs and two PNGs")
            if not math.isfinite(visuals["hdr_intensity"]) or visuals["hdr_intensity"] <= 0:
                raise ValueError("HDR intensity must be positive")

    def _validate_budgets(self):
        """Check collection quotas and the disjoint development seed pool."""
        required_development = {"generation_seed", "validation_seeds", "max_versions"}
        if not required_development <= set(self.development) or set(self.development) - required_development - {
            "max_layout_attempts"
        }:
            raise ValueError("Invalid development settings")
        if set(self.collection) != {"start_seed", "max_cases", "target_successes"}:
            raise ValueError("Collection requires start_seed, max_cases, target_successes")
        seeds = self.development["validation_seeds"]
        if len(seeds) < 2 or len(set(seeds)) != len(seeds) or not all(type(s) is int for s in seeds):
            raise ValueError("At least two distinct integer validation seeds required")
        if type(self.development["generation_seed"]) is not int:
            raise ValueError("generation_seed must be an integer")
        if self.development["generation_seed"] in seeds:
            raise ValueError("Validation seeds must differ from generation seed")
        for value in (
            self.development["max_versions"],
            self.collection["target_successes"],
        ):
            if type(value) is not int or value < 1:
                raise ValueError("Budgets must be positive integers")
        if self.development["max_versions"] > 10:
            raise ValueError("At most ten development versions")
        if type(self.collection["start_seed"]) is not int:
            raise ValueError("start_seed must be an integer")
        if type(self.collection["max_cases"]) is not int or self.collection["max_cases"] < 0:
            raise ValueError("max_cases must be a nonnegative integer; 0 means unlimited")
        if self.collection["max_cases"] and self.collection["target_successes"] > self.collection["max_cases"]:
            raise ValueError("target_successes exceeds max_cases")
        budget = self.development.get("max_layout_attempts", len(seeds) + 1)
        if type(budget) is not int or budget < len(seeds) + 1:
            raise ValueError("max_layout_attempts must cover all required development successes")
        development_pool = development_seeds(self.development)
        collecting = range(self.collection["start_seed"], self.collection["start_seed"] + self.collection["max_cases"])
        if any(
            (seed in collecting if self.collection["max_cases"] else seed >= self.collection["start_seed"])
            for seed in development_pool
        ):
            raise ValueError("Development and collection seeds must be disjoint")

    def _validate_randomization(self):
        if set(self.randomization) - {"rigid_groups", "visuals", "assembly_scatter"}:
            raise ValueError("Unknown randomization settings")
        assembly = self.randomization.get("assembly_scatter")
        if assembly is not None:
            self._validate_assembly(assembly)
        self._validate_rigid_groups()

    def _validate_assembly(self, assembly):
        """Validate a composed frame and its socket-relative scatter configuration."""
        fields = {
            "base_object",
            "parts",
            "part_targets",
            "required_scattered",
            "start_targets",
            "table_bounds_xy",
            "surface_z_m",
            "min_separation_m",
            "max_sample_tries",
        }
        if (
            not fields <= set(assembly)
            or set(assembly) - fields - {"stack_groups", "select_from_scattered"}
            or self.randomization.get("rigid_groups")
        ):
            raise ValueError("Assembly scatter requires its complete schema and no rigid_groups")
        names, base = assembly["parts"], assembly["base_object"]
        if not names or len(set(names)) != len(names) or base in names:
            raise ValueError("Assembly requires distinct part names and a separate base")
        if any(name not in self.scene["objects"] for name in [base, *names]):
            raise ValueError("Assembly objects must exist")
        if not set(assembly["required_scattered"]) <= set(names):
            raise ValueError("Required scattered parts must belong to the assembly")
        if len(set(assembly["required_scattered"])) != len(assembly["required_scattered"]):
            raise ValueError("Required scattered parts must be unique")
        if set(assembly["part_targets"]) != set(names):
            raise ValueError("Every assembly part requires a local socket target")
        for name, target_id in assembly["part_targets"].items():
            target = self.scene["targets"].get(target_id, {})
            if target.get("reference_object_id") != base or len(target.get("reference_position_m", [])) != 3:
                raise ValueError("Socket targets must reference the assembly base")
        if not set(assembly["start_targets"]) <= set(names) or any(
            target not in self.scene["targets"] for target in assembly["start_targets"].values()
        ):
            raise ValueError("Invalid scatter start target")
        candidates = assembly.get("select_from_scattered", [])
        if "select_from_scattered" in assembly:
            if not candidates or len(set(candidates)) != len(candidates) or not set(candidates) <= set(names):
                raise ValueError("Tabletop selection requires distinct assembly candidates")
            if len(self.task.goals) != 1 or self.task.milestones or not hasattr(self.task.goals[0], "target_id"):
                raise ValueError("Tabletop selection supports one placement goal without milestones")
            if self.task.goals[0].object_id not in candidates:
                raise ValueError("The goal template must reference a selection candidate")
        stacked = set()
        for group in assembly.get("stack_groups", []):
            if set(group) != {"parts", "layer_pitch_m"}:
                raise ValueError("Stack groups require ordered parts and a layer pitch")
            members, pitch = group["parts"], group["layer_pitch_m"]
            if not members or len(set(members)) != len(members) or not set(members) <= set(names):
                raise ValueError("Stack members must be distinct assembly parts")
            if stacked.intersection(members):
                raise ValueError("A part cannot belong to multiple stacks")
            stacked.update(members)
            if type(pitch) not in (int, float) or not math.isfinite(pitch) or pitch <= 0:
                raise ValueError("Stack layer pitch must be positive and finite")
            targets = [self.scene["targets"][assembly["part_targets"][name]] for name in members]
            positions = [target["reference_position_m"] for target in targets]
            if any(math.dist(position, positions[0]) > 1e-8 for position in positions):
                raise ValueError("Stack targets must declare the same bottom-layer reference position")
            rotations = [target.get("reference_quaternion_wxyz", [1, 0, 0, 0]) for target in targets]
            if any(rotation != rotations[0] for rotation in rotations):
                raise ValueError("Stack members must share the same target orientation")
        bounds = assembly["table_bounds_xy"]
        if len(bounds) != 2 or any(len(v) != 2 or not all(math.isfinite(x) for x in v) or v[0] >= v[1] for v in bounds):
            raise ValueError("Invalid assembly placement bounds")
        if not math.isfinite(assembly["surface_z_m"]) or not math.isfinite(assembly["min_separation_m"]):
            raise ValueError("Assembly heights and clearances must be finite")
        if (
            assembly["min_separation_m"] < 0
            or type(assembly["max_sample_tries"]) is not int
            or assembly["max_sample_tries"] < 1
        ):
            raise ValueError("Invalid assembly sampling limits")
        if any(not self.scene["objects"][name].get("bbox_corners") for name in [base, *names]):
            raise ValueError("Assembly scatter requires body-local bounding boxes")

    def _validate_rigid_groups(self):
        """Keep members unique and sampling ranges within declared scene bounds."""
        occupied = set()
        for group in self.randomization.get("rigid_groups", []):
            required = {"pivot_object", "objects", "targets", "x_range_m", "y_range_m", "yaw_range_deg"}
            optional = {"table_bounds_xy", "edge_margin_m", "clearance_targets", "target_clearance_m"}
            if not required <= set(group) or set(group) - required - optional:
                raise ValueError("Rigid group requires pivot, objects, targets, XY and yaw ranges")
            if "table_bounds_xy" in group:
                bounds = group["table_bounds_xy"]
                if len(bounds) != 2 or any(
                    len(pair) != 2 or not all(math.isfinite(v) for v in pair) or pair[0] >= pair[1] for pair in bounds
                ):
                    raise ValueError("Invalid table XY bounds")
                for field in ("edge_margin_m", "target_clearance_m"):
                    value = group.get(field, 0.01 if field == "edge_margin_m" else 0.06)
                    if not math.isfinite(value) or value < 0:
                        raise ValueError("Clearances must be finite and nonnegative")
                for name in group.get("clearance_targets", []):
                    if name not in self.scene.get("targets", {}) or name in group["targets"]:
                        raise ValueError("Clearance targets must exist and remain fixed")
            if group["pivot_object"] not in group["objects"]:
                raise ValueError("Pivot must belong to its rigid group")
            for category in ("objects", "targets"):
                for name in group[category]:
                    key = (category, name)
                    if key in occupied or name not in self.scene.get(category, {}):
                        raise ValueError("Unknown or duplicate rigid-group member: " + str(key))
                    occupied.add(key)
            for field in ("x_range_m", "y_range_m", "yaw_range_deg"):
                bounds = group[field]
                if len(bounds) != 2 or not all(type(v) in (int, float) and math.isfinite(v) for v in bounds):
                    raise ValueError("Ranges must contain two finite numbers")
                if bounds[0] > bounds[1]:
                    raise ValueError("Range minimum exceeds maximum")

    @property
    def grasp_object_ids(self):
        """All possible manipulated objects, including per-layout selections."""
        candidates = self.randomization.get("assembly_scatter", {}).get("select_from_scattered", [])
        return tuple(dict.fromkeys([*(goal.object_id for goal in self.task.all_goals), *candidates]))

    @classmethod
    def load(cls, path, *, catalog=None):
        return cls(read_document(path), catalog=catalog)

    def annotation_sources(self):
        """Resolve manipulated assets before source-neutral annotation discovery."""
        from isaaclab_arena.grasping.library.annotations import annotation_sources

        objects = {name: dict(self.scene["objects"][name]) for name in self.grasp_object_ids}
        for obj in objects.values():
            if obj.get("usd_path"):
                obj["usd_path"] = str(resolve_path(obj["usd_path"]))
        return annotation_sources(objects)

    def manifest(self):
        manifest = {
            "validation_mode": "execution_only",
            "taskinfo": self.document,
            "source_scene_sha256": digest_file(self.scene_path),
            "visual_assets": {str(p): digest_file(p) for p in [*self.hdrs, *self.cloths]},
        }
        annotations = self.annotation_sources()
        if annotations:
            manifest["grasp_annotations"] = {
                name: {"path": str(path), "sha256": digest_file(path)} for name, path in annotations.items()
            }
        if self.catalog_files:
            manifest["input_taskinfo"] = self.source_document
            manifest["catalog_files"] = {name: digest_file(path) for name, path in self.catalog_files.items()}
        return manifest

    def save_configuration(self, root):
        """Save reviewable input and expansion after the caller has checked its lock."""
        root = Path(root)
        write_json(root / "taskinfo.input.json", self.source_document)
        write_json(root / "taskinfo.resolved.json", self.document)
        write_json(root / "scene.resolved.json", self.scene)

    def prepare(self, directory, seed):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        scene = copy.deepcopy(self.scene)
        rng = random.Random(seed)
        changes = []
        for group in self.randomization.get("rigid_groups", []):
            pivot = list(scene["objects"][group["pivot_object"]]["position"])
            from code_gen.task.assembly import sample_rigid_group

            x, y, degrees, attempts = sample_rigid_group(scene, group, rng)
            angle = math.radians(degrees)
            c, s = math.cos(angle), math.sin(angle)
            qw, qz = math.cos(angle / 2), math.sin(angle / 2)
            for category in ("objects", "targets"):
                for name in group[category]:
                    obj = scene[category][name]
                    ox, oy, oz = obj["position"]
                    obj["position"] = [
                        x + c * (ox - pivot[0]) - s * (oy - pivot[1]),
                        y + s * (ox - pivot[0]) + c * (oy - pivot[1]),
                        oz,
                    ]
                    w, qx, qy, z = obj.get("quaternion", [1, 0, 0, 0])
                    obj["quaternion"] = [qw * w - qz * z, qw * qx - qz * qy, qw * qy + qz * qx, qw * z + qz * w]
                    # Scene grasp offsets are expressed in world axes.
                    calibrations = [obj, *obj.get("grasps_by_robot", {}).values()]
                    for calibration in calibrations:
                        if "grasp_offset_m" in calibration:
                            gx, gy, gz = calibration["grasp_offset_m"]
                            calibration["grasp_offset_m"] = [c * gx - s * gy, s * gx + c * gy, gz]
            changes.append(
                {"pivot": group["pivot_object"], "xy_m": [x, y], "yaw_deg": degrees, "sampling_attempts": attempts}
            )
        if self.randomization.get("assembly_scatter"):
            from code_gen.task.assembly import sample_assembly

            changes.append(sample_assembly(scene, self.randomization["assembly_scatter"], seed))
        if self.hdrs:
            visual_rng = random.Random(f"taskinfo-visuals-{seed}")
            hdr, cloth = visual_rng.choice(self.hdrs), visual_rng.choice(self.cloths)
            scene["visuals"] = {
                "seed": seed,
                "hdr_file": str(hdr),
                "tablecloth_texture": str(cloth),
                "hdr_intensity": self.randomization["visuals"]["hdr_intensity"],
                "sources": {"hdr_sha256": digest_file(hdr), "tablecloth_sha256": digest_file(cloth)},
            }
        scene["position_randomization"] = {"seed": seed, "groups": changes, "coordinate_frame": "source_scene"}
        scene_path = directory.resolve() / "scene.json"
        write_json(scene_path, scene)
        goals = self.task.goals
        if scene.get("task_selection"):
            selection = scene["task_selection"]
            goals = (replace(goals[0], object_id=selection["object_id"], target_id=selection["target_id"]),)
        task = replace(
            self.task,
            goals=goals,
            scene_file=str(scene_path),
            instruction=self.task.instruction
            + (
                " Use the Aloha arm selected at runtime by MGG's nearest-end-effector rule."
                if self.robot["arm"] == "auto"
                else f" Use Aloha {self.robot['arm']} arm."
            )
            + " Resolve object/target poses from the current scene; do not hard-code world poses.",
        )
        write_json(directory / "task.json", task.to_dict())
        return task
