# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Resolve small user task descriptions into the existing frozen execution spec.

Catalog files contain reusable scene, asset, robot and collection knowledge.
Resolution is deterministic and requires neither a model nor a simulator.
"""

import copy
import json
import math
import re
from pathlib import Path

from code_gen._io import PACKAGE_ROOT
from isaaclab_arena.grasping.library.orientations import MAX_GRASP_CANDIDATES, grasp_orientations

CATALOG_ROOT = PACKAGE_ROOT / "catalog"
CATEGORIES = ("scenes", "objects", "robots", "runs", "goals")

# Defaults for the short taskinfo format. Keep these values in the expanded
# document so existing task snapshots and manifests retain the same shape.
# Table placement intentionally uses a tighter Z tolerance than PlacementGoal.
DEFAULT_GOALS = {
    "placement": {},
    "lift": {"kind": "lift"},
    "table_placement": {
        "xy_tolerance_m": 0.025,
        "z_tolerance_m": 0.003,
        "max_speed_m_s": 0.02,
        "max_angular_speed_rad_s": 0.1,
        "max_other_translation_m": 0.004,
        "max_other_rotation_rad": math.radians(2),
        "kind": "table_placement",
    },
}

DEFAULT_PRESETS = {
    "robots": {
        "aloha_auto": {"name": "aloha", "arm": "auto", "assets_root": "${LISIM_ASSETS_PATH}"},
        "aloha_left": {"name": "aloha", "arm": "left", "assets_root": "${LISIM_ASSETS_PATH}"},
        "aloha_right": {"name": "aloha", "arm": "right", "assets_root": "${LISIM_ASSETS_PATH}"},
    },
    "runs": {
        "default": {
            "development": {"generation_seed": 0, "validation_seeds": [1, 2], "max_versions": 3},
            "collection": {"start_seed": 1000, "max_cases": 0, "target_successes": 1},
            "hold_seconds": 1.0,
        }
    },
}


def merge(base, updates):
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def read_document(path):
    path = Path(path)
    if path.suffix.lower() in (".yaml", ".yml"):
        import yaml

        value = yaml.safe_load(path.read_text())
    else:
        value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("Taskinfo must be a JSON/YAML mapping")
    return value


class TaskCatalog:
    def __init__(self, root=None):
        self.root = Path(root) if root is not None else CATALOG_ROOT
        self.files = {}

    def list(self):
        result = {}
        for group in CATEGORIES:
            names = {path.stem for path in (self.root / group).glob("*.json")}
            names.update(DEFAULT_GOALS if group == "goals" else DEFAULT_PRESETS.get(group, {}))
            result[group] = sorted(names)
        return result

    def get(self, group, name):
        if group not in CATEGORIES or not isinstance(name, str) or not re.fullmatch(r"[a-zA-Z0-9_-]+", name):
            raise ValueError(f"Invalid {group} preset name: {name!r}")
        path = self.root / group / (name + ".json")
        if path.is_file():
            value = read_document(path)
            self.files[f"{group}/{name}.json"] = path.resolve()
            return copy.deepcopy(value)
        defaults = DEFAULT_GOALS if group == "goals" else DEFAULT_PRESETS.get(group, {})
        if name in defaults:
            return copy.deepcopy(defaults[name])
        raise ValueError(f"Unknown {group} preset {name!r}; available: {self.list()[group]}")

    def expand(self, source):
        required = {"task_name", "scene", "instruction", "success", "episodes"}
        optional = {"schema_version", "robot", "run", "hold_seconds", "development", "collection"}
        if not required <= source.keys() or source.keys() - required - optional:
            raise ValueError(f"Simple taskinfo requires {sorted(required)}; optional fields: {sorted(optional)}")
        if source.get("schema_version", 2) != 2:
            raise ValueError("Simple taskinfo uses schema_version=2 (or omit it)")
        if not isinstance(source["task_name"], str) or not re.fullmatch(r"[a-zA-Z0-9_-]+", source["task_name"]):
            raise ValueError("task_name must contain letters, digits, underscores or hyphens")
        if not isinstance(source["instruction"], str) or not source["instruction"].strip():
            raise ValueError("instruction must be a non-empty string")
        if type(source["episodes"]) is not int or source["episodes"] < 1:
            raise ValueError("episodes must be a positive integer")

        for key in ("development", "collection"):
            if key in source and not isinstance(source[key], dict):
                raise ValueError(f"{key} must be a mapping")
        scene = self.get("scenes", source["scene"])
        robot = self.get("robots", source["robot"] if "robot" in source else scene["default_robot"])
        run = self.get("runs", source["run"] if "run" in source else scene["default_run"])
        overrides = copy.deepcopy(scene.get("overrides", {}))
        hints, profiles = [], {}
        for object_id, profile_name in scene.get("objects", {}).items():
            profile = self.get("objects", profile_name)
            profiles[object_id] = profile
            metadata = copy.deepcopy(profile.get("overrides", {}))
            for calibration in metadata.get("grasps_by_robot", {}).values():
                from isaaclab_arena.grasping.geometry import pose

                frames = calibration.get("grasp_frames", [])
                ids = [row["id"] for row in frames]
                if len(ids) != len(set(ids)) or any(not isinstance(i, str) or not i for i in ids):
                    raise ValueError("grasp_frames require unique non-empty string IDs")
                for row in frames:
                    pose(row["pose_object"])
                if calibration.get("default_grasp_id") is not None and calibration["default_grasp_id"] not in ids:
                    raise ValueError("default_grasp_id must name a declared frame")
                if "yaw_samples" in calibration:
                    count = calibration.pop("yaw_samples")
                    if type(count) is not int or not 1 <= count <= MAX_GRASP_CANDIDATES:
                        raise ValueError("yaw_samples must be an integer from 1 to 64")
                    if "yaw_candidates_deg" in calibration:
                        raise ValueError("Specify yaw_samples or yaw_candidates_deg, not both")
                    calibration["yaw_candidates_deg"] = [i * 360.0 / count for i in range(count)]
                grasp_orientations(
                    [1, 0, 0, 0],
                    calibration.get("yaw_candidates_deg", [0]),
                    calibration.get("tilt_candidates_deg", [0]),
                )
            merge(overrides.setdefault("objects", {}).setdefault(object_id, {}), metadata)

        successes = source["success"] if isinstance(source["success"], list) else [source["success"]]
        if not successes:
            raise ValueError("success must contain at least one goal")
        goals = []
        for success in successes:
            if not isinstance(success, dict) or not {"type", "object"} <= success.keys():
                raise ValueError("Each success goal requires type and object")
            object_id = success["object"]
            if not isinstance(object_id, str) or object_id not in profiles:
                raise ValueError(f"No asset profile for task object {object_id!r} in scene {source['scene']!r}")
            profile = profiles[object_id]
            template = self.get("goals", success["type"])
            aliases = {"object": "object_id", "target": "target_id", "reference": "reference_id"}
            # Detailed tolerances remain available when explicitly requested.
            # Runtime TaskSpec validates the final goal; the LLM cannot edit it.
            for key, value in success.items():
                if key == "type":
                    continue
                if key in ("kind", "object_id", "target_id", "reference_id"):
                    raise ValueError(f"Use simple goal aliases instead of {key}")
                template[aliases.get(key, key)] = copy.deepcopy(value)
            goals.append(template)
            if profile.get("instruction_hint") and profile["instruction_hint"] not in hints:
                hints.append(profile["instruction_hint"])
            for key, value in scene.get("task_overrides", {}).get(object_id, {}).items():
                if key in overrides and overrides[key] != value:
                    raise ValueError(f"Conflicting scene task overrides: {key}")
                overrides[key] = copy.deepcopy(value)

        development = merge(copy.deepcopy(run["development"]), source.get("development", {}))
        collection_updates = source.get("collection", {})
        if "target_successes" in collection_updates:
            raise ValueError("Use episodes to configure the number of successes")
        collection = merge(copy.deepcopy(run["collection"]), collection_updates)
        collection["target_successes"] = source["episodes"]
        randomization = copy.deepcopy(scene.get("randomization", {}))
        if run.get("visuals"):
            randomization["visuals"] = copy.deepcopy(run["visuals"])
        return {
            "schema_version": 1,
            "task": {
                "name": source["task_name"],
                "instruction": source["instruction"].strip() + (" " + " ".join(hints) if hints else ""),
                "goals": goals,
                "hold_seconds": source.get("hold_seconds", run.get("hold_seconds", 1.0)),
            },
            "scene_file": scene["scene_file"],
            "scene_overrides": overrides,
            "robot": robot,
            "randomization": randomization,
            "development": development,
            "collection": collection,
        }


def resolve_taskinfo(document, catalog=None):
    if not isinstance(document, dict):
        raise ValueError("Taskinfo must be a mapping")
    if document.get("schema_version") == 1:
        return copy.deepcopy(document), {}
    catalog = catalog or TaskCatalog()
    return catalog.expand(document), dict(catalog.files)
