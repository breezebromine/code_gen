# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Assembly resets through MGG's bounded placement and scatter samplers."""

import itertools
import math
import numpy as np
import random


class LayoutSamplingError(RuntimeError):
    """A seed could not fit its randomized assembly and scattered pieces."""


def sample_assembly(scene, config, seed):
    # These imports require the simulator, which the production runner starts
    # before preparing its first layout. Keep schema/config inspection lightweight.
    import torch

    from isaaclab.utils.math import quat_from_euler_xyz

    from isaaclab_arena.grasping.profiles import compose_pose
    from isaaclab_arena.relations.joint_reset import sample_non_overlapping_poses, sample_scattered_poses
    from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
    from isaaclab_arena.utils.pose import Pose

    base_id, parts = config["base_object"], config["parts"]
    base = scene["objects"][base_id]
    bounds = config["table_bounds_xy"]
    offsets = config["part_targets"]
    forced = config["required_scattered"]
    ratio = random.Random(seed).uniform(0.0, 1.0)

    def bbox(name):
        corners = np.asarray(scene["objects"][name]["bbox_corners"], dtype=float)
        return corners.min(axis=0), corners.max(axis=0)

    # Preserve the assembly footprint even when its pieces are scattered.
    # Source coordinates are body-local; the adapter's root/body offset is identity.
    corners = [np.asarray(base["bbox_corners"], dtype=float)]
    for name in parts:
        target = scene["targets"][offsets[name]]
        offset = [*target["reference_position_m"], *target.get("reference_quaternion_wxyz", [1, 0, 0, 0])]
        corners.append(
            np.asarray([compose_pose(offset, [*p, 1, 0, 0, 0])[:3] for p in scene["objects"][name]["bbox_corners"]])
        )
    corners = np.concatenate(corners)
    assembly_bbox = AxisAlignedBoundingBox(tuple(corners.min(axis=0)), tuple(corners.max(axis=0)))
    ranges = {
        "x": bounds[0],
        "y": bounds[1],
        "z": [base["position"][2]] * 2,
        "roll": [0.0, 0.0],
        "pitch": [0.0, 0.0],
        "yaw": [-math.pi, math.pi],
    }
    configs = {}
    for key, selected, scatter_ratio in (
        ("required", forced, 1.0),
        ("others", [name for name in parts if name not in forced], ratio),
    ):
        configs[key] = {
            "ratio": scatter_ratio,
            "bounds": bounds,
            "surface_z": config["surface_z_m"],
            "assembly_bbox": [assembly_bbox.min_point, assembly_bbox.max_point],
            "parts": [
                {
                    "name": name,
                    "bbox": [v.tolist() for v in bbox(name)],
                    "rotation_wxyz": scene["targets"][offsets[name]].get("reference_quaternion_wxyz", [1, 0, 0, 0]),
                    "root_body_offset": [0, 0, 0, 1, 0, 0, 0],
                }
                for name in selected
            ],
        }

    # Do not disturb simulation/planner global RNG state when preparing a seed.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        try:
            pose = sample_non_overlapping_poses(
                [assembly_bbox],
                [ranges],
                min_separation_m=config["min_separation_m"],
                max_sample_tries=config["max_sample_tries"],
                placement_bounds={0: bounds},
            )[0]
            rpy = torch.tensor(pose[3:])
            rotation = quat_from_euler_xyz(*rpy).tolist()
            assembly_pose = Pose(tuple(pose[:3]), tuple(rotation))
            occupied = [assembly_bbox.rotated_translated(pose[:3], *pose[3:])]
            scattered = sample_scattered_poses(
                {key: assembly_pose for key in configs},
                occupied,
                configs,
                min_separation_m=config["min_separation_m"],
                max_sample_tries=config["max_sample_tries"],
            )
        except RuntimeError as exc:
            raise LayoutSamplingError(str(exc)) from exc

    select_tabletop_target(scene, config, scattered, seed)
    stack_layout = pack_stacks(scene, config, scattered)
    base_pose = [*pose[:3], *rotation]
    base.update(position=base_pose[:3], quaternion=base_pose[3:])
    for name in parts:
        target = scene["targets"][offsets[name]]
        goal = compose_pose(
            base_pose, [*target["reference_position_m"], *target.get("reference_quaternion_wxyz", [1, 0, 0, 0])]
        )
        target.update(position=goal[:3], quaternion=goal[3:])
        obj_pose = [*scattered[name].position_xyz, *scattered[name].rotation_wxyz] if name in scattered else goal
        scene["objects"][name].update(position=obj_pose[:3], quaternion=obj_pose[3:])
        start = config.get("start_targets", {}).get(name)
        if start:
            scene["targets"][start].update(position=obj_pose[:3], quaternion=obj_pose[3:])
    return {
        "method": "MGG.sample_non_overlapping_poses+sample_scattered_poses",
        "reference_commit": "b00000f525553b685b8a835dfa00560d63caa5c6",
        "base_object": base_id,
        "base_pose": base_pose,
        "other_parts_scatter_ratio": ratio,
        "required_scattered": forced,
        "stacks": stack_layout,
        "scattered_parts": sorted(scattered),
        "assembled_parts": sorted(set(parts) - set(scattered)),
        "bounds_xy": bounds,
        "min_separation_m": config["min_separation_m"],
    }


def pack_stacks(scene, config, scattered):
    """Pack retained layers without floating gaps and target the next free layer."""
    stacks = []
    offsets = config["part_targets"]
    for group in config.get("stack_groups", []):
        members = group["parts"]
        retained = [name for name in members if name not in scattered]
        bottom = scene["targets"][offsets[members[0]]]["reference_position_m"][2]
        for name in members:
            index = retained.index(name) if name in retained else len(retained)
            target = scene["targets"][offsets[name]]
            target["reference_position_m"][2] = bottom + index * group["layer_pitch_m"]
            support = retained[index - 1] if index else config["base_object"]
            target["support_object_id"] = support
            for contact in scene.get("contact_surfaces", []):
                if (
                    contact["object_id"] == name
                    and contact.get("target_id") == offsets[name]
                    and "support_id" in contact
                ):
                    contact["support_id"] = support
        stacks.append({
            "parts": list(members),
            "retained_bottom_to_top": retained,
            "next_layer_z_m": bottom + len(retained) * group["layer_pitch_m"],
        })
    return stacks


def select_tabletop_target(scene, config, scattered, seed):
    """Select only after scattering; absence is a rejected layout, not relocation."""
    candidates = config.get("select_from_scattered")
    if candidates is not None:
        eligible = [name for name in candidates if name in scattered]
        if not eligible:
            raise LayoutSamplingError("No requested-shape plate was scattered onto the table")
        selected = random.Random(f"tabletop-selection-{seed}").choice(eligible)
        scene["task_selection"] = {
            "object_id": selected,
            "target_id": config["part_targets"][selected],
            "eligible_object_ids": eligible,
            "rule": "uniform choice among requested-shape objects already scattered",
        }
        scene["contact_object_id"] = selected


def group_points(scene, names, pivot):
    points = []
    for name in names:
        obj = scene["objects"][name]
        corners = obj.get("bbox_corners")
        if corners is None:
            dims = obj.get("dimensions_m", [obj.get("size", 0)] * 3)
            corners = list(itertools.product(*[[-v / 2, v / 2] for v in dims]))
        w, x, y, z = obj.get("quaternion", [1, 0, 0, 0])
        rotation = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ])
        world = np.asarray(corners) @ rotation.T + np.asarray(obj["position"])
        points.extend((world[:, :2] - np.asarray(pivot[:2])).tolist())
    return np.asarray(points)


def sample_rigid_group(scene, group, rng):
    pivot = scene["objects"][group["pivot_object"]]["position"]
    bounds = group.get("table_bounds_xy")
    points = group_points(scene, group["objects"], pivot) if bounds else None
    for attempt in range(10000):
        x, y = rng.uniform(*group["x_range_m"]), rng.uniform(*group["y_range_m"])
        yaw = rng.uniform(*group["yaw_range_deg"])
        if bounds:
            angle = math.radians(yaw)
            c, s = math.cos(angle), math.sin(angle)
            placed = points @ np.array([[c, s], [-s, c]]) + [x, y]
            low, high = placed.min(axis=0), placed.max(axis=0)
            margin = group.get("edge_margin_m", 0.01)
            if any(low[i] < bounds[i][0] + margin or high[i] > bounds[i][1] - margin for i in range(2)):
                continue
            clearance = group.get("target_clearance_m", 0.06)
            blocked = False
            for name in group.get("clearance_targets", []):
                target = np.asarray(scene["targets"][name]["position"][:2])
                closest = np.clip(target, low, high)
                if np.linalg.norm(target - closest) < clearance:
                    blocked = True
                    break
            if blocked:
                continue
        return x, y, yaw, attempt + 1
    raise LayoutSamplingError("No rigid-group layout fits table bounds and target clearance")
