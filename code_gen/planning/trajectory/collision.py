# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Release clearance checks and failure diagnostics for the live collision world."""

import math
import torch
import xml.etree.ElementTree as ET
from pathlib import Path

from code_gen._io import write_json


class CollisionChecks:
    """Check clearance and report overlaps without commanding simulation motion."""

    def __init__(self, planner, layout, output):
        self.planner = planner
        self.layout = layout
        self.output = output

    def _set_width(self, lower, width):
        lower.config.gripper_open_positions = {
            name: sign * width / 2 for name, sign in zip(self.layout.finger_names, self.layout.finger_signs)
        }
        lower._set_gripper_state(False)

    def _aperture_query(self, lower, guard, positions, start, goal):
        """Reuse endpoint FK for prismatic jaws instead of reparsing each width."""
        try:
            path = Path(lower.robot_cfg["kinematics"]["urdf_path"])
            joints = ET.parse(path).getroot().findall("joint")
            types = {joint.get("name"): joint.get("type") for joint in joints}
            prismatic = all(types.get(name) == "prismatic" for name in self.layout.finger_names)
            # Revolute mimic joints would make sphere motion nonlinear.
            prismatic &= all(
                joint.get("type") in ("prismatic", "fixed") for joint in joints if joint.find("mimic") is not None
            )
        except (OSError, KeyError, TypeError, ET.ParseError):
            prismatic = False

        def direct(width):
            self._set_width(lower, width)
            return guard.query(positions)

        if not prismatic or abs(goal - start) < 1e-9:
            return direct
        endpoints = []
        for width in (start, goal):
            self._set_width(lower, width)
            endpoints.append(lower.motion_gen.kinematics.get_state(positions).link_spheres_tensor.unsqueeze(0).clone())
        if not torch.equal(endpoints[0][..., 3], endpoints[1][..., 3]):
            return direct

        def interpolated(width):
            fraction = (width - start) / (goal - start)
            if not -1e-6 <= fraction <= 1 + 1e-6:
                raise ValueError("Gripper sweep query is outside its FK endpoints")
            # Fixed arm joints + prismatic jaws make this interpolation exact;
            # radii and inactive attachment slots are unchanged.
            return guard.query_spheres(torch.lerp(endpoints[0], endpoints[1], fraction))

        return interpolated

    def prepare_release(self, stage, snapshot, skills, event_index):
        """Find free jaw opening space around an inserted object before release."""
        lower = self.planner._get_planner(self.layout.arm)
        mg = lower.motion_gen
        lower._detach_objects()
        mg.detach_object_from_robot(link_name=lower.config.attached_object_link_name)
        lower.update_world()
        saved_open = dict(lower.config.gripper_open_positions)
        saved_closed = dict(lower.config.gripper_closed_positions)
        from code_gen.planning.constraints import INSERTION_RELEASE_CLEARANCE_M, insertion_release_widths

        guard = self.planner.create_cartesian_collision_guard(
            self.layout.arm, robot_clearance_m=INSERTION_RELEASE_CLEARANCE_M
        )
        state = lower._get_current_joint_state_for_curobo().get_ordered_joint_state(mg.kinematics.joint_names)

        trials, accepted, sweep = [], [], []
        previous_width = snapshot["gripper_width"]
        target_paths = lower._get_object_paths(skills.insertion_release_object)
        try:
            query = self._aperture_query(lower, guard, state.position, previous_width, 0.075)
            for width in insertion_release_widths(previous_width):
                steps = max(1, math.ceil((width - previous_width) / 0.001))
                blocked = False
                for i in range(steps + 1):
                    sample_width = previous_width + (width - previous_width) * i / steps
                    # The opening starts in the established target contact.
                    # Check every other obstacle throughout the finger sweep;
                    # acceptance below restores the target as an obstacle too.
                    with guard.masks(target_paths):
                        check = query(sample_width)
                    sweep.append({"width_m": sample_width, **check})
                    if not check["passed"]:
                        blocked = True
                        break
                if blocked:
                    break
                check = query(width)
                trials.append({"width_m": width, "clearance_m": width - snapshot["gripper_width"], **check})
                if check["passed"]:
                    accepted.append(width)
                previous_width = width
            if not accepted:
                # This guard has no held target or support exemptions. Identify
                # real obstructing meshes only after exhausting bounded openings.
                all_paths = {obstacle.name for obstacle in mg.world_model.objects}
                for trial in trials:
                    blocked = []
                    for path in sorted(all_paths):
                        with guard.masks(all_paths - {path}):
                            check = query(trial["width_m"])
                        if check["world_collision_m"] > 1e-6 or check["robot_clearance_collision_m"] > 1e-6:
                            blocked.append({"obstacle_path": path, **check})
                    trial["blocking_obstacles"] = blocked
                raise RuntimeError(
                    "INSERTION_RELEASE_NO_CLEARANCE: "
                    + str({
                        "trials": trials,
                        "last_sweep_check": sweep[-1] if sweep else None,
                    })
                )
            stage.target.gripper_width = max(accepted)
            skills.insertion_release_width = stage.target.gripper_width
        finally:
            lower.config.gripper_open_positions = saved_open
            lower.config.gripper_closed_positions = saved_closed
            lower._set_gripper_state(False)
            write_json(
                self.output / f"release_{event_index:03d}.json",
                {
                    "object_id": skills.insertion_release_object,
                    "robot_clearance_m": INSERTION_RELEASE_CLEARANCE_M,
                    "clearance_exempt_paths": [],
                    "trials": trials,
                    "opening_sweep": sweep,
                    "selected_width_m": max(accepted) if accepted else None,
                },
            )

    def prepare_reopen(self, stage, snapshot, event_index):
        """Check the entire finger sweep at the reached retreat pose."""
        from code_gen.planning.constraints import INSERTION_RELEASE_CLEARANCE_M

        lower = self.planner._get_planner(self.layout.arm)
        lower._detach_objects()
        lower.motion_gen.detach_object_from_robot(link_name=lower.config.attached_object_link_name)
        lower.update_world()
        saved_open = dict(lower.config.gripper_open_positions)
        saved_closed = dict(lower.config.gripper_closed_positions)
        guard = self.planner.create_cartesian_collision_guard(
            self.layout.arm, robot_clearance_m=INSERTION_RELEASE_CLEARANCE_M
        )
        state = lower._get_current_joint_state_for_curobo().get_ordered_joint_state(
            lower.motion_gen.kinematics.joint_names
        )
        start, goal = snapshot["gripper_width"], stage.target.gripper_width
        # At most 0.5 mm of travel per finger between clearance samples.
        steps = max(1, math.ceil(abs(goal - start) / 0.001))
        trials = []
        try:
            query = self._aperture_query(lower, guard, state.position, start, goal)
            for i in range(steps + 1):
                width = start + (goal - start) * i / steps
                check = query(width)
                trials.append({"width_m": width, **check})
                if not check["passed"]:
                    raise RuntimeError(f"POST_RETREAT_OPEN_COLLISION: {trials[-1]}")
        finally:
            lower.config.gripper_open_positions = saved_open
            lower.config.gripper_closed_positions = saved_closed
            lower._set_gripper_state(False)
            write_json(self.output / f"reopen_{event_index:03d}.json", {"trials": trials})

    def diagnose_start(self, attached):
        """Identify obstacle/link overlaps without changing simulation state."""
        from curobo.geom.sdf.world import CollisionQueryBuffer

        lower = self.planner._get_planner(self.layout.arm)
        active = []
        try:
            lower.update_world()
            lower._set_gripper_state(attached is not None)
            if attached and not lower._attach_object(
                attached, lower._get_object_mappings()[attached], lower._current_runtime_env_id
            ):
                raise RuntimeError("Could not reconstruct held-object collision geometry")
            checker = lower.motion_gen.world_coll_checker
            held_paths = set(lower._get_object_paths(attached)) if attached else set()
            active = [o.name for o in lower.motion_gen.world_model.objects if o.name not in held_paths]
            groups = {
                name: [p for p in lower._get_object_paths(name) if p in active]
                for name in lower._get_object_mappings()
                if name != attached
            }
            grouped = {p for paths in groups.values() for p in paths}
            groups["static_scene"] = [p for p in active if p not in grouped]
            state = lower._get_current_joint_state_for_curobo()
            spheres = lower.motion_gen.kinematics.get_state(state.position).link_spheres_tensor.unsqueeze(1)
            buffer = CollisionQueryBuffer.initialize_from_shape(
                spheres.shape, lower.tensor_args, checker.collision_types
            )
            kin = lower.motion_gen.kinematics.kinematics_config
            link_ids = kin.link_sphere_idx_map.cpu().tolist()
            names = {v: k for k, v in kin.link_name_to_idx_map.items()}
            findings = []
            for path in active:
                checker.enable_obstacle(path, enable=False)
            for name, paths in groups.items():
                for path in paths:
                    checker.enable_obstacle(path, enable=True)
                distances = checker.get_sphere_distance(
                    spheres, buffer, lower.tensor_args.to_device([1.0]), lower.tensor_args.to_device([0.0])
                ).reshape(-1)
                indices = torch.nonzero(distances > 1e-6).flatten().tolist()
                if indices:
                    findings.append({
                        "obstacle": name,
                        "links": sorted({names[link_ids[i]] for i in indices}),
                        "max_collision_cost_m": float(distances.max()),
                        "sphere_indices": indices,
                    })
                for path in paths:
                    checker.enable_obstacle(path, enable=False)
        finally:
            try:
                for path in active:
                    checker.enable_obstacle(path, enable=True)
            finally:
                # Setup (including attachment, FK and query-buffer creation) can fail too.
                lower._detach_objects()
        write_json(self.output / "start_collision.json", findings)
