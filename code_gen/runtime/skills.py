# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Task-level manipulation and measured grasp verification for code generation."""

import copy
import math
import numpy as np

from isaaclab_arena.grasping.geometry import (
    compose,
    contact_evidence,
    grasp_frame,
    inverse,
    object_goal_to_tcp,
    pose,
    rotation,
)
from isaaclab_arena.grasping.skills import OPEN_WIDTH, TabletopSkills


class TaskSkills(TabletopSkills):
    """Extend the base phase executor with task-specific manipulation skills."""

    def __init__(self, initial_snapshot, *args, home_joint_names=None, **kwargs):
        super().__init__(initial_snapshot, *args, **kwargs)
        initial_joints = dict(zip(initial_snapshot.get("joint_names", []), initial_snapshot.get("joint_positions", [])))
        self.home_joint_positions = {name: initial_joints[name] for name in home_joint_names or ()}
        self.pending_grasp = None
        self.grasp_reference = None
        self.grasp_evidence = []
        self.grasp_yaws, self.grasp_tilts = [0], [0]
        self.require_vertical = False
        self.auto_grasp_library = False
        self.pick_arguments = {}
        self.contact_gap_s = 0.0
        self.alignment_evidence = []
        self.transport_goal = None
        self.transport_evidence = []
        self.pending_release_object = None
        self.reopening_after_retreat = False

    def fail_grasp(self, snapshot, reason, evidence):
        self.grasp_evidence.append({"reason": reason, **evidence})
        self.stages.clear()
        self.target = self._hold(snapshot)
        self.holding = self.pending_grasp = None
        self.grasp_uncertain = True
        self.result = {"status": "failed", "reason": reason, "grasp_evidence": evidence}

    def _begin_contact(self, snapshot):
        evidence = contact_evidence(snapshot, self.object_id)
        if not evidence["passed"]:
            self.fail_grasp(
                snapshot, "GRASP_NOT_ESTABLISHED: require object-specific bilateral contact and jaw opening", evidence
            )
            return False
        self.pending_grasp = self.object_id
        self.contact_gap_s = 0.0
        self.grasp_reference = {
            "object": np.array(snapshot["objects"][self.object_id]["position"]),
            "tcp": np.array(snapshot["tcp_pose"][:3]),
            "object_in_tcp": rotation(snapshot["tcp_pose"][3:]).T @ (
                np.array(snapshot["objects"][self.object_id]["position"]) - snapshot["tcp_pose"][:3]
            ),
        }
        anchor = snapshot["objects"][self.object_id].get("grasp_follow_anchor_object_m")
        if anchor is not None:
            anchor = np.asarray(anchor, dtype=float)
            if anchor.shape != (3,) or not np.isfinite(anchor).all():
                raise ValueError("Grasp follow anchor must be a finite object-local point")
            obj = snapshot["objects"][self.object_id]
            point = np.asarray(obj["position"]) + rotation(obj["quaternion"]) @ anchor
            self.grasp_reference["anchor_object_m"] = anchor
            self.grasp_reference["anchor_in_tcp"] = rotation(snapshot["tcp_pose"][3:]).T @ (
                point - snapshot["tcp_pose"][:3]
            )
        self.grasp_evidence.append({"phase": "contact", **evidence})
        return True

    def check_follow(self, snapshot, require_lift=False, dt=0.0, allow_contact_gap=False):
        name = self.pending_grasp or self.holding
        if not name or self.grasp_reference is None:
            return True
        contact = contact_evidence(snapshot, name)
        self.contact_gap_s = 0.0 if contact["passed"] else self.contact_gap_s + dt
        # Only brief force flicker after proven bilateral contact is tolerated.
        # Missing sensors and a closed-empty hand never qualify.
        forces = contact["finger_contact_forces_n"]
        recent_contact = (
            isinstance(forces, (list, tuple))
            and len(forces) == 2
            and all(np.isfinite(v) for v in forces)
            and contact["gripper_width_m"] >= contact["minimum_width_m"]
            and self.contact_gap_s < 0.08
        )
        obj = np.asarray(snapshot["objects"][name]["position"])
        tcp = np.asarray(snapshot["tcp_pose"][:3])
        obj_delta = obj - self.grasp_reference["object"]
        tcp_delta = tcp - self.grasp_reference["tcp"]
        expected = tcp + rotation(snapshot["tcp_pose"][3:]) @ self.grasp_reference["object_in_tcp"]
        origin_error = float(np.linalg.norm(obj - expected))
        following_error = origin_error
        anchor_evidence = {}
        if "anchor_object_m" in self.grasp_reference:
            point = obj + rotation(snapshot["objects"][name]["quaternion"]) @ self.grasp_reference["anchor_object_m"]
            expected_point = tcp + rotation(snapshot["tcp_pose"][3:]) @ self.grasp_reference["anchor_in_tcp"]
            following_error = float(np.linalg.norm(point - expected_point))
            anchor_evidence = {
                "following_reference": "declared object-local grasp anchor",
                "anchor_object_m": self.grasp_reference["anchor_object_m"].tolist(),
                "measured_anchor_world_m": point.tolist(),
                "expected_anchor_world_m": expected_point.tolist(),
                "object_origin_following_error_m": origin_error,
            }
        lifted = float(obj_delta[2])
        moved = float(np.linalg.norm(tcp_delta))
        evidence = {
            **contact,
            "contact_gap_s": self.contact_gap_s,
            "object_lift_m": lifted,
            "tcp_displacement_m": tcp_delta.tolist(),
            "following_error_m": following_error,
            **anchor_evidence,
        }
        # Once the hand moves, a stationary object must not be treated as carried.
        if (moved >= 0.008 and following_error > 0.006) or not (contact["passed"] or recent_contact):
            self.fail_grasp(snapshot, "GRASP_LOST: object did not follow the measured gripper motion", evidence)
            return False
        if require_lift and (
            lifted < 0.008
            or following_error > 0.006
            or not (contact["passed"] or (allow_contact_gap and recent_contact))
        ):
            self.fail_grasp(snapshot, "GRASP_NOT_LIFTED: planned motion is not evidence of an actual lift", evidence)
            return False
        if lifted >= 0.008 and following_error <= 0.006 and contact["passed"]:
            self.holding, self.pending_grasp = name, None
        return True

    def _grasp_close_width(self, obj, metadata, arguments):
        """Separate the imported contact aperture from the actuator preload."""
        if "close_width_m" in arguments:
            return arguments["close_width_m"]
        width = metadata.get("close_width_m", obj.get("grasp_close_width_m", 0))
        preload = obj.get("grasp_contact_preload_m")
        span = metadata.get("contact_span_m")
        if preload is not None and span is not None and "source_validation" in metadata:
            # Imported joint positions describe blocked jaws, not a closing
            # command. Keep the aperture for planning and apply preload only
            # to execution. Explicit action widths and custom frames retain
            # their existing semantics.
            if not np.isfinite(preload) or not 0 <= preload <= 0.01:
                raise ValueError("Contact preload must be finite and within [0, 0.01] m")
            if not np.isfinite(span) or not 0 <= span <= self.max_gripper_width:
                raise ValueError("Contact aperture is outside the gripper range")
            width = max(0.0, span - preload)
        return width

    def library_candidates(self, snapshot):
        """Resolve every local frame against this attempt's measured object pose."""
        from isaaclab_arena.grasping.library.orientations import grasp_stability_score

        obj = snapshot["objects"][self.object_id]
        rows = []
        clearance = float(obj.get("grasp_table_clearance_m", 0.0))
        if not math.isfinite(clearance) or not 0 <= clearance <= 0.003:
            raise ValueError("Tabletop grasp clearance must be between 0 and 3 mm")
        table_z = snapshot.get("table_layout", {}).get("table_top_z")
        dimensions = obj.get("dimensions_m", [])
        on_table = (
            table_z is not None
            and len(dimensions) == 3
            and dimensions[2] <= 0.02
            and abs(obj.get("lowest_z", math.inf) - table_z) <= 0.0015
            and rotation(obj["quaternion"])[2, 2] >= 0.98
        )
        clearance = clearance if on_table else 0.0
        for index, frame in enumerate(obj.get("grasp_frames", [])):
            metadata, world = grasp_frame(obj, frame["id"])
            position = world[:3]
            if "grasp_offset_m" in self.pick_arguments:
                position = (np.array(obj["position"]) + self.pick_arguments["grasp_offset_m"]).tolist()
            position = [position[0], position[1], position[2] + clearance]
            width = self._grasp_close_width(obj, metadata, self.pick_arguments)
            height = self.pick_arguments.get(
                "lift_height_m", metadata.get("lift_height_m", obj.get("grasp_lift_height_m", 0.15))
            )
            if not 0 <= width <= self.max_gripper_width or not 0.08 <= height <= 0.20:
                raise ValueError("Invalid grasp-library width or lift height")
            rows.append({
                "candidate_id": f"frame_{index:04d}",
                "grasp_id": frame["id"],
                "position": position,
                "table_clearance_m": clearance,
                "approach_position": [position[0], position[1], position[2] + height],
                "quaternion": world[3:],
                "close_width_m": width,
                "predicted_held_width_m": max(width, float(metadata.get("contact_span_m", width))),
                "antipodal_alignment": metadata.get("antipodal_alignment"),
                "stability_score": grasp_stability_score(metadata, obj["quaternion"]),
                "lift_height_m": height,
                "yaw_deg": 0.0,
                "tilt_deg": 0.0,
            })
            if snapshot.get("robot") == "aloha":
                # A parallel-jaw grasp can swap fingers by turning 180 degrees
                # around its local +X approach axis. The wrist is not assumed
                # symmetric: both variants still undergo IK and collision checks.
                row = rows[-1]
                # Keep 3 mm per finger beyond the annotated contact span. A
                # fixed full opening lowers a tilted finger toward the table.
                row["open_width_m"] = min(OPEN_WIDTH, math.ceil(row["predicted_held_width_m"] * 1000) / 1000 + 0.006)
                w, x, y, z = row["quaternion"]
                rows.append({
                    **row,
                    "candidate_id": row["candidate_id"] + "_jaw_flip",
                    "quaternion": [-x, w, z, -y],
                })
        return rows

    def apply_grasp_candidate(self, row):
        """Change goals in place so the active stage and controller stay in sync."""
        updates = []
        for stage in self.stages:
            if stage.name not in ("approach", "descend", "close", "lift"):
                continue
            position = row["approach_position"] if stage.name in ("approach", "lift") else row["position"]
            width = row["close_width_m"] if stage.name in ("close", "lift") else row.get("open_width_m", OPEN_WIDTH)
            # Validate all goals before modifying any stage.
            updates.append((stage, self._move(stage.name, position, width, row["quaternion"]).target))
        for stage, target in updates:
            stage.target = target
        self.pick_lift_height = row["lift_height_m"]
        if self.stages:
            self.target = self.stages[0].target

    def start(self, name, arguments, snapshot):
        aliases = {"grasp_actor": "pick", "place_actor": "place"}
        name = aliases.get(name, name)
        if self.busy:
            super().start(name, arguments, snapshot)
            return
        self.require_vertical = False
        self.transport_goal = None
        self.transport_evidence = []
        self.reopening_after_retreat = False
        self.insertion_release_object = None
        self.insertion_release_width = None
        if name == "open_gripper" and self.skill == "insert_object" and self.holding:
            self.insertion_release_object = self.holding
            self.insertion_release_width = min(OPEN_WIDTH, snapshot["gripper_width"] + 0.002)
        if self.pending_grasp and name not in ("observe", "move_to_pose", "move_by_displacement", "open_gripper"):
            self.result = {"status": "failed", "reason": "Pending contact requires a vertical proof lift or release"}
            return
        if name == "pick":
            self._start_pick(arguments, snapshot)
            return
        if name not in (
            "move_to_pose",
            "move_object_to_pose",
            "align_object",
            "move_by_displacement",
            "open_gripper",
            "close_gripper",
        ):
            previous_width = self.target.gripper_width
            super().start(name, arguments, snapshot)
            if name == "home" and self.home_joint_positions and self.stages:
                # The joint planner checks the entire return path. Raising to
                # home height at the current XY can create an unreachable detour.
                self.stages = self.stages[-1:]
                self.target = self.stages[0].target
            if name == "place":
                for stage in self.stages:
                    stage.target.quaternion = tuple(snapshot["tcp_pose"][3:])
                self._skip_satisfied_raise(snapshot, previous_width=previous_width)
            return
        if (
            name == "open_gripper"
            and not self.holding
            and not self.pending_grasp
            and not self.grasp_uncertain
            and abs(snapshot["gripper_width"] - OPEN_WIDTH) <= 0.002
            and abs(self.target.gripper_width - OPEN_WIDTH) <= 0.002
        ):
            # Idempotent only after both the measured and commanded jaws are open.
            # Never bypass a real release, provisional contact or uncertain grasp.
            self.skill = name
            self.object_id = None
            self.result = {"status": "success", "holding": None}
            return
        self._start_explicit_action(name, arguments, snapshot)

    def _skip_satisfied_raise(self, snapshot, previous_width=None):
        """Drop only a stationary prefix; retain contact phases and their dwell."""
        if not self.stages or self.stages[0].name != "raise":
            return
        target = self.stages[0].target
        position_error, rotation_error = self._pose_error(snapshot["tcp_pose"], [*target.position, *target.quaternion])
        # A held object blocks the jaws at its own width. Compare the closing
        # command, not that measured width, to preserve the existing grip preload.
        width_satisfied = (
            previous_width is not None and bool(self.holding) and abs(previous_width - target.gripper_width) <= 0.002
        ) or (
            not self.holding
            and target.gripper_width >= OPEN_WIDTH - 0.002
            and snapshot["gripper_width"] >= target.gripper_width - 0.002
        )
        if position_error <= 0.0005 and rotation_error <= 0.005 and width_satisfied:
            self.stages.pop(0)
            self.target = self.stages[0].target

    def _start_pick(self, arguments, snapshot):
        arguments = copy.deepcopy(arguments)
        obj = snapshot["objects"].get(arguments.get("object_id"), {})
        self.grasp_yaws = arguments.pop("yaw_candidates_deg", obj.get("grasp_yaw_candidates_deg", [0]))
        self.grasp_tilts = arguments.pop("tilt_candidates_deg", obj.get("grasp_tilt_candidates_deg", [0]))
        from isaaclab_arena.grasping.library.orientations import grasp_orientations

        try:
            grasp_orientations([1, 0, 0, 0], self.grasp_yaws, self.grasp_tilts)
            selected_pose = None
            initial_candidate = None
            explicit_id = arguments.pop("grasp_id", None)
            self.auto_grasp_library = bool(obj.get("grasp_frames")) and explicit_id is None
            self.pick_arguments = copy.deepcopy(arguments)
            grasp_id = explicit_id or obj.get("default_grasp_id")
            if self.auto_grasp_library:
                self.object_id = arguments.get("object_id")
                from isaaclab_arena.grasping.geometry import workspace_rejection

                admissible = [
                    row
                    for row in self.library_candidates(snapshot)
                    if workspace_rejection(row, self.workspace_bounds) is None
                ]
                if not admissible:
                    self.result = {"status": "failed", "reason": "No grasp candidates inside workspace"}
                    self.stages.clear()
                    return
                if grasp_id not in {row["grasp_id"] for row in admissible}:
                    grasp_id = admissible[0]["grasp_id"]
                initial_candidate = next(row for row in admissible if row["grasp_id"] == grasp_id)
            if grasp_id is not None:
                metadata, selected_pose = grasp_frame(obj, grasp_id)
                if initial_candidate is not None:
                    # Bootstrap with the same clearance-adjusted pose that passed
                    # candidate workspace checks, before cuRobo searches the library.
                    selected_pose = [*initial_candidate["position"], *initial_candidate["quaternion"]]
                arguments.setdefault("grasp_offset_m", (np.array(selected_pose[:3]) - obj["position"]).tolist())
                arguments.setdefault("close_width_m", self._grasp_close_width(obj, metadata, arguments))
                if "lift_height_m" in metadata:
                    arguments.setdefault("lift_height_m", metadata["lift_height_m"])
            super().start("pick", arguments, snapshot)
            if selected_pose is not None:
                for stage in self.stages:
                    if stage.name != "raise":
                        stage.target.quaternion = tuple(selected_pose[3:])
            self._skip_satisfied_raise(snapshot)
        except (ValueError, TypeError, KeyError) as exc:
            self.result = {"status": "failed", "reason": str(exc)}
            self.stages.clear()

    def _start_explicit_action(self, name, arguments, snapshot):
        self.skill, self.object_id = name, arguments.get("object_id")
        self.result = None
        try:
            if self.grasp_uncertain:
                raise ValueError("Uncertain grasp requires a clean reset")
            current = pose(snapshot["tcp_pose"])
            width = self.target.gripper_width
            if name in ("move_to_pose", "move_object_to_pose", "align_object"):
                target = pose(arguments["pose"])
                if name in ("move_object_to_pose", "align_object"):
                    if self.pending_grasp or self.holding != self.object_id:
                        raise ValueError(f"{name} requires this object to be physically held")
                    if not self.check_follow(snapshot):
                        return
                    obj = snapshot["objects"][self.object_id]
                    if name == "move_object_to_pose":
                        self.transport_goal = target.copy()
                        self.transport_attempt = 0
                        self.transport_anchor = None
                        self.transport_samples = []
                        self.transport_settle = {"samples": 0, "max_pose_drift_m": 0.0, "max_pose_drift_rad": 0.0}
                    target = object_goal_to_tcp(current, [*obj["position"], *obj["quaternion"]], target)
                self.require_vertical = arguments.get("vertical", False)
                if type(self.require_vertical) is not bool:
                    raise ValueError("vertical must be a boolean")
            elif name == "move_by_displacement":
                delta = np.asarray(arguments["displacement"], dtype=float)
                if delta.shape != (3,) or not np.isfinite(delta).all() or np.linalg.norm(delta) > 0.5:
                    raise ValueError("Displacement must be finite XYZ within 0.5 m")
                frame = arguments.get("frame", "world")
                if frame not in ("world", "tool"):
                    raise ValueError("frame must be world or tool")
                delta = rotation(current[3:]) @ delta if frame == "tool" else delta
                target = [*(np.asarray(current[:3]) + delta), *current[3:]]
                self.require_vertical = float(np.linalg.norm(delta[:2])) < 1e-8
            else:
                target = current
            if name == "align_object":
                self.alignment_goal = pose(arguments["pose"])
                current_object = [*obj["position"], *obj["quaternion"]]
                position_error, rotation_error = self._pose_error(current_object, self.alignment_goal)
                if position_error > 0.02 or rotation_error > 0.15:
                    raise ValueError("align_object is a local correction within 2 cm and 0.15 rad; transport first")
                self.alignment_initial_tcp = target
                self.alignment_command = target
                self.alignment_attempt = 0
                self.alignment_anchor = None
                self.alignment_evidence = []
                self.stages = [self._move("align", target[:3], width, target[3:], dwell=0.25, timeout=8.0)]
            elif name in ("move_to_pose", "move_object_to_pose", "move_by_displacement"):
                if self.pending_grasp:
                    delta = np.array(target[:3]) - current[:3]
                    if np.linalg.norm(delta[:2]) > 0.001 or not 0.01 <= delta[2] <= 0.20:
                        raise ValueError("First motion after close must be a 1–20 cm vertical proof lift")
                    self.require_vertical = True
                if self.require_vertical and (
                    np.linalg.norm(np.array(target[:2]) - current[:2]) > 0.001
                    or abs(np.dot(target[3:], current[3:])) < 0.999
                ):
                    raise ValueError("Vertical motion must preserve XY and orientation")
                self.stages = [self._move("move", target[:3], width, target[3:])]
                if name == "move_object_to_pose":
                    self.transport_initial_tcp = target.copy()
                elif self.pending_release_object and not self.holding:
                    released = snapshot["objects"][self.pending_release_object]
                    if np.linalg.norm(np.asarray(target[:3]) - released["position"]) > 0.06:
                        # Insertions may allow only a narrow release between
                        # neighbors. Complete opening after the explicit retreat,
                        # at its reached pose and after a swept collision check.
                        self.reopening_after_retreat = True
                        self.stages.append(
                            self._move("reopen", target[:3], OPEN_WIDTH, target[3:], dwell=0.6, check_pose=False)
                        )
            elif name == "close_gripper":
                if self.holding or self.pending_grasp or self.object_id not in snapshot["objects"]:
                    raise ValueError("close_gripper requires an existing object and an empty gripper")
                width = arguments.get(
                    "close_width_m", snapshot["objects"][self.object_id].get("grasp_close_width_m", 0)
                )
                if type(width) not in (float, int) or not 0 <= width <= self.max_gripper_width:
                    raise ValueError("Invalid gripper width")
                self.stages = [self._move("close", current[:3], width, current[3:], dwell=0.7, check_pose=False)]
            else:
                release_width = self.insertion_release_width if self.insertion_release_width is not None else OPEN_WIDTH
                self.stages = [
                    self._move("release", current[:3], release_width, current[3:], dwell=0.6, check_pose=False)
                ]
            self.elapsed = self.settled = 0.0
            self.target = self.stages[0].target
        except (KeyError, TypeError, ValueError) as exc:
            self.stages.clear()
            self.result = {"status": "failed", "reason": str(exc)}

    def _transport_result(self, snapshot, success, reason):
        width = self.target.gripper_width
        self.stages.clear()
        if not success:
            self.target = self._hold(snapshot)
            self.target.gripper_width = width
        self.result = {
            "status": "success" if success else "failed",
            "reason": reason,
            "holding": self.holding,
            "object_goal_pose": list(self.transport_goal),
            "transport_corrections": self.transport_attempt,
            "transport_evidence": copy.deepcopy(self.transport_evidence),
            "transport_settle": copy.deepcopy(self.transport_settle),
            "transport_recent_samples": copy.deepcopy(self.transport_samples),
        }

    def _verify_object_transport(self, snapshot):
        """A reached TCP is insufficient evidence that a compliant grasp reached its goal."""
        obj = snapshot["objects"][self.object_id]
        measured = [*obj["position"], *obj["quaternion"]]
        distance, angle = self._pose_error(measured, self.transport_goal)
        self.transport_evidence.append({
            "attempt": self.transport_attempt,
            "measured_object_pose": measured,
            "object_position_error_m": distance,
            "object_rotation_error_rad": angle,
            "linear_velocity_m_s": obj.get("linear_velocity"),
            "angular_velocity_rad_s": obj.get("angular_velocity"),
            "stable_window_seconds": self.settled,
        })
        if distance <= 0.002 and angle <= 0.02:
            self._transport_result(snapshot, True, "Measured object goal reached within 2 mm and 0.02 rad")
            return
        if distance > 0.02 or angle > 0.25:
            self._transport_result(snapshot, False, "TRANSPORT_CORRECTION_LIMIT: object error exceeds 2 cm or 0.25 rad")
            return
        if self.transport_attempt >= 2:
            self._transport_result(snapshot, False, "TRANSPORT_NOT_CONVERGED: two measured corrections exhausted")
            return
        # Apply the measured object error to the last command: using only the
        # measured TCP again would resend the same target under static drive bias.
        command = compose(
            compose(self.transport_goal, inverse(measured)),
            [*self.target.position, *self.target.quaternion],
        )
        shift, turn = self._pose_error(command, self.transport_initial_tcp)
        if shift > 0.02 or turn > 0.25:
            self._transport_result(snapshot, False, "TRANSPORT_CORRECTION_LIMIT: command exceeds 2 cm or 0.25 rad")
            return
        current = pose(snapshot["tcp_pose"])
        if self.require_vertical and (
            np.linalg.norm(np.asarray(command[:2]) - current[:2]) > 0.001
            or abs(np.dot(command[3:], current[3:])) < 0.999
        ):
            self._transport_result(snapshot, False, "TRANSPORT_VERTICAL_CONSTRAINT: correction requires lateral motion")
            return
        try:
            stage = self._move("move", command[:3], self.target.gripper_width, command[3:], timeout=8.0)
        except ValueError as exc:
            self._transport_result(snapshot, False, f"TRANSPORT_CORRECTION_INVALID: {exc}")
            return
        # A new stage identity runs through the same collision-aware ActionPlanner.
        # No object pose is written and no collision/grasp check is bypassed.
        self.transport_attempt += 1
        self.transport_anchor = None
        self.stages = [stage]
        self.target = stage.target
        self.elapsed = self.settled = 0.0

    def _tick_object_transport(self, snapshot, dt, trajectory_complete):
        if not self.stages:
            return self.target
        if not trajectory_complete:
            self.transport_anchor = None
            self.settled = 0.0
            return self.target
        self.elapsed += dt
        stage = self.stages[0]
        distance, angle = self._pose_error(snapshot["tcp_pose"], [*stage.target.position, *stage.target.quaternion])
        obj = snapshot["objects"][self.object_id]
        measured = [*obj["position"], *obj["quaternion"]]
        # Contact stabilization reports solver velocities that can disagree
        # with actual pose evolution. As in the shared insertion controller,
        # require a measured stable pose window; retain raw velocities as
        # diagnostics and reject non-finite values.
        linear = np.asarray(obj.get("linear_velocity", [float("inf")] * 3), dtype=float)
        angular = np.asarray(obj.get("angular_velocity", [float("inf")] * 3), dtype=float)
        finite_velocity = np.isfinite(linear).all() and np.isfinite(angular).all()
        goal_distance, goal_angle = self._pose_error(measured, self.transport_goal)
        drift, turn = (
            self._pose_error(measured, self.transport_anchor) if self.transport_anchor is not None else (0.0, 0.0)
        )
        summary = self.transport_settle
        if summary["samples"] == 0:
            summary["first_object_pose"] = measured
        summary["samples"] += 1
        summary["last_object_pose"] = measured
        summary["max_pose_drift_m"] = max(summary["max_pose_drift_m"], drift)
        summary["max_pose_drift_rad"] = max(summary["max_pose_drift_rad"], turn)
        summary["max_stable_seconds"] = max(summary.get("max_stable_seconds", 0.0), self.settled)
        for label, value in (("object_position_error_m", goal_distance), ("object_rotation_error_rad", goal_angle)):
            summary["min_" + label] = min(summary.get("min_" + label, value), value)
            summary["max_" + label] = max(summary.get("max_" + label, value), value)
        row = {
            "attempt": self.transport_attempt,
            "settle_elapsed_s": self.elapsed,
            "object_pose": measured,
            "tcp_position_error_m": distance,
            "tcp_rotation_error_rad": angle,
            "object_position_error_m": goal_distance,
            "object_rotation_error_rad": goal_angle,
            "anchor_drift_m": drift,
            "anchor_drift_rad": turn,
            "linear_speed_m_s": float(np.linalg.norm(linear)) if finite_velocity else None,
            "angular_speed_rad_s": float(np.linalg.norm(angular)) if finite_velocity else None,
            "stable_seconds": self.settled,
        }
        self.transport_samples.append(row)
        del self.transport_samples[:-32]
        if self.elapsed > stage.timeout:
            self._transport_result(snapshot, False, "TRANSPORT_SETTLE_TIMEOUT: object or TCP did not settle")
            return self.target
        if distance > 0.006 or angle > 0.1 or not finite_velocity:
            self.transport_anchor = None
            self.settled = 0.0
            return self.target
        if self.transport_anchor is None:
            self.transport_anchor = measured
            self.settled = 0.0
            return self.target
        drift, turn = self._pose_error(measured, self.transport_anchor)
        if drift > 0.0003 or turn > 0.003:
            self.transport_anchor = measured
            self.settled = 0.0
            return self.target
        self.settled += dt
        self.transport_settle["max_stable_seconds"] = max(
            self.transport_settle.get("max_stable_seconds", 0.0), self.settled
        )
        self.transport_samples[-1]["stable_seconds"] = self.settled
        if self.settled >= 0.25:
            self._verify_object_transport(snapshot)
        return self.target

    @staticmethod
    def _pose_error(current, goal):
        current, goal = pose(current), pose(goal)
        position_error = float(np.linalg.norm(np.asarray(current[:3]) - goal[:3]))
        rotation_error = float(2 * np.arccos(np.clip(abs(np.dot(current[3:], goal[3:])), 0.0, 1.0)))
        return position_error, rotation_error

    def _alignment_failed(self, snapshot, reason):
        width = self.target.gripper_width
        self.stages.clear()
        self.target = self._hold(snapshot)
        self.target.gripper_width = width
        self.result = {
            "status": "failed",
            "reason": reason,
            "holding": self.holding,
            "alignment_evidence": copy.deepcopy(self.alignment_evidence),
        }

    def _tick_alignment(self, snapshot, dt, trajectory_complete):
        if not self.stages:
            return self.target
        stage = self.stages[0]
        self.elapsed += dt
        if self.elapsed > stage.timeout:
            self._alignment_failed(snapshot, "ALIGNMENT_TIMEOUT: object did not settle")
            return self.target
        obj = snapshot["objects"][self.object_id]
        current_object = [*obj["position"], *obj["quaternion"]]
        # Contact stabilization can leave nonzero solver velocities while the
        # measured pose is stationary. Require a stable pose window instead.
        if not trajectory_complete:
            self.alignment_anchor = None
            self.settled = 0.0
            return self.target
        if self.alignment_anchor is None:
            self.alignment_anchor = current_object
            self.settled = 0.0
            return self.target
        drift, angle = self._pose_error(current_object, self.alignment_anchor)
        if drift > 0.0001 or angle > 0.001:
            self.alignment_anchor = current_object
            self.settled = 0.0
            return self.target
        self.settled += dt
        if self.settled < stage.dwell:
            return self.target
        position_error, rotation_error = self._pose_error(current_object, self.alignment_goal)
        self.alignment_evidence.append({
            "attempt": self.alignment_attempt,
            "object_position_error_m": position_error,
            "object_rotation_error_rad": rotation_error,
            "command_tcp": list(self.alignment_command),
            "measured_object": current_object,
        })
        if position_error <= 0.0005 and rotation_error <= 0.005:
            self.stages.clear()
            self.result = {
                "status": "success",
                "reason": "Measured object aligned within 0.5 mm and 0.005 rad",
                "holding": self.holding,
                "alignment_evidence": copy.deepcopy(self.alignment_evidence),
            }
            return self.target
        if self.alignment_attempt >= 3:
            self._alignment_failed(snapshot, "ALIGNMENT_NOT_CONVERGED: bounded correction budget exhausted")
            return self.target
        # Integrate measured SE(3) error into the last commanded pose. Repeatedly
        # sending the same ideal IK target cannot compensate steady tracking bias.
        correction = compose(self.alignment_goal, inverse(current_object))
        command = compose(correction, self.alignment_command)
        translation, angle = self._pose_error(command, self.alignment_initial_tcp)
        if translation > 0.01 or angle > 0.1:
            self._alignment_failed(snapshot, "ALIGNMENT_CORRECTION_LIMIT: exceeds 1 cm or 0.1 rad")
            return self.target
        self.alignment_attempt += 1
        self.alignment_anchor = None
        self.alignment_command = command
        self.stages = [
            self._move("align", command[:3], self.target.gripper_width, command[3:], dwell=0.25, timeout=8.0)
        ]
        self.elapsed = self.settled = 0.0
        self.target = self.stages[0].target
        return self.target

    def _gripper_ready(self, snapshot, phase):
        """Keep commanding the jaws until measured contact/opening converges."""
        if phase == "close" and self.object_id:
            return contact_evidence(snapshot, self.object_id)["passed"]
        released = self.pending_release_object if phase == "reopen" else getattr(self, "insertion_release_object", None)
        if phase in ("release", "reopen") and released:
            forces = snapshot["objects"][released].get("finger_contact_forces_n")
            clear = (
                isinstance(forces, (list, tuple))
                and len(forces) == 2
                and all(np.isfinite(v) and abs(v) < 0.2 for v in forces)
            )
            return clear and abs(snapshot["gripper_width"] - self.target.gripper_width) <= 0.0002
        return True

    def tick(self, snapshot, dt, *, trajectory_complete=True):
        previous = self.phase
        if previous not in ("close", "release", "reopen") and not self.check_follow(snapshot, dt=dt):
            return self.target
        if self.skill == "move_object_to_pose":
            return self._tick_object_transport(snapshot, dt, trajectory_complete)
        if self.skill == "align_object":
            return self._tick_alignment(snapshot, dt, trajectory_complete)
        if not trajectory_complete:
            return self.target
        if self.stages and previous in ("close", "release", "reopen"):
            stage = self.stages[0]
            # Dwell is a minimum, not proof that an actuator reached its target.
            # Preserve the original validator and bound blocked-jaw waiting.
            if stage.dwell <= self.elapsed + dt < min(stage.timeout, 2.0) and not self._gripper_ready(
                snapshot, previous
            ):
                self.elapsed += dt
                return self.target
        if previous == "home" and self.home_joint_positions:
            actual = dict(zip(snapshot["joint_names"], snapshot["joint_positions"]))
            error = max(abs(actual[name] - goal) for name, goal in self.home_joint_positions.items())
            if error > 0.02:
                self.elapsed += dt
                self.settled = 0.0
                if self.elapsed > self.stages[0].timeout:
                    self.cancel(snapshot)
                    self.result = {
                        "status": "failed",
                        "reason": "Home joints did not converge",
                        "max_joint_error_rad": error,
                    }
                return self.target
        result = super().tick(snapshot, dt)
        if previous == "close" and self.phase != "close" and self.skill == "pick":
            self._begin_contact(snapshot)
            result = self.target
        if previous == "release" and self.phase != "release":
            self.pending_grasp = self.grasp_reference = None
            self.holding = None
        return result

    def _verify(self, snapshot):
        if self.skill == "home" and self.home_joint_positions:
            actual = dict(zip(snapshot["joint_names"], snapshot["joint_positions"]))
            error = max(abs(actual[name] - goal) for name, goal in self.home_joint_positions.items())
            self.result = {
                "status": "success" if error <= 0.02 else "failed",
                "reason": "Initial arm joint configuration reached",
                "max_joint_error_rad": error,
                "target_joint_positions": self.home_joint_positions,
                "holding": self.holding,
            }
        elif self.skill == "close_gripper":
            if self._begin_contact(snapshot):
                self.result = {
                    "status": "success",
                    "reason": "Contact established; vertical proof lift required",
                    "holding": None,
                }
        elif self.skill == "move_object_to_pose":
            self._verify_object_transport(snapshot)
        elif self.skill in ("move_to_pose", "move_by_displacement"):
            if self.pending_grasp and not self.check_follow(snapshot, require_lift=True):
                return
            if self.reopening_after_retreat:
                if not self._gripper_ready(snapshot, "reopen"):
                    self.result = {
                        "status": "failed",
                        "reason": "POST_RETREAT_GRIPPER_NOT_OPEN",
                        "measured_width_m": snapshot["gripper_width"],
                        "target_width_m": self.target.gripper_width,
                    }
                    return
                self.pending_release_object = None
                self.reopening_after_retreat = False
            self.result = {"status": "success", "reason": "Measured pose reached", "holding": self.holding}
        elif self.skill == "open_gripper":
            released = getattr(self, "insertion_release_object", None)
            if released:
                forces = snapshot["objects"][released].get("finger_contact_forces_n")
                # Waiting and final validation must use the same measured
                # clearance/aperture condition, including the 0.2 mm tolerance.
                ready = self._gripper_ready(snapshot, "release")
                if ready:
                    self.pending_release_object = released if self.target.gripper_width < OPEN_WIDTH - 0.0002 else None
                self.result = {
                    "status": "success" if ready else "failed",
                    "reason": "Insertion release verified" if ready else "INSERTION_RELEASE_NOT_CLEAR",
                    "holding": None,
                    "release_width_m": self.target.gripper_width,
                    "measured_width_m": snapshot["gripper_width"],
                    "width_error_m": abs(snapshot["gripper_width"] - self.target.gripper_width),
                    "finger_contact_forces_n": forces,
                }
            else:
                self.result = {"status": "success", "reason": "Gripper opened", "holding": None}
        elif self.skill == "pick":
            if self.check_follow(snapshot, require_lift=True):
                super()._verify(snapshot)
        else:
            super()._verify(snapshot)
