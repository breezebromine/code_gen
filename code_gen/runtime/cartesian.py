# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Task/recording adapter for the shared ActionPlanner insertion controller."""


import math
import numpy as np
import torch

from code_gen._io import write_json
from code_gen.planning.constraints import contact_supports
from isaaclab_arena.grasping.joint_actions import write_arm_action


def execute_insertion(backend, arguments):
    return backend.drive(insertion_steps(backend, arguments))


def insertion_steps(backend, arguments):
    from isaaclab_arena.grasping.cartesian_insertion import CartesianObservation

    skills, policy = backend.policy.skills, backend.policy
    name = arguments["object_id"]
    if backend.layout.robot != "aloha" or backend.motion_planner != "curobo":
        raise ValueError("insert_object requires Aloha with cuRobo transport")
    if skills.grasp_uncertain or skills.pending_grasp or skills.holding != name:
        raise ValueError("insert_object requires this object to be physically held")
    snapshot = policy.snapshot()
    if not skills.check_follow(snapshot):
        raise ValueError("insert_object requires a verified grasp")
    motion = backend.ensure_curobo_motion()
    width = skills.target.gripper_width
    action = backend.joint_hold.clone()
    if backend.actions:
        action = torch.as_tensor(backend.actions[-1], device=backend.env.unwrapped.device).clone()
    start = backend.layout.action_start
    action[:, start + 6] = width

    def observation(state, grasp_valid):
        obj = state["objects"][name]
        return CartesianObservation(
            object_pose=[*obj["position"], *obj["quaternion"]],
            tcp_pose=state["tcp_pose"],
            other_object_poses={
                n: [*o["position"], *o["quaternion"]] for n, o in state["objects"].items() if n != name
            },
            other_finger_contacts={
                n: o.get("finger_contact_forces_n", []) for n, o in state["objects"].items() if n != name
            },
            grasp_valid=grasp_valid,
            joint_limits_ok=backend.joint_limits_ok,
            step_index=len(backend.actions),
        )

    collision = create_collision_session(motion, snapshot, name)

    def step(joint_command):
        collision.check(joint_command)
        write_arm_action(
            action,
            0,
            arm_joint_names=backend.layout.arm_names,
            offset=start,
            gripper_width=width,
            positions=joint_command[0],
            joint_names=backend.layout.arm_names,
        )
        yield action.clone()
        state = policy.snapshot()
        return observation(state, skills.check_follow(state, dt=backend.dt))

    controller = None
    skills.skill, skills.object_id = "insert_object", name
    skills.stages.clear()
    skills.result = None
    result = {"success": False, "status": "failed"}
    try:
        controller = motion.planner.create_insertion_controller(
            backend.env_id,
            backend.layout.arm,
            tcp_offset=backend.layout.tcp_offset,
            tcp_rotation=backend.layout.tcp_rotation,
            joint_names=backend.layout.arm_names,
            initial_joint_command=action[:, start : start + 6],
            workspace_bounds=skills.workspace_bounds,
            dt=backend.dt,
        )
        # Alignment and descent share one measured-object controller. Every
        # proposed joint command is checked by step() before physics execution.
        from code_gen.task.spec import TablePlacementGoal

        terminal_goals = [
            goal
            for goal in backend.task.all_goals
            if isinstance(goal, TablePlacementGoal)
            and goal.object_id == name
            and math.dist(arguments["pose"][:3], snapshot["targets"][goal.target_id]["position"]) <= 1e-6
        ]
        commands = controller.command_steps(
            batch_ik=True,
            release_clearance_m=min(0.001, min(g.z_tolerance_m for g in terminal_goals)) if terminal_goals else 0.0,
            release_xy_tolerance_m=(
                min(0.001, min(g.xy_tolerance_m for g in terminal_goals)) if terminal_goals else 0.0005
            ),
            goal_pose=arguments["pose"],
            approach_height=arguments.get("approach_height", 0.02),
            initial_state=observation(snapshot, True),
        )
        try:
            command = next(commands)
            while True:
                from isaaclab_arena.grasping.cartesian_motion import CartesianTargetRequest

                if isinstance(command, CartesianTargetRequest):
                    state = yield command
                else:
                    state = yield from step(command)
                command = commands.send(state)
        except StopIteration as completed:
            result = completed.value
    except Exception as exc:
        result.update(success=False, status="failed", reason=str(exc), error_type=type(exc).__name__)
        raise
    finally:
        try:
            collision.close()
        finally:
            # Preserve the last actually submitted command when returning to planned motion.
            backend.joint_hold = action.clone()
            motion.stage, motion.positions, motion.index = None, None, 0
            motion.hold = action[:, start : start + 6].clone()
            skills.stages.clear()
            skills.target = skills._hold(policy.snapshot())
            skills.target.gripper_width = width
            skills.result = None
            result.update(
                object_id=name,
                holding=skills.holding,
                physics_collisions=True,
                collision_checks=collision.guard.checks,
                command_collision_check=True,
                max_support_collision_m=collision.guard.max_support_cost,
            )
            directory = backend.output / "cartesian"
            directory.mkdir(exist_ok=True)
            path = directory / f"insertion_{len(backend.events):03d}.json"
            write_json(path, {"result": result, "trace": controller.rows if controller is not None else []})
            result["trace_file"] = str(path.relative_to(backend.output))
    return result


def create_collision_session(motion, snapshot, object_id=None, *, grasp_contact_object=None):
    """Resolve task contacts and delegate collision state to ActionPlanner."""
    names, paths = contact_supports(motion.backend.scene_spec or {}, snapshot, object_id, env_id=motion.env_id)
    return motion.planner.create_cartesian_collision_session(
        motion.env_id,
        motion.layout.arm,
        joint_names=motion.arm_names,
        finger_names=motion.layout.finger_names,
        object_id=object_id,
        support_ids=names,
        support_paths=paths,
        grasp_contact_object=grasp_contact_object,
    )


class LocalSegment:
    def __init__(self, motion, stage, snapshot):
        self.motion, self.stage = motion, stage
        skills = motion.policy.skills
        held = skills.holding or getattr(skills, "pending_grasp", None)
        self.session = create_collision_session(
            motion,
            snapshot,
            held,
            grasp_contact_object=skills.object_id if stage.name == "descend" and not held else None,
        )
        try:
            self.others = {
                n: (o["position"], o["quaternion"])
                for n, o in snapshot["objects"].items()
                if n != (held or skills.object_id)
            }
            self.controller = motion.planner.create_cartesian_line_controller(
                motion.env_id,
                motion.layout.arm,
                tcp_offset=motion.layout.tcp_offset,
                tcp_rotation=motion.layout.tcp_rotation,
                joint_names=motion.arm_names,
                initial_joint_command=motion.hold,
                workspace_bounds=skills.workspace_bounds,
                dt=motion.backend.dt,
            )
            # Descent retains the selected grasp point; a lift/retreat starts vertically
            # from the measured pose, preserving the physically established grasp.
            if stage.name in ("lift", "retreat", "raise"):
                stage.target.position = (*snapshot["tcp_pose"][:2], stage.target.position[2])
                stage.target.quaternion = tuple(snapshot["tcp_pose"][3:])
            self.controller.begin(
                snapshot["tcp_pose"],
                [*stage.target.position, *stage.target.quaternion],
                # Preserve the settled drive offset for an empty arm too; its
                # own weight still deflects the joints after cuRobo approach.
                preserve_load=True,
            )
            self.record = {
                "action": skills.skill,
                "phase": stage.name,
                "source": "ActionPlanner.CartesianLine",
                "success": False,
                "attached_object": held,
                "support_paths": sorted(self.session.guard.support_paths),
            }
            motion.records.append(self.record)
            self.index = len(motion.records) - 1
            self.save()
        except BaseException:
            self.session.close()
            raise

    @property
    def done(self):
        return self.controller.done

    def advance(self, snapshot):
        from isaaclab_arena.grasping.cartesian_motion import resolve_cartesian_steps

        return resolve_cartesian_steps(self.advance_steps(snapshot))

    def advance_steps(self, snapshot):
        try:
            for name, (position, quaternion) in self.others.items():
                obj = snapshot["objects"][name]
                distance = np.linalg.norm(np.asarray(obj["position"]) - position)
                angle = 2 * np.arccos(np.clip(abs(np.dot(obj["quaternion"], quaternion)), 0, 1))
                if distance > 0.004 or angle > np.deg2rad(2):
                    raise RuntimeError(f"CARTESIAN_OTHER_OBJECT_MOVED: {name}")
            command = yield from self.controller.advance_steps(snapshot["tcp_pose"])
            self.session.check(command)
            self.record["success"] = self.done
            self.record["status"] = "Success" if self.done else "running"
            return command
        except Exception as exc:
            self.record["status"] = str(exc)
            self.record["rejected_measured_tcp"] = snapshot["tcp_pose"]
            self.record["tracking_tail"] = self.controller.rows[-5:]
            try:
                self.close()
            except Exception as cleanup_error:
                exc.add_note(f"Local collision cleanup failed: {cleanup_error}")
            raise

    def save(self):
        self.record.update(
            checks=self.session.guard.checks,
            max_world_cost_m=self.session.guard.max_world_cost,
            max_support_cost_m=self.session.guard.max_support_cost,
        )
        write_json(
            self.motion.output / f"local_{self.index:03d}.json", {"record": self.record, "trace": self.controller.rows}
        )
        write_json(self.motion.output / "plans.json", self.motion.records)

    def close(self):
        try:
            self.save()
        finally:
            self.session.close()
