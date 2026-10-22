# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Isaac Lab adapter. Import only after SimulationApp initialization."""

import copy
import gc
import hashlib
import json
import math
import numpy as np
import sys
import time
import torch
from pathlib import Path

from code_gen._io import write_json
from code_gen.runtime.episode import validate_runtime
from code_gen.task.program import Program
from code_gen.task.spec import TaskSpec
from isaaclab_arena.grasping.robot_layout import RobotLayout


class TabletopBackend:
    """Own stepping, observations, independent goals, and exact action replay."""

    def __init__(self, args, app, task, *, env=None, env_id=0, prepared_observation=None):
        self.args, self.app, self.task = args, app, task
        self.env_id, self.owns_env = env_id, env is None
        self.output = Path(args.output_dir)
        self.scene_spec = json.loads(Path(task.scene_file).read_text()) if task.scene_file else None
        self.requested_arm = getattr(args, "arm", "left")
        if self.requested_arm not in ("left", "right", "auto"):
            raise ValueError("Unknown arm request")
        self.layout = RobotLayout(
            getattr(args, "robot", "arx5"), "left" if self.requested_arm == "auto" else self.requested_arm
        )
        validate_runtime(
            self.layout.robot, getattr(args, "motion_planner", "ik"), state_only=getattr(args, "state_only", False)
        )
        if self.requested_arm == "auto" and self.layout.robot != "aloha":
            raise ValueError("Automatic arm selection requires Aloha")
        if self.layout.robot == "aloha":
            from code_gen.runtime.scene import make_env as make_aloha_env
            from code_gen.runtime.scene import prepare_scene

            self.scene_spec = prepare_scene(self.scene_spec)
            write_json(self.output / "resolved_scene.json", self.scene_spec)
            self.env = env if env is not None else make_aloha_env(args, scene_spec=self.scene_spec)
        else:
            from isaaclab_arena.environments.tabletop_environment import make_arx5_env as make_env

            self.env = env if env is not None else make_env(args, scene_spec=self.scene_spec)
        self._planner = None
        self._arm_planners = {}
        self._planner_rng_states = {}
        self._episode_index = 0
        try:
            self._initialize_episode(prepared_observation)
        except BaseException:
            if self.owns_env:
                self.env.close()
            raise

    def _initialize_episode(self, prepared_observation=None):
        """Reset physics and episode-local buffers while keeping the simulator alive."""
        self.motion = None
        self.current_action = None
        base = self.env.unwrapped
        if prepared_observation is None:
            self.obs, _ = self.env.reset(seed=self.args.seed)
            if self.scene_spec is not None:
                for name, info in self.scene_spec["objects"].items():
                    body = self.env.unwrapped.scene[name]
                    state = body.data.default_root_state.clone()
                    state[:, :3] = torch.tensor(info["position"], device=state.device)
                    state[:, :3] += self.env.unwrapped.scene.env_origins
                    state[:, 3:7] = torch.tensor(info.get("quaternion", [1, 0, 0, 0]), device=state.device)
                    state[:, 7:] = 0
                    body.write_root_state_to_sim(state)
                self.env.unwrapped.scene.write_data_to_sim()
            self.policy = self.make_policy()
            self.motion_planner = getattr(self.args, "motion_planner", "ik")
            self.arm_ids = self.policy.robot.find_joints(self.layout.arm_names, preserve_order=True)[0]
            self.joint_hold = self.measured_joint_action()
            settle_action = self.joint_hold
            if self.motion_planner == "ik":
                # Relative-EEF actions hold the measured pose with zero delta, not joint angles.
                settle_action = torch.zeros_like(self.joint_hold)
                settle_action[:, -1] = self.policy.snapshot()["gripper_width"]
            base = self.env.unwrapped
            render_interval = base.cfg.sim.render_interval
            try:
                if self.args.enable_cameras:
                    base.cfg.sim.render_interval = 10**9
                with torch.no_grad():
                    for _ in range(150 if self.scene_spec is not None else 50):
                        self.obs, _, _, _, _ = self.env.step(settle_action)
            finally:
                base.cfg.sim.render_interval = render_interval
            if self.args.enable_cameras:
                for _ in range(8):
                    base.sim.render()
                    base.scene.update(dt=0.0)
                self.obs = base.observation_manager.compute()
        else:
            self.obs = prepared_observation
            self.policy = self.make_policy()
            self.motion_planner = getattr(self.args, "motion_planner", "ik")
            self.joint_hold = self.measured_joint_action()
        self._select_episode_arm()
        self.policy = self.make_policy()
        self.arm_ids = self.policy.robot.find_joints(self.layout.arm_names, preserve_order=True)[0]
        self.initial = self.policy.snapshot()
        if getattr(self.args, "capture_initial_layout", False):
            self.capture_initial_layout()
        if self.scene_spec is not None:
            checks = {}
            for name, info in self.scene_spec["objects"].items():
                actual = self.initial["objects"][name]
                checks[name] = {
                    "position_error_m": math.dist(
                        (np.asarray(actual["position"]) - base.scene.env_origins[self.env_id].cpu().numpy()).tolist(),
                        info["position"],
                    ),
                    "linear_speed_m_s": math.sqrt(sum(v * v for v in actual["linear_velocity"])),
                }
            write_json(self.output / "initial_layout.json", checks)
            write_json(self.output / "settled_snapshot.json", self.initial)
            self.policy.images(self.obs, str(self.output))
            if any(v["position_error_m"] > 0.003 or v["linear_speed_m_s"] > 0.02 for v in checks.values()):
                failed = {
                    name: row
                    for name, row in checks.items()
                    if row["position_error_m"] > 0.003 or row["linear_speed_m_s"] > 0.02
                }
                raise ValueError(f"Configured scene did not settle in env {self.env_id}: {failed}")
        self.task.validate_scene(self.initial)
        self.dt = float(base.step_dt)
        self.actions, self.joints, self.manifest, self.events = [], [], [], []
        self.milestone_index, self.milestone_streak = 0, 0
        self.milestone_records = []
        self.max_joint_violation = 0.0
        self.joint_limits_ok = True
        self.deadline = time.monotonic() + self.args.execution_timeout
        self.timings = {"physics_step_s": 0.0, "planning_and_control_s": 0.0, "recording_s": 0.0}
        self.capture("initial")
        self.recorder = None
        if self.args.mode != "inspect" and (self.args.enable_cameras or getattr(self.args, "state_only", False)):
            from code_gen.recording.recorder import CodegenRecorder

            self.recording_epoch = time.time()
            self.recorder = CodegenRecorder(self)

    def _select_episode_arm(self):
        """Choose once per episode, using the same settled-scene rule as MGG."""
        from isaaclab_arena.grasping.action_planner import ActionPlanner

        requested = getattr(self.args, "arm", "left")
        object_ids = list(dict.fromkeys(goal.object_id for goal in getattr(self.task, "all_goals", self.task.goals)))
        if requested == "auto" and len(object_ids) != 1:
            raise ValueError("Automatic per-episode arm selection requires one task object")
        object_id = object_ids[0]
        env = self.env.unwrapped
        position = env.scene[object_id].data.root_pos_w[self.env_id] - env.scene.env_origins[self.env_id]
        transform = np.eye(4)
        transform[:3, 3] = position.cpu().numpy()
        selected = ActionPlanner.nearest_arm(env, self.env_id, transform) if requested == "auto" else requested
        self.layout = RobotLayout(self.layout.robot, selected)
        self._retain_active_planner(selected)
        self.arm_selection = {
            "requested": getattr(self.args, "requested_arm", requested),
            "selected": selected,
            "object_id": object_id,
            "rule": (
                "nearest arm first; bounded grasp/place screening"
                if requested == "auto"
                else (
                    "bounded-search fallback"
                    if getattr(self.args, "requested_arm", requested) == "auto"
                    else "explicit"
                )
            ),
            "object_position_env": transform[:3, 3].tolist(),
        }
        if self.layout.robot == "aloha":
            self.arm_selection["eef_distances_m"] = {
                arm: float(
                    torch.linalg.vector_norm(
                        env.scene[object_id].data.root_pos_w[self.env_id]
                        - env.scene[f"ee_frame_{arm}"].data.target_pos_w[self.env_id, 0]
                    )
                )
                for arm in ("left", "right")
            }
        write_json(self.output / "arm_selection.json", self.arm_selection)
        print(f"CODEGEN arm={selected} requested={requested} object={object_id}", flush=True)

    def _retain_active_planner(self, selected):
        """Release the other arm's single/batch CUDA graphs at a clean episode boundary."""
        if not self.owns_env:
            # Vectorized slots switch arms independently. Keep the two bounded
            # arm planners warm instead of rebuilding CUDA graphs on each retry.
            self._planner = self._arm_planners.get(selected)
            return
        inactive = [arm for arm in self._arm_planners if arm != selected]
        if inactive:
            cuda_initialized = torch.cuda.is_initialized()
            before = (
                {
                    "allocated_bytes": torch.cuda.memory_allocated(),
                    "reserved_bytes": torch.cuda.memory_reserved(),
                }
                if cuda_initialized
                else None
            )
            self._planner = None
            for arm in inactive:
                self._planner_rng_states[arm] = self._arm_planners[arm]._rng.getstate()
                del self._arm_planners[arm]
            # Kit caches failed import stat calls with their exception tracebacks.
            # Those frames can pin the first native planner and its CUDA graphs.
            # Invalidate through Kit before collecting otherwise-unowned planners.
            stat_cache = sys.modules.get("omni.ext._impl.stat_cache")
            if stat_cache is not None:
                stat_cache.reset_stat_cache()
            # Old motion/local-segment objects can form cycles. No reset helper
            # locals may retain their lower planner or MotionGen objects here.
            gc.collect()
            if cuda_initialized:
                torch.cuda.empty_cache()
            after = (
                {
                    "allocated_bytes": torch.cuda.memory_allocated(),
                    "reserved_bytes": torch.cuda.memory_reserved(),
                }
                if cuda_initialized
                else None
            )
            print(
                "CODEGEN planner_cache_release "
                + json.dumps({
                    "selected": selected,
                    "released": inactive,
                    "cuda_before": before,
                    "cuda_after": after,
                }),
                flush=True,
            )
        self._planner = self._arm_planners.get(selected)

    def _reset_active_planner(self):
        """End all temporary planner references before initialization can switch arms."""
        if self._planner is not None:
            from curobo.rollout.cost.pose_cost import PoseCostMetric

            lower = self._planner._get_planner(self.layout.arm)
            for mg in (lower.motion_gen, getattr(lower, "_motion_gen_batch", None)):
                if mg is not None:
                    mg.detach_object_from_robot(link_name=lower.config.attached_object_link_name)
                    mg.update_pose_cost_metric(PoseCostMetric.reset_metric())
                    mg.reset(reset_seed=True)
            lower.attached_objects.clear()
            lower.reset_plan()

    def reset_episode(self, args, task, *, prepared_observation=None):
        """Same-topology reset, with fresh goals, layout, visuals and observation origins."""
        from code_gen.runtime.scene import prepare_scene
        from isaaclab_arena.environments.tabletop_environment import reset_scene_configuration, scene_topology

        source = json.loads(Path(task.scene_file).read_text())
        scene = prepare_scene(source)
        if scene_topology(scene) != scene_topology(self.scene_spec):
            raise ValueError("Persistent episode changed asset topology; use a new production session")
        self._reset_active_planner()
        self.args, self.task, self.output, self.scene_spec = args, task, Path(args.output_dir), scene
        self._episode_index += 1
        if prepared_observation is None:
            reset_scene_configuration(self.env.unwrapped, scene)
        write_json(self.output / "resolved_scene.json", scene)
        self._initialize_episode(prepared_observation)

    def measured_joint_action(self):
        robot = self.policy.robot
        action = torch.zeros((1, self.layout.action_dim), device=self.env.unwrapped.device)
        arms = ("left", "right") if self.layout.robot == "aloha" else ("left",)
        for arm in arms:
            layout = RobotLayout(self.layout.robot, arm)
            ids = robot.find_joints(layout.arm_names, preserve_order=True)[0]
            fingers = robot.find_joints(layout.finger_names, preserve_order=True)[0]
            signs = torch.tensor(layout.finger_signs, device=action.device)
            start = layout.action_start
            action[:, start : start + 6] = robot.data.joint_pos[self.env_id : self.env_id + 1, ids]
            action[:, start + 6] = (
                (robot.data.joint_pos[self.env_id : self.env_id + 1, fingers] * signs).sum(dim=-1).clamp(0, 0.088)
            )
        return action

    def ensure_curobo_motion(self):
        """Initialize the shared planner without advancing the skill state."""
        if self.motion_planner != "curobo":
            raise ValueError("This action requires the cuRobo motion backend")
        if self.motion is None:
            from code_gen.planning.motion import CuroboMotion

            self.motion = CuroboMotion(self, planner=self._planner)
            self._planner = self.motion.planner
            if self.layout.arm in self._planner_rng_states:
                self._planner._rng.setstate(self._planner_rng_states.pop(self.layout.arm))
            self._arm_planners[self.layout.arm] = self._planner
        return self.motion

    def default_action(self):
        if self.motion_planner == "curobo":
            if self.args.mode == "execute" and self.motion is None and hasattr(self, "dt"):
                self.ensure_curobo_motion()
            return self.motion.get_action() if self.motion is not None else self.joint_hold.clone()
        return self.policy.get_action(self.env, self.obs)

    def make_policy(self):
        if self.layout.robot == "aloha":
            from code_gen.runtime.policy import AlohaPolicy

            return AlohaPolicy(self.env, self.scene_spec, self.layout.arm, env_id=self.env_id)
        if self.scene_spec is not None:
            from code_gen.runtime.policy import ConfiguredARX5Policy

            return ConfiguredARX5Policy(self.env, self.scene_spec)
        from code_gen.runtime.policy import ARX5Policy

        return ARX5Policy(self.env)

    def capture_initial_layout(self):
        """Render the settled layout once, before any task action or planning."""
        import isaaclab.utils.math as math_utils

        from isaaclab_arena.utils.layout_preview import annotate_layout, layout_items

        base = self.env.unwrapped
        name = "head_camera" if self.args.enable_cameras else "layout_camera"
        camera = base.scene[name]
        # Render-only warmup updates textures/sensors without advancing physics.
        for _ in range(8):
            base.sim.render()
            camera.update(0.0, force_recompute=True)
        rgb = camera.data.output["rgb"][0, ..., :3].detach().cpu().numpy()
        if rgb.size == 0 or not np.isfinite(rgb).all():
            raise RuntimeError("Invalid initial-layout camera frame")
        items = layout_items(self.initial, self.scene_spec or {}, self.task.goals)
        camera_position = camera.data.pos_w[0].detach().cpu().tolist()
        camera_rotation = math_utils.matrix_from_quat(camera.data.quat_w_ros[0]).detach().cpu().tolist()
        intrinsics = camera.data.intrinsic_matrices[0].detach().cpu().tolist()
        annotate_layout(
            rgb,
            items,
            camera_position,
            camera_rotation,
            intrinsics,
            seed=self.args.seed,
            arm=self.layout.arm,
        ).save(self.output / "initial_layout.png")
        write_json(
            self.output / "initial_layout_view.json",
            {
                "seed": self.args.seed,
                "arm": self.layout.arm,
                "coordinate_frame": "world",
                "position_units": "m",
                "angle_units": "deg",
                "items": items,
                "arm_selection": self.arm_selection,
            },
        )

    def capture(self, label):
        index = len(self.manifest)
        directory = self.output / "observations" / f"{index:03d}_{label}"
        directory.mkdir(parents=True, exist_ok=False)
        snapshot = self.policy.snapshot()
        write_json(directory / "state.json", snapshot)
        self.policy.images(self.obs, str(directory))
        self.manifest.append({
            "index": index,
            "label": label,
            "step": len(self.actions),
            "simulation_time": snapshot["simulation_time"],
            "state": str((directory / "state.json").relative_to(self.output)),
            "images": [str(p.relative_to(self.output)) for p in sorted(directory.glob("*.png"))],
        })
        write_json(self.output / "manifest.json", self.manifest)
        return snapshot

    @property
    def maximum_steps(self):
        """Use one bounded trajectory length for execution and action replay."""
        return max(15000, 2500 * len(self.task.goals))

    def prepare_step(self, action=None):
        from isaaclab_arena.grasping.batch_planner import resolve_planning_steps

        return resolve_planning_steps(self.prepare_step_steps(action))

    def prepare_step_steps(self, action=None):
        """Yield planning requests before committing a single recorded action."""
        if not self.app.is_running() or time.monotonic() > self.deadline:
            raise TimeoutError("Simulation stopped or execution time budget exhausted")
        if len(self.actions) >= self.maximum_steps:
            raise TimeoutError("Simulation step budget exhausted")
        started = time.monotonic()
        if action is None:
            if self.motion_planner == "curobo" and self.args.mode == "execute":
                if self.motion is None:
                    self.ensure_curobo_motion()
                action = yield from self.motion.action_steps()
            else:
                action = self.default_action()
        self.timings["planning_and_control_s"] += time.monotonic() - started
        if tuple(action.shape) != (1, self.layout.action_dim) or not torch.isfinite(action).all():
            raise ValueError("Invalid action tensor")
        self.actions.append(action.detach().cpu().numpy().copy())
        return action

    def step(self, action=None):
        action = self.prepare_step(action)
        started = time.monotonic()
        with torch.no_grad():
            obs, _, terminated, truncated, _ = self.env.step(action)
        self.consume_step(action, obs, terminated, truncated, time.monotonic() - started)

    def consume_step(self, action, obs, terminated, truncated, physics_seconds):
        """Record and validate only this episode after a shared simulator step."""
        self.obs = obs
        self.timings["physics_step_s"] += physics_seconds
        robot = self.policy.robot
        joints = robot.data.joint_pos[self.env_id : self.env_id + 1]
        if not torch.isfinite(joints).all():
            raise RuntimeError("Non-finite robot joint state")
        self.joints.append(joints.detach().cpu().numpy().copy())
        if self.recorder is not None:
            started = time.monotonic()
            self.recorder.record_step(action)
            self.timings["recording_s"] += time.monotonic() - started
        limits = robot.data.joint_pos_limits[self.env_id : self.env_id + 1]
        violation = torch.maximum(limits[..., 0] - joints, joints - limits[..., 1]).clamp_min(0)
        self.max_joint_violation = max(self.max_joint_violation, float(violation.max()))
        tolerance = torch.full_like(violation, 0.02)
        tolerance[:, self.policy.finger_ids] = 0.001
        self.joint_limits_ok &= bool(torch.all(violation <= tolerance))
        if self.milestone_index < len(self.task.milestones):
            milestone = self.task.milestones[self.milestone_index]
            check = milestone.evaluate(
                self.policy.snapshot(), movable_object_ids={g.object_id for g in self.task.all_goals}
            )
            self.milestone_streak = self.milestone_streak + 1 if check["success"] else 0
            if self.milestone_streak * self.dt >= milestone.hold_seconds:
                self.milestone_records.append({
                    "name": milestone.name,
                    "step": len(self.actions),
                    "stable_seconds": self.milestone_streak * self.dt,
                    "check": check,
                })
                self.milestone_index += 1
                self.milestone_streak = 0
                write_json(self.output / "milestones.json", self.milestone_records)
        if terminated.any() or truncated.any():
            raise RuntimeError("Environment terminated; candidate invalidated")

    def event(self, action, result):
        self.events.append(
            {"action": action.name, "arguments": action.arguments, "result": result, "step": len(self.actions)}
        )
        active = getattr(self, "current_action", None) or {}
        if active.get("held_object_id") is not None:
            self.events[-1]["held_object_id"] = active["held_object_id"]
        write_json(self.output / "events.json", self.events)
        self.capture(f"{action.name}_{result['status']}")
        print(f"CODEGEN action={action.name} result={result['status']} steps={len(self.actions)}", flush=True)

    def drive(self, steps):
        """Drive resumable task/control steps in the single-environment runtime."""
        from isaaclab_arena.grasping.cartesian_motion import CartesianTargetRequest

        value = None
        try:
            while True:
                command = steps.send(value)
                if isinstance(command, CartesianTargetRequest):
                    value = command.solve_single()
                else:
                    self.step(command)
                    value = None
        except StopIteration as result:
            return result.value
        finally:
            steps.close()

    def execution_control_steps(self, program):
        """Resolve local-control yields before producing one physical action.

        Returns (action, None), or (None, report) when the task completes. Only
        actual physical actions go through prepare_step_steps and the recorder.
        """
        from isaaclab_arena.grasping.cartesian_motion import CartesianTargetRequest

        value = None
        while True:
            try:
                request = program.send(value)
            except StopIteration as completed:
                return None, completed.value
            if isinstance(request, CartesianTargetRequest):
                value = yield request
            else:
                action = yield from self.prepare_step_steps(request)
                return action, None

    def execute(self, source):
        return self.drive(self.execution_steps(source))

    def verify_hold(self):
        return self.drive(self.hold_steps())

    def execution_steps(self, source):
        program = Program(source, max_actions=max(24, 12 * len(self.task.goals)))
        request = program.advance(self.policy.snapshot())
        while request is not None:
            self.current_action = {"action": request.name, "arguments": copy.deepcopy(request.arguments)}
            held_object_id = getattr(self.policy.skills, "holding", None)
            if held_object_id is not None:
                self.current_action["held_object_id"] = held_object_id
            if self.layout.robot == "arx5" and request.name not in (
                "pick",
                "grasp_actor",
                "place",
                "place_actor",
                "home",
                "observe",
                "wait",
            ):
                raise ValueError(f"ARX5 legacy IK does not support {request.name}; use Aloha for advanced skills")
            if request.name == "insert_object":
                from code_gen.runtime.cartesian import insertion_steps

                result = yield from insertion_steps(self, request.arguments)
            elif request.name == "wait":
                for _ in range(math.ceil(request.arguments["seconds"] / self.dt)):
                    yield None
                result = {"success": True, "status": "success", "reason": "Physics wait completed"}
            else:
                self.policy.skills.start(request.name, request.arguments, self.policy.snapshot())
                while self.policy.skills.result is None:
                    phase = self.policy.skills.phase
                    yield None
                    if phase in ("descend", "close") and self.policy.skills.phase != phase:
                        self.capture(f"{request.name}_after_{phase}")
                result = dict(self.policy.skills.result)
                self.policy.skills.result = None
                result["success"] = result["status"] == "success"
            self.event(request, result)
            self.current_action = None
            # Failed skills end this candidate. Repair happens from a clean reset.
            if not result["success"]:
                return {"program_completed": False, "failed_action": self.events[-1], "physical_success": False}
            request = program.advance(self.policy.snapshot(), result)
        return {"program_completed": True, **(yield from self.hold_steps())}

    def hold_steps(self):
        checks = [self.task.evaluate(self.policy.snapshot())]
        for _ in range(math.ceil(self.task.hold_seconds / self.dt)):
            yield None
            checks.append(self.task.evaluate(self.policy.snapshot()))
        write_json(self.output / "hold_checks.json", checks)
        return {
            "physical_success": all(c["success"] for c in checks) and self.milestone_index == len(self.task.milestones),
            "milestones_completed": self.milestone_records,
            "milestones_required": len(self.task.milestones),
            "hold_samples": len(checks),
            "hold_seconds": (len(checks) - 1) * self.dt,
            "goal_verification": checks[-1],
            "hold_failed_samples": sum(not c["success"] for c in checks),
            "first_failed_hold": next(
                ({"sample_index": i, "check": check} for i, check in enumerate(checks) if not check["success"]), None
            ),
        }

    def replay(self, source_dir):
        source_dir = Path(source_dir)
        original = json.loads((source_dir / "simulation.json").read_text())
        if TaskSpec.from_dict(original["task"]) != self.task or original["seed"] != self.args.seed:
            raise ValueError("Replay task or seed differs from original run")
        if not math.isclose(original["step_dt"], self.dt, abs_tol=1e-10):
            raise ValueError("Replay timestep differs from original")
        if original.get("motion_planner", "ik") != self.motion_planner:
            raise ValueError("Replay action semantics differ from the original motion backend")
        if original.get("robot", "arx5") != self.layout.robot or original.get("arm", "left") != self.layout.arm:
            raise ValueError("Replay robot or active arm differs from original")
        previous = original["initial_snapshot"]
        for name in self.initial["objects"]:
            if math.dist(self.initial["objects"][name]["position"], previous["objects"][name]["position"]) > 1e-4:
                raise ValueError("Replay initial object positions differ")
        if max(abs(a - b) for a, b in zip(self.initial["joint_positions"], previous["joint_positions"])) > 1e-4:
            raise ValueError("Replay initial robot joints differ")
        with np.load(source_dir / "trajectory.npz", allow_pickle=False) as archive:
            actions = archive["actions"]
            original_joints = archive["joints"]
        if original_joints.shape != (len(actions), 1, len(self.initial["joint_positions"])):
            raise ValueError("Invalid saved joint trajectory")
        if (
            actions.ndim != 3
            or actions.shape[1:] != (1, self.layout.action_dim)
            or not 1 <= len(actions) <= self.maximum_steps
        ):
            raise ValueError("Invalid saved trajectory")
        checks = []
        hold_steps = math.ceil(self.task.hold_seconds / self.dt)
        for index, value in enumerate(actions):
            self.step(torch.as_tensor(value, device=self.env.unwrapped.device))
            if index >= len(actions) - hold_steps - 1:
                checks.append(self.task.evaluate(self.policy.snapshot()))
        write_json(self.output / "hold_checks.json", checks)
        return {
            "program_completed": True,
            "physical_success": (
                len(checks) == hold_steps + 1
                and all(c["success"] for c in checks)
                and self.milestone_index == len(self.task.milestones)
            ),
            "milestones_completed": self.milestone_records,
            "milestones_required": len(self.task.milestones),
            "goal_verification": checks[-1] if checks else None,
            "replanning": False,
            "replayed_steps": len(actions),
            "source_replay": {
                "directory": str(source_dir.resolve()),
                "trajectory_sha256": hashlib.sha256((source_dir / "trajectory.npz").read_bytes()).hexdigest(),
                "physical_success": bool(original.get("physical_success")),
                "max_joint_position_difference": float(np.max(np.abs(np.asarray(self.joints) - original_joints))),
            },
            "hold_samples": len(checks),
        }

    def finish(self, report, *, close=True):
        try:
            report["events"] = copy.deepcopy(self.events)
            active = getattr(self, "current_action", None)
            if active is not None:
                report["action_context"] = {**copy.deepcopy(active), "phase": self.policy.skills.phase}
            report["final_snapshot"] = self.capture("final")
            report["initial_snapshot"] = self.initial
            report["grasp_evidence"] = getattr(self.policy.skills, "grasp_evidence", [])
            report["alignment_evidence"] = getattr(self.policy.skills, "alignment_evidence", [])
            report["step_dt"] = self.dt
            report["motion_planner"] = self.motion_planner
            report["robot"] = self.layout.robot
            report["arm"] = self.layout.arm
            report["arm_selection"] = self.arm_selection
            report["action_space"] = (
                "joint_position[6], gripper_width_m"
                + (" per arm, left then right" if self.layout.robot == "aloha" else "")
                if self.motion_planner == "curobo"
                else "relative_flange_pose[6], gripper_width_m"
            )
            if self.motion is not None:
                report["planning"] = {
                    "calls": len(self.motion.records),
                    "grasp_selections": self.motion.grasp_selections,
                    "directory": str(self.motion.output),
                    "segments": [
                        {
                            **{
                                key: row.get(key)
                                for key in ("phase", "success", "status", "waypoints", "attached_object")
                            },
                            **{
                                key: row[key]
                                for key in ("rejected_measured_tcp", "tracking_tail", "source")
                                if key in row
                            },
                        }
                        for row in self.motion.records
                    ],
                }
                diagnostic = self.motion.output / "start_collision.json"
                if diagnostic.exists():
                    report["planning"]["start_collisions"] = [
                        {key: row[key] for key in ("obstacle", "links", "max_collision_cost_m")}
                        for row in json.loads(diagnostic.read_text())
                    ]

            report["steps"] = len(self.actions)
            report["timings"] = dict(self.timings)
            report["session_episode_index"] = self._episode_index
            report["joint_limits_ok"] = self.joint_limits_ok
            report["max_joint_limit_excess"] = self.max_joint_violation
            report["joint_limit_tolerances"] = {"arm_rad": 0.02, "finger_m": 0.001}
            physically_valid = bool(report.get("completed") and report.get("physical_success") and self.joint_limits_ok)
            if self.actions and physically_valid:
                np.savez_compressed(
                    self.output / "trajectory.npz", actions=np.asarray(self.actions), joints=np.asarray(self.joints)
                )
            if self.recorder is not None:
                from code_gen.recording.artifacts import finish_recording

                report["recording"] = finish_recording(self, self.recorder.writer, report)
            write_json(self.output / "simulation.json", report)
        finally:
            try:
                if self.motion is not None and self.motion.local is not None:
                    self.motion.local.close()
                    self.motion.local = None
            finally:
                if close:
                    self.env.close()
