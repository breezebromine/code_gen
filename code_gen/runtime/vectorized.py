# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Cooperative generated-task execution on one Isaac Lab vectorized environment.

Each slot owns its program, controller, collision state and recorder. Only this
scheduler calls env.step(); ready episodes are reset without restarting Kit.
"""

import copy
import json
import math
import random
import time
import torch
import traceback
from functools import partial
from pathlib import Path

from code_gen._io import source_hash, write_json
from code_gen.recording.artifacts import retain_case
from code_gen.runtime.backend import TabletopBackend
from code_gen.runtime.episode import episode_report
from code_gen.runtime.session import PersistentWorker, ProductionSession
from isaaclab_arena.grasping.batch_planner import BatchPosePlanner
from isaaclab_arena.grasping.robot_layout import RobotLayout


def slice_observation(value, env_id):
    """Retain the batch dimension required by the shared MGG recorder."""
    if isinstance(value, dict):
        return {key: slice_observation(item, env_id) for key, item in value.items()}
    return value[env_id : env_id + 1] if isinstance(value, torch.Tensor) else value


def measured_actions(env):
    robot = env.scene["robot"]
    actions = torch.zeros((env.num_envs, 14), device=env.device)
    for arm in ("left", "right"):
        layout = RobotLayout("aloha", arm)
        joints = robot.find_joints(layout.arm_names, preserve_order=True)[0]
        fingers = robot.find_joints(layout.finger_names, preserve_order=True)[0]
        signs = torch.tensor(layout.finger_signs, device=env.device)
        start = layout.action_start
        actions[:, start : start + 6] = robot.data.joint_pos[:, joints]
        actions[:, start + 6] = (robot.data.joint_pos[:, fingers] * signs).sum(-1).clamp(0, 0.08)
    return actions


def rotation_audit(recorder, limit=540.0):
    """Apply the existing cumulative shortest-arc EE rotation criterion."""
    samples = recorder.save_trajectory["samples"]
    angles = {}
    for arm in samples[0]["arms"] if samples else ():
        total = 0.0
        for before, after in zip(samples, samples[1:]):
            a, b = before["arms"][arm]["ee_quat"], after["arms"][arm]["ee_quat"]
            dot = abs(sum(x * y for x, y in zip(a, b)))
            norm = math.sqrt(sum(x * x for x in a) * sum(x * x for x in b))
            if not math.isfinite(norm) or norm <= 0:
                return {"passed": False, "reason": "Invalid recorded EE quaternion"}
            total += math.degrees(2 * math.acos(min(1.0, dot / norm)))
        angles[arm] = total
    return {
        "passed": bool(samples) and all(v <= limit for v in angles.values()),
        "limit_deg": limit,
        "per_arm_deg": angles,
    }


def prepare_controls(planner, controllers, actions, pending, failed, completed, hold):
    """Resolve all planning yields before advancing the shared physics clock."""
    prepared, errors = planner.solve_steps(controllers)
    for env_id, (action, report) in prepared.items():
        if report is not None:
            completed(env_id, report)
        else:
            pending[env_id] = action
            actions[env_id] = action[0]
    for env_id, error in errors.items():
        failed(env_id, error)
    for env_id in planner.waiting_envs:
        try:
            action = hold(env_id)
            pending[env_id] = action
            actions[env_id] = action[0]
        except Exception as error:
            planner.cancel(env_id)
            failed(env_id, error)


def episode_backend(slot, env_id, *, env, args, app, obs, source, cached_backends):
    """Create one episode's controller while retaining its warmed native planner."""
    if "backend" not in slot:
        episode_args = copy.copy(args)
        episode_args.output_dir, episode_args.seed, episode_args.mode = (
            str(slot["output"]),
            slot["seed"],
            "execute",
        )
        episode_args.execution_timeout = 1200
        episode_args.arm = slot["worker"].arm
        episode_args.requested_arm = args.arm
        backend = cached_backends.pop(env_id, None)
        if backend is None:
            backend = TabletopBackend(
                episode_args,
                app,
                slot["task"],
                env=env,
                env_id=env_id,
                prepared_observation=slice_observation(obs, env_id),
            )
        else:
            backend.reset_episode(episode_args, slot["task"], prepared_observation=slice_observation(obs, env_id))
        slot["backend"] = backend
        slot["steps"] = backend.execution_steps(source)
        slot["report"] = episode_report(
            episode_args,
            slot["task"],
            source,
            execution_backend="vectorized",
            env_id=env_id,
            num_envs=env.num_envs,
        )
    return slot["backend"]


def reset_environment_slot(env, slot, env_id, actions):
    ids = torch.tensor([env_id], device=env.device)
    env.reset(env_ids=ids)
    for name, data in slot["scene"]["objects"].items():
        body = env.scene[name]
        pose = body.data.default_root_state[env_id : env_id + 1].clone()
        pose[:, :3] = torch.tensor(data["position"], device=env.device) + env.scene.env_origins[env_id]
        pose[:, 3:7] = torch.tensor(data.get("quaternion", [1, 0, 0, 0]), device=env.device)
        pose[:, 7:] = 0
        body.write_root_state_to_sim(pose, env_ids=ids)
    env.scene.write_data_to_sim()
    env.sim.forward()
    env.scene.update(dt=0.0)
    if hasattr(env, "scene_state_reader"):
        env.scene_state_reader.invalidate()
    actions[env_id] = measured_actions(env)[env_id]


def account_execution_service(slots, participants, seconds):
    for env_id, slot in slots.items():
        backend = slot.get("backend")
        if backend is None:
            continue
        if env_id in participants:
            key = "scheduler_active_execution_s"
        else:
            # Other environments and scheduler bookkeeping must not spend
            # this episode's execution budget. Its own timeout is unchanged.
            backend.deadline += seconds
            key = "scheduler_wait_credit_s"
        backend.timings[key] = backend.timings.get(key, 0.0) + seconds


def collect_vectorized(info, root, source, *, num_envs, device):
    """Collect fresh independent layouts; publish only audited physical successes."""
    root = Path(root)
    target = info.collection["target_successes"]
    maximum = info.collection["max_cases"]
    state = {
        "num_envs": num_envs,
        "status": "running",
        "started_epoch": time.time(),
        "attempts": 0,
        "successes": 0,
        "failures": 0,
        "physics_steps": 0,
        "physics_seconds": 0.0,
        "control_seconds": 0.0,
        "observation_and_recording_seconds": 0.0,
        "results": [],
    }
    started = time.monotonic()
    slots = {}
    randomizer = random.Random()
    shared_visuals = None
    cached_backends = {}
    pending = {}
    env = None

    batch_planner = BatchPosePlanner(
        coalesce_steps=4 if num_envs > 1 else 0, account_service=partial(account_execution_service, slots)
    )
    state["execution_budget"] = "per_env_active_service_with_shared_physics_and_planning"
    state["batch_planning"] = batch_planner.stats
    (root / "candidate.py").write_text(source)

    def progress():
        state["elapsed_seconds"] = time.monotonic() - started
        state["updated_epoch"] = time.time()
        state["active_envs"] = {
            str(env_id): {
                "episode": slot["index"],
                "seed": slot["seed"],
                "settling": slot["settling"],
                "arm": slot["backend"].layout.arm if "backend" in slot else None,
                "phase": slot["backend"].policy.skills.phase if "backend" in slot else "settling",
                "steps": len(slot["backend"].actions) if "backend" in slot else 0,
            }
            for env_id, slot in slots.items()
        }
        write_json(root / "progress.json", state)

    def reserve(env_id):
        nonlocal shared_visuals
        if state["successes"] + len(slots) >= target or (maximum and state["attempts"] >= maximum):
            return False
        index = state["attempts"]
        state["attempts"] += 1
        seed = randomizer.randrange(2**31)
        inputs = root / "logs" / "inputs" / f"episode_{index:06d}"
        task = info.prepare(inputs, seed)
        scene_path = Path(task.scene_file)
        scene = json.loads(scene_path.read_text())
        # Dome/Distant lights are process-global. Keep a recorded shared visual
        # realization; per-env layout randomization remains independent.
        if shared_visuals is None:
            shared_visuals = copy.deepcopy(scene.get("visuals", {}))
            state["shared_visuals"] = shared_visuals
        scene["visuals"] = copy.deepcopy(shared_visuals)
        write_json(scene_path, scene)
        scene = prepare_scene(scene)
        directory = root / "logs" / "collection" / f"episode_{index:06d}"
        output = directory / "simulation"
        output.mkdir(parents=True)
        slots[env_id] = {
            "index": index,
            "seed": seed,
            "task": task,
            "scene": scene,
            "directory": directory,
            "output": output,
            "settling": 150,
            "started": time.monotonic(),
            "worker": PersistentWorker(session, inputs / "task.json", seed),
        }
        progress()
        return True

    def reset_slot(env_id):
        reset_environment_slot(env, slots[env_id], env_id, actions)

    def finish_slot_impl(env_id, report):
        batch_planner.cancel(env_id)
        slot = slots[env_id]
        backend = slot.get("backend")
        report["elapsed_seconds"] = time.monotonic() - slot["started"]
        if backend is not None:
            audit = rotation_audit(backend.recorder.writer) if backend.recorder else {"passed": False}
            report["rotation_audit"] = audit
            report["validation_success"] = bool(
                report.get("physical_success") and audit["passed"] and backend.joint_limits_ok
            )
            backend.finish(report, close=False)
            cached_backends[env_id] = backend
            write_json(root / "logs" / f"episode_{slot['index']:06d}.{backend.layout.arm}.json", report)
            if slot["worker"].advance_arm(report):
                pending.pop(env_id, None)
                slot.pop("backend")
                slot.pop("steps", None)
                slot["output"] = slot["directory"] / ("simulation_" + slot["worker"].arm)
                slot["output"].mkdir()
                slot["settling"] = 150
                state["arm_retries"] = state.get("arm_retries", 0) + 1
                print(f"MULTIENV env={env_id} retry_arm={slot['worker'].arm} same_seed={slot['seed']}", flush=True)
                reset_slot(env_id)
                progress()
                return

        else:
            report["validation_success"] = False
        report.update(
            seed=slot["seed"],
            source_sha256=source_hash(source),
            successful_attempt=0,
            feedback={"simulation": copy.deepcopy(report)},
        )
        recording = slot["output"] / "demo_0000"
        report["data"] = {"recording": str(recording)} if report["validation_success"] else None
        accepted = bool(report.get("physical_success") and report["validation_success"])
        compact = retain_case(slot["directory"], report)
        state["successes" if accepted else "failures"] += 1
        state["results"].append({
            "env_id": env_id,
            "episode": slot["index"],
            "seed": slot["seed"],
            "success": accepted,
            "error": report.get("error"),
            "rotation_audit": report.get("rotation_audit"),
            "data": compact.get("data"),
            "elapsed_seconds": report["elapsed_seconds"],
        })
        print(
            f"MULTIENV env={env_id} episode={slot['index']} success={accepted} total={state['successes']}/{target}",
            flush=True,
        )
        pending.pop(env_id, None)
        slots.pop(env_id)
        progress()
        if reserve(env_id):
            reset_slot(env_id)

    def finish_slot(env_id, report):
        with batch_planner.service(()):
            return finish_slot_impl(env_id, report)

    def hold_action(env_id):
        with batch_planner.service((env_id,)):
            return slots[env_id]["backend"].prepare_step(actions[env_id : env_id + 1].clone())

    def failed_control(env_id, exc):
        slot = slots[env_id]
        slot["steps"].close()
        finish_slot(
            env_id,
            {
                **slot["report"],
                "completed": False,
                "physical_success": False,
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": "".join(traceback.format_exception(exc)),
                },
            },
        )

    try:
        session = ProductionSession(info, recording="state_only", device=device)
        session.args.capture_initial_layout = False
        with session:
            from isaaclab_arena.environments.tabletop_environment import make_aloha_env, prepare_scene

            args = session.args
            args.num_envs, args.capture_initial_layout = num_envs, False
            args.env_spacing = 4.0
            for env_id in range(min(num_envs, target)):
                reserve(env_id)
            env = make_aloha_env(args, scene_spec=next(iter(slots.values()))["scene"]).unwrapped
            env.cfg.sim.render_interval = 10**9
            from isaaclab_arena.grasping.scene_state import BatchedSceneState

            env.scene_state_reader = BatchedSceneState(env, next(iter(slots.values()))["scene"]["objects"])
            state["state_readbacks"] = env.scene_state_reader.stats
            env.reset()
            actions = measured_actions(env)
            for env_id in slots:
                reset_slot(env_id)
            obs = env.observation_manager.compute()
            while slots:
                with batch_planner.execution_tick():
                    control_started = time.monotonic()
                    controllers = {}
                    for env_id, slot in list(slots.items()):
                        if slot["settling"] or env_id in batch_planner.waiting_envs:
                            continue
                        try:
                            with batch_planner.service((env_id,)):
                                backend = episode_backend(
                                    slot,
                                    env_id,
                                    env=env,
                                    args=args,
                                    app=session.app,
                                    obs=obs,
                                    source=source,
                                    cached_backends=cached_backends,
                                )
                                controllers[env_id] = backend.execution_control_steps(slot["steps"])
                        except StopIteration as completed:
                            report = {**slot["report"], **completed.value, "completed": True}
                            finish_slot(env_id, report)
                        except Exception as exc:
                            if "steps" in slot:
                                slot["steps"].close()
                            report = {
                                **slot.get("report", {}),
                                "mode": "execute",
                                "completed": False,
                                "physical_success": False,
                                "error": {
                                    "type": type(exc).__name__,
                                    "message": str(exc),
                                    "traceback": traceback.format_exc(),
                                },
                            }
                            finish_slot(env_id, report)
                    prepare_controls(
                        batch_planner,
                        controllers,
                        actions,
                        pending,
                        failed_control,
                        lambda env_id, report: finish_slot(
                            env_id, {**slots[env_id]["report"], **report, "completed": True}
                        ),
                        hold_action,
                    )
                    state["control_seconds"] += time.monotonic() - control_started
                    if not slots:
                        break
                    before = time.monotonic()
                    with torch.no_grad():
                        obs, _, terminated, truncated, _ = env.step(actions)
                    elapsed = time.monotonic() - before
                    batch_planner.account(tuple(slots), elapsed)
                    env.scene_state_reader.invalidate()
                    state["physics_steps"] += 1
                    state["physics_seconds"] += elapsed
                    observation_started = time.monotonic()
                    for env_id, slot in list(slots.items()):
                        if slot["settling"]:
                            slot["settling"] -= 1
                            continue
                        backend = slot.get("backend")
                        if backend is None or env_id not in pending:
                            continue
                        try:
                            with batch_planner.service((env_id,)):
                                backend.consume_step(
                                    pending[env_id],
                                    slice_observation(obs, env_id),
                                    terminated[env_id : env_id + 1],
                                    truncated[env_id : env_id + 1],
                                    elapsed,
                                )
                        except Exception as exc:
                            slot["steps"].close()
                            finish_slot(
                                env_id,
                                {
                                    **slot["report"],
                                    "completed": False,
                                    "physical_success": False,
                                    "error": {
                                        "type": type(exc).__name__,
                                        "message": str(exc),
                                        "traceback": traceback.format_exc(),
                                    },
                                },
                            )
                    state["observation_and_recording_seconds"] += time.monotonic() - observation_started
                    if state["physics_steps"] % 100 == 0:
                        progress()
            state["status"] = "complete" if state["successes"] >= target else "budget_exhausted"
            progress()
            batch_planner.close()
            env.close()
            env = None
    finally:
        batch_planner.close()
        progress()
        if env is not None:
            env.close()
    return 0 if state["successes"] >= target else 1
