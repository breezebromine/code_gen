# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared episode input, execution and report contract for both worker lifecycles."""

import hashlib
import json
import time
import traceback
from pathlib import Path

from code_gen._io import framework_hashes, write_json
from code_gen.task.program import validate_program
from code_gen.task.spec import TaskSpec


def validate_runtime(robot, motion_planner, *, state_only=False):
    """Reject unsupported adapters before creating a simulation environment."""
    if robot == "arx5" and motion_planner != "ik":
        raise ValueError("ARX5 codegen supports the legacy IK adapter only; use --motion-planner ik or Aloha/cuRobo")
    if robot == "arx5" and state_only:
        raise ValueError("ARX5 state-only recording is unsupported; use online RGB recording or Aloha")


def load_episode(task_file, directory, candidate=None):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    task = TaskSpec.from_dict(json.loads(Path(task_file).read_text()))
    source = Path(candidate).read_text() if candidate else None
    if source is not None:
        validate_program(source)
        (directory / "candidate.py").write_text(source)
    return directory, task, source


def episode_report(args, task, source, **details):
    return {
        "task": task.to_dict(),
        "seed": args.seed,
        "mode": args.mode,
        "device": str(args.device),
        "robot": args.robot,
        "arm": args.arm,
        "motion_planner": args.motion_planner,
        "completed": False,
        "physical_success": False,
        "joint_limits_ok": False,
        "simulation_started": False,
        "framework_sha256": framework_hashes(),
        "source_sha256": hashlib.sha256(source.encode()).hexdigest() if source else None,
        **details,
    }


def run_episode(prepare_backend, report, directory, *, source=None, replay_dir=None, close=True):
    """Execute one prepared/reset episode; the caller owns SimulationApp lifetime."""
    write_json(directory / "simulation.json", report)
    started = time.monotonic()
    backend = None
    try:
        backend = prepare_backend()
        report["simulation_started"] = True
        report["initialization_seconds"] = time.monotonic() - started
        if report["mode"] == "execute":
            report.update(backend.execute(source))
        elif report["mode"] == "replay":
            report.update(backend.replay(replay_dir))
        elif report["mode"] != "inspect":
            raise ValueError("Unknown worker mode")
        report["completed"] = True
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        if backend is not None:
            save_start = time.monotonic()
            try:
                backend.finish(report, close=close)
            except Exception as exc:
                report["completed"] = False
                report["finalization_error"] = traceback.format_exc()
                report.setdefault("error", {"type": type(exc).__name__, "message": str(exc)})
            report["save_seconds"] = time.monotonic() - save_start
        write_json(directory / "simulation.json", report)
    return report
