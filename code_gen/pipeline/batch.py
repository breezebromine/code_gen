# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Generate, validate across seeds, freeze, then collect without model calls."""

import itertools
import json
from pathlib import Path

from code_gen._io import source_hash, write_json
from code_gen.llm.prompts import PROGRAM_CONTRACT, motion_planner_contract
from code_gen.pipeline.attempts import execute_attempts
from code_gen.pipeline.failure_evidence import cartesian_rejection, completed_task_rejection, release_rejection
from code_gen.pipeline.grasp_retry import is_planning_failure
from code_gen.recording.artifacts import case_location, publish_success_index, retain_case, success_entry
from code_gen.runtime.worker import SimulationWorker
from code_gen.task.config import development_seeds
from code_gen.task.program import ProgramError, extract_program
from isaaclab_arena.grasping.robot_layout import aloha_robot_path


class RenderYieldRequested(Exception):
    """Scheduler requested GPU release at a completed case boundary."""


def collection_seeds(start, max_cases=0):
    """Zero means keep sampling until the caller reaches its success quota."""
    if type(start) is not int or type(max_cases) is not int or max_cases < 0:
        raise ValueError("Expected integer start and nonnegative max_cases")
    return itertools.count(start) if max_cases == 0 else range(start, start + max_cases)


def passed(report):
    return report.get("physical_success") is True and report.get("validation_success") is True


def failure_kind(report):
    """Route model repair only on evidence of a defective task program."""
    status = report.get("status")
    if status == "layout_sampling_failed":
        return "runtime"
    if status in {"infrastructure_failed", "scene_failed"}:
        return "infrastructure"
    feedback = report.get("feedback") or {}
    if not isinstance(feedback, dict):
        return "unknown"
    simulation = feedback.get("simulation") or {}
    if not isinstance(simulation, dict):
        return "unknown"
    if feedback.get("error") or simulation.get("timed_out") or simulation.get("finalization_error"):
        return "unknown"
    error = simulation.get("error") or {}
    if not isinstance(error, dict):
        return "unknown"
    error_type, message = error.get("type"), error.get("message", "")
    if not isinstance(message, str):
        return "unknown"
    if is_planning_failure(error):
        return "runtime"
    if error_type == "ProgramError":
        return "program"
    # These are explicit action-contract violations, not arbitrary ValueErrors
    # from scene setup, physics, or planner configuration.
    if error_type == "ValueError" and message in {
        "vertical must be a boolean",
        "Displacement must be finite XYZ within 0.5 m",
        "frame must be world or tool",
        "Vertical motion must preserve XY and orientation",
        "Invalid gripper width",
    }:
        return "program"
    if error_type == "RuntimeError" and message in {
        "Grasp search budget exhausted; reachability is undetermined",
        "MGG grasp candidate search budget exhausted; reachability is undetermined",
        "Grasp search budget exhausted after solver retries; reachability is undetermined",
        "No remaining grasp candidates",
    }:
        return "runtime"
    # A guarded Cartesian segment can reject the measured motion even after
    # planning passes. Retry the layout, not the program or an unrelated grasp.
    context = simulation.get("action_context") or {}
    if not isinstance(context, dict):
        return "unknown"
    if cartesian_rejection(context, error) or release_rejection(context, error):
        return "runtime"
    if error:
        return "unknown"
    if (
        status in {"offline_candidate_failed", "budget_exhausted"}
        and report.get("physical_success") is False
        and report.get("validation_success") is False
        and completed_task_rejection(simulation)
    ):
        return "runtime"
    action = simulation.get("failed_action") or {}
    if not isinstance(action, dict):
        return "unknown"
    result = action.get("result") or {}
    if not isinstance(result, dict):
        return "unknown"
    if (
        status in {"offline_candidate_failed", "budget_exhausted"}
        and simulation.get("execution_ok") is True
        and result.get("success") is False
        and result.get("status") == "failed"
        and result.get("reason")
    ):
        return "runtime"
    return "unknown"


def layout_failure(report):
    """A rejected execution/layout can be retried without changing task code."""
    return failure_kind(report) == "runtime"


def validate_development(runner, directory, seeds, required, source, *, skip_layout_failures=False):
    reports = []
    for seed in seeds:
        print(f"VALIDATE {directory.name} seed={seed}", flush=True)
        report = cached_case(runner, directory / f"seed_{seed}", seed, source)
        reports.append(report)
        write_json(directory / "validation.json", reports)
        accepted = [r for r in reports if passed(r)]
        write_json(
            case_location(directory / "seed_0")[0] / "progress.json",
            {
                "phase": "developing",
                "successes": 0,
                "target_reached": False,
                "validation_successes": len(accepted),
                "validation_required": required,
                "validation_attempted_seeds": [r["seed"] for r in reports],
                "layout_failures": sum(layout_failure(r) for r in reports),
                "source_sha256": source_hash(source),
            },
        )
        if len(accepted) == required:
            break
        if not passed(report):
            if skip_layout_failures and layout_failure(report):
                print(
                    f"VALIDATION_LAYOUT_REJECTED seed={seed}; reuse program, sample next seed; API_calls=0", flush=True
                )
                continue
            break
    return reports


class BatchWorker(SimulationWorker):
    def __call__(self, mode, directory, **kwargs):
        self.cameras = mode != "inspect"
        return super().__call__(mode, directory, **kwargs)


class SeedRunner:
    def __init__(self, info):
        self.info = info
        self.visual_client = None

    def worker(self, task_file, seed):
        return BatchWorker(
            task_file,
            aloha_robot_path(self.info.assets),
            urdf=False,
            seed=seed,
            timeout=1800,
            execution_timeout=1200,
            cameras=True,
            device="cpu",
            motion_planner="curobo",
            robot=self.info.robot["name"],
            arm=self.info.robot["arm"],
            aloha_assets_root=str(self.info.assets),
        )

    def inspect(self, directory, seed):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        task = self.info.prepare(directory / "input", seed)
        worker = self.worker(directory / "input/task.json", seed)
        report = worker("inspect", directory / "inspection")
        if not report.get("execution_ok"):
            raise RuntimeError(f"Generation scene inspection failed: {directory}")
        task.validate_scene(report["initial_snapshot"])
        return task, report["initial_snapshot"]

    def prepare_task(self, directory, seed):
        return self.info.prepare(directory, seed)

    def __call__(self, directory, seed, source):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        from code_gen.task.assembly import LayoutSamplingError

        try:
            task = self.prepare_task(directory / "input", seed)
        except LayoutSamplingError as exc:
            report = {
                "status": "layout_sampling_failed",
                "physical_success": False,
                "validation_success": False,
                "error": str(exc),
                "attempts": [],
            }
            (directory / "execution").mkdir(parents=True, exist_ok=True)
            write_json(directory / "execution/summary.json", report)
            return report
        worker = self.worker(directory / "execution/task.json", seed)
        return execute_attempts(
            task,
            directory / "execution",
            worker,
            source=source,
            rounds=3,
            visual_client=self.visual_client,
        )


def generation_context(info, runner, root):
    root = root / "logs"
    root.mkdir(parents=True, exist_ok=True)
    path = root / "generation_context.json"
    if path.exists():
        return json.loads(path.read_text())
    # Preserve interrupted inspections; retry into a fresh directory.
    index = len(list(root.glob("inspection_*")))
    task, snapshot = runner.inspect(root / f"inspection_{index:03d}", info.development["generation_seed"])
    context = {
        "task": task.to_dict(),
        "scene": snapshot,
        "robot": info.robot["name"],
        "active_arm": info.robot["arm"],
        "motion_planner": "curobo",
        "max_actions": max(24, 12 * len(task.goals)),
        "randomization": info.randomization,
    }
    write_json(path, context)
    return context


def cached_case(runner, directory, seed, source):
    """A receipt is written only after execution finishes. Partial runs are preserved."""
    directory = Path(directory)
    digest = source_hash(source)
    dataset_root, cache_key = case_location(directory)
    if (dataset_root / ".yield_for_render").exists():
        print("RENDER_YIELD: case boundary reached; release simulator for queued rendering", flush=True)
        raise RenderYieldRequested()
    cache = dataset_root / "logs/cases.json"
    if cache.exists():
        saved = json.loads(cache.read_text()).get(cache_key)
        if saved is not None:
            if saved["source_sha256"] != digest or saved["seed"] != seed:
                raise ValueError("Case receipt belongs to a different program or seed")
            if passed(saved) and not Path(saved["data"]["recording"]).is_dir():
                raise ValueError("Cached successful recording is missing")
            return saved
    failures = directory.parent / "failed_cases.json"
    if failures.exists():
        saved = json.loads(failures.read_text()).get(directory.name)
        if saved is not None:
            if saved["source_sha256"] != digest or saved["seed"] != seed:
                raise ValueError("Failed case belongs to a different program or seed")
            return saved
    directory.mkdir(parents=True, exist_ok=True)
    receipt = directory / "result.json"
    if receipt.exists():
        saved = json.loads(receipt.read_text())
        if saved["source_sha256"] != digest or saved["seed"] != seed:
            raise ValueError("Case receipt belongs to a different program or seed")
        return retain_case(directory, saved)
    # Recover a completed run after interruption between execution and receipt.
    runs = sorted(directory.glob("run_*"))
    report, run = None, None
    if runs:
        last = runs[-1]
        summary = last / "execution/summary.json"
        if summary.exists():
            candidate = last / "execution/attempt_00/candidate.py"
            saved = json.loads(summary.read_text())
            terminal = {
                "validated",
                "physical_success_audit_incomplete",
                "offline_candidate_failed",
                "scene_failed",
                "infrastructure_failed",
                "budget_exhausted",
                "layout_sampling_failed",
            }
            if candidate.exists() and source_hash(candidate.read_text()) != digest:
                raise ValueError("Recovered run used a different program")
            if saved.get("status") in terminal and (candidate.exists() or not passed(saved)):
                report, run = saved, last
    if report is None:
        run = directory / f"run_{len(runs):03d}"
        report = runner(run, seed, source)
    attempts = report.get("attempts", [])
    attempt = attempts[-1] if attempts else {}
    simulation = attempt.get("simulation", {})
    saved = {
        "seed": seed,
        "data": success_entry(run / "execution", report),
        "successful_attempt": report.get("successful_attempt"),
        "validation_mode": report.get("validation_mode", "execution_only"),
        "source_sha256": digest,
        "directory": str(run.resolve()),
        "status": report.get("status"),
        "physical_success": report.get("physical_success", False),
        "validation_success": report.get("validation_success", False),
        "feedback": {
            "error": report.get("error") or attempt.get("error"),
            "simulation": {
                k: v
                for k, v in simulation.items()
                if k not in ("framework_sha256", "initial_snapshot", "final_snapshot")
            },
            "observation": attempt.get("observation"),
            "replay": {
                k: v
                for k, v in attempt.get("replay", {}).items()
                if k in ("error", "execution_ok", "physical_success", "joint_limits_ok")
            },
        },
    }
    write_json(receipt, saved)
    return retain_case(directory, saved)


def save_collection_progress(root, records, successes, collection, digest, *, phase, **details):
    """One success-only publication/progress contract for both collection entrypoints."""
    publish_success_index(root, successes)
    write_json(root / "cases.json", records)
    write_json(root / "successful_cases.json", successes)
    reached = len(successes) >= collection["target_successes"]
    progress = {
        "phase": phase,
        "successes": len(successes),
        "target": collection["target_successes"],
        "attempted_seeds": len(records),
        "max_cases": collection["max_cases"],
        "target_reached": reached,
        "source_sha256": digest,
        **details,
    }
    write_json(root / "progress.json", progress)
    return reached


def run_batch(info, root, client_factory, runner=None, phase="all", visual_feedback=False, visual_client_factory=None):
    """Client is created lazily; resuming a frozen program requires no credentials."""
    root = Path(root)
    log_root = root / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    for name in ("development", "collection", "generation_context.json"):
        legacy, target = root / name, log_root / name
        if legacy.exists():
            if target.exists():
                raise ValueError(f"Both legacy and current log locations exist: {name}")
            legacy.rename(target)
    runner = runner or SeedRunner(info)
    ready_file, program_file = root / "program.json", root / "task.py"
    development = info.development
    seeds = [development["generation_seed"], *development["validation_seeds"]]
    seed_pool = development_seeds(development)
    retry_layouts = len(seed_pool) > len(seeds)
    source = None
    if ready_file.exists():
        ready = json.loads(ready_file.read_text())
        source = program_file.read_text()
        actual_seeds = ready["validated_seeds"]
        if ready["source_sha256"] != source_hash(source):
            raise ValueError("Frozen program was modified")
        if len(actual_seeds) != len(seeds) or len(set(actual_seeds)) != len(seeds):
            raise ValueError("Frozen program lacks distinct successful validation seeds")
        if not retry_layouts and actual_seeds != seeds:
            raise ValueError("Frozen program or seed validation manifest was modified")
        if retry_layouts:
            attempts = ready.get("validation_attempts", [])
            attempt_seeds = [r["seed"] for r in attempts]
            if (
                ready.get("requested_seeds") != seeds
                or attempt_seeds != seed_pool[: len(attempts)]
                or [r["seed"] for r in attempts if passed(r)] != actual_seeds
                or any(r.get("source_sha256") != ready["source_sha256"] for r in attempts)
                or any(not passed(r) and not layout_failure(r) for r in attempts)
            ):
                raise ValueError("Invalid development layout retry evidence")
        if not all(passed(r) and r["source_sha256"] == ready["source_sha256"] for r in ready["validation"]):
            raise ValueError("Frozen program lacks successful validation")
        if [r["seed"] for r in ready["validation"]] != actual_seeds:
            raise ValueError("Frozen program lacks required validation seeds")
    else:
        if phase == "collect":
            raise ValueError("No validated task.py; run --phase all or develop first")
        context = generation_context(info, runner, root)
        client = None
        visual_client = None
        previous_source, feedback = None, None
        for version in range(development["max_versions"]):
            directory = root / "logs" / "development" / f"version_{version:02d}"
            directory.mkdir(parents=True, exist_ok=True)
            candidate = directory / "candidate.py"
            error_file = directory / "generation_error.json"
            if error_file.exists():
                feedback = json.loads(error_file.read_text())
                continue
            if not candidate.exists():
                prompt = {
                    **context,
                    "previous_source": previous_source,
                    "latest_feedback": feedback,
                    "requirement": (
                        "One reusable program for all randomized layouts. Use current object/target "
                        "poses and scene metadata. Never hard-code sampled world positions, change "
                        "goals, or disable collisions. Every validation seed must pass with identical code."
                    ),
                }
                if client is None:
                    client = client_factory()
                api_dir = directory / f"generation_{len(list(directory.glob('generation_*'))):03d}"
                try:
                    response = client.complete(
                        [
                            {"role": "system", "content": PROGRAM_CONTRACT + motion_planner_contract("curobo")},
                            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
                        ],
                        api_dir,
                    )
                    candidate.write_text(extract_program(response))
                except ProgramError as exc:
                    write_json(error_file, {"error": f"{type(exc).__name__}: {exc}"})
                    feedback = json.loads(error_file.read_text())
                    continue
            source = candidate.read_text()
            if previous_source is not None and source_hash(source) == source_hash(previous_source):
                feedback = {
                    "error": "Unchanged failed candidate; change the task code using the previous diagnostics.",
                    "previous_feedback": feedback,
                }
                write_json(error_file, feedback)
                continue
            previous_source = source
            if visual_feedback and visual_client is None:
                if visual_client_factory is not None:
                    visual_client = visual_client_factory()
                else:
                    client = client or client_factory()
                    visual_client = client
            runner.visual_client = visual_client if visual_feedback else None
            reports = validate_development(
                runner, directory, seed_pool, len(seeds), source, skip_layout_failures=retry_layouts
            )
            accepted = [r for r in reports if passed(r)]
            if len(accepted) == len(seeds):
                program_file.write_text(source)
                ready = {
                    "source_sha256": source_hash(source),
                    "version": version,
                    "validated_seeds": [r["seed"] for r in accepted],
                    "validation": accepted,
                    "requested_seeds": seeds,
                    "validation_attempts": reports,
                }
                write_json(ready_file, ready)
                print("PROGRAM_FROZEN " + ready["source_sha256"], flush=True)
                break
            if (
                retry_layouts
                and len(reports) == len(seed_pool)
                and all(passed(r) or layout_failure(r) for r in reports)
            ):
                write_json(
                    root / "progress.json",
                    {
                        "phase": "development_layout_budget_exhausted",
                        "successes": 0,
                        "target_reached": False,
                        "validation_successes": len(accepted),
                        "validation_required": len(seeds),
                        "validation_attempted_seeds": [r["seed"] for r in reports],
                        "reason": "Runtime layout budget exhausted; reuse task code and inspect execution diagnostics",
                    },
                )
                return 4
            failed = next((r for r in reversed(reports) if not passed(r)), None)
            kind = failure_kind(failed) if failed else "unknown"
            if kind != "program":
                write_json(
                    root / "progress.json",
                    {
                        "phase": (
                            "development_runtime_budget_exhausted"
                            if kind == "runtime"
                            else "development_needs_attention"
                        ),
                        "failure_kind": kind,
                        "successes": 0,
                        "target_reached": False,
                        "validation_successes": len(accepted),
                        "validation_required": len(seeds),
                        "validation_attempted_seeds": [r["seed"] for r in reports],
                        "source_sha256": source_hash(source),
                    },
                )
                return 4 if kind == "runtime" else 3
            feedback = {"validation_results": reports}
        else:
            write_json(root / "progress.json", {"phase": "development_failed", "successes": 0, "target_reached": False})
            return 3
    runner.visual_client = None
    if phase == "develop":
        write_json(root / "progress.json", {"phase": "program_ready", "successes": 0, "target_reached": False})
        return 0
    collection = info.collection
    records = []
    for seed in collection_seeds(collection["start_seed"], collection["max_cases"]):
        print(f"COLLECT seed={seed} API_calls=0", flush=True)
        report = cached_case(runner, root / "logs" / "collection" / f"seed_{seed}", seed, source)
        records.append(report)
        successes = [r for r in records if passed(r)]
        reached = len(successes) >= collection["target_successes"]
        save_collection_progress(
            root,
            records,
            successes,
            collection,
            source_hash(source),
            phase="complete" if reached else "collecting",
        )
        print(f"PROGRESS {len(successes)}/{collection['target_successes']}", flush=True)
        if reached:
            return 0
        if not passed(report) and failure_kind(report) != "runtime":
            progress = json.loads((root / "progress.json").read_text())
            progress.update(phase="collection_needs_attention", failure_kind=failure_kind(report))
            write_json(root / "progress.json", progress)
            return 3
    progress = json.loads((root / "progress.json").read_text())
    progress["phase"] = "collection_budget_exhausted"
    write_json(root / "progress.json", progress)
    return 2


class ProductionSeedRunner(SeedRunner):
    def __init__(self, info, session):
        super().__init__(info)
        self.session = session

    def worker(self, task_file, seed):
        return self.session.worker(task_file, seed)
