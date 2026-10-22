# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Publish validated executions only; keep failed-attempt reports for diagnosis."""

import copy
import json
import os
import shutil
from pathlib import Path

from code_gen._io import write_json

POLICY = "validated_execution_only_v1"
TERMINAL = {
    "validated",
    "physical_success_audit_incomplete",
    "offline_candidate_failed",
    "scene_failed",
    "infrastructure_failed",
    "budget_exhausted",
    "interrupted",
}


def _recording_metadata(backend):
    """Describe task execution separately from the shared MGG data schema."""
    if backend.layout.robot == "aloha":
        return {
            "active_arm": backend.layout.arm,
            "control_action": {
                "fields": ["left_joints_rad[6]", "left_width_m", "right_joints_rad[6]", "right_width_m"],
                "motion_planner": "ActionPlanner/CuroboPlanner/MotionGen",
            },
            "coordinate_frames": {
                "left": "fl_base_link",
                "right": "fr_base_link",
                "ee": "link6 origin, X=approach Y=jaw-axis; xyz mm, quaternion xyzw",
                "tcp_offset_link6_m": [0.0, 0.0, 0.0],
                "planning_tcp_offset_link6_m": list(backend.layout.tcp_offset),
                "tcp_rotation_link6_wxyz": list(backend.layout.tcp_rotation),
            },
            "recording_mode": "online_rgb" if backend.args.enable_cameras else "state_only",
        }
    control_action = {
        "fields": ["relative_flange_position[3]", "relative_flange_axis_angle[3]", "gripper_width_m"],
        "arm_action_scale": 0.5,
    }
    if backend.motion_planner == "curobo":
        control_action = {
            "fields": ["joint_position_rad[6]", "gripper_width_m"],
            "joint_names": [f"joint{i}" for i in range(1, 7)],
            "motion_planner": "ActionPlanner/CuroboPlanner/MotionGen",
        }
    return {
        "coordinate_frames": {
            "left": "ARX5 base_link; single arm uses the existing left-arm schema",
            "ee": "link6 + [0.14, 0, 0] m TCP; XYZ in mm, quaternion xyzw",
            "objects": "world and base_link; XYZ in m, quaternion wxyz",
            "camera": "ROS optical coordinates; extrinsics map the named reference frame into camera coordinates",
        },
        "control_action": control_action,
    }


def finish_recording(backend, recorder, report):
    """Publish successful MGG recordings and task-specific diagnostic receipts."""
    if not (report.get("completed") and report.get("physical_success") and report.get("joint_limits_ok")):
        return {"saved": False, "reason": "Physical execution did not pass validation"}
    if report.get("validation_success") is False:
        return {"saved": False, "reason": "Episode did not pass dataset validation"}
    if recorder.count < 2:
        return {"saved": False, "reason": "Fewer than two recorded control steps"}
    visuals = (backend.scene_spec or {}).get("visuals", {})
    metadata = {
        **_recording_metadata(backend),
        "recording": {
            "mode": report["mode"],
            "physical_success": bool(report.get("physical_success")),
            "execution_completed": bool(report.get("completed")),
            "source_replay": report.get("source_replay"),
            "step_dt": backend.dt,
            "sample_phase": "after env.step(action)",
        },
        "scene_visuals": copy.deepcopy(visuals),
        "milestones_completed": copy.deepcopy(report.get("milestones_completed", [])),
    }
    for field in ("support_contacts", "contact_surfaces", "attachment_collision", "diagnostic_contact_departure"):
        if field in (backend.scene_spec or {}):
            metadata[field] = copy.deepcopy(backend.scene_spec[field])
    if metadata.get("diagnostic_contact_departure"):
        metadata["diagnostic_only"] = True
    # Task diagnostics belong to the run logs, not the shared MGG schema.
    write_json(backend.output / "recording.json", metadata)
    recorder.save_episode(
        environment_name="codegen_" + backend.task.name,
        hdr_file=visuals.get("hdr_file"),
        generation_hdr_file=visuals.get("hdr_file"),
    )
    manifest = recorder.save_root.parent / "path_tracing_jobs.json"
    write_json(
        manifest,
        {
            "version": 1,
            "demos": [str(recorder.save_root.resolve())],
            "purpose": "Physically successful execution; published validation receipts live in logs/",
        },
    )
    return {"saved": True, "directory": str(recorder.save_root), "frames": recorder.count, "manifest": str(manifest)}


def execution_evidence(simulation):
    """Keep acceptance and guard evidence even when failed trajectories are discarded."""
    return {
        key: simulation[key]
        for key in (
            "error",
            "finalization_error",
            "execution_ok",
            "timed_out",
            "failed_action",
            "action_context",
            "program_completed",
            "physical_success",
            "joint_limits_ok",
            "goal_verification",
            "hold_samples",
            "hold_seconds",
            "hold_failed_samples",
            "first_failed_hold",
            "milestones_completed",
            "milestones_required",
        )
        if key in simulation
    }


def success_marker(recording):
    """Resolve the validation receipt, including recordings from older runs."""
    recording = Path(recording)
    marker = recording.parent / "logs" / (recording.name + ".success.json")
    return marker if marker.is_file() else recording / "SUCCESS.json"


def _remove(path):
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def retain_attempt(directory, record):
    """Call after observation/replay/audit, never while the worker is running."""
    directory = Path(directory)
    if directory.is_symlink():
        raise ValueError("Refusing to clean a symlinked attempt directory")
    accepted = record.get("physical_success") is True and record.get("validation_success") is True
    if record.get("artifacts", {}).get("policy") == POLICY:
        return False
    paths = {}
    for name in ("simulation", "replay"):
        folder = directory / name
        if not accepted:
            if name == "simulation":
                for filename in ("initial_layout.png", "initial_layout_view.json"):
                    image = folder / filename
                    if image.is_file():
                        shutil.copy2(image, directory.parent / filename)
            _remove(folder)
            report = record.get(name)
            if isinstance(report, dict):
                recording = report.get("recording")
                if isinstance(recording, dict):
                    recording.update(saved=False, removed=True, reason="Attempt did not pass complete validation")
                report["artifacts_removed"] = True
                sidecar = directory / f"{name}.worker.json"
                if sidecar.is_file() and not sidecar.is_symlink():
                    write_json(sidecar, report)
            continue
        if folder.is_symlink():
            raise ValueError("Refusing to publish a symlinked simulation directory")
        if not folder.is_dir():
            continue
        # Candidate previews and per-segment plans are not recorded executions.
        planning = folder / "planning"
        if planning.is_symlink():
            planning.unlink()
        elif planning.exists():
            for path in planning.rglob("*.npz"):
                _remove(path)
        write_json(
            folder / "SUCCESS.json",
            {
                "policy": POLICY,
                "physical_success": True,
                "validation_success": True,
                "source_sha256": record.get("source_sha256"),
                "attempt": record.get("index"),
                "report": "../result.json",
                "trajectory": "trajectory.npz" if (folder / "trajectory.npz").is_file() else None,
            },
        )
        paths[name] = str(folder.resolve())
        if name == "simulation" and (folder / "demo_0000").is_dir():
            paths["recording"] = str((folder / "demo_0000").resolve())
    record["artifacts"] = {
        "policy": POLICY,
        "validated": accepted,
        "paths": paths,
        "planning_npz_retained": False,
    }
    return True


def clean_case(directory):
    """Migrate a completed historical pipeline case. In-flight cases are untouched."""
    directory = Path(directory)
    summary = directory / "summary.json"
    report = json.loads(summary.read_text())
    if report.get("status") not in TERMINAL or not isinstance(report.get("attempts"), list):
        return None
    if report.get("artifact_policy") == POLICY and all(
        row.get("artifacts", {}).get("policy") == POLICY for row in report["attempts"]
    ):
        return report
    for record in report["attempts"]:
        index = record.get("index")
        if type(index) is not int or index < 0:
            raise ValueError("Invalid attempt index in summary")
        attempt = directory / f"attempt_{index:02d}"
        if not attempt.is_dir():
            continue
        if retain_attempt(attempt, record):
            write_json(attempt / "result.json", record)
    report["artifact_policy"] = POLICY
    write_json(summary, report)
    return report


def success_entry(case, report):
    """Canonical data paths; never infer success from an NPZ's existence."""
    if report.get("physical_success") is not True or report.get("validation_success") is not True:
        return None
    index = report.get("successful_attempt")
    if type(index) is not int or index < 0:
        return None
    attempt = Path(case) / f"attempt_{index:02d}"
    simulation = attempt / "simulation"
    if not (simulation / "SUCCESS.json").is_file():
        return None
    return {
        "case": str(Path(case).resolve()),
        "successful_attempt": index,
        "simulation": str(simulation.resolve()),
        "trajectory": (
            str((simulation / "trajectory.npz").resolve()) if (simulation / "trajectory.npz").is_file() else None
        ),
        "recording": str((simulation / "demo_0000").resolve()) if (simulation / "demo_0000").is_dir() else None,
        "path_tracing_manifest": (
            str((simulation / "path_tracing_jobs.json").resolve())
            if (simulation / "path_tracing_jobs.json").is_file()
            else None
        ),
    }


def publish_success_index(root, records):
    """Task-independent dataset index, including legacy cases recovered on resume."""
    root = Path(root).resolve()
    entries = []
    for record in records:
        directory = record.get("directory")
        entry = record.get("data")
        if directory:
            for case in (Path(directory), Path(directory) / "execution"):
                if not case.resolve().is_relative_to(root):
                    continue
                if (case / "summary.json").is_file():
                    report = clean_case(case)
                    if report is not None:
                        entry = success_entry(case, report)
                        record["data"] = entry
                    break
        if record.get("physical_success") is True and record.get("validation_success") is True and entry:
            entries.append(entry)
    # Resume walks cached receipts from the first seed. Do not retract later
    # already-published successes while rebuilding that prefix.
    existing = root / "successful_data.json"
    known = {e["case"] for e in entries}
    if existing.exists():
        for old in json.loads(existing.read_text()).get("cases", []):
            case = Path(old["case"])
            if str(case) in known or not case.resolve().is_relative_to(root):
                continue
            marker = success_marker(case)
            if marker.is_file() and (case / "path_tracing_bundle.pkl").is_file():
                evidence = json.loads(marker.read_text())
                if evidence.get("physical_success") is True and evidence.get("validation_success") is True:
                    entries.append(old)
                    known.add(str(case))
                continue
            if (case / "summary.json").is_file():
                report = clean_case(case)
                recovered = success_entry(case, report) if report is not None else None
                if recovered:
                    entries.append(recovered)
                    known.add(str(case))
    publish_dataset(root, entries)


def publish_dataset(root, entries):
    """Expose MGG demo_NNNN directories without copying videos or relocating bundles.

    Relative links keep the batch movable as a whole. Existing successful demo
    numbers remain stable on resume, including interruption before index writes.
    Only links recorded in our previous index can be removed.
    """
    root = Path(root).resolve()
    index_path = root / "successful_data.json"
    previous = json.loads(index_path.read_text()).get("cases", []) if index_path.exists() else []
    recordings = {str(Path(e["recording"]).resolve()) for e in entries if e.get("recording")}
    for old in previous:
        if not old.get("demo") or old.get("recording") in recordings:
            continue
        link = Path(old["demo"])
        if (
            link.parent == root
            and link.name.startswith("demo_")
            and link.is_symlink()
            and str(link.resolve()) == old.get("recording")
        ):
            link.unlink()

    links = {}
    occupied = set()
    for path in root.glob("demo_*"):
        suffix = path.name.removeprefix("demo_")
        if suffix.isdigit():
            occupied.add(int(suffix))
            if path.is_symlink():
                links[str(path.resolve())] = path
    demos, seen = [], set()
    published = []
    for entry in entries:
        if entry["case"] in seen:
            continue
        seen.add(entry["case"])
        recording = entry.get("recording")
        if recording:
            recording = Path(recording).resolve()
            if not recording.is_relative_to(root) or not recording.is_dir():
                raise ValueError(f"Recording must exist inside the batch: {recording}")
            if not (recording.parent / "SUCCESS.json").is_file() and not success_marker(recording).is_file():
                raise ValueError(f"Recording lacks complete validation: {recording}")
            link = recording if recording.parent == root else links.get(str(recording))
            if link is None:
                index = 0
                while index in occupied:
                    index += 1
                link = root / f"demo_{index:04d}"
                link.symlink_to(os.path.relpath(recording, root), target_is_directory=True)
                occupied.add(index)
                links[str(recording)] = link
            entry["demo"] = str(link)
            if (recording / "path_tracing_bundle.pkl").is_file():
                demos.append(str(link))
        published.append(entry)
    result = {"policy": POLICY, "cases": published}
    if not index_path.exists() or json.loads(index_path.read_text()) != result:
        write_json(index_path, result)
    manifest = {"version": 1, "demos": demos}
    path = root / "path_tracing_jobs.json"
    if not path.exists() or json.loads(path.read_text()) != manifest:
        write_json(path, manifest)


def case_location(directory):
    """Resolve the task root and stable cache key for transient seed work."""
    directory = Path(directory)
    if directory.parent.name.startswith("version_"):
        root = directory.parent.parent.parent
        key = f"development/{directory.parent.name}/{directory.name}"
    elif directory.parent.name == "collection":
        root = directory.parent.parent
        key = f"collection/{directory.name}"
    else:
        # Standalone callers keep their receipts adjacent to the case.
        return directory.parent, directory.name
    if root.name == "logs":
        root = root.parent
    return root, key


def search_summary(row):
    """Read current counters, falling back only to an actually present legacy list."""
    result = {
        k: row.get(k)
        for k in (
            "object_id",
            "search_seconds",
            "path_candidates_evaluated",
            "full_path_candidates_evaluated",
            "search_budget",
            "exhausted",
            "selected_candidate_id",
            "stopping_reason",
            "endpoint_seconds",
            "endpoint_stopping_reason",
            "endpoint_candidates_total",
            "endpoint_unchecked",
            "endpoint_scan_complete",
            "endpoint_scan_stopping_reason",
            "endpoint_nonfirst_solution_passes",
            "endpoint_solution_policy",
            "approach_batches",
            "approach_evaluations",
            "approach_candidates_evaluated",
            "endpoint_evaluations",
            "endpoint_rejected",
            "preflight_scope",
        )
    }
    legacy = row.get("endpoint_screen")
    result["endpoint_screened"] = row.get("endpoint_screened", len(legacy) if isinstance(legacy, list) else None)
    result["endpoint_passed"] = row.get(
        "endpoint_passed", sum(e.get("success") is True for e in legacy) if isinstance(legacy, list) else None
    )
    return result


def retain_search_diagnostics(directory, logs, key, report):
    """Preserve every arm attempt before case cleanup, without predicted trajectories."""
    paths = []
    for summary in sorted(directory.glob("run_*/execution/summary.json")):
        run = summary.parent.parent.name
        execution = json.loads(summary.read_text())
        for attempt in execution.get("attempts", []):
            simulation = attempt.get("simulation") or {}
            index = attempt["index"]
            if type(index) is not int or index < 0:
                raise ValueError("Invalid attempt index in summary")
            destination = logs / "grasp_search" / key / run / f"attempt_{index:02d}.json"
            destination.parent.mkdir(parents=True, exist_ok=True)
            write_json(
                destination,
                {
                    "case_key": key,
                    "seed": report.get("seed"),
                    "source_sha256": report.get("source_sha256"),
                    "attempt": index,
                    "arm": simulation.get("arm"),
                    "error": simulation.get("error"),
                    "attempt_error": attempt.get("error"),
                    "execution": execution_evidence(simulation),
                    "action_context": simulation.get("action_context"),
                    "execution_ok": simulation.get("execution_ok"),
                    "physical_success": attempt.get("physical_success"),
                    "validation_success": attempt.get("validation_success"),
                    "grasp_selections": simulation.get("planning", {}).get("grasp_selections", []),
                },
            )
            paths.append(str(destination.resolve()))
    return paths


def retain_case(directory, report):
    """Publish a successful MGG bundle directly as demo_NNNN; discard work dirs.

    Logs and resume receipts live under logs/. Only finalized cases may enter.
    The receipt is committed before deleting staging files.
    """
    directory = Path(directory)
    if directory.is_symlink():
        raise ValueError("Refusing to clean a symlinked case directory")
    root, key = case_location(directory)
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    accepted = report.get("physical_success") is True and report.get("validation_success") is True
    path = logs / "cases.json"
    ledger = json.loads(path.read_text()) if path.exists() else {}
    if key in ledger:
        saved = ledger[key]
        if saved.get("source_sha256") != report.get("source_sha256"):
            raise ValueError("Case cache belongs to another program")
        if directory.exists():
            _remove(directory)
        return saved
    compact = {
        k: report.get(k)
        for k in ("seed", "source_sha256", "status", "physical_success", "validation_success", "validation_mode")
    }
    feedback = report.get("feedback") or {}
    simulation = feedback.get("simulation") or {}
    compact.update(
        directory=None,
        data=None,
        successful_attempt=report.get("successful_attempt"),
        feedback={
            "error": feedback.get("error"),
            "simulation": execution_evidence(simulation),
        },
    )
    selections = simulation.get("planning", {}).get("grasp_selections", [])
    compact["search_summary"] = [search_summary(row) for row in selections]
    compact["grasp_search_diagnostics"] = retain_search_diagnostics(directory, logs, key, report)
    if accepted:
        data = report.get("data") or {}
        recording = Path(data["recording"]) if data.get("recording") else None
        if recording is None or not recording.is_dir():
            raise ValueError("Successful case has no MGG recording; refusing to remove it")
        for name in ("meta.json", "trajectory.json", "output.json", "scene_states.npz", "path_tracing_bundle.pkl"):
            if not (recording / name).is_file():
                raise ValueError(f"Successful case lacks MGG recording file: {name}")
        # Recover relocation interrupted before the cache write.
        demo = None
        for existing in root.glob("demo_*"):
            marker = success_marker(existing)
            if marker.is_file() and not existing.is_symlink():
                evidence = json.loads(marker.read_text())
                if evidence.get("case_key") == key:
                    demo = existing
                    break
        if demo is None:
            index = 0
            while (root / f"demo_{index:04d}").exists():
                index += 1
            demo = root / f"demo_{index:04d}"
            # Copy to a staging folder, then rename: original survives interruption.
            staging = root / f".demo_{index:04d}.publishing"
            if staging.exists():
                _remove(staging)
            shutil.copytree(recording, staging)
            for extra in ("task.json", "result.json", "scene_states_index.json"):
                if (staging / extra).is_file():
                    write_json(logs / (demo.name + "." + extra), json.loads((staging / extra).read_text()))
                    (staging / extra).unlink()
            write_json(
                staging / "SUCCESS.json",
                {
                    "physical_success": True,
                    "validation_success": True,
                    "source_sha256": report["source_sha256"],
                    "seed": report["seed"],
                    "case_key": key,
                    "phase": key.split("/")[0],
                },
            )
            staging.rename(demo)
        # The temporary marker also makes an interrupted publication recoverable.
        # Keep only MGG data in the finalized demo directory.
        if (demo / "SUCCESS.json").is_file():
            (demo / "SUCCESS.json").replace(logs / (demo.name + ".success.json"))
        metadata = recording.parent / "recording.json"
        if metadata.is_file():
            shutil.copy2(metadata, logs / (demo.name + ".recording.json"))
        compact["data"] = {
            "case": str(demo.resolve()),
            "recording": str(demo.resolve()),
            "demo": str(demo.resolve()),
            "simulation": str(demo.resolve()),
            "trajectory": str((demo / "trajectory.json").resolve()),
            "successful_attempt": report.get("successful_attempt"),
            "path_tracing_manifest": str((root / "path_tracing_jobs.json").resolve()),
        }
        compact["directory"] = str(demo.resolve())
        # Full execution diagnostics are logs, never another dataset hierarchy.
        write_json(logs / (demo.name + ".json"), report)
    if not accepted:
        images = sorted(directory.glob("run_*/execution/initial_layout.png"))
        if images:
            image_dir = logs / "failed_layouts"
            image_dir.mkdir(exist_ok=True)
            image = image_dir / (key.replace("/", "_") + ".png")
            shutil.copy2(images[0], image)
            compact["initial_layout_image"] = str(image.resolve())
            details = images[0].with_name("initial_layout_view.json")
            if details.is_file():
                compact["initial_layout"] = json.loads(details.read_text())
        else:
            compact["initial_layout_image"] = None
            compact["initial_layout_image_status"] = "No captured frame; layout may have failed before scene creation"
    ledger[key] = compact
    write_json(path, ledger)
    demos = sorted(
        str(p.resolve())
        for p in root.glob("demo_*")
        if not p.is_symlink() and success_marker(p).is_file() and (p / "path_tracing_bundle.pkl").is_file()
    )
    write_json(root / "path_tracing_jobs.json", {"version": 1, "demos": demos})
    _remove(directory)
    return compact
