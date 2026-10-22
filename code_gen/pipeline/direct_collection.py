# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Success-quota collection with durable, fresh-layout episodes and no validation gate."""

import json
import secrets
from pathlib import Path

from code_gen._io import digest_file, framework_hashes, source_hash, write_json
from code_gen.pipeline.batch import ProductionSeedRunner, cached_case, failure_kind, passed, save_collection_progress
from code_gen.recording.artifacts import execution_evidence, success_marker
from code_gen.task.program import extract_program, validate_program

MODE = "direct_random_v1"
TERMINAL = {
    "validated",
    "physical_success_audit_incomplete",
    "offline_candidate_failed",
    "scene_failed",
    "infrastructure_failed",
    "budget_exhausted",
    "layout_sampling_failed",
}


def read_json(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def published_successes(root, digest):
    """Count real published recordings, including development successes, exactly once."""
    root = Path(root).resolve()
    ledger = read_json(root / "logs/cases.json", {})
    records, seen, keys = [], set(), set()
    for demo in sorted(root.glob("demo_*")):
        recording = demo.resolve()
        if not recording.is_relative_to(root):
            raise ValueError(f"Recording escaped dataset: {demo}")
        marker = success_marker(recording)
        if not marker.is_file():
            raise ValueError(f"Published demo lacks success evidence: {demo}")
        evidence = read_json(marker)
        if not passed(evidence):
            raise ValueError(f"Published demo did not pass complete validation: {demo}")
        if evidence.get("source_sha256") != digest:
            raise ValueError(f"Successful recording belongs to another program: {demo}")
        if recording in seen:
            continue
        for name in ("trajectory.json", "path_tracing_bundle.pkl"):
            path = recording / name
            if not path.is_file() or not path.stat().st_size:
                raise ValueError(f"Successful recording is incomplete: {path}")
        key, seed = evidence.get("case_key"), evidence.get("seed")
        if not isinstance(key, str) or not key or type(seed) is not int or not 0 <= seed < 2**31 or key in keys:
            raise ValueError(f"Invalid or duplicate successful case identity: {demo}")
        saved = ledger.get(key)
        if saved is not None and (
            not passed(saved) or saved.get("source_sha256") != digest or saved.get("seed") != seed
        ):
            raise ValueError(f"Success marker disagrees with case receipt: {demo}")
        seen.add(recording)
        keys.add(key)
        records.append({
            **evidence,
            "status": "validated",
            "directory": str(recording),
            "data": {
                "case": str(recording),
                "recording": str(recording),
                "demo": str(recording),
                "simulation": str(recording),
                "trajectory": str(recording / "trajectory.json"),
                "path_tracing_manifest": str(root / "path_tracing_jobs.json"),
            },
        })
    # Missing demos must not silently reduce the count and trigger replacement runs.
    for key, saved in ledger.items():
        if passed(saved) and saved.get("source_sha256") == digest and key not in keys:
            raise ValueError(f"Successful case has no published recording: {key}")
    return records


class DirectSeedRunner(ProductionSeedRunner):
    """Keep layout evidence outside staging, including failed and interrupted episodes."""

    def __init__(self, info, session, root):
        super().__init__(info, session)
        self.root = Path(root)

    def evidence_directory(self, run):
        path = self.root / "logs/direct_layouts" / Path(run).parent.name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def prepare_task(self, directory, seed):
        task = super().prepare_task(directory, seed)
        evidence = self.evidence_directory(Path(directory).parent)
        write_json(evidence / "scene.json", read_json(Path(task.scene_file)))
        write_json(evidence / "task.json", task.to_dict())
        return task

    def __call__(self, directory, seed, source):
        report = super().__call__(directory, seed, source)
        save_execution_evidence(self.evidence_directory(directory), seed, source, report)
        return report


def save_execution_evidence(directory, seed, source, report):
    directory.mkdir(parents=True, exist_ok=True)
    initial = [
        {
            "attempt": a["index"],
            "arm": (a.get("simulation") or {}).get("arm"),
            "initial_snapshot": (a.get("simulation") or {}).get("initial_snapshot"),
            "execution": execution_evidence(a.get("simulation") or {}),
            "attempt_error": a.get("error"),
        }
        for a in report.get("attempts", [])
    ]
    write_json(
        directory / "execution.json",
        {
            "seed": seed,
            "source_sha256": source_hash(source),
            "status": report.get("status"),
            "initial_states": initial,
        },
    )


def initialize(root, source, collection):
    root = Path(root)
    validate_program(source)
    if extract_program(source) != source:
        raise ValueError("Program must be plain source with one trailing newline; refusing to change its hash")
    digest = source_hash(source)
    config = {
        "mode": MODE,
        "source_sha256": digest,
        "target_successes": collection["target_successes"],
        "max_cases": collection["max_cases"],
        "seed_policy": "system_random_unique_31bit_reserved_before_execution",
        "taskinfo_sha256": digest_file(root / "taskinfo.lock.json"),
    }
    path = root / "direct_collection.lock.json"
    if path.exists() and read_json(path) != config:
        raise ValueError("Direct collection program or configuration changed")
    state_path = root / "logs/direct_collection.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    program = root / "logs/direct_program.py"
    if program.exists() and program.read_text() != source:
        raise ValueError("Direct collection program snapshot changed")
    if not path.exists():
        write_json(path, config)
    if not program.exists():
        program.write_text(source)
    state = read_json(state_path, {"mode": MODE, "source_sha256": digest, "next_episode": 0, "episodes": []})
    if state.get("mode") != MODE or state.get("source_sha256") != digest:
        raise ValueError("Direct collection journal belongs to another mode/program")
    rows, count = state.get("episodes"), state.get("next_episode")
    if not isinstance(rows, list) or type(count) is not int or count < 0 or len(rows) != count:
        raise ValueError("Invalid direct episode journal")
    seeds = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or type(row.get("episode")) is not int or row["episode"] != index:
            raise ValueError("Invalid direct episode sequence")
        seed = row.get("seed")
        if type(seed) is not int or not 0 <= seed < 2**31 or seed in seeds:
            raise ValueError("Invalid or duplicate direct layout seed")
        seeds.add(seed)
        status = row.get("status")
        if status not in {"reserved", "finished", "interrupted"}:
            raise ValueError("Invalid direct episode status")
        if status == "reserved" and index != count - 1:
            raise ValueError("Only the final episode may be reserved")
        report = row.get("report")
        if status == "finished" and (
            not isinstance(report, dict) or report.get("seed") != seed or report.get("source_sha256") != digest
        ):
            raise ValueError("Finished episode receipt disagrees with journal")
    known = {f"episode_{row['episode']:06d}" for row in rows}
    existing = {p.name for p in (root / "logs/collection").glob("episode_*")}
    existing.update(p.name for p in (root / "logs/direct_layouts").glob("episode_*"))
    ledger = read_json(root / "logs/cases.json", {})
    existing.update(k.split("/")[-1] for k in ledger if k.startswith("collection/episode_"))
    for demo in root.glob("demo_*"):
        key = read_json(success_marker(demo.resolve()), {}).get("case_key", "")
        if key.startswith("collection/episode_"):
            existing.add(key.split("/")[-1])
    if existing - known:
        raise ValueError("Direct episode evidence has no journal reservation; restore the journal before resuming")
    write_json(state_path, state)
    return state_path, state


def recover_episode(root, row, source):
    """Recover terminal work only; unfinished work stays on disk and is never rerun."""
    case = root / "logs/collection" / f"episode_{row['episode']:06d}"
    key = f"collection/{case.name}"
    # A published marker may precede the final receipt or staging cleanup.
    for record in published_successes(root, source_hash(source)):
        if record["case_key"] == key:
            if record["seed"] != row["seed"]:
                raise ValueError("Published episode seed disagrees with reservation")
            return record
    ledger = read_json(root / "logs/cases.json", {})
    terminal = key in ledger or (case / "result.json").exists()
    runs = sorted(case.glob("run_*"))
    if runs:
        report = read_json(runs[-1] / "execution/summary.json", {})
        if report.get("status") in TERMINAL:
            terminal = True
            save_execution_evidence(root / "logs/direct_layouts" / case.name, row["seed"], source, report)
    if not terminal:
        return None

    def never_execute(*args):
        raise RuntimeError("Terminal recovery unexpectedly requested new simulation")

    return cached_case(never_execute, case, row["seed"], source)


def run_direct_collection(info, root, source, runner):
    """No model calls or three-seed gate; the ordinary pipeline still audits every success."""
    root = Path(root).resolve()
    digest = source_hash(source)
    state_path, state = initialize(root, source, info.collection)
    for row in state["episodes"]:
        if row["status"] == "reserved":
            report = recover_episode(root, row, source)
            row.update(status="finished" if report is not None else "interrupted", report=report)
    write_json(state_path, state)
    successes = published_successes(root, digest)
    invocation = root / "logs" / f"direct_invocation_{secrets.token_hex(8)}.json"
    write_json(invocation, {"mode": MODE, "source_sha256": digest, "framework_sha256": framework_hashes()})

    def progress(phase, **extra):
        records = [r["report"] for r in state["episodes"] if r.get("report") is not None]
        save_collection_progress(
            root,
            records,
            successes,
            info.collection,
            digest,
            phase=phase,
            mode=MODE,
            attempted_seeds=len(state["episodes"]),
            episodes=len(state["episodes"]),
            **extra,
        )

    if len(successes) >= info.collection["target_successes"]:
        progress("complete")
        return 0
    ledger = read_json(root / "logs/cases.json", {})
    used = {r["seed"] for r in state["episodes"]} | {r["seed"] for r in successes}
    used.update(r["seed"] for r in ledger.values() if type(r.get("seed")) is int)
    # Retain interrupted historical layouts too, even if no final receipt exists.
    for base in (root, root / "logs"):
        for pattern in ("development/version_*/seed_*", "collection/seed_*"):
            for p in base.glob(pattern):
                if p.name.removeprefix("seed_").isdigit():
                    used.add(int(p.name.removeprefix("seed_")))
    progress("collecting")
    while not info.collection["max_cases"] or len(state["episodes"]) < info.collection["max_cases"]:
        seed = secrets.randbelow(2**31)
        while seed in used:
            seed = secrets.randbelow(2**31)
        used.add(seed)
        episode = state["next_episode"]
        row = {"episode": episode, "seed": seed, "status": "reserved", "invocation": str(invocation)}
        state["episodes"].append(row)
        state["next_episode"] += 1
        write_json(state_path, state)
        progress("collecting", current_episode=episode, current_seed=seed)
        print(
            "DIRECT_EPISODE"
            f" episode={episode} seed={seed} successes={len(successes)}/{info.collection['target_successes']}",
            flush=True,
        )
        try:
            report = cached_case(runner, root / "logs/collection" / f"episode_{episode:06d}", seed, source)
        except Exception:
            progress(
                "collection_needs_attention", current_episode=episode, current_seed=seed, failure_kind="infrastructure"
            )
            raise
        row.update(status="finished", report=report)
        write_json(state_path, state)
        successes = published_successes(root, digest)
        reached = len(successes) >= info.collection["target_successes"]
        progress("complete" if reached else "collecting")
        print(f"DIRECT_PROGRESS {len(successes)}/{info.collection['target_successes']} episode={episode}", flush=True)
        if reached:
            return 0
        if not passed(report) and failure_kind(report) != "runtime":
            progress("collection_needs_attention", failure_kind=failure_kind(report))
            return 3
    progress("collection_budget_exhausted")
    return 2
