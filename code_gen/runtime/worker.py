"""Isolated simulator process lifecycle and sanitized worker environment."""

import json
import os
import signal
import subprocess
import sys
from contextlib import suppress
from pathlib import Path

from code_gen._io import write_json


def worker_environment():
    """Keep simulator setup but remove model credentials from the child process."""
    return {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("OPENAI_", "DEEPSEEK_", "ANTHROPIC_"))
        and not any(term in key.upper() for term in ("API_KEY", "TOKEN", "SECRET", "PASSWORD"))
        and key not in ("LOOP_API_ENV", "CODEGEN_API_AUDIT_DIR")
    }


class SimulationWorker:
    def __init__(
        self,
        task_file,
        robot_path,
        *,
        urdf=True,
        seed=0,
        timeout=1200,
        cameras=True,
        device="cuda:0",
        motion_planner="curobo",
        execution_timeout=600,
        robot="aloha",
        arm="left",
        aloha_assets_root=None,
    ):
        self.task_file = Path(task_file).resolve()
        self.robot_path = str(Path(robot_path).resolve())
        self.urdf, self.seed, self.timeout, self.cameras = urdf, seed, timeout, cameras
        self.robot, self.arm, self.aloha_assets_root = robot, arm, aloha_assets_root
        self.device = device
        self.excluded_grasp_candidates = {}
        self.preferred_grasp_candidates = {}
        self.motion_planner, self.execution_timeout = motion_planner, execution_timeout

    def __call__(self, mode, directory, candidate=None, replay_dir=None):
        directory = Path(directory).resolve()
        command = [
            sys.executable,
            "-m",
            "isaaclab_arena.evaluation.codegen_worker",
            "--task-file",
            str(self.task_file),
            "--output-dir",
            str(directory),
            "--mode",
            mode,
            "--robot-urdf" if self.urdf else "--robot-usd",
            self.robot_path,
            "--seed",
            str(self.seed),
            "--headless",
            "--device",
            self.device,
            "--robot",
            self.robot,
            "--arm",
            self.arm,
            "--motion-planner",
            self.motion_planner,
            "--execution-timeout",
            str(self.execution_timeout),
        ]
        command += ["--excluded-grasp-candidates", json.dumps(self.excluded_grasp_candidates)]
        command += ["--preferred-grasp-candidates", json.dumps(self.preferred_grasp_candidates)]
        if self.aloha_assets_root:
            command += ["--aloha-assets-root", self.aloha_assets_root]
        if candidate is not None:
            command += ["--candidate", str(Path(candidate).resolve())]
        if replay_dir is not None:
            command += ["--replay-dir", str(Path(replay_dir).resolve())]
        if not self.cameras:
            command += ["--no-cameras"]
        timed_out = False
        with directory.with_suffix(".log").open("x") as log:
            process = subprocess.Popen(
                command, env=worker_environment(), stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
            try:
                code = process.wait(timeout=self.timeout)
            except BaseException:
                # Reap the entire child group on timeout, cancellation, or interrupt.
                for sig, grace in ((signal.SIGTERM, 8), (signal.SIGKILL, None)):
                    with suppress(ProcessLookupError):
                        os.killpg(process.pid, sig)
                    try:
                        code = process.wait(timeout=grace)
                        break
                    except subprocess.TimeoutExpired:
                        continue
                if sys.exc_info()[0] is not subprocess.TimeoutExpired:
                    raise
                timed_out = True
        report_path = directory / "simulation.json"
        try:
            report = json.loads(report_path.read_text())
        except (OSError, ValueError):
            report = {"completed": False, "physical_success": False, "error": "Missing or malformed simulation report"}
        report.update(worker_exit_code=code, timed_out=timed_out)
        report["execution_ok"] = bool(report.get("completed") and code == 0 and not timed_out)
        # Preserve measured physical success even if later shutdown failed.
        write_json(directory.with_suffix(".worker.json"), report)
        return report
