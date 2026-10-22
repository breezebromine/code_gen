# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Persistent MGG-style execution for generated task programs.

One SimulationApp, environment and ActionPlanner serve all same-topology episodes.
The existing pipeline owns acceptance, candidate retry and publication.
"""

import copy
import os

from code_gen._io import write_json
from code_gen.runtime.episode import episode_report, load_episode, run_episode


class PersistentWorker:
    motion_planner = "curobo"

    def __init__(self, session, task_file, seed):
        self.session, self.task_file, self.seed = session, task_file, seed
        self.excluded_grasp_candidates = {}
        self.preferred_grasp_candidates = {}
        self.arm = session.args.arm
        self.requested_arm = session.args.arm
        self.arm_history = []
        self.arm_retry_state = {}

    def advance_arm(self, simulation):
        """Only restart a failed, resettable episode; never switch a live grasp."""
        if self.requested_arm != "auto" or simulation.get("physical_success"):
            return False
        from code_gen.pipeline.grasp_retry import failed_grasp

        _, kind = failed_grasp(simulation)
        selections = simulation.get("planning", {}).get("grasp_selections", [])
        exhausted = any(row.get("exhausted") for row in selections)
        if kind == "infrastructure" or not (
            exhausted
            or kind
            in {
                "pick_execution",
                "grasp_lost",
                "placement_ik",
                "placement_planning",
                "placement_path_collision",
                "placement_release_clearance",
                "placement_release_retreat_collision",
            }
        ):
            return False
        active = simulation.get("arm")
        if active not in ("left", "right"):
            return False
        other = "left" if active == "right" else "right"
        if active not in self.arm_history:
            self.arm_history.append(active)
        if other in self.arm_history:
            return False
        self.arm_retry_state[active] = {
            "exclusions": copy.deepcopy(self.excluded_grasp_candidates),
            "preferences": copy.deepcopy(self.preferred_grasp_candidates),
        }
        self.arm = other
        self.arm_history.append(other)
        self.excluded_grasp_candidates = {}
        self.preferred_grasp_candidates = {}
        return True

    def __call__(self, mode, directory, candidate=None, replay_dir=None):
        return self.session.execute(
            self.task_file,
            self.seed,
            mode,
            directory,
            candidate,
            replay_dir,
            self.excluded_grasp_candidates,
            self.preferred_grasp_candidates,
            arm_override=self.arm if self.arm != "auto" else None,
        )


class ProductionSession:
    def __init__(self, info, *, recording="state_only", device="cpu", batch_planning=True):
        if recording not in ("state_only", "online_rgb"):
            raise ValueError("Unknown recording mode")
        from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
        from isaaclab_arena.grasping.robot_layout import aloha_robot_path

        parser = get_isaaclab_arena_cli_parser()
        self.args = parser.parse_args(["--headless", "--device", device])
        self.args.enable_cameras = recording == "online_rgb"
        self.args.state_only = recording == "state_only"
        self.args.capture_initial_layout = True
        self.args.enable_pinocchio = False
        self.args.solve_relations = False
        self.args.robot = info.robot["name"]
        self.args.arm = info.robot["arm"]
        self.args.robot_usd = str(aloha_robot_path(info.assets))
        self.args.robot_urdf = None
        self.args.aloha_assets_root = str(info.assets)
        self.args.motion_planner = "curobo"
        self.args.batch_planning = batch_planning
        self.args.execution_timeout = 1200
        os.environ["LISIM_ASSETS_PATH"] = str(info.assets)
        self.backend = None
        self.calls = 0

    def __enter__(self):
        from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext

        launcher_args = copy.copy(self.args)
        # Enable renderer support for one initial image; state-only execution
        # still has no online RGB recording or per-step camera rendering.
        launcher_args.enable_cameras = self.args.enable_cameras or self.args.capture_initial_layout
        self.context = SimulationAppContext(launcher_args)
        self.app = self.context.__enter__()
        return self

    def __exit__(self, *exc):
        try:
            if self.backend is not None:
                self.backend.env.close()
        finally:
            result = self.context.__exit__(*exc)
        return result

    def worker(self, task_file, seed):
        return PersistentWorker(self, task_file, seed)

    def execute(
        self,
        task_file,
        seed,
        mode,
        directory,
        candidate,
        replay_dir,
        exclusions,
        preferences=None,
        *,
        arm_override=None
    ):
        from code_gen.runtime.backend import TabletopBackend

        directory, task, source = load_episode(task_file, directory, candidate)
        args = copy.copy(self.args)
        args.mode, args.seed, args.output_dir = mode, seed, str(directory)
        if arm_override is not None:
            args.arm = arm_override
        args.requested_arm = self.args.arm
        args.excluded_grasp_candidates = copy.deepcopy(exclusions)
        args.preferred_grasp_candidates = copy.deepcopy(preferences or {})
        report = episode_report(
            args,
            task,
            source,
            execution_backend="persistent_mgg",
            recording_mode="state_only" if args.state_only else "online_rgb",
            session_call=self.calls,
            environment_reused=self.backend is not None,
            planner_reused=self.backend is not None and self.backend._planner is not None,
        )
        self.calls += 1

        def prepare_backend():
            if self.backend is None:
                self.backend = TabletopBackend(args, self.app, task)
            else:
                self.backend.reset_episode(args, task)
            # reset_episode may have selected the other arm; report that arm's cache.
            report["planner_reused"] = self.backend._planner is not None
            return self.backend

        run_episode(prepare_backend, report, directory, source=source, replay_dir=replay_dir, close=False)
        report.update(
            worker_exit_code=0 if report["completed"] else 1, timed_out=False, execution_ok=report["completed"]
        )
        write_json(directory / "simulation.json", report)
        write_json(directory.with_suffix(".worker.json"), report)
        return report
