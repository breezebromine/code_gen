# code_gen

LLM-assisted robotic task program generation and demonstration collection for Isaac Lab.

`code_gen` turns a declared tabletop task and a natural-language instruction into a reusable Python task program. Simulation feedback guides program revisions; once validated, a frozen program is reused across randomized layouts to collect successful robot demonstrations without making a model request for every episode.

中文：面向机器人桌面操作的任务代码生成与仿真数据采集模块，连接自然语言任务、受限技能程序、物理验证和可复用的数据采集流程。

## What it provides

- Declarative JSON/YAML task descriptions and reusable scene/object catalogs.
- A model-driven development loop with execution diagnostics and optional image-based feedback.
- A restricted task-language interpreter with manipulation skills such as pick, place, transport and insertion.
- Grasp-library preparation, bounded grasp search and adapters for shared cuRobo planning and Cartesian controllers.
- Fixed-program retries, randomized collection and success-only demonstration publication.
- MGG-compatible observations, timestamps, scene-state recording and offline replay integration.
- Direct collection from an existing task program, including a state-only multi-environment path.

## Scope and runtime dependencies

This repository contains the `code_gen/` module extracted from a LiSimLab working tree. It is an integration module, and this checkout alone is not a standalone Isaac Lab application or an installable simulator distribution.

Execution requires a compatible full LiSimLab environment, including its `isaaclab_arena` scene builders, grasping/planning modules, robot configuration, common recorder and replay utilities. It also requires Isaac Sim/Isaac Lab, an NVIDIA GPU, cuRobo, and separately supplied robot/object USD assets, scene definitions and optional HDR/texture assets. The included catalogs are examples; they reference assets and generated files that are not shipped here. The Chinese tutorial also refers to scripts available in the full LiSimLab tree.

Place the `code_gen/` directory beside the compatible `isaaclab_arena/` package in that full tree and run commands in its configured Docker environment. Use the LiSimLab version that matches these interfaces; an unmodified upstream IsaacLab-Arena installation is not asserted to be compatible.

The task interpreter restricts the supported language and skill calls. These checks do not establish isolation for arbitrary untrusted code or safety for a physical robot.

## Getting started in the full environment

From the full repository root, inside its Docker container:

```bash
/isaac-sim/python.sh -m code_gen --help
```

Prepare and mount your assets and scene catalog, then configure a compatible model endpoint in [`code_gen/api_config.py`](code_gen/api_config.py). The current client uses a Chat Completions-compatible HTTP interface; the checked-in default provider is DeepSeek. Configure a model supported by your provider. Keep API keys outside Git; environment variables `DEEPSEEK_API_KEY` and `CODEGEN_VISION_API_KEY` are supported by the current client.

Run a declared task after updating the example catalog paths:

```bash
/isaac-sim/python.sh -m code_gen \
  --taskinfo code_gen/examples/rectangle_to_table_50.taskinfo.json \
  --catalog code_gen/examples/catalog \
  --output logs/codegen/my_task
```

To reuse an existing program without model calls:

```bash
/isaac-sim/python.sh -m code_gen \
  --taskinfo my_tasks/taskinfo.json \
  --direct-collect --program my_tasks/candidate.py \
  --recording state_only --output logs/codegen/direct_trial
```

See the [module documentation](code_gen/README.md) and [中文完整教程](code_gen/TUTORIAL.zh-CN.md) for development seeds, task schemas, grasp preparation, retries, validation and recording contracts.

## Source and attribution

This publication is a source snapshot of `code_gen/`, including current working-tree changes, rather than a copy of the full LiSimLab repository or its Git history. Runtime assets, datasets, logs and credentials are not included. No benchmark, adoption or independent end-to-end reproduction claim accompanies this extraction.

Existing copyright and SPDX notices are retained. The module uses interfaces and components from the Isaac Lab Arena ecosystem; the original developer attribution remains in the source headers. See [LICENSE.md](LICENSE.md) for the Apache License 2.0 text inherited from the source repository. Third-party runtime software and assets retain their own licenses.
