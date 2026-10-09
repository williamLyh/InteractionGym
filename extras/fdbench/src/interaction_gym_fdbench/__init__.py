# SPDX-License-Identifier: CC-BY-NC-4.0
"""interaction-gym-fdbench: Full-Duplex-Bench material (github.com/DanielLin94144/Full-Duplex-Bench), (c) the
Full-Duplex-Bench authors, as an optional component of InteractionGym.

Licensed under the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0, see LICENSE
at the root of this package, https://creativecommons.org/licenses/by-nc/4.0/): non-commercial use only. Unlike
InteractionGym itself (Apache-2.0), this package is NOT Apache-2.0; it is distributed separately and only
``interaction_gym.benchmarks.full_duplex_bench`` / ``fdb2`` / ``fdb3`` load it, on first use. Changes are described at the top of each file (ports of the
official Python scripts and prompts at upstream commit 3e799c4, May 2026).

- ``v1``: v1 / v1.5 metric rules (eval_*.py, get_timing.py) and judge prompts
- ``v2``: v2 judge rubric prompts (eval/eval_prompts.json) and prompt layout (eval_single_item.py)
- ``v3``: v3 mock APIs, tool schemas, agent instructions and judges (mock_apis.py, cascaded_agent.py, evaluate_*.py)
"""
