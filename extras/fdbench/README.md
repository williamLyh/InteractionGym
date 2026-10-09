# interaction-gym-fdbench

Optional component of [InteractionGym](https://github.com/williamLyh/InteractionGym): the material ported from
[Full-Duplex-Bench](https://github.com/DanielLin94144/Full-Duplex-Bench) that the benchmark loaders
`interaction_gym.benchmarks.full_duplex_bench`, `fdb2` and `fdb3` need (official metric rules, judge prompts,
v3 mock APIs and tool schemas).

**License: CC BY-NC 4.0 (non-commercial use only)**, unlike InteractionGym itself (Apache-2.0). See
[LICENSE](LICENSE) and [NOTICE](NOTICE). It is a separate distribution so that the main package stays Apache-2.0;
install it only if your use is non-commercial.

| Module | Ported from |
|---|---|
| `v1` | v1 / v1.5 `v1_v1.5/evaluation/` (metric rules, thresholds, judge prompts) |
| `v2` | v2 judge rubric prompts and prompt layout, examinee prompt, examiner suffix |
| `v3` | v3 mock APIs, tool schemas, agent instructions, latency profiles, judges and pass logic |

Install (from a checkout of the repository):

```bash
uv sync --extra fdbench                 # in the repository (the dev group installs it too)
pip install ./extras/fdbench            # or with pip, next to interaction-gym
```
