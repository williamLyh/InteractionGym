"""ColBench Backend Programming adapter (Meta, SWEET-RL; arXiv 2503.15478; github.com/facebookresearch/sweet_rl).

A ColBench task: write a Python function (up to ~50 lines) together with a human. The agent starts from a
high-level description and the function's signature; the details (conditions, edge cases) are only in a reference
implementation that the human sees and the agent does not, so the agent has to ask. The simulated human answers in
two sentences from that hidden information, never writes code, and says it does not know when the answer is not
there. At most 10 back-and-forth rounds; the agent hands in its function once and it is scored by 10 hidden test
cases (each: the call's output equals the reference function's): ``tests_passed`` (fraction) and ``success`` (all).
The agent cannot run code (the original setting).

Here it is spoken:

- **The user** says the description (``scenario["first_turn"]``, the description without its signature line) and
  answers questions with ColBench's own human-simulator prompt (``ColBenchUser``, an ``LLMSource``), plus a note that
  this is a spoken conversation. It hangs up only once the agent has said the code is done (``DONE_WORDS``), or
  after ``max_rounds`` answers.
- **The agent** sees the signature as text (``scenario["agent_context"]``, as if shared on screen: identifiers are
  not something to dictate), and hands in the code with the one tool ``submit_code(code)``: code is never spoken.
- **Scoring** (``evaluate``) runs the hidden tests after the episode, in a separate Python process with a time limit
  per call; the same rejections as the original (``import os`` / ``sys``, ``exit(``, ...) score 0.

Licences: ColBench's code and prompts are CC BY-NC 4.0 and its data is CC BY-NC (an output of Llama 3.1, also under
the Llama 3.1 licence); none of it is part of this package. Point ``COLBENCH_PATH`` at a sweet_rl checkout (for
``prompts/human_simulator_code_prompt.txt``) and ``COLBENCH_DATA`` at the downloaded dataset
(``huggingface-cli download facebook/collaborative_agent_bench backend_tasks/test.jsonl``), or put them at
``third_party/sweet_rl`` and ``third_party/colbench``. The test split has 10,000 records; the paper (and
``tasks``'s default) uses the first 1,000.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

from ..core import Node, Task
from ..tools import ToolBackend, ToolResult, ToolSpec, ToolWorld, tool_log
from ..user import STOP, LLMSource, Utterance

ROOT = Path(__file__).resolve().parents[3]
SIGNATURE_MARK = "The signature of the function is:"

INSTRUCTIONS = (
    "You are helping a caller write a Python function, in a spoken conversation. The function is personal to the caller: "
    "the details that decide whether it is right (the conditions, edge cases, exact values and output format) are known "
    "only to them and they tell you only what you ask. So ask clarification questions before writing anything: short "
    "ones, one or two at a time, since they answer briefly. You have at most {rounds} rounds of questions. "
    "The function's signature is in your context. You cannot run code. When you know enough, call submit_code once with "
    "the complete function (plain Python source, exactly that signature); never read code aloud. After submitting, tell "
    "the caller in one or two sentences what the function does."
)

# words by which the agent says the code is in; before them the simulated user may not hang up
DONE_WORDS = ("submit", "finished", "written", "wrote", "done", "ready", "here's the function", "the function takes", "the function returns")

SPOKEN_NOTE = (
    "\n\nThis is a spoken conversation (a phone call): say your answers naturally, never read code aloud, and stay "
    "within two sentences. Once the agent has written the function and told you what it does, thank it in a few words "
    "and end that turn with {stop}."
)


def _path(env: str, default: Path) -> Path:
    p = Path(os.environ.get(env) or default)
    return p


def data_dir() -> Path:
    return _path("COLBENCH_DATA", ROOT / "third_party" / "colbench")


def human_prompt() -> str:
    p = _path("COLBENCH_PATH", ROOT / "third_party" / "sweet_rl") / "prompts" / "human_simulator_code_prompt.txt"
    if not p.is_file():
        raise FileNotFoundError(f"ColBench's human-simulator prompt not found at {p}: set COLBENCH_PATH to a sweet_rl checkout")
    return p.read_text()


def records(split: str = "test", n: int | None = 1000) -> list[dict]:
    p = data_dir() / "backend_tasks" / f"{split}.jsonl"
    if not p.is_file():
        raise FileNotFoundError(f"{p} not found: set COLBENCH_DATA to the downloaded facebook/collaborative_agent_bench")
    out = []
    with open(p) as f:
        for line in f:
            if n is not None and len(out) >= n:
                break
            out.append(json.loads(line))
    return out


def to_task(rec: dict, task_id: str) -> Task:
    desc = str(rec["problem_description"])
    spoken, _, sig = desc.partition(SIGNATURE_MARK)
    return Task(
        id=task_id,
        scenario={
            "first_turn": spoken.strip(),
            "persona": "Someone who needs a Python function written and explains it over the phone.",
            "colbench": {"problem_description": desc, "hidden_information": str(rec["hidden_information"])},
            "agent_context": {"function_signature": sig.strip()} if sig else {},
        },
        criteria={"ground_truth": rec["ground_truth"], "test_cases": rec["test_cases"]},
    )


def tasks(split: str = "test", n: int | None = 1000) -> list[Task]:
    return [to_task(r, f"colbench-{split}-{i}") for i, r in enumerate(records(split, n))]


class ColBenchUser(LLMSource):
    """ColBench's human simulator (its own prompt, filled with the task's description and hidden information), with the
    conversation as chat messages and a spoken-conversation note; ends the call after ``max_rounds`` answers."""

    def __init__(self, llm, max_rounds: int = 10, prompt: str | None = None):
        super().__init__(llm)
        self.max_rounds = max_rounds
        self.prompt = prompt

    def profile(self) -> dict:
        return {"mode": "online", "simulator": "ColBench human simulator (CC BY-NC prompt) + spoken note",
                "max_rounds": self.max_rounds, **{k: v for k, v in super().profile().items() if k != "mode"}}

    def messages(self, task: Task, convo: list[Utterance], barge_in: bool = False) -> list[dict]:
        cb = task.scenario["colbench"]
        system = (self.prompt or human_prompt()).format(
            problem_description=cb["problem_description"], hidden_information=cb["hidden_information"],
            dialogue_history="(the dialogue so far follows as chat messages: yours are the assistant's, the agent's are the user's)",
        ) + SPOKEN_NOTE.format(stop=STOP)
        msgs = [m for m in super().messages(task, convo, barge_in) if m["role"] != "system"]
        return [{"role": "system", "content": system}] + msgs

    async def next(self, i, task, convo, barge_in=False):
        if i > self.max_rounds:  # the description plus max_rounds answers: the rounds are used up
            return None
        turn = await super().next(i, task, convo, barge_in)
        submitted = any(u.role == "agent" and any(w in u.text.lower() for w in DONE_WORDS) for u in convo)
        if turn is not None and turn.final and not submitted:  # the simulator may not hang up before the code is in
            turn = replace(turn, final=False)
        return turn


class ColBenchBackend(ToolBackend):
    """The one agent tool, ``submit_code``: the first submission is the answer."""

    name = "colbench"
    description = "Hand in the finished Python function."

    def __init__(self, max_rounds: int = 10):
        self.max_rounds = max_rounds

    def tools(self, caller="agent"):
        return [ToolSpec("submit_code", "Hand in the complete Python function (source code, with the given signature). Only the first submission counts.",
                         {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]})] if caller == "agent" else []

    def instructions(self, caller="agent"):
        return INSTRUCTIONS.format(rounds=self.max_rounds) if caller == "agent" else ""

    def reset(self, task, rng):
        return {"code": None}

    async def call(self, state, caller, call):
        if call.name != "submit_code":
            return ToolResult(call.id, call.name, f"Error: unknown tool {call.name}", error=True)
        if state["code"] is not None:
            return ToolResult(call.id, call.name, "Error: the function was already submitted", error=True)
        state["code"] = str(call.arguments.get("code", ""))
        return ToolResult(call.id, call.name, "Submitted. It will be tested after the call.")


def tool_world(max_rounds: int = 10, latency_ms: int = 300) -> ToolWorld:
    return ToolWorld(ColBenchBackend(max_rounds), latency_ms, callers=("agent",))


def nodes(llm, max_rounds: int = 10, voice=None, **user_kwargs) -> dict[str, Node]:
    """``{"user": UserSim(ColBenchUser), "tools": ToolWorld}``. ``voice`` (a ``Voice``) makes the user speak."""
    from ..user import UserSim
    user = UserSim(ColBenchUser(llm, max_rounds), voice, **user_kwargs) if voice is not None else UserSim(ColBenchUser(llm, max_rounds), **user_kwargs)
    return {"user": user, "tools": tool_world(max_rounds)}


# ---------------------------------------------------------------- scoring

_REJECT = ("import os", "from os", "import sys", "from sys", "sudo", "transformers", "exit(", "quit(", "argparse")

_RUNNER = r"""
import json, signal, sys
job = json.load(sys.stdin)
def run(src, expr):
    ns = {}
    try:
        exec(src, ns)
        return eval(expr, ns)
    except BaseException:
        return None
def alarm(*_):
    raise TimeoutError
signal.signal(signal.SIGALRM, alarm)
out = []
for expr in job["tests"]:
    want = run(job["reference"], expr)
    signal.setitimer(signal.ITIMER_REAL, job["timeout"])
    try:
        got = run(job["code"], expr)
    except TimeoutError:
        got = None
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    try:
        ok = want is not None and bool(want == got)
    except BaseException:
        ok = False
    out.append(ok)
sys.__stdout__.write("\n__RESULT__" + json.dumps(out))
"""


def _strip_fences(code: str) -> str:
    if "```python" in code:
        code = code.split("```python")[1].split("```")[0]
    if "```" in code:
        code = code.split("```")[1].split("```")[0]
    return code


VERIFIER = {
    "name": "ColBench hidden tests",
    "description": ("After the call, the first function handed in with submit_code is run on ColBench's 10 hidden test cases, in a "
                    "separate Python process with a time limit per call; each test passes if the submission returns what the "
                    "reference function returns. Reward 1 only if all tests pass. The agent never sees the tests or the reference."),
    "checks": {
        "tests_passed": "fraction of the hidden test cases on which the submission's output equals the reference function's",
        "success": "all hidden test cases pass (the reward)",
        "submitted": "a function was handed in with submit_code",
    },
}


def run_tests(code: str | None, reference: str, test_cases: dict, timeout_s: float = 1.0) -> list[bool]:
    """Per test case: does ``code`` give the reference function's (non-None) output? Runs in a separate Python
    process (isolated mode, a per-call time limit); generated code is untrusted, so run this in a sandbox."""
    tests = list(test_cases.values())
    if not code or any(w in code for w in _REJECT):
        return [False] * len(tests)
    job = {"code": _strip_fences(code), "reference": reference, "tests": tests, "timeout": timeout_s}
    try:
        p = subprocess.run([sys.executable, "-I", "-c", _RUNNER], input=json.dumps(job), capture_output=True, text=True,
                           timeout=5 + 2 * timeout_s * len(tests))
    except subprocess.TimeoutExpired:
        return [False] * len(tests)
    tail = p.stdout.rsplit("\n__RESULT__", 1)
    return json.loads(tail[1]) if len(tail) == 2 else [False] * len(tests)


def submitted_code(env, node: str = "tools") -> str | None:
    sim = getattr(env, "sim", env)
    return sim.states[node]["services"][ColBenchBackend.name]["code"]


def evaluate(env, task: Task, node: str = "tools") -> dict:
    """``{"reward", "tests_passed", "success", "submitted", "tests", "submit_time"}``: reward = success (all tests pass)."""
    code = submitted_code(env, node)
    ok = run_tests(code, task.criteria["ground_truth"], task.criteria["test_cases"])
    frac = sum(ok) / len(ok) if ok else 0.0
    t_sub = next((t for t, caller, c, r in tool_log(env.log) if caller == "agent" and c.name == "submit_code" and r and not r.error), None)
    return {"reward": float(frac == 1.0), "tests_passed": frac, "success": frac == 1.0, "submitted": code is not None, "tests": ok,
            "submit_time": t_sub}
