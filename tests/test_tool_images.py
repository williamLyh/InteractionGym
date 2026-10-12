"""Tool results can carry images (e.g. a computer-use screenshot): stored as media and referenced from the call."""

import asyncio

from interaction_gym import AgentSpec, Env, Frame, Task
from interaction_gym.media import MediaStore
from interaction_gym.tools import CALL, RESULT, ToolBackend, ToolCall, ToolResult, ToolSpec, ToolWorld
from interaction_gym.traj import episode

PNG = b"\x89PNG\r\n\x1a\nfake"


class Screen(ToolBackend):
    name = "screen"

    def tools(self, caller="agent"):
        return [ToolSpec("look")]

    async def call(self, state, caller, call):
        return ToolResult(call.id, call.name, "a page", images=(PNG,))


def test_images_are_saved_as_media(tmp_path):
    env = Env({"tools": ToolWorld(Screen(), 50)}, AgentSpec(chunk_ms=100, obs=(RESULT["agent"],)))

    async def go():
        await env.reset(Task(id="t"))
        await env.step([Frame(CALL["agent"], 0, ToolCall("c1", "look"))])
        await env.step([])

    asyncio.run(go())
    ep = episode(env, "t", media=MediaStore(tmp_path))
    call = next(c for t in ep["turns"] for c in t.get("tool_calls", []))
    assert call["images"][0]["kind"] == "image" and call["images"][0]["uri"].endswith(".png")
    assert (tmp_path / call["images"][0]["uri"]).read_bytes() == PNG
    assert "images" not in next(c for t in episode(env, "t")["turns"] for c in t.get("tool_calls", []))  # no store: not written
