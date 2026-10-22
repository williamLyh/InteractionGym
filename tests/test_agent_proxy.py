"""The agent WebSocket proxy: one address in front of several duplex servers, one server per session."""

import asyncio
import json
import time
import urllib.request

import pytest

websockets = pytest.importorskip("websockets")

from websockets.asyncio.client import connect  # noqa: E402
from websockets.asyncio.server import serve  # noqa: E402
from websockets.exceptions import ConnectionClosed  # noqa: E402

from interaction_gym.agent_proxy import AgentProxy  # noqa: E402


def backend(name, seen):
    """A stand-in duplex server: says who it is (and what it was asked), then echoes; "close" closes with 4000."""
    async def handler(ws):
        seen.append(name)
        await ws.send(json.dumps({"backend": name, "path": ws.request.path, "auth": ws.request.headers.get("Authorization")}))
        try:
            async for m in ws:
                if m == "close":
                    await ws.close(4000, "server done")
                    return
                await ws.send(m)
        except ConnectionClosed:
            pass
        seen.append(f"{name}:closed")
    return handler


async def cluster(n=2, **kw):
    seen, servers = [], []
    for i in range(n):
        servers.append(await serve(backend(f"b{i}", seen), "127.0.0.1", 0, max_size=None))
    urls = [f"ws://127.0.0.1:{s.sockets[0].getsockname()[1]}" for s in servers]
    proxy = AgentProxy(kw.pop("backends", urls), log=lambda *_: None, **kw)
    ps = await proxy.serve("127.0.0.1", 0)
    return proxy, ps, servers, seen, f"ws://127.0.0.1:{ps.sockets[0].getsockname()[1]}"


async def hello(ws):
    return json.loads(await ws.recv())


def run(coro):
    return asyncio.run(coro)


def test_sessions_spread_over_servers_by_open_sessions():
    async def main():
        proxy, ps, servers, seen, url = await cluster(2)
        c = [await connect(url + "/v1/realtime?duplex=1") for _ in range(4)]
        names = [(await hello(x))["backend"] for x in c]
        assert names == ["b0", "b1", "b0", "b1"] and proxy.open == dict(zip(proxy.backends, [2, 2]))
        await c[0].close()  # b0 now has one: the next session goes there
        await asyncio.sleep(0.05)
        d = await connect(url + "/v1/realtime?duplex=1")
        assert (await hello(d))["backend"] == "b0"
        for x in c[1:] + [d]:
            await x.close()
        ps.close()
        [s.close() for s in servers]
    run(main())


def test_path_query_and_key_are_forwarded_and_messages_relay_both_ways():
    async def main():
        proxy, ps, servers, seen, url = await cluster(1)
        async with connect(url + "/v1/realtime?duplex=1", additional_headers={"Authorization": "Bearer k"}, max_size=None) as c:
            h = await hello(c)
            assert h["path"] == "/v1/realtime?duplex=1" and h["auth"] == "Bearer k"
            big = "x" * 2_000_000  # larger than websockets' default 1 MiB limit
            await c.send(big)
            assert await c.recv() == big
            await c.send(b"\x00\x01")
            assert await c.recv() == b"\x00\x01"
        ps.close()
        servers[0].close()
    run(main())


def test_a_close_on_either_side_reaches_the_other():
    async def main():
        proxy, ps, servers, seen, url = await cluster(1)
        c = await connect(url + "/x")
        await hello(c)
        await c.send("close")
        with pytest.raises(ConnectionClosed):
            await c.recv()
        assert c.close_code == 4000 and c.close_reason == "server done"
        c2 = await connect(url + "/x")
        await hello(c2)
        await c2.close()
        await asyncio.sleep(0.1)
        assert "b0:closed" in seen and proxy.open[proxy.backends[0]] == 0
        ps.close()
        servers[0].close()
    run(main())


def test_full_servers_make_a_session_wait_and_a_long_wait_ends_it():
    async def main():
        proxy, ps, servers, seen, url = await cluster(2, max_sessions=1, queue_timeout_s=5)
        a, b = await connect(url + "/x"), await connect(url + "/x")
        await hello(a), await hello(b)
        waiting = asyncio.ensure_future(connect(url + "/x"))
        await asyncio.sleep(0.2)
        assert proxy.waiting == 1
        await a.close()  # a slot frees up: the waiting session gets it
        c = await waiting
        assert (await hello(c))["backend"] in ("b0", "b1") and proxy.waiting == 0
        proxy.queue_timeout_s = 0.3
        late = await connect(url + "/x")
        with pytest.raises(ConnectionClosed):
            await late.recv()
        assert late.close_code == 1013  # try again later
        for x in (b, c):
            await x.close()
        ps.close()
        [s.close() for s in servers]
    run(main())


def test_a_server_that_is_down_is_skipped():
    async def main():
        proxy, ps, servers, seen, url = await cluster(1)
        live = proxy.backends[0]
        proxy2 = AgentProxy(["ws://127.0.0.1:9", live], log=lambda *_: None)  # port 9: nothing listens
        ps2 = await proxy2.serve("127.0.0.1", 0)
        url2 = f"ws://127.0.0.1:{ps2.sockets[0].getsockname()[1]}"
        for _ in range(3):
            async with connect(url2 + "/x") as c:
                assert (await hello(c))["backend"] == "b0"
        await asyncio.sleep(0.1)
        assert proxy2.open == {"ws://127.0.0.1:9": 0, live: 0}
        ps.close(), ps2.close()
        servers[0].close()
    run(main())


def test_plain_http_reports_the_open_sessions():
    async def main():
        proxy, ps, servers, seen, url = await cluster(2, max_sessions=16)
        c = await connect(url + "/x")
        await hello(c)
        port = ps.sockets[0].getsockname()[1]
        body = await asyncio.to_thread(lambda: urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5).read())
        st = json.loads(body)
        assert [b["open"] for b in st["backends"]] == [1, 0] and st["max_sessions"] == 16
        await c.close()
        ps.close()
        [s.close() for s in servers]
    run(main())


def test_the_extra_hop_costs_little(capsys):
    async def main():
        proxy, ps, servers, seen, url = await cluster(1)
        direct = proxy.backends[0]
        msg = "a" * 8600  # a 200 ms, 16 kHz PCM16 append, base64 in JSON

        async def rtt(u):
            async with connect(u + "/x", max_size=None) as c:
                await hello(c)
                for _ in range(20):
                    await c.send(msg), await c.recv()
                t = time.perf_counter()
                for _ in range(300):
                    await c.send(msg)
                    await c.recv()
                return (time.perf_counter() - t) / 300 * 1000

        d, p = await rtt(direct), await rtt(url)
        print(f"round trip: direct {d:.3f} ms, through the proxy {p:.3f} ms (+{p - d:.3f})")
        assert p - d < 5  # generous: well under a step (80-200 ms) or a lockstep acknowledgement
        ps.close()
        servers[0].close()
    run(main())
    print(capsys.readouterr().out)


def test_a_client_can_ask_how_many_servers_are_behind_an_address():
    from interaction_gym.agent_proxy import servers_behind

    async def main():
        proxy, ps, servers, seen, url = await cluster(3)
        n_proxy = await asyncio.to_thread(servers_behind, url + "/v1/realtime?duplex=1")
        n_server = await asyncio.to_thread(servers_behind, proxy.backends[0] + "/v1/realtime?duplex=1")  # not a proxy
        n_none = await asyncio.to_thread(servers_behind, "ws://127.0.0.1:9/v1/realtime", 1.0)
        # warm-up sessions opened at once land one per server
        c = [await connect(url + "/x") for _ in range(3)]
        names = sorted([(await hello(x))["backend"] for x in c])
        for x in c:
            await x.close()
        ps.close()
        [s.close() for s in servers]
        return n_proxy, n_server, n_none, names

    assert run(main()) == (3, 1, 1, ["b0", "b1", "b2"])
