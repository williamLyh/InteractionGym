"""Tiny least-in-flight load balancer over TTS replicas (vLLM-Omni): TTS_BACKENDS = comma-separated base URLs.
Forwards any path/method verbatim; skips replicas that fail to connect. Run by scripts/run_tts_proxy.sh."""
import os
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import Response, JSONResponse

BACKENDS = os.environ.get("TTS_BACKENDS", "http://127.0.0.1:8002,http://127.0.0.1:8003").split(",")
inflight = {b: 0 for b in BACKENDS}
client = httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=5.0), limits=httpx.Limits(max_connections=512))
app = FastAPI()

@app.api_route("/{path:path}", methods=["GET", "POST"])
async def proxy(path: str, request: Request):
    body = await request.body()
    headers = {k: v for k, v in request.headers.items() if k.lower() in ("content-type", "authorization", "accept")}
    last_err = None
    for b in sorted(BACKENDS, key=lambda x: inflight[x]):
        inflight[b] += 1
        try:
            r = await client.request(request.method, f"{b}/{path}", content=body, headers=headers, params=request.query_params)
            return Response(content=r.content, status_code=r.status_code,
                            media_type=r.headers.get("content-type"))
        except httpx.ConnectError as e:
            last_err = e
        finally:
            inflight[b] -= 1
    return JSONResponse({"error": f"no TTS backend reachable: {last_err}"}, status_code=503)
