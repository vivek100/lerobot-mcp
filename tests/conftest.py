"""E2E fixtures: launch the real MCP server (sim backend, or the real arm with SO101_REAL=1) and talk to it over MCP/HTTP."""

from __future__ import annotations

import base64
import io
import json
import os
import signal
import socket
import subprocess
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "artifacts" / "e2e"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"


def _launch_server(backend_args: list[str], log_name: str):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    port = _free_port()
    log_path = LOG_DIR / log_name
    log = open(log_path, "w", encoding="utf-8")
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "MCP_USE_ANONYMIZED_TELEMETRY": "false"}
    proc = subprocess.Popen(
        [sys.executable, "-m", "so101_mcp.server", *backend_args, "--port", str(port)],
        cwd=ROOT,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
    )
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            log.close()
            raise RuntimeError(f"Server exited early; see {log_path}:\n{log_path.read_text(encoding='utf-8')[-3000:]}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                break
        except OSError:
            time.sleep(0.5)
    else:
        proc.kill()
        raise TimeoutError(f"Server did not start; see {log_path}")
    yield f"http://127.0.0.1:{port}/mcp"
    # terminate() is TerminateProcess on Windows (no cleanup: the real arm would not park and release).
    proc.send_signal(signal.CTRL_BREAK_EVENT)
    try:
        proc.wait(40)
    except subprocess.TimeoutExpired:
        proc.kill()
    log.close()


@pytest.fixture(scope="session")
def server_url():
    yield from _launch_server(["--backend", "sim", "--seed", "0", "--mute"], "server.log")


@pytest.fixture(scope="session")
def real_server_url():
    if os.environ.get("SO101_REAL") != "1":
        pytest.skip("real-arm tests need SO101_REAL=1, the arm powered, and someone watching it")
    yield from _launch_server(["--backend", "real"], "server_real.log")


class Robot:
    """Thin MCP client wrapper: every call goes through the real MCP protocol."""

    def __init__(self, session: ClientSession):
        self.session = session

    async def call(self, tool: str, args: dict | None = None, expect_error: bool = False):
        result = await self.session.call_tool(tool, args or {})
        if expect_error:
            assert result.isError, f"{tool} should have failed but returned {result}"
            return " ".join(c.text for c in result.content if c.type == "text")
        assert not result.isError, f"{tool} failed: {[getattr(c, 'text', c) for c in result.content]}"
        return result

    async def data(self, tool: str, args: dict | None = None) -> dict:
        result = await self.call(tool, args)
        if result.structuredContent:
            return result.structuredContent
        texts = [c.text for c in result.content if c.type == "text"]
        return json.loads(texts[0])

    async def status(self) -> dict:
        state, _ = await self.observe(width=320, previews=False)
        return state

    async def joint(self, name: str) -> float:
        return (await self.status())["joints"][name]

    async def move(self, **request) -> dict:
        return await self.data("move", {"request": request})

    async def observe(
        self, cameras: list[str] | None = None, width: int = 320, previews: bool = False
    ) -> tuple[dict, dict[str, np.ndarray]]:
        request: dict = {"image_width": width, "previews": previews}
        if cameras:
            request["cameras"] = cameras
        result = await self.call("observe", {"request": request})
        texts = [c for c in result.content if c.type == "text"]
        images = [c for c in result.content if c.type == "image"]
        state = json.loads(texts[0].text)
        frames = {
            name: np.asarray(Image.open(io.BytesIO(base64.b64decode(img.data))).convert("RGB"))
            for name, img in zip(state["cameras"], images)
        }
        assert all(img.mimeType == "image/jpeg" for img in images)
        return state, frames

    async def reset(self, seed: int | None = 0) -> dict:
        await self.call("stop")
        return await self.data("go_rest")


@pytest.fixture(scope="session")
def kinematics():
    """Independent forward kinematics (same MuJoCo model) to check where the arm really is."""
    from so101_mcp.kinematics import ArmKinematics

    return ArmKinematics()


@pytest.fixture
async def robot(server_url):
    @asynccontextmanager
    async def connect():
        async with streamable_http_client(server_url) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield Robot(session)

    async with connect() as r:
        await r.call("stop")
        await r.data("go_rest")
        yield r


@pytest.fixture
async def real_robot(real_server_url):
    async with streamable_http_client(real_server_url) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            r = Robot(session)
            await r.call("stop")
            yield r
