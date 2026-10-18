"""Scripted agent session: call the SO-101 MCP tools over HTTP and record the MuJoCo viewer side by side.

Start the server first (separate terminal):   uv run so101-mcp --viewer
Then run:                                      uv run python scripts/demo_mcp_session.py
Outputs: artifacts/demo/demo.gif, artifacts/demo/frames/*.png, artifacts/demo/key_*.png
"""

from __future__ import annotations

import asyncio
import base64
import ctypes
import ctypes.wintypes as wt
import io
import json
import shutil
import time
from pathlib import Path

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from PIL import Image, ImageDraw, ImageFont, ImageGrab

URL = "http://127.0.0.1:8000/mcp"
OUT = Path("artifacts/demo")
VIEWER_TITLE = "MuJoCo : so101_tabletop"
CANVAS = (1600, 760)
PANEL_W = 600
FPS_CAPTURE = 6

user32 = ctypes.windll.user32
ctypes.windll.shcore.SetProcessDpiAwareness(2)
FONT = ImageFont.truetype("consola.ttf", 17)
FONT_B = ImageFont.truetype("consolab.ttf", 19)


def find_viewer() -> int:
    hwnd = user32.FindWindowW(None, VIEWER_TITLE)
    if not hwnd:
        raise SystemExit(f"Viewer window '{VIEWER_TITLE}' not found; start the server with --viewer")
    return hwnd


def set_topmost(hwnd: int, on: bool) -> None:
    """Screen capture grabs whatever is on top, so keep the viewer above other windows while recording."""
    user32.ShowWindow(hwnd, 9)  # SW_RESTORE
    user32.SetWindowPos(hwnd, -1 if on else -2, 0, 0, 0, 0, 0x0001 | 0x0002)  # (NO)TOPMOST, NOSIZE|NOMOVE


def capture(hwnd: int) -> Image.Image:
    set_topmost(hwnd, True)
    rect = wt.RECT()
    user32.GetWindowRect(hwnd, ctypes.byref(rect))
    return ImageGrab.grab(bbox=(rect.left + 8, rect.top + 31, rect.right - 8, rect.bottom - 8), all_screens=True)


class Recorder:
    def __init__(self, hwnd: int):
        self.hwnd = hwnd
        self.log: list[tuple[str, str]] = []
        self.inset: Image.Image | None = None
        self.frames: list[Path] = []
        (OUT / "frames").mkdir(parents=True, exist_ok=True)

    def say(self, kind: str, text: str) -> None:
        self.log.append((kind, text))
        self.log = self.log[-30:]

    def frame(self, key: str | None = None) -> None:
        canvas = Image.new("RGB", CANVAS, (18, 20, 26))
        draw = ImageDraw.Draw(canvas)
        draw.text((16, 12), "Agent (MCP client)  ->  so101-robot MCP server", font=FONT_B, fill=(140, 200, 255))
        y = 48
        colors = {"step": (255, 214, 102), "call": (130, 220, 150), "result": (200, 200, 210), "warn": (255, 120, 120)}
        lines: list[tuple[str, tuple]] = []
        for kind, text in self.log:
            prefix = {"step": "## ", "call": "-> ", "result": "<- ", "warn": "!! "}[kind]
            wrapped = [text[i : i + 58] for i in range(0, len(text), 58)] or [""]
            for j, part in enumerate(wrapped):
                lines.append(((prefix if j == 0 else "   ") + part, colors[kind]))
        for text, color in lines[-34:]:
            draw.text((16, y), text, font=FONT, fill=color)
            y += 20

        shot = capture(self.hwnd)
        right_w, right_h = CANVAS[0] - PANEL_W - 16, CANVAS[1] - 50
        shot.thumbnail((right_w, right_h))
        canvas.paste(shot, (PANEL_W + 8, 44))
        draw.text((PANEL_W + 12, 12), "MuJoCo viewer (live sim)", font=FONT_B, fill=(140, 200, 255))
        if self.inset is not None:
            inset = self.inset.copy()
            inset.thumbnail((300, 225))
            x, y0 = CANVAS[0] - inset.width - 20, CANVAS[1] - inset.height - 20
            canvas.paste(inset, (x, y0))
            draw.rectangle((x - 2, y0 - 24, x + inset.width + 2, y0 + inset.height + 2), outline=(255, 214, 102), width=2)
            draw.text((x + 4, y0 - 22), "observe() -> front camera", font=FONT, fill=(255, 214, 102))

        path = OUT / "frames" / f"{len(self.frames):04d}.png"
        canvas.save(path)
        self.frames.append(path)
        if key:
            canvas.save(OUT / f"key_{key}.png")


class Agent:
    def __init__(self, session: ClientSession, rec: Recorder):
        self.s, self.rec = session, rec

    async def call(self, tool: str, args: dict | None = None, show_args: str | None = None):
        self.rec.say("call", f"{tool}({show_args if show_args is not None else json.dumps(args or {})})")
        result = await self.s.call_tool(tool, args or {})
        if result.isError:
            self.rec.say("warn", " ".join(c.text for c in result.content if c.type == "text"))
        return result

    async def run_job(self, request: dict, key: str | None = None, stop_after: float | None = None) -> dict:
        res = await self.call("jog_joints", {"request": request}, show_args=json.dumps(request))
        job = res.structuredContent
        self.rec.say("result", f"job {job['job_id']} state={job['state']}")
        started = time.monotonic()
        while True:
            self.rec.frame()
            await asyncio.sleep(1 / FPS_CAPTURE)
            job = (await self.s.call_tool("get_job", {"job_id": job["job_id"]})).structuredContent
            if stop_after is not None and time.monotonic() - started > stop_after and job["state"] == "running":
                stopped = (await self.call("stop")).structuredContent
                j = stopped["joints"]
                self.rec.say("result", f"stopped={stopped['stopped']} job={stopped['job']['state']} pan={j['shoulder_pan']}")
                self.rec.frame(key)
                return stopped["job"]
            if job["state"] in ("completed", "failed", "cancelled"):
                break
        errs = ", ".join(f"{e['joint']}={e['error']}" for e in job["final_errors"])
        self.rec.say("result", f"get_job -> {job['state']} reached_target={job['reached_target']} ({job['duration_s']}s)")
        if errs:
            self.rec.say("result", f"final_errors: {errs}")
        if job["notes"]:
            self.rec.say("result", f"notes: {', '.join(job['notes'])}")
        for _ in range(3):
            self.rec.frame()
        if key:
            self.rec.frame(key)
        return job


async def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    hwnd = find_viewer()
    set_topmost(hwnd, True)
    user32.SetForegroundWindow(hwnd)
    await asyncio.sleep(1.5)
    rec = Recorder(hwnd)
    try:
        await run_session(rec)
    finally:
        set_topmost(hwnd, False)

    frames = [Image.open(p).convert("P", palette=Image.ADAPTIVE) for p in rec.frames]
    frames[0].save(OUT / "demo.gif", save_all=True, append_images=frames[1:], duration=int(1000 / FPS_CAPTURE), loop=0)
    print(f"{len(frames)} frames -> {OUT / 'demo.gif'}")


async def run_session(rec: Recorder) -> None:
    async with streamable_http_client(URL) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            agent = Agent(session, rec)
            tools = [t.name for t in (await session.list_tools()).tools]
            rec.say("step", "Discover tools")
            rec.say("result", ", ".join(tools))

            rec.say("step", "1. Reset scene and check status")
            await agent.call("reset_scene", {"request": {"seed": 0}}, show_args='{"seed": 0}')
            st = (await agent.call("robot_status")).structuredContent
            rec.say("result", f"mode={st['mode']} hz={st['control_hz_measured']} cubes_in_box={st['task']['cubes_in_box']}")
            rec.say("result", f"joints pan={st['joints']['shoulder_pan']} lift={st['joints']['shoulder_lift']} elbow={st['joints']['elbow_flex']}")
            rec.frame("1_status")

            rec.say("step", "2. Turn the base 45 deg left")
            await agent.run_job({"mode": "relative", "shoulder_pan": 45, "speed_deg_per_s": 45}, key="2_pan")

            rec.say("step", "3. Raise the arm and open the gripper")
            await agent.run_job(
                {"mode": "absolute", "shoulder_lift": -15, "elbow_flex": 15, "wrist_flex": 0, "gripper": 95, "speed_deg_per_s": 50},
                key="3_raise",
            )

            rec.say("step", "4. Wave the wrist")
            for offset in (35, -70, 70, -35):
                await agent.run_job({"mode": "relative", "wrist_flex": offset, "speed_deg_per_s": 90})
            rec.frame("4_wave")

            rec.say("step", "5. Look with the camera (observe)")
            obs = await agent.call("observe", {"request": {"cameras": ["front"], "image_width": 640}}, show_args='{"cameras": ["front"]}')
            state = json.loads(next(c.text for c in obs.content if c.type == "text"))
            img = next(c for c in obs.content if c.type == "image")
            rec.inset = Image.open(io.BytesIO(base64.b64decode(img.data))).convert("RGB")
            rec.say("result", f"1 jpeg image + state: pan={state['joints']['shoulder_pan']} gripper={state['joints']['gripper']}")
            rec.frame("5_observe")

            rec.say("step", "6. Slow sweep right, then STOP mid-motion")
            await agent.run_job({"mode": "relative", "shoulder_pan": -120, "speed_deg_per_s": 20}, key="6_stop", stop_after=2.0)
            rec.inset = None

            rec.say("step", "7. Push into the table (verification test)")
            await agent.run_job(
                {"mode": "absolute", "shoulder_pan": 0, "shoulder_lift": 60, "elbow_flex": 0, "wrist_flex": 60, "speed_deg_per_s": 45},
                key="7_contact",
            )

            rec.say("step", "8. Reset scene")
            await agent.call("reset_scene", {"request": {"seed": 0}}, show_args='{"seed": 0}')
            await asyncio.sleep(0.3)
            rec.say("result", "scene and arm restored")
            for _ in range(4):
                rec.frame()
            rec.frame("8_reset")


if __name__ == "__main__":
    asyncio.run(main())
