"""mcp-use MCP server exposing the SO-101 runtime to agents.

Run (PowerShell, repo root):
    uv run so101-mcp --backend sim --viewer
    uv run so101-mcp --backend real
Then connect an MCP client to http://127.0.0.1:8000/mcp (Cursor: see .cursor/mcp.json).
"""

from __future__ import annotations

import os

os.environ.setdefault("MCP_USE_ANONYMIZED_TELEMETRY", "false")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import argparse  # noqa: E402
import asyncio  # noqa: E402
import base64  # noqa: E402
import io  # noqa: E402
import signal  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Optional, get_args  # noqa: E402

from mcp import types as mcp_types  # noqa: E402
from mcp.server.fastmcp import Image  # noqa: E402
from mcp.types import ToolAnnotations  # noqa: E402
from mcp_use.server import MCPServer  # noqa: E402
from PIL import Image as PILImage  # noqa: E402

from so101_mcp.backends import make_backend  # noqa: E402
from so101_mcp.joints import JOINT_KEYS, JOINTS  # noqa: E402
from so101_mcp.models import (  # noqa: E402
    DetectRequest,
    TrackRequest,
    AnimationInfo,
    AnimationList,
    ExpressState,
    GestureName,
    GripperPose,
    GripperRequest,
    JointDetail,
    JointValues,
    MoveRequest,
    ObserveRequest,
    PlayRequest,
    PoseSnapshot,
    RoomDir,
)
from so101_mcp.runtime import RobotRuntime  # noqa: E402
from so101_mcp.view_log import ViewLog  # noqa: E402
from so101_mcp.runtime.animation import LIBRARY  # noqa: E402
from so101_mcp.runtime.runtime import Snapshot  # noqa: E402
from so101_mcp.runtime.talk import MOOD_BODY, prepare_talk  # noqa: E402
from so101_mcp.runtime.tracking import ColorDetector, FaceDetector, yunet_path  # noqa: E402
from so101_mcp.guide import ROBOT_GUIDE  # noqa: E402
from so101_mcp.vision import project_to_plane  # noqa: E402

INSTRUCTIONS = "Start with robot_guide for coordinates, camera interpretation and target selection.\n" + ROBOT_GUIDE
server = MCPServer(name="so101-robot", version="0.3.0", instructions=INSTRUCTIONS, dns_rebinding_protection=True)
runtime: RobotRuntime | None = None
mute_audio = False
view_log: ViewLog | None = None

READ_ONLY = ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False)
MOTION = ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=True)
WRIST_APP_URI = "ui://so101/wrist-camera.html"


@server.resource(WRIST_APP_URI, name="wrist_camera_view", mime_type="text/html;profile=mcp-app",
                 meta={"ui": {"prefersBorder": True, "csp": {"connectDomains": [], "resourceDomains": []}}})
def wrist_camera_view() -> str:
    """Sandboxed read-only camera viewer; frames travel through the MCP App host bridge."""
    return (Path(__file__).parent / "ui" / "wrist_camera.html").read_text(encoding="utf-8")


@server.tool(annotations=READ_ONLY, meta={"ui": {"resourceUri": WRIST_APP_URI}, "openai/outputTemplate": WRIST_APP_URI}, structured_output=False)
def show_wrist_camera() -> mcp_types.CallToolResult:
    """Open a live wrist-camera MCP App for the user. Read-only: does not start tracking or move the arm.

    Requires an MCP Apps-compatible host; otherwise open the returned local browser viewer.
    The live UI is for monitoring; use observe/detect_target for agent decisions.
    """
    return mcp_types.CallToolResult(
        content=[mcp_types.TextContent(type="text", text="Live wrist-camera viewer. If this client cannot render MCP Apps, open http://127.0.0.1:8765/index.html. Viewing does not move the arm.")],
        structuredContent={"camera": "wrist", "mode": _rt().status().mode, "viewer_url": "http://127.0.0.1:8765/index.html", "read_only": True},
    )


@server.tool(annotations=READ_ONLY, meta={"ui": {"visibility": ["app"]}}, structured_output=False)
async def wrist_camera_frame() -> mcp_types.CallToolResult:
    """Internal read-only MCP App frame refresh. Use show_wrist_camera to open the viewer."""
    frame = await asyncio.to_thread(_rt().camera_frame, "wrist")
    encoded = _encode(frame, 640)
    return mcp_types.CallToolResult(content=[mcp_types.ImageContent(type="image", data=base64.b64encode(encoded.data).decode("ascii"), mimeType="image/jpeg")])


def _rt() -> RobotRuntime:
    if runtime is None:
        raise RuntimeError("Robot runtime is not running")
    return runtime


def _joints(values) -> JointValues:
    return JointValues(**{name: round(float(v), 2) for name, v in zip(JOINTS, values)})


def _joint_details(values, limits) -> list[JointDetail]:
    out = []
    for name, value, lo, hi in zip(JOINTS, values, limits.lower, limits.upper):
        margin = min(float(value) - float(lo), float(hi) - float(value))
        out.append(JointDetail(
            joint=name, position_deg=round(float(value), 2),
            lower_deg=round(float(lo), 1), upper_deg=round(float(hi), 1),
            margin_deg=round(float(margin), 1),
        ))
    return out


def _encode(frame, width: int) -> Image:
    img = PILImage.fromarray(frame)
    if img.width != width:
        img = img.resize((width, round(img.height * width / img.width)), PILImage.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return Image(data=buf.getvalue(), format="jpeg")


@server.custom_route("/camera/frame/{camera}", methods=["GET"])
async def camera_frame_route(request):
    """Read-only local camera capture for video; does not command motion."""
    from starlette.responses import Response

    camera = request.path_params["camera"]
    if camera not in _rt().status().cameras:
        return Response("Unknown camera", status_code=404)
    frame = await asyncio.to_thread(_rt().camera_frame, camera)
    return Response(_encode(frame, 640).data, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


def _pose_payload(snap: Snapshot, result: str, message: str, requested: str | None = None, achieved: str | None = None, extra_cameras: list[str] | None = None) -> PoseSnapshot:
    cameras = list(snap.frames)
    if extra_cameras:
        cameras = cameras + extra_cameras
    return PoseSnapshot(
        result=result,  # type: ignore[arg-type]
        message=message,
        mode=snap.mode,  # type: ignore[arg-type]
        requested=requested,
        achieved=achieved,
        pose=GripperPose(
            forward_cm=snap.forward_cm, left_cm=snap.left_cm, up_cm=snap.up_cm,
            tilt_down_deg=snap.tilt_down_deg, roll_deg=snap.roll_deg,
            gripper=round(float(snap.joints[5]), 1),
        ),
        room_left=[RoomDir(direction=n, remaining_cm=cm, limited_by=why) for n, cm, why in snap.room],  # type: ignore[arg-type]
        joints=_joints(snap.joints),
        joint_details=_joint_details(snap.joints, _rt().limits),
        table_clearance_cm=round(snap.lowest_m * 100, 1),
        warnings=[],
        faults=list(snap.faults),
        cameras=cameras,
    )


def _observe_content(
    snap: Snapshot, result: str, message: str, image_width: int,
    requested: str | None = None, achieved: str | None = None,
    tool: str = "", request: dict | None = None,
    target: dict | None = None, execution: dict | None = None, view_mode: str = "focused",
) -> list:
    extra = list(snap.previews)
    if view_mode == "focused":
        extra = [name for name in extra if name == "preview_side"]
    payload = _pose_payload(snap, result, message, requested, achieved, extra_cameras=extra)
    payload.target = target
    payload.execution = execution
    images = [_encode(snap.frames[name], image_width) for name in snap.frames]
    images += [_encode(snap.previews[name], 320) for name in extra]
    payload_json = payload.model_dump_json(exclude_none=True)
    if view_log is not None:
        names = list(snap.frames) + extra
        verification = None
        if snap.mode == "sim":
            verification = {"sim_objects_m": {name: xyz.tolist() for name, xyz in _rt().status().objects.items()}}
        view_log.write(tool, request, payload_json, [(name, image.data or b"") for name, image in zip(names, images)], verification=verification)
    return [payload_json, *images]


async def _wait_job(job, timeout: float = 90.0):
    deadline = __import__("time").monotonic() + timeout
    while not job.done and __import__("time").monotonic() < deadline:
        await asyncio.sleep(0.05)
    if not job.done:
        await asyncio.to_thread(_rt().stop)
        raise TimeoutError("Motion wait expired; stop requested. Observe the actual pose before continuing.")
    return job


@server.tool(annotations=READ_ONLY)
def robot_guide() -> str:
    """Read first: base-fixed directions, camera interpretation, safe moves, target selection and tracking."""
    return ROBOT_GUIDE


@server.prompt(name="robot_operation_guide", description="How to interpret SO-101 views and use movement and tracking tools.")
def robot_operation_guide() -> str:
    return ROBOT_GUIDE


def _detect_snapshot(snap: Snapshot, request: DetectRequest) -> dict:
    import cv2

    if request.target == "color" and request.color is None:
        raise ValueError("target=color requires color (red, yellow, green or blue)")
    if request.target == "face" and request.color is not None:
        raise ValueError("color only applies to target=color")
    if request.camera not in snap.frames:
        raise ValueError(f"Camera {request.camera!r} unavailable; choose from {list(snap.frames)}")
    detector = ColorDetector(request.color) if request.target == "color" else FaceDetector(yunet_path())
    frame = snap.frames[request.camera].copy()
    detection = detector(frame)
    info = {
        "camera": request.camera, "kind": request.target, "color": request.color,
        "visible": detection is not None, "selection": "largest_match_each_frame",
        "coordinate_frame": "image_normalized_top_left", "image_size_px": [frame.shape[1], frame.shape[0]],
        "warning": "May switch between matching targets. Pixel positions are not robot-space coordinates.",
    }
    if detection is not None:
        info.update(bbox_xyxy=detection.bbox, center_error_xy=[detection.x, detection.y], score=detection.score)
        if request.plane_height_cm is not None:
            geometry = snap.camera_geometry.get(request.camera)
            info["projection"] = {"available": False, "reason": "Camera is not calibrated; no world coordinates inferred."}
            if geometry is not None:
                point = project_to_plane(
                    ((detection.x + 1)/2, (detection.y + 1)/2), (frame.shape[1], frame.shape[0]),
                    geometry, _rt().planner.kin.table_top_z + request.plane_height_cm/100,
                )
                if point is not None:
                    fwd, left = float(point[0]*100), float(point[1]*100)
                    info["projection"] = {
                        "available": True, "source": geometry["source"], "method": "ray_to_assumed_horizontal_plane",
                        "assumed_height_cm": request.plane_height_cm, "measured_depth": False,
                        "position_cm": {"forward": round(fwd, 2), "left": round(left, 2), "up": request.plane_height_cm},
                        "delta_from_gripper_cm": {"forward": round(fwd-snap.forward_cm, 2), "left": round(left-snap.left_cm, 2), "up": round(request.plane_height_cm-snap.up_cm, 2)},
                        "warning": "Estimate depends on chosen height and visible centroid; verify with wrist image before closing.",
                    }
                else:
                    info["projection"]["reason"] = "Camera ray does not intersect that plane in front of the camera."
        if detection.bbox:
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = detection.bbox
            cv2.rectangle(frame, (int(x1*w), int(y1*h)), (min(w-1, int(x2*w)), min(h-1, int(y2*h))), (0, 255, 160), 2)
    label = f"{request.camera}: {request.color or request.target} {'VISIBLE' if detection else 'NOT VISIBLE'}"
    cv2.putText(frame, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
    cv2.putText(frame, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 160), 1)
    snap.frames[request.camera] = frame
    return info


@server.tool(annotations=READ_ONLY, structured_output=False)
async def detect_target(request: DetectRequest) -> list:
    """Locate the largest matching color blob or face without moving. Returns a normalized box and annotated camera image.

    Call before track to verify the intended target. This is not arbitrary-object selection or a 3D grasp pose.
    """
    snap = await asyncio.to_thread(_rt().snapshot, False)
    target = await asyncio.to_thread(_detect_snapshot, snap, request)
    return _observe_content(snap, "ok", "Target visible." if target["visible"] else "Target not visible; aim camera before tracking.",
                            640, tool="detect_target", request=request.model_dump(), target=target)


@server.tool(annotations=MOTION, structured_output=False)
async def track(request: TrackRequest) -> list:
    """Follow the largest matching face or color for a bounded duration; waits and returns outcome, metrics and annotated images.

    Wrist: first makes ~5 degree calibration nudges, then centers the target. Front: approximate horizontal pointing only.
    Use detect_target first. No approach or grasp, no instance lock. Lost target stops after 3 seconds. stop cancels.
    """
    rt = _rt()
    snap = await asyncio.to_thread(rt.snapshot, False)
    target = await asyncio.to_thread(_detect_snapshot, snap, request)
    if not target["visible"]:
        return _observe_content(snap, "refused", "Target not visible; no tracking motion started.", 640,
                                tool="track", request=request.model_dump(), target=target)
    job = await asyncio.to_thread(rt.start_track, request.target, request.color, request.camera, request.seconds, request.invert)
    job = await _wait_job(job, request.seconds + 15)
    snap = await asyncio.to_thread(rt.snapshot, True)
    target = await asyncio.to_thread(_detect_snapshot, snap, request)
    result = "tracked" if job.state == "completed" else ("stopped" if job.state == "cancelled" else "failed")
    execution = {
        "state": job.state, "reached_target": job.reached,
        "objective": "center_target_in_wrist_image" if request.camera == "wrist" else "approximate_horizontal_pointing",
        "reached_target_meaning": "final image error under 0.2 on each normalized axis" if request.camera == "wrist" else "pan within 3 degrees of approximate bearing; not image centering or calibrated 3D alignment",
        "metrics": dict(job.metrics), "notes": list(job.notes), "error": job.error,
    }
    return _observe_content(snap, result, job.error or f"Tracking {job.state}; inspect reached_target and camera image.", 640,
                            tool="track", request=request.model_dump(), target=target, execution=execution)


@server.tool(annotations=READ_ONLY, structured_output=False)
async def observe(request: Optional[ObserveRequest] = None) -> list:
    """Look at the arm and the table. Returns pose, remaining room in each direction, then labelled images.

    Images follow the JSON in `cameras` order: physical cameras first (front = laptop / scene,
    wrist = jaws), then optional MuJoCo previews of the current joints (preview_front, preview_side,
    preview_top) so you can see the whole arm even when the wrist camera is aimed at the table.
    """
    request = request or ObserveRequest()
    rt = _rt()
    snap = await asyncio.to_thread(rt.snapshot, False if request.previews is False else True)
    if request.cameras:
        unknown = [c for c in request.cameras if c not in snap.frames]
        if unknown:
            raise ValueError(f"Unknown cameras {unknown}; available: {list(snap.frames)}")
        snap.frames = {c: snap.frames[c] for c in request.cameras}
    width = request.image_width or 640
    return _observe_content(snap, "ok", "Current pose and camera views.", width, tool="observe", request=request.model_dump(exclude_none=True), view_mode="all")


@server.tool(annotations=MOTION, structured_output=False)
async def move(request: MoveRequest) -> list:
    """Move the gripper in centimetres from where it is now. Waits until the arm stops, then returns a new observe.

    Omit any axis you do not want to change. The server picks the joint angles and a fixed safe speed.
    result=refused means nothing moved (out of reach, would hit the table, or a motor is busy).
    """
    deltas = {
        "forward_cm": request.forward_cm, "left_cm": request.left_cm, "up_cm": request.up_cm,
        "tilt_down_deg": request.tilt_down_deg, "roll_deg": request.roll_deg,
    }
    if all(v is None for v in deltas.values()):
        raise ValueError("Specify at least one of forward_cm, left_cm, up_cm, tilt_down_deg, roll_deg")
    requested = ", ".join(f"{k.replace('_', ' ')}={v}" for k, v in deltas.items() if v is not None)
    rt = _rt()
    job, message, ok = await asyncio.to_thread(
        rt.start_cartesian,
        (request.forward_cm or 0.0) / 100.0,
        (request.left_cm or 0.0) / 100.0,
        (request.up_cm or 0.0) / 100.0,
        request.tilt_down_deg or 0.0,
        request.roll_deg or 0.0,
    )
    if not ok or job is None:
        snap = await asyncio.to_thread(rt.snapshot, True)
        return _observe_content(
            snap, "refused", message, 640, requested, "no motion",
            tool="move", request={k: v for k, v in deltas.items() if v is not None},
        )
    job = await _wait_job(job)
    snap = await asyncio.to_thread(rt.snapshot, True)
    achieved = (
        f"forward {snap.forward_cm} cm, left {snap.left_cm} cm, up {snap.up_cm} cm, "
        f"tilt {snap.tilt_down_deg}°, roll {snap.roll_deg}°"
    )
    result = "moved" if job.state == "completed" else ("stopped" if job.state == "cancelled" else "refused")
    note = message if message != "ok" else (job.error or "Move finished.")
    if job.state == "failed":
        note = job.error or note
        result = "failed"
    return _observe_content(
        snap, result, note, 640, requested, achieved,
        tool="move", request={k: v for k, v in deltas.items() if v is not None},
    )


@server.tool(annotations=MOTION, structured_output=False)
async def set_gripper(request: GripperRequest) -> list:
    """Open or close the jaws only. 0 = closed, 100 = open. Waits, then returns a new observe."""
    rt = _rt()
    job = await asyncio.to_thread(rt.start_gripper, request.openness)
    job = await _wait_job(job)
    snap = await asyncio.to_thread(rt.snapshot, True)
    result = "moved" if job.state == "completed" else ("stopped" if job.state == "cancelled" else "failed")
    return _observe_content(
        snap, result, job.error or f"Gripper {job.state}: requested {request.openness:.0f}, measured {snap.joints[5]:.0f}. Closure alone does not verify a grasp.",
        640, f"openness={request.openness:.0f}", f"gripper={snap.joints[5]:.0f}",
        tool="set_gripper", request={"openness": request.openness},
    )


@server.tool(annotations=MOTION, structured_output=False)
async def stop() -> list:
    """Cancel the active motion immediately; the arm holds. Returns a new observe."""
    rt = _rt()
    job = await asyncio.to_thread(rt.stop)
    snap = await asyncio.to_thread(rt.snapshot, True)
    if job is None:
        return _observe_content(snap, "ok", "Nothing was moving.", 640, tool="stop", request={})
    return _observe_content(snap, "stopped", f"Cancelled {job.kind} {job.id}.", 640, tool="stop", request={})


@server.tool(annotations=MOTION, structured_output=False)
async def go_rest() -> list:
    """Lift, then fold to the rest pose (on the real arm, on the table). Waits, then returns a new observe."""
    rt = _rt()
    job = await asyncio.to_thread(rt.start_rest)
    job = await _wait_job(job, timeout=60.0)
    snap = await asyncio.to_thread(rt.snapshot, True)
    result = "rested" if job.state == "completed" else ("stopped" if job.state == "cancelled" else "failed")
    return _observe_content(
        snap, result, job.error or "At rest." if job.state == "completed" else f"go_rest {job.state}", 640,
        tool="go_rest", request={},
    )


@server.tool(annotations=READ_ONLY)
async def list_animations() -> AnimationList:
    """Gestures, held states, and how to speak. Pass these to run_animation."""
    return AnimationList(
        animations=[
            AnimationInfo(
                name=a.name, kind="state" if a.loopable else "gesture",
                description=a.description, seconds=a.cycle_s,
            )
            for a in LIBRARY.values()
        ],
        moods=list(MOOD_BODY),
        how_to_play=(
            'gesture: {"type":"gesture","name":"wave"} (optional repeat). '
            'state: {"type":"state","name":"thinking","seconds":8} — returns immediately; stop or the next motion ends it. '
            'talk: {"type":"talk","text":"Hello","mood":"happy"} — waits until the speech finishes.'
        ),
    )


@server.tool(annotations=MOTION, structured_output=False)
async def run_animation(request: PlayRequest) -> list:
    """Play a gesture, hold a state, or speak. Gestures and talk wait until finished; states return immediately."""
    rt = _rt()
    if request.type == "talk":
        if not request.text:
            raise ValueError("talk requires text")
        voice = True if request.voice is None else request.voice
        script = await asyncio.to_thread(
            prepare_talk, request.text, request.mood or "neutral", rt.status().fps, voice, None, mute_audio
        )
        try:
            job = await asyncio.to_thread(rt.start_talk, script)
        except BaseException:
            script.finish()
            raise
        job = await _wait_job(job, timeout=min(script.duration + 15, 90))
        snap = await asyncio.to_thread(rt.snapshot, True)
        return _observe_content(
            snap, "played" if job.state == "completed" else ("stopped" if job.state == "cancelled" else "failed"),
            job.error or f"Speech {job.state}.", 640, request.text[:80],
            tool="run_animation", request=request.model_dump(exclude_none=True),
        )
    if request.type == "gesture":
        name = request.name or ""
        if name not in get_args(GestureName):
            raise ValueError(f"Unknown gesture {name!r}; see list_animations")
        job = await asyncio.to_thread(rt.start_animation, name, request.repeat or 1)
        job = await _wait_job(job)
        snap = await asyncio.to_thread(rt.snapshot, True)
        return _observe_content(
            snap, "played" if job.state == "completed" else ("stopped" if job.state == "cancelled" else "failed"),
            job.error or f"Gesture {name} {job.state}.", 640, name,
            tool="run_animation", request=request.model_dump(exclude_none=True),
        )
    name = request.name or ""
    if name not in get_args(ExpressState):
        raise ValueError(f"Unknown state {name!r}; see list_animations")
    job = await asyncio.to_thread(rt.start_animation, name, 1, request.seconds or 8.0)
    snap = await asyncio.to_thread(rt.snapshot, True)
    return _observe_content(
        snap, "holding",
        f"Holding {name} ({job.id}) until stop or the next motion.",
        640, name,
        tool="run_animation", request=request.model_dump(exclude_none=True),
    )


def _warm_tool_cache() -> None:
    handler = server._mcp_server.request_handlers[mcp_types.ListToolsRequest]
    asyncio.run(handler(mcp_types.ListToolsRequest(method="tools/list")))


def main() -> None:
    global runtime, mute_audio, view_log
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="SO-101 MCP server")
    parser.add_argument("--backend", choices=["sim", "real"], default="sim")
    parser.add_argument("--viewer", action="store_true", help="open the MuJoCo 3D viewer (sim only)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--serial-port", default="COM3", help="real arm motor bus")
    parser.add_argument("--robot-id", default="follower_01", help="real arm calibration id")
    parser.add_argument("--no-front-camera", action="store_true", help="real arm: don't open the laptop webcam")
    parser.add_argument("--mute", action="store_true", help="talk: synthesize speech for the mouth but don't play it")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--debug", action="store_true", help="enable mcp-use inspector at /inspector")
    parser.add_argument("--save-views", default=None, help="folder for each reply's JSON and camera JPEGs")
    args = parser.parse_args()
    mute_audio = args.mute
    view_log = ViewLog(Path(args.save_views)) if args.save_views else None

    if args.backend == "real":
        cameras = {"wrist": "Innomaker"} if args.no_front_camera else {"wrist": "Innomaker", "front": "Integrated Webcam"}

        def factory():
            return make_backend("real", port=args.serial_port, robot_id=args.robot_id, cameras=cameras)

        from so101_mcp.backends.real import REAL_SPEED

        runtime = RobotRuntime(factory, speed=REAL_SPEED, reach_tolerance_deg=4.0)
    else:
        runtime = RobotRuntime(lambda: make_backend("sim", seed=args.seed), viewer=args.viewer)
    runtime.start()
    status = runtime.status()
    print(f"[so101-mcp] {status.mode} backend '{status.backend}' ready; joints={dict(zip(JOINT_KEYS, status.joints.round(1).tolist()))}", flush=True)
    print(f"[so101-mcp] MCP endpoint: http://{args.host}:{args.port}/mcp", flush=True)
    _warm_tool_cache()
    for sig in (signal.SIGTERM, getattr(signal, "SIGBREAK", None)):
        if sig is not None:
            signal.signal(sig, _raise_keyboard_interrupt)
    try:
        server.run(transport="streamable-http", host=args.host, port=args.port, debug=args.debug)
    except KeyboardInterrupt:
        pass
    finally:
        print("[so101-mcp] shutting down" + (" (parking the arm first)" if status.mode == "real" else ""), flush=True)
        runtime.shutdown()
        print("[so101-mcp] stopped", flush=True)


def _raise_keyboard_interrupt(signum, frame) -> None:
    raise KeyboardInterrupt


if __name__ == "__main__":
    main()
