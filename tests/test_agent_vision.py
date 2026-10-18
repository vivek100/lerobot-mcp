"""Target metadata, annotations, native discovery and bounded tracking on the real sim stack."""
import json
import time

import numpy as np
import pytest

from so101_mcp.runtime.tracking import ColorDetector


def test_color_bbox_is_normalized_and_selects_largest():
    frame = np.zeros((200, 400, 3), dtype=np.uint8)
    frame[40:120, 80:200] = [255, 0, 0]
    frame[150:160, 300:310] = [255, 0, 0]
    detection = ColorDetector("red")(frame)
    assert detection is not None
    assert detection.bbox == pytest.approx((0.2, 0.2, 0.5, 0.6), abs=0.01)


@pytest.mark.anyio
async def test_guide_prompt_and_detect_native_mcp(robot):
    prompts = await robot.session.list_prompts()
    assert "robot_operation_guide" in [p.name for p in prompts.prompts]
    prompt = await robot.session.get_prompt("robot_operation_guide")
    assert "Image-left is NOT necessarily robot-left" in prompt.messages[0].content.text
    guide = await robot.call("robot_guide")
    assert "RELATIVE centimetres" in guide.content[0].text
    result = await robot.call("detect_target", {"request": {"target": "color", "color": "red", "camera": "front"}})
    state = json.loads(next(c.text for c in result.content if c.type == "text"))
    assert state["target"]["visible"]
    assert len(state["target"]["bbox_xyxy"]) == 4
    assert len([c for c in result.content if c.type == "image"]) == 2
    await robot.call("detect_target", {"request": {"target": "color"}}, expect_error=True)


@pytest.mark.anyio
async def test_sim_projection_returns_estimate_and_offsets(robot):
    state = await robot.data("detect_target", {"request": {"target": "color", "color": "red", "camera": "front", "plane_height_cm": 1.5}})
    projection = state["target"]["projection"]
    assert projection["available"]
    assert projection["measured_depth"] is False
    assert projection["source"] == "simulator_calibration"
    assert 5 < projection["position_cm"]["forward"] < 50


def test_planar_projection_math_and_behind_camera_rejection():
    from so101_mcp.vision import project_to_plane

    geometry = {"vertical_fov_deg": 60, "position_m": [0.2, 0.1, 1.0], "rotation_camera_to_world": np.eye(3).tolist()}
    assert project_to_plane((0.5, 0.5), (640, 480), geometry, 0) == pytest.approx([0.2, 0.1, 0])
    assert project_to_plane((0.5, 0.5), (640, 480), geometry, 2) is None


@pytest.mark.anyio
async def test_tracking_native_mcp_reports_actual_outcome(robot):
    before = await robot.joint("shoulder_pan")
    state = await robot.data("track", {"request": {"target": "color", "color": "red", "camera": "front", "seconds": 5}})
    assert state["result"] == "tracked", state
    assert state["execution"]["state"] == "completed"
    assert state["execution"]["metrics"]["detections"] > 10
    assert state["execution"]["objective"] == "approximate_horizontal_pointing"
    assert state["execution"]["metrics"]["image_centering_applicable"] == 0
    assert "centered_detections" not in state["execution"]["metrics"]
    assert state["target"]["visible"]
    assert abs(state["joints"]["shoulder_pan"] - before) > 1


def test_wrist_servo_follows_moving_target_and_stops_on_loss():
    # Env.close() terminates process-global GLFW; keep this complete runtime lifecycle
    # separate from planner render tests and the session-scoped MCP server.
    import subprocess
    import sys
    from pathlib import Path

    path = Path(__file__).resolve()
    code = f"import runpy; runpy.run_path({str(path)!r})['_wrist_tracking_trial']()"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr


def _wrist_tracking_trial():
    """Independent runtime rig moves cube in sim; detector/servos/safety are production code."""
    from so101_mcp.backends import make_backend
    from so101_mcp.runtime import RobotRuntime

    rt = RobotRuntime(lambda: make_backend("sim", seed=0))
    rt.start()

    def wait(predicate, seconds):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(0.05)
        raise AssertionError("Timed out waiting for tracking condition")

    try:
        for i, name in enumerate(("green_cube", "blue_cube", "yellow_cube")):
            rt.place_object(name, -0.15, 0.25 + i * 0.05)
        rt.place_object("red_cube", 0.25, 0)
        look = [0, -20, 20, 75, 0, 30]
        job = rt.start_jog(dict(enumerate(look)), False, 60)
        wait(lambda: job.done, 10)
        assert job.state == "completed", job.error
        job = rt.start_track("color", "red", "wrist", 45)
        wait(lambda: job.metrics.get("calibrated") == 1 or job.done, 10)
        assert not job.done, job.error
        pans = []
        for path in ((0.04, 0.08), (0.04, 0, -0.04, -0.08)):
            for y in path:
                rt.place_object("red_cube", 0.25, y)
                time.sleep(1.5)
            time.sleep(1)
            assert job.state == "running", job.error
            assert abs(job.metrics["error_x"]) < 0.15, job.metrics
            assert abs(job.metrics["error_y"]) < 0.15, job.metrics
            pans.append(rt.status().joints[0])
        assert abs(pans[0] - pans[1]) > 15
        rt.place_object("red_cube", -0.15, -0.3)
        wait(lambda: job.done, 7)
        assert job.state == "failed" and "target lost" in job.error
        assert not [f for f in rt.status().faults if "target lost" in f]
    finally:
        rt.shutdown()
