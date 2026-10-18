"""End-to-end: an MCP client drives the SO-101 MuJoCo sim through the real server, runtime and backend."""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.anyio

EXPECTED_TOOLS = {
    "observe", "move", "set_gripper", "stop", "go_rest", "list_animations", "run_animation",
    "robot_guide", "detect_target", "track",
    "show_wrist_camera", "wrist_camera_frame",
}
JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def image_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean())


async def test_discovery_and_status(robot):
    tools = {t.name for t in (await robot.session.list_tools()).tools}
    assert tools == EXPECTED_TOOLS

    status = await robot.status()
    assert status["mode"] == "sim" and status["result"] == "ok"
    assert "front" in status["cameras"] and "wrist" in status["cameras"]
    assert status["pose"]["up_cm"] > 5
    down = next(r for r in status["room_left"] if r["direction"] == "down")
    assert down["remaining_cm"] >= 0.5


async def test_observe_returns_camera_images(robot):
    state, frames = await robot.observe(width=640, previews=True)
    assert state["mode"] == "sim"
    assert frames["front"].shape[2] == 3 and frames["wrist"].shape[2] == 3
    assert frames["front"].std() > 10, "front camera should show a textured scene, not a blank frame"
    assert "preview_front" in frames and "preview_side" in frames and "preview_top" in frames

    _, small = await robot.observe(cameras=["wrist"], width=320, previews=False)
    assert list(small) == ["wrist"] and small["wrist"].shape == (240, 320, 3)


async def test_move_changes_pose_and_is_verifiable(robot):
    before = await robot.status()
    _, before_frames = await robot.observe(cameras=["front"], width=320, previews=False)

    result = await robot.move(left_cm=4)
    assert result["result"] == "moved", result
    assert result["pose"]["left_cm"] == pytest.approx(before["pose"]["left_cm"] + 4, abs=1.5)

    _, after_frames = await robot.observe(cameras=["front"], width=320, previews=False)
    assert image_diff(before_frames["front"], after_frames["front"]) > 1.0

    back = await robot.move(left_cm=-4)
    assert back["result"] == "moved"
    assert back["pose"]["left_cm"] == pytest.approx(before["pose"]["left_cm"], abs=1.5)


async def test_set_gripper(robot):
    opened = await robot.data("set_gripper", {"request": {"openness": 80}})
    assert opened["result"] == "moved"
    assert opened["pose"]["gripper"] == pytest.approx(80, abs=5)
    closed = await robot.data("set_gripper", {"request": {"openness": 20}})
    assert closed["pose"]["gripper"] == pytest.approx(20, abs=5)


async def test_stop_cancels_held_state(robot):
    holding = await robot.data("run_animation", {"request": {"type": "state", "name": "listening", "seconds": 30}})
    assert holding["result"] == "holding"
    stopped = await robot.data("stop")
    assert stopped["result"] == "stopped"
    idle = await robot.status()
    assert idle["result"] == "ok"


async def test_table_guard_refuses_before_touching_table(robot, kinematics):
    start = await robot.status()
    down = next(r for r in start["room_left"] if r["direction"] == "down")
    refused = await robot.move(up_cm=-(down["remaining_cm"] + 4))
    assert refused["result"] == "refused"
    assert "table" in refused["message"].lower() or "reach" in refused["message"].lower()
    now = await robot.status()
    assert now["pose"]["up_cm"] == pytest.approx(start["pose"]["up_cm"], abs=1.5)
    pose = np.array([now["joints"][j] for j in JOINT_NAMES])
    lowest, link = kinematics.lowest_point(pose)
    assert lowest > 0.002, f"{link} got within {lowest * 100:.1f} cm of the table"


async def test_go_rest(robot):
    await robot.move(left_cm=5, forward_cm=3)
    rested = await robot.data("go_rest")
    assert rested["result"] == "rested"
    joints = rested["joints"]
    for joint, value in zip(JOINT_NAMES, (0, -60, 60, 35, 0, 30)):
        assert joints[joint] == pytest.approx(value, abs=5)


async def test_invalid_requests_rejected(robot):
    await robot.call("move", {"request": {}}, expect_error=True)
    await robot.call("move", {"request": {"up_cm": 50}}, expect_error=True)
    await robot.call("set_gripper", {"request": {"openness": 200}}, expect_error=True)
    await robot.call("observe", {"request": {"cameras": ["top"]}}, expect_error=True)
    await robot.call("run_animation", {"request": {"type": "gesture", "name": "backflip"}}, expect_error=True)
