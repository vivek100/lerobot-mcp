"""End-to-end: gestures, held states and talk through run_animation on the sim."""

from __future__ import annotations

import time

import pytest

from so101_mcp.models import ExpressState, GestureName
from so101_mcp.runtime.animation import LIBRARY

pytestmark = pytest.mark.anyio

JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


async def test_list_animations_matches_tool_schemas(robot):
    tools = {t.name for t in (await robot.session.list_tools()).tools}
    assert {"list_animations", "run_animation"} <= tools
    listing = await robot.data("list_animations")
    gestures = {a["name"] for a in listing["animations"] if a["kind"] == "gesture"}
    states = {a["name"] for a in listing["animations"] if a["kind"] == "state"}
    assert gestures == set(GestureName.__args__)
    assert states == set(ExpressState.__args__)
    assert set(listing["moods"]) == {"neutral", "happy", "excited", "sad", "curious"}


@pytest.mark.parametrize("name", ["nod", "wave", "tada"])
async def test_gesture_plays_and_ends_on_its_pose(robot, kinematics, name):
    result = await robot.data("run_animation", {"request": {"type": "gesture", "name": name}})
    assert result["result"] == "played", result
    assert result["table_clearance_cm"] > 3
    final = result["joints"]
    base = LIBRARY[name].base
    for i, j in enumerate(JOINT_NAMES):
        assert final[j] == pytest.approx(base[i], abs=6), f"{name} should end on its base pose ({j})"
    pose = [final[j] for j in JOINT_NAMES]
    lowest, _ = kinematics.lowest_point(pose)
    assert lowest > 0.03


async def test_repeat_plays_longer(robot):
    t0 = time.monotonic()
    await robot.data("run_animation", {"request": {"type": "gesture", "name": "nod"}})
    t1 = time.monotonic() - t0
    t0 = time.monotonic()
    await robot.data("run_animation", {"request": {"type": "gesture", "name": "nod", "repeat": 2}})
    t2 = time.monotonic() - t0
    assert t2 > t1 + LIBRARY["nod"].cycle_s * 0.5


async def test_state_holds_until_stop(robot):
    holding = await robot.data("run_animation", {"request": {"type": "state", "name": "thinking", "seconds": 30}})
    assert holding["result"] == "holding"
    joints = holding["joints"]
    assert joints["wrist_roll"] > 8 or abs(joints["wrist_flex"] - LIBRARY["thinking"].base[3]) > 3
    stopped = await robot.data("stop")
    assert stopped["result"] == "stopped"


async def test_talk_without_voice(robot):
    result = await robot.data(
        "run_animation",
        {"request": {"type": "talk", "text": "Hello. I am an SO-101.", "mood": "happy", "voice": False}},
    )
    assert result["result"] == "played"


async def test_invalid_expressive_requests_rejected(robot):
    await robot.call("run_animation", {"request": {"type": "gesture", "name": "backflip"}}, expect_error=True)
    await robot.call("run_animation", {"request": {"type": "state", "name": "wave"}}, expect_error=True)
    await robot.call("run_animation", {"request": {"type": "talk"}}, expect_error=True)
    await robot.call("run_animation", {"request": {"type": "talk", "text": "hi", "mood": "angry"}}, expect_error=True)
