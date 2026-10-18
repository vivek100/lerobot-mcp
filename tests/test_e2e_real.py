"""End-to-end on the REAL arm through the MCP server. Opt-in: $env:SO101_REAL = "1".

The arm must start folded at rest on the table, powered, with someone watching it and a hand near the
motor power switch. Camera frames from every step are saved to artifacts/e2e/real/ as evidence.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytestmark = pytest.mark.anyio

JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
OUT = Path(__file__).resolve().parents[1] / "artifacts" / "e2e" / "real"


async def snap(robot, name: str) -> dict:
    state, frames = await robot.observe(width=640, previews=False)
    OUT.mkdir(parents=True, exist_ok=True)
    for cam, frame in frames.items():
        Image.fromarray(frame).save(OUT / f"{name}_{cam}.jpg")
    return state


async def test_real_arm_session(real_robot, kinematics):
    robot = real_robot
    status = await robot.status()
    assert status["mode"] == "real"
    assert "wrist" in status["cameras"]
    await snap(robot, "00_rest")

    raised = await robot.move(up_cm=8)
    assert raised["result"] == "moved", raised
    await snap(robot, "01_raised")

    start_left = (await robot.status())["pose"]["left_cm"]
    panned = await robot.move(left_cm=4)
    if panned["result"] == "moved":
        assert panned["pose"]["left_cm"] == pytest.approx(start_left + 4, abs=2.5)
        await snap(robot, "02_left")
        await robot.move(left_cm=-4)

    opened = await robot.data("set_gripper", {"request": {"openness": 70}})
    assert opened["result"] == "moved"
    await snap(robot, "03_gripper_open")
    await robot.data("set_gripper", {"request": {"openness": 20}})

    start = await robot.status()
    refused = await robot.move(up_cm=-12)
    assert refused["result"] == "refused"
    assert "table" in refused["message"].lower() or "reach" in refused["message"].lower()
    now = await robot.status()
    pose = np.array([now["joints"][j] for j in JOINT_NAMES])
    lowest, link = kinematics.lowest_point(pose)
    await snap(robot, "04_table_guard_stop")
    assert lowest > -0.01, f"{link} at {lowest * 100:.1f} cm by the model"
    assert now["pose"]["up_cm"] == pytest.approx(start["pose"]["up_cm"], abs=2)

    rested = await robot.data("go_rest")
    assert rested["result"] == "rested", rested
    await snap(robot, "05_back_at_rest")


async def test_real_arm_expressive(real_robot, kinematics):
    robot = real_robot
    await snap(robot, "10_rest")

    for i, name in enumerate(["wave", "nod", "tada"]):
        result = await robot.data("run_animation", {"request": {"type": "gesture", "name": name}})
        assert result["result"] == "played", result
        await snap(robot, f"1{i + 1}_{name}")

    holding = await robot.data("run_animation", {"request": {"type": "state", "name": "thinking", "seconds": 20}})
    assert holding["result"] == "holding"
    await snap(robot, "14_thinking")

    talked = await robot.data(
        "run_animation",
        {"request": {"type": "talk", "text": "Hello! I am your robot arm. Nice to meet you.", "mood": "happy"}},
    )
    assert talked["result"] == "played", talked
    await snap(robot, "15_talk")

    faults = talked.get("faults") or []
    assert not [f for f in faults if "contact stop" in f or "overheat" in f], faults

    rested = await robot.data("go_rest")
    assert rested["result"] == "rested", rested
    await snap(robot, "16_back_at_rest")
