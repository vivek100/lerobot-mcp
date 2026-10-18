"""End-to-end: visual tracking through the MCP server on the sim (cubes moved with place_object)."""

from __future__ import annotations

import asyncio
import time

import numpy as np
import pytest

pytestmark = [pytest.mark.anyio, pytest.mark.skip(reason="legacy async tracking/place_object surface; current blocking track covered by test_agent_vision.py")]

JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
# Wrist camera looking down at the table in front of the arm.
LOOK_DOWN = {"shoulder_pan": 0, "shoulder_lift": -20, "elbow_flex": 20, "wrist_flex": 75, "wrist_roll": 0, "gripper": 30}


async def place(robot, name: str, x: float, y: float) -> dict:
    return await robot.data("place_object", {"request": {"name": name, "x": x, "y": y}})


async def clear_table(robot, keep: str) -> None:
    """Park every other cube at the back-left corner, out of the cameras' view."""
    others = [n for n in ("red_cube", "green_cube", "blue_cube", "yellow_cube") if n != keep]
    for i, name in enumerate(others):
        await place(robot, name, -0.15, 0.25 + 0.05 * i)


def metrics(job: dict) -> dict[str, float]:
    return {m["name"]: m["value"] for m in job["metrics"]}


async def wait_for(robot, job_id: str, predicate, timeout: float) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = await robot.data("get_job", {"job_id": job_id})
        if predicate(job):
            return job
        await asyncio.sleep(0.1)
    raise AssertionError(f"condition not met within {timeout}s: {job}")


async def test_place_object_reports_ground_truth(robot):
    status = await place(robot, "blue_cube", 0.3, -0.1)
    blue = next(o for o in status["task"]["objects"] if o["name"] == "blue_cube")
    assert blue["x"] == pytest.approx(0.3, abs=0.005) and blue["y"] == pytest.approx(-0.1, abs=0.005)
    assert {o["name"] for o in status["task"]["objects"]} == {"red_cube", "green_cube", "blue_cube", "yellow_cube"}


async def test_wrist_tracking_follows_a_moving_cube(robot):
    await clear_table(robot, "red_cube")
    await place(robot, "red_cube", 0.25, 0.0)
    await robot.jog(mode="absolute", speed_deg_per_s=60, **LOOK_DOWN)

    job = await robot.data("track", {"request": {"target": "color", "color": "red", "camera": "wrist", "seconds": 40}})
    assert job["kind"] == "track"
    job = await wait_for(robot, job["job_id"], lambda j: metrics(j).get("calibrated") == 1.0, 10)
    assert any("image Jacobian" in n for n in job["notes"]) or job["state"] == "running"

    pans = []
    for path in ((0.04, 0.08), (0.04, 0.0, -0.04, -0.08)):
        for y in path:
            await place(robot, "red_cube", 0.25, y)
            await asyncio.sleep(1.5)
        await asyncio.sleep(1.0)
        job = await robot.data("get_job", {"job_id": job["job_id"]})
        assert job["state"] == "running", job
        m = metrics(job)
        assert abs(m["error_x"]) < 0.12 and abs(m["error_y"]) < 0.12, f"cube at y={y} should be re-centred: {m}"
        pans.append(await robot.joint("shoulder_pan"))
    assert abs(pans[0] - pans[1]) > 15, f"the base should turn to follow the cube: {pans}"

    stopped = await robot.data("stop")
    assert stopped["job"]["state"] == "cancelled"
    assert metrics(stopped["job"])["centered_detections"] > 20


async def test_tracking_fails_when_target_not_visible(robot):
    await clear_table(robot, "blue_cube")
    await place(robot, "blue_cube", -0.15, -0.3)  # behind the arm
    await robot.jog(mode="absolute", speed_deg_per_s=60, **LOOK_DOWN)
    job = await robot.data("track", {"request": {"target": "color", "color": "blue", "seconds": 20}})
    job = await robot.data("get_job", {"job_id": job["job_id"], "wait_seconds": 10})
    assert job["state"] == "failed" and job["error"] == "target not visible", job


async def test_tracking_fails_after_target_lost(robot):
    await clear_table(robot, "yellow_cube")
    await place(robot, "yellow_cube", 0.25, 0.0)
    await robot.jog(mode="absolute", speed_deg_per_s=60, **LOOK_DOWN)
    job = await robot.data("track", {"request": {"target": "color", "color": "yellow", "seconds": 40}})
    await wait_for(robot, job["job_id"], lambda j: metrics(j).get("calibrated") == 1.0, 10)

    await place(robot, "yellow_cube", -0.15, -0.3)
    lost_at = time.monotonic()
    job = await wait_for(robot, job["job_id"], lambda j: j["state"] != "running", 8)
    assert job["state"] == "failed" and "target lost" in job["error"], job
    assert 2.5 < time.monotonic() - lost_at < 6
    assert not [f for f in (await robot.status())["faults"] if "target lost" in f], "a lost target is not a safety fault"


async def test_fixed_camera_turns_toward_target(robot, kinematics):
    """front camera (fixed, facing the arm): the arm pans toward the side the cube is on."""
    await clear_table(robot, "red_cube")
    await robot.jog(mode="absolute", shoulder_pan=0, shoulder_lift=-45, elbow_flex=35, wrist_flex=15, speed_deg_per_s=60)
    sides = []
    for y in (0.15, -0.15):
        await place(robot, "red_cube", 0.25, y)
        job = await robot.data("track", {"request": {"target": "color", "color": "red", "camera": "front", "seconds": 4}})
        job = await robot.data("get_job", {"job_id": job["job_id"], "wait_seconds": 10})
        assert job["state"] == "completed" and job["reached_target"] is True, job
        status = await robot.status()
        gripper_y = kinematics.gripper_position(np.array([status["joints"][j] for j in JOINT_NAMES]))[1]
        sides.append(gripper_y)
        assert np.sign(gripper_y) == np.sign(y) and abs(gripper_y) > 0.03, f"cube at y={y}, gripper at y={gripper_y:.3f}"


async def test_invalid_track_requests_rejected(robot):
    await robot.call("track", {"request": {"target": "cat"}}, expect_error=True)
    await robot.call("track", {"request": {"target": "color", "color": "purple"}}, expect_error=True)
    await robot.call("track", {"request": {"target": "face", "camera": "top"}}, expect_error=True)
    await robot.call("place_object", {"request": {"name": "red_cube", "x": 2.0, "y": 0}}, expect_error=True)
