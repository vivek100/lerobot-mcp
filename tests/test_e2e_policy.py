"""End-to-end: a pretrained LeRobot ACT checkpoint drives the sim arm through the MCP server.

The checkpoint was trained on another SO-101 MuJoCo scene, so it is not expected to solve our task; the
test proves the pipeline (observation format, camera mapping, chunked async inference, safety path).
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

pytestmark = [pytest.mark.anyio, pytest.mark.skip(reason="list_policies/start_policy were removed from the agent MCP surface")]

JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
POLICY = "act_cube_sim_chaenn"


def metrics(job: dict) -> dict[str, float]:
    return {m["name"]: m["value"] for m in job["metrics"]}


async def test_list_policies(robot):
    listing = await robot.data("list_policies")
    policy = next(p for p in listing["policies"] if p["name"] == POLICY)
    assert policy["repo"] == "Chaenn/act_policy_so101_cube_multitask_sim_0827"
    assert policy["cameras"] == ["front", "wrist"] and policy["real_ok"] is False


async def test_policy_runs_with_async_chunks_inside_the_safety_envelope(robot, kinematics):
    start = await robot.status()
    job = await robot.data("start_policy", {"request": {"name": POLICY, "seconds": 10}})
    assert job["kind"] == "policy"

    poses = []
    while True:
        status = await robot.status()
        poses.append([status["joints"][j] for j in JOINT_NAMES])
        job = await robot.data("get_job", {"job_id": job["job_id"]})
        if job["state"] != "running" and job["state"] != "accepted":
            break
        await asyncio.sleep(0.1)

    assert job["state"] == "completed", job
    m = metrics(job)
    assert m["chunks"] >= 3, "a new chunk is computed before the current one runs out"
    assert m["p50_ms"] < 1000, "ACT inference on this CPU should take well under a second"
    poses = np.array(poses)
    assert np.ptp(poses[:, :5], axis=0).max() > 15, "the policy should move the arm"
    lowest = min(kinematics.lowest_point(p)[0] for p in poses)
    assert lowest > -0.004, f"the arm got {lowest * 100:.1f} cm into the table"
    end = await robot.status()
    assert end["overruns"] - start["overruns"] <= 5, "inference must not stall the 30 Hz control loop"
    assert end["control_hz_measured"] > 25


async def test_stop_cancels_policy(robot):
    job = await robot.data("start_policy", {"request": {"name": POLICY, "seconds": 30}})
    await asyncio.sleep(2.0)
    stopped = await robot.data("stop")
    assert stopped["job"]["job_id"] == job["job_id"] and stopped["job"]["state"] == "cancelled"
    held = stopped["joints"]
    await asyncio.sleep(1.0)
    now = (await robot.status())["joints"]
    for j in JOINT_NAMES[:5]:
        assert now[j] == pytest.approx(held[j], abs=3), "the arm holds after the policy is stopped"


async def test_unknown_policy_rejected(robot):
    await robot.call("start_policy", {"request": {"name": "pi0_magic"}}, expect_error=True)
