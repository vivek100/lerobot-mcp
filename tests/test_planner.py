"""Planner unit tests: centimetre IK, remaining room, leaving the real rest pose."""

from __future__ import annotations

import numpy as np
import pytest

from so101_mcp.backends.real import REST_POSE
from so101_mcp.kinematics import REAL_TABLE_TOP_Z, ArmKinematics
from so101_mcp.planner import UNPARK, MotionPlanner

LOWER = np.array([-99.0, -101.0, -93.0, -96.0, -176.0, 0.0])
UPPER = np.array([99.0, 101.0, 93.0, 96.0, 176.0, 100.0])
CLEARANCE = 0.003


def test_unpark_from_real_rest_then_lift():
    planner = MotionPlanner()
    planned = planner.plan(REST_POSE.copy(), 0.0, 0.0, 0.08, 0.0, 0.0, LOWER, UPPER, CLEARANCE)
    assert planned.ok, planned.message
    assert len(planned.poses) >= 2
    assert np.allclose(planned.poses[0][:5], np.array([REST_POSE[0], *UNPARK[1:5]]), atol=1.0)
    raised_xyz = planner.gripper_xyz(planned.poses[0])
    goal_xyz = planner.gripper_xyz(planned.poses[-1])
    assert goal_xyz[2] == pytest.approx(raised_xyz[2] + 0.08, abs=0.015)
    assert planner.kin.lowest_point(planned.poses[-1])[0] >= CLEARANCE


def test_side_step_from_home_yaws():
    planner = MotionPlanner()
    seed = UNPARK.copy()
    planned = planner.plan(seed, 0.0, 0.04, 0.0, 0.0, 0.0, LOWER, UPPER, CLEARANCE)
    assert planned.ok, planned.message
    now = planner.describe_cm(seed)
    goal = planner.describe_cm(planned.poses[-1])
    assert goal[1] == pytest.approx(now[1] + 4, abs=1.5)


def test_hover_waypoint_does_not_change_requested_destination(monkeypatch):
    planner = MotionPlanner()
    start = planner.gripper_xyz(UNPARK)
    checks = iter([False, True, True])
    monkeypatch.setattr(planner, "_path_clear", lambda *args: next(checks))
    planned = planner.plan(UNPARK.copy(), 0.02, 0, 0, 0, 0, LOWER, UPPER, CLEARANCE)
    assert planned.ok, planned.message
    assert len(planned.poses) == 2
    assert planner.gripper_xyz(planned.poses[-1]) == pytest.approx(start + [0.02, 0, 0], abs=0.004)


def test_through_table_from_home_is_refused():
    planner = MotionPlanner()
    planned = planner.plan(UNPARK.copy(), 0.0, 0.0, -0.15, 0.0, 0.0, LOWER, UPPER, CLEARANCE)
    assert not planned.ok
    assert "table" in planned.message.lower()


def test_stretched_near_table_does_not_unpark():
    planner = MotionPlanner()
    seed = np.array([23.96, 52.31, -50.11, 70.81, 6.9, 42.2])
    planned = planner.plan(seed, 0.0, 0.0, 0.01, 0.0, 0.0, LOWER, UPPER, CLEARANCE)
    if planned.ok:
        assert not np.allclose(planned.poses[0][:5], np.array([seed[0], *UNPARK[1:5]]), atol=2.0)
    else:
        assert "rest pose" not in planned.message.lower()
        assert "left the rest pose" not in planned.message.lower()


def test_real_desk_offset_matches_ruler():
    kin = ArmKinematics(table_top_z=REAL_TABLE_TOP_Z)
    seed = np.array([23.96, 53.71, -49.67, 70.81, 6.9, 40.33])
    low, _ = kin.lowest_point(seed)
    assert low == pytest.approx(0.025, abs=0.005)


def test_reported_gripper_height_uses_real_table_plane():
    planner = MotionPlanner(ArmKinematics(table_top_z=REAL_TABLE_TOP_Z))
    expected = (planner.gripper_xyz(UNPARK)[2] - REAL_TABLE_TOP_Z) * 100
    assert planner.describe_cm(UNPARK)[2] == pytest.approx(expected, abs=0.1)


def test_real_desk_offset_allows_descend_from_stretch():
    planner = MotionPlanner(ArmKinematics(table_top_z=REAL_TABLE_TOP_Z))
    seed = np.array([23.96, 53.71, -49.67, 70.81, 6.9, 40.33])
    planned = planner.plan(seed, 0.0, 0.0, -0.01, 0.0, 0.0, LOWER, UPPER, CLEARANCE)
    assert planned.ok, planned.message
    assert not np.allclose(planned.poses[0][:5], np.array([seed[0], *UNPARK[1:5]]), atol=2.0)


def test_previews_leave_out_the_sim_bin_and_cubes():
    planner = MotionPlanner()
    frames = planner.previews(np.zeros(6), width=160)
    assert set(frames) == {"preview_front", "preview_side", "preview_top"}
    front = frames["preview_front"]
    green = (front[:, :, 1] > 140) & (front[:, :, 0] < 90) & (front[:, :, 2] < 90)
    assert float(green.mean()) < 0.002


def test_real_desk_offset_still_unparks_from_rest():
    planner = MotionPlanner(ArmKinematics(table_top_z=REAL_TABLE_TOP_Z))
    planned = planner.plan(REST_POSE.copy(), 0.0, 0.0, 0.08, 0.0, 0.0, LOWER, UPPER, CLEARANCE)
    assert planned.ok, planned.message
    assert len(planned.poses) >= 2
    assert np.allclose(planned.poses[0][:5], np.array([REST_POSE[0], *UNPARK[1:5]]), atol=1.0)
