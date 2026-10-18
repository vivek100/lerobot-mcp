"""Phase 0 probe: load the SO-101 tabletop MuJoCo sim through LeRobot, report its schema,
save camera frames, time the control step, and optionally play a wave in the 3D viewer.

Usage (PowerShell, from the repo root):
    uv run python scripts/phase0_sim_probe.py
    uv run python scripts/phase0_sim_probe.py --viewer
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from PIL import Image

from lerobot.envs.factory import make_env

HUB_PATH = "lerobot/so101-tabletop-mujoco"
OUT_DIR = Path("artifacts/phase0")


def load_env():
    envs = make_env(HUB_PATH, n_envs=1, trust_remote_code=True)
    suite = next(iter(envs.values()))
    vector_env = suite[0]
    return vector_env, vector_env.envs[0].unwrapped


def set_camera_shadows(env, enabled: bool) -> None:
    """Shadows cost ~40 ms per camera frame on this laptop's Intel GPUs regardless of resolution."""
    import mujoco

    flags = env._ensure_world().renderer.scene.flags
    flags[mujoco.mjtRndFlag.mjRND_SHADOW] = int(enabled)
    flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = int(enabled)


def describe_observation(obs: dict) -> dict:
    return {
        "agent_pos": {"shape": list(obs["agent_pos"].shape), "dtype": str(obs["agent_pos"].dtype)},
        "pixels": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in obs["pixels"].items()},
    }


def wave_action(home: np.ndarray, t: float) -> np.ndarray:
    """Raise the arm and wave the wrist; a placeholder until the animation library exists."""
    target = home.copy()
    raise_blend = min(t / 1.0, 1.0)
    target[1] = home[1] + raise_blend * 40.0          # shoulder_lift up
    target[2] = home[2] - raise_blend * 50.0          # elbow_flex open
    if t > 1.0:
        target[3] = 0.0 + 30.0 * math.sin(2 * math.pi * 1.2 * (t - 1.0))   # wrist_flex wave
        target[5] = 50.0 + 40.0 * math.sin(2 * math.pi * 2.4 * (t - 1.0))  # gripper "mouth"
    return target.astype(np.float32)


def run_probe(steps: int, shadows: bool) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    _, env = load_env()
    set_camera_shadows(env, shadows)
    obs, info = env.reset(seed=0)
    load_s = time.perf_counter() - t0

    world = env._ensure_world()
    joint_names = [f"{j}.pos" for j in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")]
    lo_deg = np.rad2deg(world.lo).round(1).tolist()
    hi_deg = np.rad2deg(world.hi).round(1).tolist()

    for cam, frame in obs["pixels"].items():
        Image.fromarray(frame).save(OUT_DIR / f"reset_{cam}.png")

    home = obs["agent_pos"].copy()
    step_times = []
    trace = []
    for i in range(steps):
        action = wave_action(home, i / env.fps)
        s = time.perf_counter()
        obs, reward, terminated, truncated, info = env.step(action)
        step_times.append(time.perf_counter() - s)
        trace.append({"t": round(i / env.fps, 3), "action": action.round(2).tolist(), "state": obs["agent_pos"].round(2).tolist()})
    for cam, frame in obs["pixels"].items():
        Image.fromarray(frame).save(OUT_DIR / f"after_wave_{cam}.png")

    tracking_err = np.abs(np.array(trace[-1]["action"]) - np.array(trace[-1]["state"]))
    report = {
        "hub_path": HUB_PATH,
        "task": env.task,
        "task_description": env.task_description,
        "fps": env.fps,
        "camera_shadows": shadows,
        "substeps_per_control_step": world.substeps,
        "joint_names": joint_names,
        "action_space": {"shape": list(env.action_space.shape), "low": float(env.action_space.low[0]), "high": float(env.action_space.high[0])},
        "observation": describe_observation(obs),
        "home_state": home.round(2).tolist(),
        "actuator_ctrlrange_deg_arm": dict(zip(joint_names[:5], zip(lo_deg[:5], hi_deg[:5]))),
        "gripper_units": "0-100 (mapped onto actuator range)",
        "load_and_reset_s": round(load_s, 2),
        "step_ms": {
            "mean": round(1000 * float(np.mean(step_times)), 2),
            "p95": round(1000 * float(np.percentile(step_times, 95)), 2),
            "max_realtime_fps": round(1.0 / float(np.mean(step_times)), 1),
        },
        "final_reward": float(reward),
        "final_info": {k: (v.item() if hasattr(v, "item") else v) for k, v in info.items()},
        "final_tracking_error_deg": tracking_err.round(2).tolist(),
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2))
    (OUT_DIR / "wave_trace.json").write_text(json.dumps(trace))
    env.close()
    return report


def run_viewer(seconds: float) -> None:
    import mujoco
    import mujoco.viewer

    _, env = load_env()
    obs, _ = env.reset(seed=0)
    world = env._ensure_world()
    home = obs["agent_pos"].copy()
    period = 1.0 / env.fps
    with mujoco.viewer.launch_passive(world.model, world.data) as viewer:
        start = time.perf_counter()
        i = 0
        while viewer.is_running() and time.perf_counter() - start < seconds:
            tick = time.perf_counter()
            action = wave_action(home, (i % (env.fps * 6)) / env.fps)
            world.data.ctrl[world.actuators] = world.to_ctrl(action)
            mujoco.mj_step(world.model, world.data, nstep=world.substeps)
            viewer.sync()
            i += 1
            time.sleep(max(0.0, period - (time.perf_counter() - tick)))
    env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--viewer", action="store_true", help="open the MuJoCo 3D viewer and loop a wave")
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--shadows", action="store_true", help="render observation cameras with shadows (slow)")
    args = parser.parse_args()
    if args.viewer:
        run_viewer(args.seconds)
    else:
        print(json.dumps(run_probe(args.steps, args.shadows), indent=2))


if __name__ == "__main__":
    main()
