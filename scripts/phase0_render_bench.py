"""Split the sim control-step cost into physics vs camera rendering, and find where render time goes.

Usage (PowerShell): uv run python scripts/phase0_render_bench.py
"""

from __future__ import annotations

import json
import time

import mujoco

from lerobot.envs.factory import make_env


def timeit(fn, n: int) -> float:
    fn()
    s = time.perf_counter()
    for _ in range(n):
        fn()
    return 1000 * (time.perf_counter() - s) / n


def main() -> None:
    envs = make_env("lerobot/so101-tabletop-mujoco", n_envs=1, trust_remote_code=True)
    env = next(iter(envs.values()))[0].envs[0].unwrapped
    env.reset(seed=0)
    world = env._ensure_world()
    m, d = world.model, world.data

    r = mujoco.Renderer(m, 480, 640)
    r.update_scene(d, camera="front")
    r.render()
    from OpenGL import GL  # noqa: PLC0415

    results = {
        "gl_vendor": GL.glGetString(GL.GL_VENDOR).decode(),
        "gl_renderer": GL.glGetString(GL.GL_RENDERER).decode(),
        "physics_ms_per_control_step": round(timeit(lambda: mujoco.mj_step(m, d, nstep=world.substeps), 300), 3),
        "ngeom": int(m.ngeom),
        "nmesh": int(m.nmesh),
        "update_scene_ms": round(timeit(lambda: r.update_scene(d, camera="front"), 30), 2),
        "render_ms_default": round(timeit(r.render, 30), 2),
    }
    r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
    r.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
    results["render_ms_no_shadow_no_reflection"] = round(timeit(r.render, 30), 2)
    r.close()
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
