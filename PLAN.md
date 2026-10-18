# LeRobot MCP — Implementation Plan

Companion to [`lerobot_mcp_codex_brief.md`](lerobot_mcp_codex_brief.md). The brief defines *what*; this plan answers the open questions and sets the order of work.

**Current interface and demo work:** see [AGENT_USABILITY.md](AGENT_USABILITY.md). This plan contains historical design proposals (including the older async job tools) and results; current tools use blocking movement feedback, guidance, target detection and tracking. Native Codex MCP attachment has now been tested directly.

Evidence labels used below: **verified** (checked in source/docs on 2026-10-17), **documented** (stated in official docs, not yet tested here), **estimate** (reasoned expectation, must be benchmarked), **proposal** (design choice requiring validation).

## Decisions so far (2026-10-17)

| Topic | Decision |
|---|---|
| Hardware | SO-101 **follower only** (no leader arm) with **one camera attached**. Laptop webcam is available as a second camera. Details follow once the arm is connected. |
| Order | **Simulation first.** Build and test the runtime, MCP tools, animations, talk mode, data recording, and a learned pick-and-drop policy in MuJoCo, then switch the backend to the real arm. |
| MCP clients | **Cursor** for development and testing; **ChatGPT** as the second client. |
| Policies | Prefer existing checkpoints where they match; expect to train our own ACT for both sim and real. |

---

## 0. This laptop (measured)

| Item | Value |
|---|---|
| CPU | Intel i7-13700H — 14 cores / 20 threads |
| RAM | 31.7 GB |
| GPU | Intel Arc A370M (4 GB) + Intel UHD; **no NVIDIA / CUDA** |
| OS / shell | Windows 11 (build 26200), PowerShell |
| Python | 3.12.12 and 3.11 available via `uv` 0.10.0; `python` not on PATH |
| Cameras | Integrated webcam + IR webcam only |
| Serial ports | None yet (arm never connected) |

LeRobot `main` is v0.6.2 and needs **Python ≥ 3.12** (**verified**, `pyproject.toml`). Latest PyPI stable is v0.6.1 (**documented**). Python 3.12 via `uv` is fine.

---

## 1. Can this laptop run 1 or 2 local policies?

**Short answer:** Yes for **ACT** — it can keep two policies loaded and run one at a time. **SmolVLA** might run on the CPU but needs benchmarking. **Pi0 / Pi0.5** need a remote GPU.

| Policy | Size | Local on this laptop? | Notes |
|---|---|---|---|
| ACT | ~50 M params, ResNet-18 backbone | **Yes (estimate)** | Expect roughly 50–200 ms per inference on CPU, depending on camera count and resolution. ACT predicts a chunk of actions (default 100), so it only needs to run every few seconds at 30 fps. Use LeRobot async inference, even locally, so the arm does not pause while a new chunk is computed. |
| SmolVLA | ~450 M params | **Maybe (estimate)** | Expect roughly 1–3 s per chunk on CPU. That may work if a chunk covers more time than it takes to compute the next one. Benchmark before relying on it. |
| Pi0 / Pi0.5 | ~3 B params | **No** | Use a remote GPU with the LeRobot policy server. |

**"Two policies":** Memory is not a problem: ACT uses about 0.2 GB and SmolVLA about 2 GB in fp32. Only **one policy can drive the arm at a time**, because there is one motion owner (brief §9). Two policies can stay *loaded* so switching is instant, for example `pick_cube_act` and `stack_act`. Running a vision tracker at the same time competes for CPU, so cap `torch.set_num_threads()`.

**Intel Arc GPU (proposal):** PyTorch has native `xpu` wheels for Windows, so the Arc GPU could speed up ACT or SmolVLA. However, the 4 GB of VRAM is small, and I have not confirmed in source that LeRobot accepts `--policy.device=xpu`. OpenVINO on the Arc GPU is a stronger option for the tracking detectors (§2). Treat both as optimizations.

**Major caveat (estimate):** Prehosted SO-101 checkpoints are trained on someone else's cameras, scene, lighting, and calibration, so they usually transfer poorly. The realistic route is to record about 50 of our own episodes, train ACT on a rented cloud GPU for a few hours, then run inference locally.

**Existing checkpoints are still worth trying, in sim first:**
- Running a foreign checkpoint in simulation is risk-free. Even when it fails the task, it proves the pipeline end to end: observation format, preprocessing, action chunks, and joint order.
- Hub search shows ACT checkpoints labelled as sim-trained, for example `Chaenn/act_policy_so101_cube_multitask_sim_0827` and `Chaenn/act_policy_so101_multitask_newsim_pick_place_stack_onehot`. Their simulators are unknown, so check the model cards against §1a.
- Real-data checkpoints such as those trained on `lerobot/svla_so101_pickplace` are worth trying on the real arm only if the camera placement can be reproduced.

**Benchmark task (Phase 5 entry gate):** Write a script that loads a candidate ACT checkpoint, feeds it dummy frames with the checkpoint's camera shapes, and reports p50/p95 latency on CPU and, if available, `xpu`.

---

## 1a. Simulation first: one runtime, two backends

### Is it "the same software" for sim and the real arm?
**Mostly yes, by design.**
- The official SO-101 sim uses the **same joint names and units as the real arm**:
  - `shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex` and `wrist_roll` in degrees
  - `gripper` from 0 to 100
  - 30 fps control
- These match LeRobot's `so101_follower` with `use_degrees=True` (**documented**, model card).
- Policies, animations, talk mode, tracking, MCP tools, and the recorder all speak those units, so they don't need to know which backend is active.

What is *not* shared out of the box:
- LeRobot's real arm is a `Robot` class with `get_observation()` and `send_action()`.
- The sim is a Gymnasium env with `reset()` and `step()`, loaded with `make_env`.
- The model card mentions `lerobot-env-server` and `--robot.type=sim` to bridge the two. Neither is in LeRobot `main` today: there is no `robots/sim` folder and no script entry (**verified**, repo tree and `pyproject.toml`, 2026-10-17).

So we write a ~100-line **`SimBackend` adapter** that makes the gym env look like the real robot. If LeRobot ships `robot.type=sim`, we switch to it.

```python
class RobotBackend(Protocol):                    # proposal
    mode: Literal["sim", "real"]
    def connect(self) -> None: ...
    def disconnect(self) -> None: ...
    def get_observation(self) -> dict: ...       # {"shoulder_pan.pos": float, ..., "front": HxWx3, "wrist": HxWx3}
    def send_action(self, action: dict) -> dict: # {"shoulder_pan.pos": float, ...} -> action actually applied
        ...
    def reset(self, seed: int | None) -> None: ...        # sim: new scene; real: move to rest pose
    def task_signals(self) -> dict: ...                   # sim: reward, cubes_in_box, success; real: empty

RealBackend -> wraps lerobot SO101Follower(use_degrees=True)
SimBackend  -> wraps make_env("lerobot/so101-tabletop-mujoco"); maps agent_pos[i] <-> joint names,
               steps physics once per control tick (30 Hz real time, or faster when generating data)
```

### Phase 0 results (2026-10-17, this laptop, native Windows — no WSL needed)
- **Environment:** `uv sync` installs LeRobot 0.6.1, MuJoCo 3.3.7, and torch 2.11 CPU on Python 3.12.12. `make_env("lerobot/so101-tabletop-mujoco", trust_remote_code=True)` loads the env (**verified**, run).
- **Schema parity with the real arm (verified in source and run):**
  - Sim joint names are `shoulder_pan.pos` … `gripper.pos`. The installed `so_follower.py` builds `f"{motor}.pos"` keys.
  - `config_so_follower.py` defaults to `use_degrees: bool = True`, and the gripper uses `MotorNormMode.RANGE_0_100`.
  - Sim units are degrees with a 0–100 gripper.
  - `max_relative_target` exists in the real config as a per-step safety cap.
- **Sim details:**
  - Observation: `agent_pos` float32[6], plus `pixels.front` and `pixels.wrist` as uint8[480,640,3].
  - Action: float32[6] absolute targets.
  - 30 fps control with 7 physics substeps.
  - Home pose: about `(0, -60, 60, 35, 0, 30)`.
- **Sim actuator ranges (degrees):** pan ±110, lift ±100, elbow ±96.8, wrist_flex ±95, wrist_roll −157…+163. The real arm's ranges come from its own calibration and must be read separately.
- **Performance:**
  - Physics: 0.2 ms per control step.
  - Rendering one camera **with shadows** costs about 50–63 ms on both Intel UHD and Arc A370M, independent of resolution. Shadow-map size does not change it.
  - Rendering **without shadows/reflections** costs about 10.5 ms per camera.
  - Full step with 2 cameras and shadows off: **20.8 ms (48 fps max)**. With shadows: 127 ms (8 fps).
  - **Decision:** observation cameras render without shadows; the 3D viewer keeps full visuals. Datasets and policies must use the same render settings at train and run time.
- **GPU preference:** Python is set to the Arc GPU via `HKCU\Software\Microsoft\DirectX\UserGpuPreferences` (`GpuPreference=2;`) for the uv base interpreter and `.venv\Scripts\python.exe`. Gain is small (about 10 ms with shadows); remove the two values to revert.
- **Torch:** `2.11.0+cpu` with 14 threads. `torch.xpu.is_available()` is False because this is the CPU build. Testing the Arc GPU for policy inference would need the separate PyTorch XPU wheel index in a second environment, deferred to Phase 3 benchmarking.
- **Viewer:** `mujoco.viewer.launch_passive` runs on Windows from a stepping loop, using `scripts/phase0_sim_probe.py --viewer`.
- **Artifacts:** `artifacts/phase0/` holds `report.json`, `wave_trace.json`, and reset/after-wave PNGs for both cameras.

### Phase 1 results (2026-10-17)
- **Package:** `src/so101_mcp`, a mcp-use `MCPServer` over streamable HTTP with tools `robot_status`, `observe`, `jog_joints`, `get_job`, `stop`, `reset_scene`. Registered in `.cursor/mcp.json`.
- **Runtime:**
  - One control thread creates the backend, because MuJoCo GL contexts and serial ports are thread-bound.
  - It runs a 30 Hz loop: drain commands, read joints, behavior target, SafetyLimiter, command, publish status.
  - MCP handlers are async and only post commands to the thread. Cameras render on demand on the control thread.
- **Sim backend:** Steps the env's MuJoCo world directly and renders only when asked. The env's own `step` renders both cameras on every call. Env revision pinned to `12e504e…`.
- **Safety:**
  - One motion owner at a time.
  - Soft limits are the actuator range minus 3°.
  - Speed is capped at 90 °/s for arm joints and 150 units/s for the gripper.
  - Non-finite targets are rejected.
  - A blocked motion reports `reached_target: false` and the runtime then holds the measured pose instead of pushing.
- **No mocks (decision):** All tests are end-to-end. A real server subprocess runs on the sim and is driven by the official MCP client. 10 tests pass in about 25 s (`uv run pytest -q`).
- **Upstream issues found and worked around:**
  1. mcp-use 1.7.1 + mcp 1.30: a `tools/call` before `tools/list` fails with `'NoneType' object has no attribute 'params'`. The CallTool cache refresh invokes the wrapped ListTools handler with `None`. Fix: warm the tool cache at startup (`_warm_tool_cache`).
  2. The mcp-use startup banner crashes on Windows when stdout is redirected, because of the cp1252 encoding. Fix: reconfigure stdout/stderr to UTF-8.
- **Pydantic note:** `ConfigDict(exclude_none=True)` is accepted but ignored by Pydantic. `StrictModel.model_dump` honors it, so structured tool output omits `None` fields.

### Which simulator
| Env | Use it for | Why |
|---|---|---|
| [`lerobot/so101-tabletop-mujoco`](https://huggingface.co/lerobot/so101-tabletop-mujoco) (official) | **Default backend:** viewer, animations, talk, tracking, recording, MCP testing | Real-arm units; `front` + `wrist` 480×640 cameras; 4 cubes + box; reward and `cubes_in_box` give ground-truth success; Apache-2.0 |
| [`namenu/so101-pick-place-env`](https://huggingface.co/namenu/so101-pick-place-env) | **First learned policy** (single pick-and-drop) | One cube, 200-step episodes, explicit success rule. Grasp is *assisted*: the cube attaches when the gripper closes nearby, so it is easy to learn but not physically honest. 20 Hz, cameras `up`/`side`. |

Both load with `lerobot.envs.factory.make_env(..., trust_remote_code=True)` (**documented**). The official env's 4-cube, 2-minute task is long-horizon. We'll start with "put **one** cube in the box" using its `make_env` overrides, or use the namenu env.

### 3D viewer on Windows
Use `mujoco.viewer.launch_passive` for a live 3D window while the runtime steps physics. The agent gets camera renders from `get_observation()` as images. Rerun is optional for timelines. On Windows `MUJOCO_GL` is not needed; on headless Linux it would be `egl`.

### How a model picks and drops in sim without a leader arm
1. **Scripted expert:** Because the sim knows the true cube pose, a script plans approach, grasp, lift, move, and release. MuJoCo-model inverse kinematics turns that into joint targets.
2. **Automatic recording:** Run the expert for about 100 to 200 seeded episodes. The Recorder saves them as a `LeRobotDataset`, keeping successful episodes only.
3. **Training:** Run `lerobot-train --policy.type=act` on a cloud GPU. A few hours is typical (estimate).
4. **Local rollout:** Run the trained policy on the laptop CPU through the MCP `start_policy` tool. The sim's success signal verifies the result.

This exercises the exact real-arm pipeline: observation, policy, action chunk, `send_action`, recording, and verification.

### What sim will not prove
- Real calibration, servo latency and backlash, gripper friction (especially with the namenu assisted grasp), lighting, and camera appearance.
- **A policy trained only in sim will not transfer reliably to the real arm.** Plan on real demos for the real task.
- Real demos without a leader arm are possible in two ways:
  - Gamepad or keyboard end-effector control, reusing the MuJoCo model's IK. This avoids `placo` on Windows.
  - Kinesthetic teaching: turn torque off, move the follower by hand, and record. The operator's hand appears in camera view.
- **Match camera layout.** To make the sim useful for real, configure the sim cameras to resemble the real setup: the one attached camera, plus the laptop webcam if used.

---

## 2. Face / object tracking — how

Tracking is a **reactive local control loop**, not a learned policy. The pipeline is: camera frame, then detector, then target pixel, then a bounded joint-delta controller, then `send_action`. It runs at 15–30 Hz in a local thread. The agent only starts it, stops it, and inspects it.

### Camera placement decides the controller
- **Eye-in-hand (recommended for "follow"):** Mount a small USB camera on the gripper or wrist. Common SO-101 wrist-camera mounts exist. The controller keeps the target centered in the image, so **no inverse kinematics or camera calibration is needed**:
  - horizontal error → `shoulder_pan`
  - vertical error → `wrist_flex` (and optionally `elbow_flex`)
- **Eye-to-hand (fixed camera, such as the laptop webcam):** Requires mapping pixels to joint angles. A rough approximation of pan from horizontal pixel offset is enough for "look at me" if the camera sits next to the robot base. The integrated webcam in the laptop lid works for **offline detector testing**, not for good control.
- Policies (ACT/SmolVLA) usually want **two cameras: front + wrist**. Buying one wrist camera and one fixed USB camera serves both tracking and training.

### Detectors (all CPU real-time, estimate)
| Target | Library | Cost |
|---|---|---|
| Colored ball | OpenCV HSV threshold + contour | ~1 ms; start here |
| Face | OpenCV `FaceDetectorYN` (YuNet ONNX) or MediaPipe Face Detection | ~5–15 ms |
| Hand | MediaPipe Hands | ~15–30 ms |
| Generic objects ("the cup") | YOLO11n / YOLOv8n (Ultralytics; AGPL licence) optionally exported to OpenVINO for Arc | ~20–50 ms CPU |
| Re-acquire between detections | OpenCV CSRT/KCF or ByteTrack | cheap |

### Controller sketch (proposal)
```
err_x, err_y = (target_px - image_center) / half_size       # in [-1, 1]
if |err| < deadband (≈0.05): hold
delta_pan  = clamp(-Kp * err_x, ±max_step_deg)              # e.g. max 2°/tick at 20 Hz
delta_wfx  = clamp(+Kp * err_y, ±max_step_deg)
target     = clamp(current + EMA(delta), soft_joint_limits)
lost target > 0.5 s → hold;  > 3 s → end job, optionally return to rest
```
LeRobot's `max_relative_target` on the follower config also caps each step (**documented**; confirm field name in `so_follower` config).

### Testing tracking in sim (proposal)
- **Object follow:** In the tabletop scene, a script moves a colored cube along a path. The tracker uses the sim **wrist** camera. This tests the full controller, including target-loss handling.
- **Face follow (hybrid mode):** The detector reads the **laptop webcam** while commands drive the **sim arm**. You can move in front of the laptop and watch the virtual arm turn toward you before the real arm is involved. Pixel-to-pan mapping is approximate here, the same as with a fixed camera on the real arm.

### Cartesian option (later)
LeRobot has kinematics support (`placo`, SO-101 URDF) for end-effector control (**documented**). Check that `placo` installs on Windows before depending on it.

---

## 3. How the agent controls the arm, gets feedback, and creates training data

### Architecture (proposal)

```
MCP client (Cursor / Claude / ChatGPT / our own mcp-use MCPAgent)
        │  MCP (streamable-http, localhost)
        ▼
mcp-use Python MCPServer  ── thin: validation, job API, images, notifications
        │  in-process calls
        ▼
Robot Runtime (single process, owns the arm)
  ├─ MotionOwner lock     – exactly one active job per arm
  ├─ SafetyLimiter        – joint soft limits, max step, speed, stale-action watchdog
  ├─ Control thread       – fixed-rate loop (30 Hz) calling robot.send_action()
  ├─ Camera grabber       – latest frames, shared by tracker / policy / recorder
  ├─ Behaviours           – Animation | Expressive(talk) | Tracking | Policy | DirectJog
  ├─ Recorder             – LeRobotDataset writer, opt-in per job
  └─ Backend              – real SO101Follower | MuJoCo sim
```

Why Python: LeRobot is Python, and mcp-use's Python `MCPServer` supports `@server.tool()`, resources, notifications, stdio and streamable-http (**verified**, docs). The LLM never times motors. It picks a behavior, and the runtime's loop executes it.

### Tool pattern: jobs, not blocking calls
1. `start_*` tools validate inputs, take the motion lock, and return a `job_id` immediately.
2. `get_job(job_id)` returns `state` (accepted, running, completed, failed, cancelled), progress, faults, and verification evidence.
3. `stop()` is always available and never needs the lock.
4. `observe(include_images)` returns joint state plus camera frames as MCP **image content**, so a multimodal agent can see the scene.
5. `verify(job_id)` returns before/after frames, joint deltas, and task-specific checks. "Command accepted", "execution completed", and "task succeeded" are reported separately.
6. Progress is pushed with MCP notifications where the client supports them; otherwise the agent polls.

Candidate tools (**not frozen**): `robot_status`, `observe`, `list_capabilities`, `run_animation`, `express` / `talk` (§5), `start_tracking`, `start_policy`, `jog_joints` (bounded), `set_gripper`, `stop`, `get_job`, `verify`, `start_recording`, `stop_recording`, `label_episode`, `list_datasets`, `register_policy`.

Tool inputs and outputs are Pydantic models that follow the project rules. Example:
```python
from typing import Literal, Optional
from pydantic import BaseModel, ConfigDict, Field

class StartTrackingInput(BaseModel):
    model_config = ConfigDict(extra="forbid", exclude_none=True)
    target: Literal["ball", "face", "hand"] = Field(..., description="What to follow")
    color: Optional[Literal["red", "green", "blue", "yellow"]] = None
    max_seconds: int = Field(..., ge=1, le=120, description="Hard time limit")
    record: bool = Field(..., description="Save this run as a dataset episode")
```

### Training data (proposal; LeRobot APIs **documented**, verify in source)
- Any job can set `record=true`. The Recorder writes frames at a fixed fps into a `LeRobotDataset` using `observation.state`, `observation.images.<cam>`, `action`, and `task`, then calls `save_episode()` with metadata.
- Episode success labels come from `verify()` plus human confirmation through `label_episode`.
- Data sources, best first:
  1. **Leader-arm teleop**: best for manipulation policies. Requires an SO-101 leader arm.
  2. **Policy rollouts**: successful runs can be added back to the dataset; failures help with evaluation.
  3. **Tracking runs**: valid data for learning a "follow" policy.
  4. **Animations / agent jog**: useful for the motion library, poor for manipulation learning.
- Pipeline: dataset → push to HF Hub → `lerobot-train` on a cloud GPU → evaluate in sim or on held-out episodes → `register_policy` stores checkpoint, cameras, features, fps, calibration id, and eval results in a local registry (JSON/SQLite). Only registered policies appear in `list_capabilities`.
- Windows note: dataset video encoding needs `ffmpeg` and Windows `torchcodec ≥ 0.7` (**verified**, `pyproject.toml`).

### mcp-use: what to use where
| Need | mcp-use piece | Language |
|---|---|---|
| Robot MCP server | `mcp_use.server.MCPServer` | Python (**verified**) |
| Debugging tools | built-in Inspector at `/inspector` | Python (**verified**) |
| Our own agent loop (voice / auto "thinking" / "talking" gestures) | `MCPClient` + `MCPAgent` | Python (**documented**) |
| Rich in-chat UI (live camera, joint sliders, job timeline) | **MCP Apps** widgets | **TypeScript only** (**verified**, docs index) |

Plan: build the Python server first. Use Rerun or a small web page for live visualization. If in-chat UI is wanted later, add a TypeScript mcp-use MCP App that calls the Python runtime over HTTP.

### Connecting the two clients (proposal; confirm current client docs at setup time)
- **Cursor (dev loop):** Run the server with `streamable-http` on `http://localhost:8000/mcp` and register it in the project's `.cursor/mcp.json`. Cursor agents can call tools and receive image content, so `observe` shows the sim render or camera frame in chat.
- **ChatGPT:** ChatGPT connects to remote MCP servers over **public HTTPS**, so the local server needs a tunnel such as `cloudflared` or `ngrok`, plus auth before exposing anything that moves the real arm. ChatGPT also renders MCP Apps, which makes the TypeScript widget phase more worthwhile there.
- **Safety:**
  - Physical-motion tools require `mode == "real"` and a human-armed flag set locally, not by the agent.
  - In sim mode, everything is allowed.

---

## 4. First-time arm setup on this Windows laptop

Do this whenever the arm is first connected; it can run **in parallel with** the sim phases. Phase 6 depends on it. LeRobot's guide is written for Mac/Linux, so the commands below are PowerShell equivalents. Use backticks for line continuation and `COMx` ports.

### 4.1 Prerequisites
```powershell
winget install --id Gyan.FFmpeg -e
winget install --id Git.Git -e
```
- The motor controller board uses a USB-serial chip, often CH340/CH343. If no new COM port appears in Device Manager → *Ports (COM & LPT)*, install the WCH CH343/CH340 driver.
- Waveshare board: set both jumpers to channel **B (USB)** (**documented**).
- Use the power supply that matches the motors (5 V standard vs 12 V "Pro" kits). Put a switch on the power line within reach as a physical **e-stop**; MCP `stop()` does not replace it.

### 4.2 Python environment
```powershell
cd d:\projectsnew\makesomethingagent
uv venv --python 3.12 .venv
.\.venv\Scripts\Activate.ps1
uv pip install "lerobot[feetech]==0.6.1"
```
Pin the stable release for hardware work and upgrade deliberately. Windows PyPI `torch` wheels are CPU-only, which is correct here.

### 4.3 Bring-up sequence
```powershell
# 1. Find the arm's COM port (unplug USB when prompted)
lerobot-find-port

# 2. ONLY if motors are new/unconfigured (IDs not yet set) — one motor at a time
lerobot-setup-motors --robot.type=so101_follower --robot.port=COM5

# 3. Calibrate (middle-of-range pose, then sweep every joint through full range)
lerobot-calibrate `
  --robot.type=so101_follower `
  --robot.port=COM5 `
  --robot.id=follower_01

# 4. Cameras
lerobot-find-cameras opencv

```
No leader arm, so `lerobot-teleoperate` with `so101_leader` does not apply. Teleop for real demos comes from our own gamepad, keyboard, or kinesthetic recorder (§1a).
Calibration is saved under `%USERPROFILE%\.cache\huggingface\lerobot\calibration\` (**documented**; confirm path). Keep `--robot.id` stable because policies depend on matching calibration.

### 4.4 Our own smoke tests (Phase 0 deliverables)
1. **Read-only:** Connect, print `get_observation()` joint values at 10 Hz while moving the arm by hand. Confirm joint names, units, and direction signs.
2. **Single-joint nudge:** Move `shoulder_pan` ±5° at low speed with `max_relative_target` set, then return.
3. **Rest pose:** Move to a known safe rest pose and disable torque on exit.

Windows gotchas: COM numbers change when you switch USB ports; OpenCV camera opening on Windows can be slow (try the MSMF vs DSHOW backends); close other apps that use the webcam.

---

## 5. Expressive "talking" animations (new)

The arm can act as an embodied presence for the agent, like a desk lamp or puppet that moves while ChatGPT talks.

### Motion vocabulary (proposal)
- **Gripper as a mouth:** Open and close the gripper in sync with speech amplitude. This is the cheapest, clearest effect.
- **Idle / attentive:** Slow breathing motion using small sinusoid or noise motion on `wrist_flex` and `shoulder_lift`.
- **Thinking:** Tilt the head with `wrist_roll` and look up and away while the LLM is generating.
- **Gestures:** Nod (yes), shake (no), shrug, point at the camera, a small "ta-da".
- **Look at the user:** Combine with face tracking (§2) so the "head" faces the speaker while talking.

### Driving it
- **Text-only:** `talk(text, mood)` → the server estimates duration from word count, sets syllable-rate gripper "mouth" motion, nods at punctuation, and picks gestures from `mood`.
- **With voice:** Local TTS (for example Piper, which runs on CPU) generates audio. The server computes an RMS envelope every 33 ms to drive gripper aperture, plays the audio, and runs the motion on the same clock.
- **Trigger:**
  - In generic MCP clients (Cursor/Claude), the agent must call `express("thinking")` and `talk(...)` explicitly; MCP servers cannot see when the model is generating.
  - In **our own mcp-use `MCPAgent` app**, we control the loop: "thinking" starts automatically when generation starts and "talking" while tokens or audio stream. This gives the best result.

### Safety
Expressive motion runs through the same SafetyLimiter as other animations. Use small amplitudes around a "presentation" pose and low speed. Limit duty cycle, because STS3215 servos heat up under continuous motion; monitor temperature if the motor bus exposes it.

---

## 6. Revised phases

**Simulation first.** Phases 0–5 run entirely in MuJoCo on this laptop and are tested from Cursor. Arm bring-up (§4) can happen any time in parallel; Phase 6 switches the backend to the real arm.

| Phase | Scope | Acceptance |
|---|---|---|
| **0. Sim env + source check** | `uv` env with LeRobot 0.6.1 + MuJoCo; load `so101-tabletop-mujoco` via `make_env`; open passive viewer; dump one observation (keys, shapes, units, camera images) and step a few actions; source-verify `so_follower` features, `LeRobotDataset`, async inference, mcp-use server | Arm visible in 3D viewer on Windows; printed observation/action schema matches real `so101_follower` names; notes on any mismatch |
| **1. Runtime core + MCP skeleton** | `RobotBackend` (Sim), MotionOwner, SafetyLimiter, 30 Hz control thread, job model; MCP tools `robot_status`, `observe`, `jog_joints`, `stop`, `get_job`, `reset_scene`; register in `.cursor/mcp.json` | From Cursor: agent nudges a joint, sees the sim render in chat, `stop` interrupts mid-motion |
| **2. Animations + talk** | Keyframe library (wave, nod, dance, rest, presentation), interpolation, start-pose check; `talk` / `express` with gripper "mouth", optional Piper TTS | Agent says hello with a wave in sim; `talk` animates in sync with text/audio |
| **3. Run existing ACT checkpoints in sim** | Policy loader + compatibility check (cameras, state/action names, fps); `start_policy` job; CPU latency benchmark; try candidate sim-trained checkpoints | Pipeline proven end-to-end (obs → policy → chunk → action) with measured latency, even if the foreign policy fails the task |
| **4. Data + own pick-and-drop policy** | Scripted expert (true cube pose + MuJoCo IK); Recorder → `LeRobotDataset`; ~100–200 episodes; cloud `lerobot-train` ACT; registry; `verify` from sim success signal | Agent calls `start_policy("pick_and_drop")` in sim; cube ends in box; verified by sim signal, not just "policy finished" |
| **5. Tracking in sim** | Cube-follow with wrist cam; face-follow hybrid (laptop webcam → sim arm) | Sim arm follows target in bounded workspace; holds/stops on target loss |
| **6. Real arm** | §4 bring-up; `RealBackend`; replay the *same* animations, talk, tracking; collect real demos (gamepad / kinesthetic); fine-tune ACT on real data | Same MCP calls work with `mode=real`, behind the local arm switch and e-stop |
| **7. ChatGPT + MCP App UI** | Tunnel + auth for ChatGPT; optional TypeScript mcp-use MCP App (live camera, joints, job timeline) | ChatGPT drives sim (then real) via the same server; widget renders |

---

### Phase 2 results (2026-10-18)
- **Tools:** `list_animations`, `run_animation` (10 gestures), `express` (5 held states, preemptible by the next motion), and `talk` (text, mood, voice).
- **Animations:** each is a base pose plus smooth offsets (sines with eased envelopes, or min-jerk keyframes).
  - Before playing, every animation is fitted to the backend's per-joint speed and acceleration budget. Amplitude is scaled per joint first, with a 50% floor; then the whole animation is slowed if still needed.
  - Playback glides from the commanded pose to the first sample, plays on the control clock, then settles. It all runs through the same SafetyLimiter and table guard.
- **Presentation pose** `(0, -45, 35, 15, 0, 20)`: the gripper points at the viewer and its jaws open up and down like a mouth. The wave uses a raised-hand pose `(0, -10, -40, -30, 0, 60)`. Every animation keeps at least 7 cm of table clearance by kinematics.
- **Talk:**
  - Voice is Windows SAPI (built in, about 0.1 s to synthesize a sentence) instead of Piper. No model download, and David and Zira are installed.
  - The mouth follows the WAV loudness at 30 Hz, opens 80 ms early, and is rate-limited to the gripper's speed. Without voice it follows a syllable-rate estimate from the text.
  - The head dips after each sentence, and the mood adds sway or a tilt. Audio starts on the control tick the motion starts, and talk is never slowed down.
- **Real-arm speed budget (`REAL_SPEED`):**
  - Speed: pan 40, lift 25, elbow 30, wrist 90 °/s; gripper 150 units/s.
  - Acceleration budget: (200, 120, 150, 900, 900, 2000).
  - The servo Goal_Velocity and Acceleration registers are derived from this budget with 1.5× headroom. Otherwise servo lag on fast wrist motion would read as contact, and the old acceleration value of 20 (about 180 °/s²) would smear nods.
- **Tests:** 17 sim e2e tests in `tests/test_e2e_expressive.py`. The real expressive e2e (wave, nod, ta-da, thinking, then spoken talk, then rest) passed with no contact stops.
- **Bring-up note:** one startup failed because shoulder_lift missed a status packet on Torque_Enable. The setup writes now retry 3 times, and a failed connect closes the port.

### Phase 5 results (2026-10-18)
- **Tools:**
  - `track(target=face|color, color, camera=wrist|front|webcam, seconds, invert)` runs as a job with live metrics (`error_x`, `error_y`, `target_area`, detections, `calibrated`).
  - `place_object` (sim) moves cubes, and `robot_status.task.objects` gives their ground-truth positions.
- **Wrist camera:** the sim's wrist camera turned out to be mounted rotated 90°. Pan moves the target vertically in the image, and wrist_flex moves it horizontally. So the §2 sketch (horizontal error to pan, vertical error to wrist_flex) would have failed. Instead each job probes pan and wrist_flex by ±5° toward mid-range and estimates the 2×2 image Jacobian. It refuses ill-conditioned results (condition number over 25). It then steers with `-0.35 · J⁻¹ · error` per detection, capped at 3° per detection, at 15 Hz.
- **Fixed cameras:** a proportional loop on pixel error can't converge when the arm doesn't move the target in the image, so the mapping is absolute: pan = ±40° × image x, smoothed.
  - Sign convention: positive pan turns the arm toward its right (−y).
  - The front camera faces the arm, so its default sign is −1. The laptop webcam in hybrid sim mode counts as standing beside the arm, so its sign is +1.
- **Detectors:** HSV colour blobs, and YuNet faces (`opencv/face_detection_yunet` @ `3cc26e7f`). The face detector was checked on YuNet's example selfie and runs at about 12 ms per 640×480 webcam frame.
- **Tests:** 6 sim e2e tests in `tests/test_e2e_tracking.py`.
- **Still to try on the real arm** (needs a person in front of the camera):
  - Face tracking with the Innomaker wrist camera, from the presentation pose.
  - Hybrid face-follow (`camera: "webcam"` in sim).
  - The real webcam's `front` sign. With the webcam facing the arm it should be −1 as in sim; `invert` overrides it.

### Phase 3 results (2026-10-18)
- **Checkpoint:** no Hub checkpoint was trained on `lerobot/so101-tabletop-mujoco`. `Chaenn/act_policy_so101_cube_multitask_sim_0827` (pinned at `96e4b70b`) matches our observation format exactly: a 6-value state in degrees, plus `side` and `wrist` images at 480×640, which map to our `front` and `wrist`.
- **Loading:** `PreTrainedConfig.from_pretrained`, then `config.device = "cpu"` and `pretrained_backbone_weights = None` (otherwise ACT downloads ImageNet ResNet-18 weights it then overwrites, and this network drops that download). Then `get_policy_class(...).from_pretrained` and `make_pre_post_processors` with device overrides. Hub downloads use `max_workers=1` and retries.
- **Latency (CPU, i7-13700H), 100-step chunk:**

  | Threads | p50 | p95 |
  |---|---|---|
  | 14 | 241 ms | 252 ms |
  | 8 | 300 ms | 310 ms |
  | 4 | 286 ms | 320 ms |

  - The server uses 4 threads, which leaves cores for physics, rendering and the control loop. A 10–30 s run shows 0 control-loop overruns.
  - The installed torch build has no `xpu` support (Arc GPU unused).
- **Async execution:** a refill is requested when about 1 s of chunk remains, and actions already overtaken by inference latency are skipped (about 9 per chunk).
- **Safety additions:**
  - Per-job table clearance (0 cm for policies, so grasps are possible).
  - A hold-on-guard mode for streaming behaviours: one refused tick doesn't kill the job, and `table_guard_holds` counts them.
  - Policies not marked `real_ok` are refused on the real arm.
- **Outcome:** in our scene the foreign checkpoint reaches toward the cubes but solves nothing (reward 0). It pushes into the table often (374 holds in 30 s), which is the scene mismatch §1 predicted. Our own policy needs our own data (Phase 4).

## 6a. Real-arm bring-up status (2026-10-17)

- **Arm camera (verified):**
  - Innomaker U20CAM-1080p-S1 (USB VID 0x0C45 / PID 0x6366), mounted on the wrist with the gripper jaw visible at the bottom of the frame.
  - Selected **by name**, never by index; DirectShow index is 1 today.
  - 640x480 at 29.6 fps. 1280x720 gives 9.8 fps and 1920x1080 gives 4.9 fps. Use 640x480 at 30 fps, which matches the sim `wrist` camera.
  - Probe: `scripts/hw_probe_cameras.py`.
  - **The laptop webcam is not to be opened** (user decision).
- **Motor controller (verified, read-only probe `scripts/hw_probe_motors.py`):**
  - **COM3**, WCH CH343 (USB VID 0x1A86 / PID 0x55D3), Windows inbox driver.
  - All 6 motors answer at 1 Mbaud with model 777 (STS3215), IDs 1–6. Firmware is uniform.
  - **Torque off** on all motors. About 4.9 V supply, which means the standard 5 V kit with lower torque than 12 V. Temperatures are 26–30 °C.
  - `sync_read` of 6 positions takes about 1 ms. Position jitter is 0 ticks at rest.
  - The motors already hold homing offsets and limits from calibration. Gripper torque limit is 500.
- **Hazard found (verified):** `Goal_Position` reads **0 on every motor** while present positions are about 2048.
  - Enabling torque could drive each joint toward 0, a swing of about 180° at full speed.
  - Stock `lerobot-calibrate` does enable torque briefly: `connect()` → `configure()` → `torque_disabled()` re-enables torque on exit, *before* calibration.
  - **Do not use stock `lerobot-calibrate` on this arm.** Use `scripts/hw_calibrate.py`. It runs LeRobot's `calibrate()` with torque never enabled and closes the port without torque writes. **Never write Goal_Position with torque off: on STS3215 a goal write auto-enables torque** (found during the first calibration run; harmless only because the goal written was the present position). Align goal to present only immediately before deliberately enabling torque. The script also flags ranges that hit the 0/4095 encoder edge (wrap-around from an off-center middle pose).
- **Calibration file location (LeRobot 0.6.1):** `%USERPROFILE%\.cache\huggingface\lerobot\calibration\robots\so_follower\follower_01.json`. The robot name is `so_follower`, not `so101_follower`.
- **Calibration accepted 2026-10-17** (`scripts/hw_calibrate.py`, torque stayed off). Second recording, kept because it matches the first:
  - shoulder_pan 878–3231 (~207°), shoulder_lift 817–3213 (~211°), elbow_flex 843–3055 (~194°), wrist_flex 915–3202 (~201°). Midpoints are within about 9° of the center pose. None of them touch the 0/4095 encoder edges.
  - wrist_roll is the full 0–4095 by design.
  - gripper 2037–3485 (~127°). The low end is the mechanical stop at the pose held in step 1; both recordings agree within about 1°. Gripper 0 is that stop and gripper 100 is the other stop.

- **First powered test:** `scripts/hw_first_motion.py`.
  - Sets goal to present, then enables torque, then verifies the hold (drift under 20 ticks).
  - Servo-side caps: `Goal_Velocity` 200 steps/s (about 18 °/s) and `Acceleration` 10.
  - 5° nudges, one joint at a time, toward the open side of each range and then back, gravity-heavy joints last.
  - Load stop at 700/1000 and a tracking-error stop at 150 ticks.
  - On abort the arm holds its pose; torque is released only on user confirmation.
  - Torque is capped at 40% (`Torque_Limit` 400) for these tests.
  - `--auto --webcam` runs unattended and saves laptop-webcam before/after/diff frames to `artifacts/hw/first_motion/`.
- **Power (fixed 2026-10-18):** these are the **12 V** STS3215 servos (factory `Max_Voltage_Limit` = 14.0 V). The first adapter only gave about 4.9 V: the motors answered and enabled torque but could not move. With the 12 V adapter they read 11.9–12.1 V.
- **First powered test passed (2026-10-18):**
  - Hold: 0 ticks of drift on every joint.
  - Every joint moved 5° and came back. Encoder error was under 1° on all of them, except the elbow, which reached about 2.5° short of its target under the 40% torque cap (gravity droop) and then returned exactly.
  - Peak load was 14% or less.
  - The webcam diffs match the expected parts: the whole arm for the base, the jaw for the gripper, the forearm for the elbow.
  - Torque was released with the arm back at rest.

### Phase 6 core results (2026-10-18)
- `RealBackend` (`src/so101_mcp/backends/real.py`) runs behind the same runtime and MCP tools as the sim. `uv run so101-mcp --backend real`.
- **Startup:** goal is set to present, torque is enabled, and the arm just holds.
- **Servo caps:**
  - Torque: shoulder_lift 80%, elbow 70%, the others 50%.
  - Speed: 40 °/s on the servos; the runtime caps motion at 20 °/s.
- **Cameras:** `wrist` (Innomaker) and `front` (laptop webcam), opened by name and read in background threads.
- **Table guard (sim and real):** forward kinematics from the sim's MuJoCo model gives the lowest mesh point of every moving link for each tick's target. Within 0.5 cm of the table, the job fails and the arm holds.
  - The joint conventions match: the real rest pose rendered in sim matches the webcam photo.
  - The model reads the real arm about 1.6 cm lower than it really is, so the real gap is larger, which is the safe direction.
- **Contact stop:** fires when a joint is loaded and lags its goal by more than 8°, or is pinned at its torque cap.
- **Parking:** `go_rest` and shutdown lift to a raised waypoint, then fold to rest, then release torque. Ctrl+C in the server triggers this (uvicorn re-raises the signal; the server turns that into KeyboardInterrupt so `finally` runs).
- **Bugs found only on the real arm, now fixed and covered by sim tests:**
  1. The guard's "may not go lower" check compared consecutive ticks, so a slow descent (under 2 mm per tick) crept through. The floor now ratchets.
  2. Jobs started from the gravity-sagged measured pose, which made the arm creep downward. Jobs now start from the commanded pose.
  3. Move durations let the min-jerk peak (1.875× the average) exceed the speed cap. The limiter then clipped joints independently and bent the path toward the table. Durations now keep the peak under the cap.
  4. Load alone was used as contact. The extended arm loads the shoulder to about 46% just holding itself up.
  5. LeRobot's servo gains (P=16) leave about 3° of steady-state error. Kept, since policies are trained with those dynamics; the real-arm reach tolerance is 4°.
- **Real e2e** (`tests/test_e2e_real.py`, opt-in): passes. Evidence frames are in `artifacts/e2e/real/`.

### Real-arm safety rules (must be in place before the first powered motion)
1. **No motion during setup:** `lerobot-find-port` and `lerobot-calibrate` run with torque **off**; the user moves joints by hand.
2. **First powered step is "hold":**
   - **Always write Goal_Position = Present_Position before enabling torque.** The goal register can hold stale values; it was 0 here.
   - Enable torque at the current measured pose, verify nothing moved, then try 5° nudges on one joint at a time.
3. **Slow by default on hardware:**
   - Runtime speed cap of 20 °/s, instead of the 90 °/s used in sim.
   - LeRobot `max_relative_target` per step as a second, independent cap.
4. **Table guard:**
   - Forward kinematics from the MuJoCo SO-101 model (same joint convention) computes the gripper-tip and wrist heights for every target, including each interpolated step.
   - Targets that would put any of them below table height plus a margin are refused or clamped **before** being sent.
   - Applies in sim too, so it is e2e-tested there first.
5. **Contact stop:** Watch STS3215 present load/current each tick. Above a threshold, cancel the job and hold the measured pose. Sim already holds the measured pose when a target is not reached.
6. **Gravity on shutdown:** LeRobot disables torque on disconnect, and a raised arm then **drops onto the table**. The runtime must move to the rest pose before releasing torque. Never release torque while the arm is raised.
7. **Physical e-stop:** Keep a power switch on the motor supply within reach.
8. **Calibration ranges:** Soft limits come from the real calibration file (`%USERPROFILE%\.cache\huggingface\lerobot\calibration\robots\so_follower\<id>.json`) minus a margin, not from sim ranges.

## 7. Open hardware / product questions

1. ~~**Attached camera**~~: Answered. Wrist-mounted Innomaker U20CAM, 640x480 at 30 fps (§6a).
2. Is the follower assembled, and are **motor IDs already set**? Do the motors use 5 V or 12 V? Answer after connecting.
3. Is **voice (TTS)** wanted for talk mode, or text-only gestures?
4. Is the real-demo method gamepad end-effector control or kinesthetic hand-guiding? Decide in Phase 6.
