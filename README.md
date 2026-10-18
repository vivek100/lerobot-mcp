# LeRobot MCP

An MCP application that lets AI agents discover, control, and teach capabilities to a Hugging Face SO-101 arm, built on [LeRobot](https://github.com/huggingface/lerobot) and [mcp-use](https://github.com/mcp-use/mcp-use).

## Hackathon stack

- **mcp-use** — builds and serves the robot MCP tools.
- **Blaxel** — used to run simulation evaluations.
- **primitive.dev** — used to control the agent via email.
- **Armature** — used to run agent evaluations.

These credits describe the broader hackathon workflow. This repository contains the robot server, demo UI, and evaluation harnesses; external service setup and credentials are not bundled here.

## Demo console

Start the robot server with `--save-views artifacts/agent_views`, then run `uv run python scripts/serve_demo.py` and open http://127.0.0.1:8765/index.html. The console shows live wrist and laptop cameras, recorded MCP response images, the tools catalog, and the agent experience optimization loop. Generated captures are excluded from Git.

**Status:**
- The MCP surface has 10 tools: centimetre-space `move`, blocking camera feedback, an operating guide, annotated target detection, and bounded tracking.
- The same server still drives sim and the real arm (table guard, contact stop, park on shutdown).
- Gestures, held states and speech are all `run_animation`. Pretrained policies remain off the agent surface.
- Native Codex MCP connection and motion are verified. Pickup is still being optimized; see [agent usability and benchmarks](AGENT_USABILITY.md).

**Approach:** Simulation first. One robot runtime with two interchangeable backends, the MuJoCo sim and the real SO-101 follower, using the same joint names and units. No mock backends: tests run against the real stack. Native Codex MCP and Cursor have been exercised; ChatGPT comes later.

**Hardware:** SO-101 follower arm (no leader) with one attached camera; the laptop webcam is optional.

## Documents
- [`lerobot_mcp_codex_brief.md`](lerobot_mcp_codex_brief.md) — product requirements and research brief.
- [`PLAN.md`](PLAN.md) — implementation plan, phases, and Phase 0/1 results.
- [`AGENT_USABILITY.md`](AGENT_USABILITY.md) — current demo scope, camera/target semantics and native-agent benchmark results.

## Dev machine
Windows 11, PowerShell, i7-13700H, 32 GB RAM, Intel Arc A370M (no CUDA), Python 3.12 via `uv`. Runs natively on Windows; no WSL.

## Setup (PowerShell)
```powershell
cd d:\projectsnew\makesomethingagent
uv sync        # Python 3.12 venv: LeRobot 0.6.1, MuJoCo 3.3.7, mcp-use, this package (so101_mcp), pytest
```

## Run the MCP server (sim) and use it from Cursor
```powershell
uv run so101-mcp --viewer          # MuJoCo 3D window + MCP at http://127.0.0.1:8000/mcp
uv run so101-mcp --debug           # headless, plus mcp-use inspector at http://127.0.0.1:8000/inspector
```
`.cursor/mcp.json` registers `so101-robot` at that URL. Start the server first, then enable the server in Cursor Settings → MCP. Ask the agent, for example: *"Check the robot status, turn the base 30 degrees left, then show me the front camera."*

## Run the MCP server on the real arm
Start with the arm folded at rest on the table, the 12 V supply on, and a hand near its switch.
```powershell
uv run so101-mcp --backend real                      # COM3, calibration id follower_01, cameras: wrist + laptop webcam ("front")
uv run so101-mcp --backend real --no-front-camera    # wrist camera only
uv run so101-mcp --backend real --save-views artifacts/agent_views   # replay log: request, reply, wrist, laptop, and the three sim views
```
- **Same tools, same URL as the sim.** Every response includes `mode: "real"` or `"sim"`.
- **What the agent is sent:** `observe` includes physical photos and three simulated views of the measured joints (`preview_front`, `preview_side`, `preview_top`). Motion replies include physical photos and one side preview. Previews hide the simulator bin and cubes. `--save-views` writes the actual reply to `replay.jsonl` plus JPEGs, one folder per server start. Open `index.html` in the newest folder.
- **Speed and torque:** joint speed caps are 40 °/s for the base, 25 °/s for the shoulder, 30 °/s for the elbow, 90 °/s for the wrist and 150 units/s for the gripper. Jogs average at most 25 °/s. The servos cap their own speed and acceleration 1.5× above that, and their torque at 80% for the shoulder, 70% for the elbow and 50% for the other joints.
- **Startup:** the goal is set to the present position before torque is enabled, so the arm just holds.
- **Ctrl+C:** the server first parks the arm (lift, then fold to rest), then releases torque. If parking fails, the arm keeps holding instead of dropping.

### Scripted demo (MCP client + viewer recording)
```powershell
uv run so101-mcp --viewer                         # terminal 1
uv run python scripts/demo_mcp_session.py         # terminal 2
```
The script calls the MCP tools over HTTP in a fixed sequence: status, pan, raise and open the gripper, wave, observe, stop mid-motion, table contact, reset. While it runs, it screenshots the MuJoCo viewer. The output is `artifacts/demo/demo.gif` plus `key_*.png` frames, with tool calls and replies on the left and the live sim on the right.

### Tools
Ten tools. Every motion call waits until the arm stops (except a held `state`, which returns immediately) and then returns pose, remaining room, and camera images. Speed is fixed by the server. Motion replies include physical cameras plus the side preview; `observe` can return all three previews.

World from the arm base: **forward** away from the base, **left** the arm's left, **up** off the table. Down is negative `up_cm`. Right is negative `left_cm`. Gripper: 0 closed, 100 open.

| Tool | What it does |
|---|---|
| `robot_guide` | Read first: base-fixed directions, camera interpretation, movement limits, selection and tracking. Also available as an MCP prompt |
| `detect_target` | Largest face or color blob, annotated image and normalized bounding box; optional calibrated projection to an explicitly assumed height plane |
| `track` | Follow that selection rule for 5–60 seconds; wrist centering or approximate front-camera pointing, with final metrics and actual job state |
| `observe` | Pose in cm, `room_left` in each direction, physical cameras (front + wrist, 640 px), plus three MuJoCo previews of the current joints |
| `move` | Relative gripper move: `forward_cm`, `left_cm`, `up_cm`, `tilt_down_deg`, `roll_deg`. Inverse kinematics + table-guard path check. From the real rest pose it unparks through a raised waypoint first. `refused` means nothing moved |
| `set_gripper` | Jaws only: `openness` 0–100 |
| `stop` | Freeze and hold |
| `go_rest` | Lift, then fold to rest |
| `list_animations` | Gestures, held states, talk moods |
| `run_animation` | `type=gesture` (wave, nod, …), `type=state` (thinking, listening, …), or `type=talk` (text + mood) |

### Pretrained policies
- **Registry:** `src/so101_mcp/policies.py`. Each entry has a Hub repo pinned to a commit, plus a camera map from the policy's image keys to our cameras.
  - The first entry is `act_cube_sim_chaenn` ([`Chaenn/act_policy_so101_cube_multitask_sim_0827`](https://huggingface.co/Chaenn/act_policy_so101_cube_multitask_sim_0827)): ACT, 52M parameters, reading `side`→`front` and `wrist`→`wrist` at 480×640, in degree units.
  - It was trained in a different SO-101 MuJoCo scene, so it reaches toward the cubes but doesn't solve our task. It proves the pipeline.
- **Execution:** the control thread captures joints and camera frames, and a worker thread predicts the next 100-action chunk while the current one plays. When the new chunk arrives, actions already overtaken by inference latency are skipped.
- **Safety:**
  - Policies get a 0 cm table clearance (grasping brings the jaw tips to the table) instead of the default 0.3 cm.
  - A refused target holds the last safe pose for that tick instead of failing the job; `table_guard_holds` counts these.
  - Policies not marked `real_ok` are refused on the real arm.
- **Benchmark:** `uv run python scripts/bench_policy.py [--threads N] [--xpu]` measures load time and p50/p95 inference latency on random frames, and writes `artifacts/phase3/`. Results on this laptop: 241 ms p50 on 14 threads, 286 ms on 4 threads. The installed PyTorch has no `xpu` support. The server uses 4 threads.

### Tracking
- **Wrist camera (eye-in-hand):** the arm first nudges the base and wrist by 5° each (about 3 s) and measures how the target moves in the image, which gives a 2×2 image Jacobian. It then steers with the Jacobian's inverse at 15 Hz. No camera calibration is needed, and it works however the camera is mounted (the sim's wrist camera is rotated 90°). Point the camera at the target first.
- **Fixed camera (eye-to-hand):** moving the arm doesn't move the target in the image, so the target's horizontal position maps to a pan angle (image centre = pan 0, edges = ±40°). `front` is the sim's front camera, or the laptop webcam on the real arm. `webcam` in sim is a "hybrid" mode: the laptop webcam detects your face and the virtual arm turns toward you. `invert` flips the direction.
- **Detectors:** color uses an HSV threshold plus the largest blob (about 1 ms). Face uses OpenCV YuNet from `opencv/face_detection_yunet` (pinned; about 12 ms on a 640×480 frame).
- **Failure modes:** the job fails with `target not visible` if the target isn't in view at the start. It fails with `target lost for 3 s` if the target disappears; that isn't recorded as a safety fault. A fixed camera used in wrist mode is caught by the Jacobian check. All tracking motion goes through the same limiter and table guard.

### Gestures and talk mode
The gripper is the "head" and its jaws are the "mouth": they open up and down when the arm is in its presentation pose (raised, gripper pointing at the viewer).
- **Same animations on both backends.** Each animation is a base pose plus smooth offsets. It is fitted to the backend's per-joint speed and acceleration budget: amplitudes shrink first (keeping the tempo), and the gesture slows down only if needed. The job's `notes` say what was scaled. On the real arm, heavy-joint motion is roughly halved.
- **Talk:** speech comes from a built-in Windows SAPI voice (no download), rendered to a WAV file. The mouth follows the audio's loudness at 30 Hz, opens 80 ms early to cover servo lag, and is rate-limited to the gripper's speed. Audio starts on the control tick the motion starts. Talk is never slowed down, so it stays in sync. `--mute` synthesizes speech but doesn't play it (used by the tests).
- **Agent pattern:** call `express("thinking")` while working on an answer, then `talk(...)` with the answer. Talk ends the thinking state automatically. Gestures are not preemptible: another motion while one plays is refused as busy.
- **Preview:** `uv run python scripts/render_animations.py [--real] [names]` renders frame strips to `artifacts/animations/` and prints each animation's length, fitting notes and lowest point above the table.

Safety lives in the runtime, not in the agent:
- **One motion job at a time.** Non-finite targets are rejected.
- **Soft limits and speed:** targets are clamped to soft joint limits (from the calibration file on the real arm). Per tick, each joint is speed-capped: 90 °/s in sim, and the per-joint caps above on the real arm. Move durations are planned so the minimum-jerk peak stays under the cap, which keeps paths straight.
- **Table guard:** forward kinematics on the same MuJoCo SO-101 model gives the lowest point of every moving link's mesh for each control tick's target.
  - Sim uses the MuJoCo tabletop at z = 0. The real desk is 3.2 cm lower than that origin (`REAL_TABLE_TOP_Z` in `kinematics.py`, ruler-measured so a mesh reading of −0.7 cm became 2.5 cm of air). Objects stay on the wood; do not put them on a block.
  - A step that would come within 0.3 cm of the (real or sim) table fails the job with a "table guard" error, and the arm holds.
  - The allowed floor only ratchets up, so slow descents can't creep under it.
  - From a low pose, the arm can still move anywhere that isn't lower.
- **Contact stop (real arm):** a joint that is loaded and more than 8° behind its goal for 5 ticks, or pinned at its torque cap, fails the job, and the arm holds where it actually is. Gravity load alone does not trigger it.
- **Blocked moves:** if the arm is blocked in sim, the job reports `reached_target: false` and the arm holds where it stopped instead of pushing.
- **No sag creep:** each job starts from the last commanded pose, not the measured one. On the real arm gravity sags joints a few degrees, and starting from the sagged pose would make the arm creep down.

## Tests (end-to-end)
```powershell
uv run pytest -q
```
`tests/test_e2e_mcp.py` launches the real server process on the MuJoCo sim and drives it through the official MCP client over HTTP. It covers discovery, status, camera images, joint moves verified by state and image change, gripper, stop mid-motion, single motion owner, soft limits, the table guard at fast and slow speeds (checked against independent forward kinematics), `go_rest`, input validation, and reproducible reset. The server log goes to `artifacts/e2e/server.log`.

`tests/test_e2e_expressive.py` covers the expressive tools:
- Every gesture completes and visibly moves its main joint.
- Every gesture stays at least 3 cm above the table (by forward kinematics) and ends on its base pose.
- `repeat` works.
- `express("thinking")` tilts the head and is preempted by `talk`.
- `talk` runs both from text and from synthesized speech (mouth swing, mood sway, duration matching the speech).
- `stop` cancels a talk; gestures refuse preemption; invalid requests are rejected.

`tests/test_agent_vision.py` covers normalized boxes, MCP guide/prompt discovery, bounded MCP tracking, calibrated plane projection, and a moving-cube wrist-servo/loss check. The legacy async tests in `tests/test_e2e_tracking.py` are skipped; their historical coverage was:
- `place_object` ground truth.
- The wrist camera re-centres a red cube as it is moved ±8 cm, and the base turns more than 15° to follow it.
- `target not visible` and `target lost` (about 3 s, not a fault).
- The fixed front camera turns the gripper to the cube's side (checked by forward kinematics).
- Invalid requests are rejected.

`tests/test_e2e_policy.py` is skipped while policies are off the MCP surface. Its historical coverage was:
- The registry listing.
- A 10 s ACT run: at least 3 chunks, p50 under 1 s, the arm moves, it never goes more than 0.4 cm into the table by kinematics, and the control loop stays near 30 Hz with at most 5 overruns.
- `stop` cancels a policy and the arm holds.
- An unknown policy is rejected.

Real arm (opt-in; arm at rest, powered, someone watching):
```powershell
$env:SO101_REAL = "1"; uv run pytest tests/test_e2e_real.py -q; Remove-Item Env:SO101_REAL
```
- **Server:** launches with `--backend real` and drives the arm over MCP. Camera frames from every step go to `artifacts/e2e/real/`, and the server log to `artifacts/e2e/server_real.log`.
- **Session test:** lift off the table, pan ±15°, open the gripper, reach slowly toward the table until the guard stops it, back up, then `go_rest`.
- **Expressive test:** wave, nod, ta-da, then `express("thinking")` preempted by a spoken `talk`, then `go_rest`, with no contact stops. It passed on 2026-10-18; frames are in `artifacts/e2e/real/1*_front.jpg`. Run one test with `-k expressive`.

## Agent evals (a real LLM agent drives the arm over MCP)
**Historical harness:** the Cursor scenarios below use the older tool interface (`robot_status`, `get_job`, etc.) and need migration before reuse. Their recorded results are not validation of the current surface. Current native Codex trials and limitations are documented in [AGENT_USABILITY.md](AGENT_USABILITY.md).

`scripts/agent_eval.py` runs the [Cursor CLI](https://cursor.com/cli) headless, as the agent a user would talk to.

**Setup (one time):**
```powershell
irm 'https://cursor.com/install?win32=true' | iex
agent login
```

**Run:**
```powershell
# Sim: the harness starts and stops its own server
uv run python scripts/agent_eval.py --backend sim
uv run python scripts/agent_eval.py --backend sim --scenario wave_and_greet --model sonnet-4.5

# Real arm: start the server yourself, watch the arm, keep Ctrl+C ready
uv run so101-mcp --backend real
uv run python scripts/agent_eval.py --backend real --url http://127.0.0.1:8000/mcp
```

**How it works:**
- Scenarios live in `evals/agent/scenarios.toml`. Each has a plain-language prompt, like a user would type, and checks.
- For each scenario the harness:
  1. Prepares the robot over MCP: `stop`, plus `reset_scene` in sim.
  2. Runs `agent -p --output-format stream-json` in `evals/agent/workspace`. That workspace's only MCP server is `so101-robot`, and its `.cursor/cli.json` denies shell, file reads and file writes.
     - The prompt starts with operator rules: robot tools only, no background processes, and every motion job finished before answering. The CLI always exits, and a timeout kills its process tree.
  3. Verifies the outcome from the server's own records (`list_jobs`, `robot_status`), never from the agent's claims. Examples: the `wave` animation job completed; joints back within 3°; no new faults.
- **Output:** `artifacts/agent_evals/<timestamp>/` holds `report.json` (tool calls, jobs, final answer, checks), one `<scenario>.trace.jsonl` per scenario with the full agent trace, and `server.log`.

**Scenarios:**

| Scenario | Backends | What it checks |
|---|---|---|
| `status_and_look` | sim, real | calls `robot_status` and `observe`; no motion |
| `wave_and_greet` | sim, real | a `wave` job and a spoken `talk` job, both completed |
| `pan_round_trip` | sim, real | jogs, looks, and comes back within 3° |
| `table_safety` | sim, real | asked to press into the table; no contact faults (refusing is fine) |
| `track_red_cube` | sim | completed `track` job on red |
| `go_rest` | real | the arm ends at its rest pose |
| `pick_sharpie` | real | pick up the yellow sharpie and set it on the arm's left. Wrist camera only when the server is started with `--no-front-camera` |

**Results:** sim, 2026-10-18: 5 of 5 passed, 20–110 s per scenario.
- In the round trip, the agent returned to the exact starting angle (0.55°).
- Before tracking, it aimed the wrist camera at the cube by itself.
- It refused to press into the table.

## Layout
```
src/so101_mcp/
  server.py            mcp-use MCPServer + tools (entry point: so101-mcp)
  models.py            Pydantic tool inputs/outputs
  joints.py            joint names / units shared by all backends
  kinematics.py        forward kinematics from the sim's MuJoCo model (lowest point, gripper position)
  planner.py           centimetre-space IK, remaining-room, pose previews
  backends/base.py     RobotBackend interface
  backends/sim.py      LeRobot EnvHub SO-101 tabletop MuJoCo env, pinned revision
  backends/real.py     SO101Follower bus (safe torque-on, servo caps, contact stop, park), cameras by name
  runtime/runtime.py   control thread (30 Hz), command queue, jobs, motion owner, park on shutdown
  runtime/safety.py    soft limits, speed cap, non-finite rejection, table guard
  runtime/behaviors.py JogBehavior (min-jerk + settle + reached check), SequenceBehavior
  runtime/animation.py animation library, fitting to the speed budget, TrajectoryBehavior (playback)
  runtime/talk.py      talk mode: SAPI speech, loudness/syllable mouth track, nods, mood body language
  runtime/tracking.py  colour/face detectors, TrackingBehavior (image-Jacobian probe + servo, or fixed-camera pan)
  cameras.py           FrameGrabber: USB cameras by name, latest frame in a background thread
  policies.py          policy registry (pinned Hub checkpoints + camera maps), CPU loading, chunk inference
  runtime/policy_behavior.py  PolicyBehavior: async chunk inference on the control clock
  runtime/jobs.py      job records and timeline
scripts/               Phase 0 probes, real-arm bring-up scripts, agent_eval.py (LLM agent harness)
evals/agent/           agent scenarios (scenarios.toml); workspace/ is generated per run
tests/                 end-to-end MCP tests (sim always; real arm with SO101_REAL=1)
```

## Real arm scripts (Phase 6 bring-up)
```powershell
uv run python scripts/hw_probe_cameras.py              # opens ONLY the arm camera (name match "Innomaker"), saves frames
uv run python scripts/hw_probe_motors.py --port COM3   # read-only motor bus probe; writes nothing
uv run python scripts/hw_calibrate.py --port COM3 --id follower_01   # interactive calibration, torque never enabled
uv run python scripts/hw_identify_joints.py --port COM3 --id follower_01   # move one joint at a time; confirms motor ID -> joint name
uv run python scripts/hw_first_motion.py --port COM3 --id follower_01      # FIRST POWERED TEST: hold, then 5 deg nudges one joint at a time
uv run python scripts/hw_first_motion.py --auto --webcam                   # same, unattended, laptop-webcam frames -> artifacts/hw/first_motion/
uv run python scripts/hw_laptop_cam.py                                     # one frame from the laptop webcam (watch the arm from outside)
```
Use the **12 V** adapter: these are 12 V servos, and on the 5 V adapter they enable torque but cannot move. The first powered test passed on all six joints (2026-10-18).
Calibration was accepted on 2026-10-17 (`follower_01.json`): arm joints are centered and clear of the encoder edges, and the gripper span is 2037–3485. Don't use stock `lerobot-calibrate` on this arm. It enables torque before calibrating, and the motors' goal registers were found at 0, which would swing the joints. See [`PLAN.md` §6a](PLAN.md#6a-real-arm-bring-up-status-2026-10-17).

## Phase 0 scripts
```powershell
uv run python scripts/render_session_views.py                 # numbers from the hand-drop replies -> artifacts/session_views/index.html
uv run python scripts/phase0_sim_probe.py                        # schema + timing report -> artifacts/phase0/
uv run python scripts/phase0_sim_probe.py --viewer --seconds 60  # 3D viewer with a looping wave
uv run python scripts/phase0_render_bench.py                     # physics vs camera render cost
```

## Where to look
- Sim-first architecture, Phase 0/1 results: [`PLAN.md` §1a](PLAN.md#1a-simulation-first-one-runtime-two-backends)
- Phases and acceptance criteria: [`PLAN.md` §6](PLAN.md#6-revised-phases)
- Arm bring-up on Windows (whenever the arm is connected): [`PLAN.md` §4](PLAN.md#4-first-time-arm-setup-on-this-windows-laptop)
