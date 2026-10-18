# Agent usability and hackathon demo

Current work recorded on October 18, 2026. This is the current product/benchmark note;
older results in PLAN.md and the original brief describe earlier MCP interfaces.

## Intended demo

1. An agent observes the audience and waves.
2. A visible face is detected in the wrist image; the arm follows it while the user moves.
3. An agent identifies an object, approaches, grasps, lifts and moves it, then verifies the result.

The user requested native Codex MCP integration trials, not just SDK scripts or existing tests.
Each trial should record task success, wall time, tool latency, call count, movement attempts,
refusals and recovery. A completed movement/closed gripper never counts as a verified pickup.
The user authorized slow tests with the physical yellow Sharpie. Simulation trials come first;
record actual hardware outcomes separately. Phone setup and an MCP App live camera UI are deferred.

## Current interface

Ten tools: observe, move, set_gripper, stop, go_rest, list_animations, run_animation,
robot_guide, detect_target, track. The same guide is available as the robot_operation_guide MCP prompt
and in server initialization instructions, so clients that do not surface prompts still get guidance.

- Base-fixed forward/left/up axes, centimetres, relative moves, per-call limits and gesture recovery
  are explicit. Camera screen directions must not be confused with robot directions.
- Preview axes and an outlined wrist camera frustum explain orientation. The frustum uses the
  simulator camera mount/FOV; on hardware it is illustrative, not physical calibration or depth.
- observe returns all requested physical images and optionally all three model previews.
- Motion replies use physical images plus one side preview to reduce repeated image context.
- detect_target produces a normalized image bounding box and annotated photo. It selects the
  largest matching color blob or face each frame, not a persistent instance or an arbitrary object.
- track preflights visibility, then performs wrist image-Jacobian calibration/servoing or fixed-camera
  approximate horizontal pointing. It reports actual execution state, reached_target and metrics.
- Optional plane_height_cm projects a detection centre to an assumed horizontal plane when camera
  calibration is available. Sim cameras have exact simulator calibration; real cameras currently do
  not. Sim cubes are 3 cm tall, so a 1.5 cm centre-plane assumption is useful. This is not measured
  depth or a guaranteed grasp location. No hidden object ground-truth positions are supplied to agents.
- Failed/partially executed motions no longer masquerade as a refusal before motion; animation
  replies check completion. A motion wait timeout requests stop. Physical camera frames older than
  one second raise an error rather than silently feeding a frozen image to tracking.
- Reported real gripper height now uses the same table plane as clearance checks.

## Established evidence

| Trial | Task time | Calls | Outcome |
|---|---:|---:|---|
| Official MCP SDK agent baseline (not native Codex) | 116.4 s | 14, including 9 arm moves | Two failed grasp attempts; image interpretation dominated. Tool execution 19.1 s. |
| Native Codex wave + detection | about 41 s, excluding startup | See native report | Native guide, images, wave and target detection verified; pickup did not run after a validation rejection. |
| Native Codex pickup, boxes without metric projection | about 89 s, excluding startup | 15, including 11 movement attempts | Two refusals recovered; closed and lifted 2 cm, but cube stayed on table. |
| Native Codex pickup, projected offsets and focused replies | 68.7 s, excluding startup | 15, including 10 movement attempts | Cube visibly held above table; final image independently inspected. |
| Native Codex front tracking | 28.7 s session; includes 5 s tracking | See native report | 75/75 target detections; approximate pan pointing, not camera centering. |
| Native Codex pick/move/place | 92.9 s, excluding startup | 23, including 18 movement attempts | Independently logged 3.17 cm rise and 4.46 cm horizontal displacement; returned to table, jaws open. |

Native connection logs include actual mcp_tool_call events. Automatic approval is scoped to the
isolated simulator MCP configuration. The initial native run could read images but motion was
blocked by the default tool-approval policy; the approved run executed native motion successfully.
Raw supporting artifacts are under artifacts/agent_usability; native reports distinguish session
time from simulator startup. These are individual trials, not statistically stable comparisons.
The consolidated [native trial report](artifacts/agent_usability/native/REPORT.md) links the evidence.
Motion-call totals include gripper operations and refused attempts. Prompt/source fingerprints are
captured by the reusable harness for future trials; early exploratory runs predate that feature.

The fixed-camera trial exposed ambiguous reached_target semantics. The current response explicitly
names the objective (approximate horizontal pointing versus wrist image centering), omits irrelevant
centering counts for fixed cameras, and checks pan against the approximate bearing within 3 degrees.
A final fresh native tracking recheck passed in 32.0 seconds and 5 calls; the agent correctly
interpreted the revised objective. Its report includes source/prompt fingerprints captured at launch.
The wrist frustum and axes remain small at the default preview width; legibility needs more work.
After the native trials, large overlapping world labels were replaced by a fixed colored legend.
The physical server was subsequently restarted with these changes. A fresh native Codex hardware
smoke session completed seven calls in 46.84 seconds: guide, observation, animation listing,
front-camera yellow detection, one up_cm=2 move, go_rest, and final observation. The move correctly
unparked first (joint-derived height 2.9→15.9 cm including waypoint travel, not a net 2 cm rise).
The arm returned to rest with no reported faults; jaws stayed unchanged. Both physical cameras,
initial previews and annotated detection arrived through native MCP. Real-camera plane projection
was correctly unavailable without calibration. No physical pickup or face tracking was attempted.
Evidence: artifacts/restart_real/native-report.json and native-final.txt. The updated real server
remains running on port 8000; the existing preview HTTP server remains on port 8765.

The production tracking pipeline has also followed a simulated cube moved across ±8 cm and stopped
when it disappeared. That is a separate deterministic control check, not a native agent benchmark
and not proof of real wrist face-following performance.

## Running a native benchmark

```powershell
uv run python scripts/native_agent_bench.py --scenario pickup
uv run python scripts/native_agent_bench.py --scenario tracking
uv run python scripts/native_agent_bench.py --scenario pick_move_place --timeout 300
```

The launcher starts an isolated simulator and attaches its URL to a fresh Codex CLI session.
It records native MCP events and never sends robot tool calls itself. Prompts live in
evals/native/scenarios.toml. Port 8000 and existing listeners are refused. The CLI's existing
login is used. Automatic approval is scoped to that simulator's MCP tools:

```toml
[mcp_servers.so101]
url = "http://127.0.0.1:8012/mcp"
default_tools_approval_mode = "approve"
```

This configuration is supplied only to the fresh invocation; global settings are not changed.
Reports include session/startup time, per-tool latency, call counts, image counts and available
simulator verification. Missing verification is null, never an inferred success. The motion-count
limit is an agent instruction; the process timeout is enforced. Independent object positions are
logged only to disk for post-session assessment, not sent in MCP replies. Review camera evidence
alongside elevation/displacement: object elevation alone does not prove a stable grasp. Multiple
repetitions are needed for performance claims. Current native MCP configuration is described in
the [official OpenAI documentation](https://developers.openai.com/learn/docs-mcp).

## Next stages

Repeat native pickup/place trials across seeds, target arrangements and model settings. The first
comparison with projected offsets and smaller image replies succeeded, but one trial cannot establish
a success rate or isolate which change helped. Judge grasp success from visible lift plus independent
simulation object motion, not the agent's claim.
Add a marked gripper contact point and more specific IK refusal diagnostics if ambiguity persists.
room_left is an endpoint-based directional estimate; it does not guarantee a collision-free path
or an exact feasible move, particularly after gripper/orientation changes.

For physical metric localisation: fix the overhead phone mount, calibrate the table relative to
the robot base, measure camera intrinsics/mounting and check against measured table points.
A robot bounding rectangle alone cannot establish scale or the camera-to-robot transform.
Add instance selection/tracking if multiple similarly colored objects or multiple faces matter.
MCP Apps live wrist feed and richer user-visible tool result rendering remain a later increment.
