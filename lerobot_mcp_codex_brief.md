# LeRobot MCP — Product Requirements and Codex Research Brief

**Status:** Discovery / planning, not implementation  
**Hardware:** Hugging Face SO-101 follower arm  
**Development machine:** CPU-only laptop  
**MCP SDK preference:** [mcp-use](https://github.com/mcp-use/mcp-use)  
**Core robotics stack:** [Hugging Face LeRobot](https://github.com/huggingface/lerobot)

## Mission

Build an MCP application that allows AI agents to discover, control, and teach capabilities to an SO-101 robot arm. Reuse LeRobot's hardware APIs, policy inference, datasets, training, recording, and simulation instead of recreating robotics infrastructure. Start with a compelling demo: wave/dance animations, pick-and-place using a compatible trained policy, and visually follow a moving object. Agents should observe and verify outcomes, not merely assume a command succeeded.

**Codex instruction: research and propose an architecture first. Do not implement or finalize MCP tools until the relevant LeRobot APIs have been verified in source.**

## 1. User experiences

- **Show off:** “Say hello” or “do something cool” runs predefined, safely validated wave/dance/presentation trajectories.
- **Pick and place:** “Pick up the cube” selects a compatible trained policy, runs it, and verifies whether the cube was grasped or moved successfully.
- **Follow:** “Follow this ball” uses camera-based tracking and bounded reactive control; this is *not* a prerecorded animation or necessarily a learned policy.
- **Inspect:** The agent can inspect camera frames, joint state, running capability, progress, faults, and verification evidence.
- **Teach:** Record teleoperated or otherwise valid demonstrations as LeRobot datasets; train/fine-tune with existing LeRobot tools, evaluate, and register validated new policies as discoverable capabilities.
- **Direct control (research):** Determine a safe bounded way to command poses, joints, and gripper without giving agents unrestricted motor access.

## 2. Two explicit execution modes

### Simulation mode

Run an SO-101 in an existing MuJoCo/LeRobot-compatible environment, preferably on CPU. Show 3D movement, camera observations, joint states, task progress, and agent actions. Support reset, reproducible scenarios, safe animation replay, compatible policy execution, recording, and verification. Simulation action/observation semantics must be checked rather than assumed identical to hardware.

### Real robot mode

Run the physical SO-101 through LeRobot on the local laptop, including motor bus, calibration, and attached cameras (camera setup still to be confirmed). Execute validated animations, policies, tracking, and bounded direct commands. Keep time-sensitive control and safety authority local. Remote GPU inference is allowed; network loss or stale action chunks must fail safely.

Both modes should expose comparable agent-level capabilities, but explicitly report the mode and backend-specific limitations.

## 3. Capability families and requirements

### Predefined animations

List, inspect, run, interrupt, record, and replay named joint trajectories: wave, dance, nod, presentation pose, rest pose. These are not neural policies. Investigate LeRobot recording/replay, interpolation, motion timing, initial pose checks, joint/speed limits, and collision risks. Animation completion does not imply a semantic task succeeded.

### Learned policies

Discover and select trained ACT, SmolVLA, Pi0/Pi0.5, or other LeRobot-compatible checkpoints. Capture model metadata: task, checkpoint, architecture, required cameras, feature names/shapes, normalization, action order/units, expected frequency, robot/calibration compatibility, LeRobot version, hardware needs, inference location, and evaluation evidence. Support loading, execution status, cancellation, observation, recording, and post-execution verification. A downloadable model is not necessarily usable on this robot without matching the training setup or fine-tuning.

### Reactive tracking

Use existing camera/vision and inverse-kinematics/visual-servoing libraries where suitable. Start with following a colored ball within a constrained workspace, then consider hand/person tracking. Handle lost target, camera calibration, speed/position limits, and stop conditions. Do not assume a VLA is required.

### Observation and verification

Read joint/gripper state and camera images; expose execution logs and faults. Distinguish command accepted, execution completed, and task succeeded. Use task-specific success criteria (e.g. simulation rewards/object location), geometric checks, or visual verification with uncertainty. Never declare success based only on the policy returning actions.

### Recording, training, and policy registration

Use LeRobot's dataset and episode recording mechanisms; capture observation/action streams, task annotations, metadata, and success labels. Investigate recording with leader/follower teleoperation, scripted actions, simulation, and policy rollouts. Reuse LeRobot training and evaluation, with optional cloud GPU. Only validated checkpoints become available for physical execution.

## 4. LeRobot interfaces to inspect in source

- [LeRobot repository](https://github.com/huggingface/lerobot)
- [SO follower implementation](https://github.com/huggingface/lerobot/blob/main/src/lerobot/robots/so_follower/so_follower.py): inspect `connect`, `disconnect`, `get_observation`, `send_action`, `observation_features`, `action_features`, calibration, motor buses, position units, and camera naming. SO-101 is a six-motor arm (five joints and gripper); confirm actual feature names/units in the checked-out version.
- [Policy base](https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/pretrained.py): inspect `select_action`, `predict_action_chunk`, action queues, preprocessing/postprocessing, feature metadata, checkpoint loading, task text, and normalization.
- [ACT inference example](https://github.com/huggingface/lerobot/blob/main/examples/tutorial/act/act_using_example.py): trace observation -> tensor/features -> preprocess -> policy -> postprocess -> motor action.
- [Remote inference](https://huggingface.co/docs/lerobot/main/remote_inference) and [asynchronous inference](https://huggingface.co/docs/lerobot/async): inspect server/client, action chunk buffering, network transport, policy loading, compatibility checks, latency, reconnect behavior, and multi-model hosting. Prefer existing LeRobot transport to inventing one.
- [Recording](https://huggingface.co/docs/lerobot/main/record): investigate `LeRobotDataset`, episodes, teleoperation, rollout recording, and Hugging Face Hub upload.
- [Training](https://huggingface.co/docs/lerobot/main/train): investigate ACT/SmolVLA/Pi0 training and fine-tuning support and evaluation workflows.
- [Inference/rollout](https://huggingface.co/docs/lerobot/main/inference): inspect current rollout, interactive execution, recording, evaluation, and visualization.
- [LeRobot simulation tutorial](https://huggingface.co/docs/lerobot/main/il_sim): inspect demonstrations, datasets, training, and evaluation in simulation.
- Inspect EnvHub, `make_env`, Gymnasium adapters, and Rerun integration in the current repository.

**Core model:** robot `get_observation()` produces raw joint states and configured camera frames; policy inference consumes appropriately preprocessed observations and predicts joint action(s)/chunks; `send_action()` sends joint-position targets to hardware. A policy is not automatically a Cartesian planner or a semantic object detector. Verify exact contracts and timing against current code.

## 5. Simulation references — compare, do not blindly adopt

1. [Official LeRobot SO-101 tabletop MuJoCo environment](https://huggingface.co/lerobot/so101-tabletop-mujoco) — preferred first investigation. Verify download, license, CPU performance, observation/action spaces, cameras, 3D viewer, tasks, reset, and EnvHub integration.
2. [SO-101 pick-and-place environment](https://huggingface.co/namenu/so101-pick-place-env) — examine Gymnasium interface, task rewards/success checks, camera inputs, and joint actions.
3. [SO-101 Sim2Real](https://github.com/isaka1022/so101-sim2real) — examine MuJoCo model, visualization, trajectory control, and compatibility.
4. [LeRobot imitation learning in simulation](https://huggingface.co/docs/lerobot/main/il_sim) — official workflows.
5. Investigate [LeIsaac](https://huggingface.co/docs/lerobot/main/en/envhub_leisaac) as a possible later alternative; do not add a heavier simulator without need.

Deliver a working *research-level* API sketch for initializing, observing, stepping, resetting, and rendering the recommended environment, with source links. Identify what it would take to replay the same validated trajectory in sim and on hardware.

## 6. Candidate prehosted/pretrained policies — verify before relying on them

These are **candidate Hugging Face Hub checkpoints**, not confirmed plug-and-play policies for our exact hardware or simulation. Check repository availability, model card, weights, processor config, datasets, task, cameras, state/action schemas, license, LeRobot version, and evaluation evidence.

| Candidate | URL | What to investigate |
|---|---|---|
| SO-101 ACT pick-and-place | https://huggingface.co/BoyuZhao/so101-act-pick-and-place | Trained pick/place task; CPU inference feasibility; required cameras and training scene |
| SO-101 ACT follower | https://huggingface.co/jenyao/so101_act_policy | `handeye`/`front` cameras, six-dimensional state, action mapping |
| SO-101 ACT pick-place | https://huggingface.co/chohyeongju/so101_act_pick_place | Dataset, checkpoint completeness, inference setup |
| SO-101 Pi0 multi-task | https://huggingface.co/Harrysunshine/so101-pi0-9task | Tasks, remote GPU serving, cameras, checkpoint compatibility, claimed results |
| SmolVLA base | https://huggingface.co/lerobot/smolvla_base | Base model vs task-trained checkpoint, zero-shot limitations, fine-tuning needs |

Search for more suitable, recently maintained **task-trained SO-101 SmolVLA/ACT** checkpoints if needed. Prefer a reproducible demo with known camera/scene constraints to an unvalidated promise of general manipulation.

### Compute and serving

- CPU-only laptop is sufficient for hardware I/O and often for lightweight scripted motion, vision, and MuJoCo; benchmark rather than assume real-time performance.
- ACT may be feasible locally on CPU depending on checkpoint, image size, and control frequency.
- Pi0 and other larger VLAs should generally use a cloud GPU for practical inference.
- Investigate one server per policy vs dynamic checkpoint loading, action chunking, caching, policy switch safety, and network latency.
- Do not assume a generic hosted LLM API can execute LeRobot weights.
- [InsForge](https://insforge.dev/) is a possible application backend for auth, metadata, policy registry, and orchestration; verify GPU/long-running compute availability before considering it for inference.

## 7. Existing MCP and robot projects to learn from

- [robot_MCP](https://github.com/IliaLarchenko/robot_MCP) — SO-100/SO-101 MCP patterns and version compatibility.
- [ros-mcp-plugin](https://github.com/anshuman-dev/ros-mcp-plugin) — ROS 2 / Gazebo / SO-101 integration; avoid ROS unless justified.
- [chem-0](https://github.com/JacobFV/chem-0) — SO-101, camera feedback, MCP, and GUI patterns.
- [robotics-so101-mcp](https://github.com/TitanSage02/robotics-so101-mcp) — motion sequences, recording/replay, mock robot.
- [mcp-use](https://github.com/mcp-use/mcp-use) — preferred MCP SDK; verify server APIs and transport support.

Compare actual code, maintenance, licensing, and reusable components. Do not copy tool designs uncritically.

## 8. Proposed boundaries (subject to research)

- **Agent/MCP client:** selects high-level intent and inspects results.
- **mcp-use server:** discovery, validation, sessions, status, cancellation, and orchestration; no real-time motor loop in the LLM.
- **Capability runtime:** predefined animations, learned policies, tracking, bounded manual control.
- **LeRobot hardware runtime:** calibrated observations/actions, local control timing, motor and camera connections.
- **Policy service:** local ACT or remote GPU policy server; checkpoint compatibility validation.
- **Simulation runtime:** MuJoCo/EnvHub, viewer, reset, rendering, success signals.
- **Dataset/training pipeline:** existing LeRobot recorder, dataset, trainer, evaluator, registry.
- **Visualization:** physical camera or 3D sim, agent/tool timeline, trajectory/state and verification evidence; reuse Rerun or existing viewers where feasible.

Do not freeze MCP tool names yet. First decide if animations, policies, and reactive behaviors can share an execution/session abstraction while keeping their different input, timing, and safety requirements visible.

## 9. Safety and correctness

- One active motion owner per arm; prevent conflicting commands.
- Joint, velocity, workspace, and gripper limits; validate starting pose and trajectories.
- Explicit simulated vs physical mode and human authorization for physical motion.
- Local cancellation/stop and a physical emergency-stop provision; network or MCP cancellation is not a substitute for hardware safety.
- Handle connection loss, stale actions, model loading delays, camera loss, lost tracking target, and faults.
- Keep policy action scaling, joint order, calibration, and camera features exact; mismatches can produce unsafe motions.
- Verify task success separately from successful dispatch/completion; expose uncertainty.

## 10. Proposed phases and acceptance criteria

**Phase 0 — Source-backed discovery (do this first):** map exact current APIs, simulator choice, candidate checkpoints, remote inference, recording, motion safety, and mcp-use integration. Cite source files and functions. Output architecture and open decisions; no code yet.

**Phase 1 — Simulation foundation:** initialize SO-101 sim on CPU, render/view, read state, execute bounded joint trajectories, reset, and log actions. Acceptance: repeatable safe wave animation with visible movement and inspectable execution state.

**Phase 2 — Real-arm animation:** connect SO-101, validate calibration and safety, play wave/dance/rest sequences under local control, cancel and report faults. Acceptance: controlled animation with no agent-driven motor timing.

**Phase 3 — Pretrained policy:** load a validated task-trained SO-101 checkpoint, verify input schema, run against matching recorded/sim observations, then carefully trial hardware if compatible. Acceptance: actual observed pick/place result with verification, not just an action stream.

**Phase 4 — Follow demo:** track a colored object and command bounded reactive motion. Acceptance: arm follows within a safe workspace and stops on target loss.

**Phase 5 — Teaching loop:** record demonstrations into LeRobot datasets, train/fine-tune via existing pipeline, evaluate, register new capability. Acceptance: newly validated policy appears in discovery and can be run with explicit safeguards.

## 11. Required Codex deliverables

1. Source-backed map of SO-101 hardware, policy, rollout, recording, simulation, and remote inference APIs.
2. Compatibility matrix for simulation environments and candidate checkpoints, with required observations, actions, LeRobot versions, and hardware.
3. Recommended simulator and 3D viewer, with CPU feasibility and minimal example/API sketch.
4. Animation vs policy vs tracking vs direct-control execution model, including safety and cancellation.
5. Proposed mcp-use integration and *then* justified MCP tool/resource schema, not speculative tools.
6. Local CPU / cloud GPU deployment and multi-policy serving options; whether InsForge belongs in the stack.
7. Reuse analysis of existing SO-101 MCP repositories and LeRobot components.
8. Phased implementation plan, dependencies, acceptance tests, risks, and unresolved hardware questions (cameras, leader arm, OS, robot calibration).

For each key finding, label **verified in source**, **documented**, **demonstrated by third party**, or **proposal requiring validation**. Provide direct file/function links. Do not start implementing until the research and design are reviewed.

**Guiding principle:** Build the agent-facing robot experience, not another robotics framework.
