"""Benchmark native Codex MCP calls against an isolated SO-101 simulator.

This launcher does not invoke robot tools. Only the fresh Codex session does.
No real backend option is offered, and approval overrides apply only to its sim.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import shutil
import signal
import socket
import subprocess
import threading
import time
import tomllib

ROOT = Path(__file__).resolve().parents[1]


def sanitize(value):
    if isinstance(value, dict):
        if value.get("type") == "image" and "data" in value:
            encoded = value.get("data", "")
            return {"type": "image", "mimeType": value.get("mimeType"),
                    "bytes": len(base64.b64decode(encoded))}
        return {key: sanitize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    return value


def shutdown(process):
    if process is None or process.poll() is not None:
        return
    if os.name == "nt":
        process.send_signal(signal.CTRL_BREAK_EVENT)
    else:
        process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def collect_metrics(calls, output):
    movement = {"move", "set_gripper", "go_rest", "run_animation", "track"}
    metrics = {"movement_attempts": 0, "refusals": 0, "validation_errors": 0,
               "image_count": 0, "image_bytes": 0, "tool_latency_total_s": 0,
               "per_tool": {}, "automated_lift_verified": None}
    for call in calls:
        tool = call.get("tool")
        metrics["movement_attempts"] += tool in movement
        latency = call.get("latency_s", 0)
        entry = metrics["per_tool"].setdefault(tool, {"calls": 0, "latency_total_s": 0})
        entry["calls"] += 1
        entry["latency_total_s"] = round(entry["latency_total_s"] + latency, 3)
        metrics["tool_latency_total_s"] += latency
        for content in (call.get("result") or {}).get("content", []):
            if content.get("type") == "image":
                metrics["image_count"] += 1
                metrics["image_bytes"] += content.get("bytes", 0)
            elif content.get("type") == "text":
                text = content.get("text", "")
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError:
                    payload = {}
                if isinstance(payload, dict):
                    metrics["refusals"] += payload.get("result") == "refused"
                metrics["validation_errors"] += "validation error" in text.lower()
    metrics["tool_latency_total_s"] = round(metrics["tool_latency_total_s"], 3)
    # Ground-truth data is read AFTER the model session, never supplied to it.
    snapshots = []
    final_gripper = None
    for replay in sorted((output / "views").rglob("replay.jsonl")):
        for line in replay.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            final_gripper = (record.get("reply") or {}).get("pose", {}).get("gripper", final_gripper)
            objects = (record.get("verification") or {}).get("sim_objects_m", {})
            red = next((xyz for name, xyz in objects.items() if "red" in name.lower()), None)
            if red is not None:
                snapshots.append(red)
    if snapshots:
        initial, final = snapshots[0], snapshots[-1]
        rise = max(xyz[2] for xyz in snapshots) - initial[2]
        metrics["verification"] = {
            "source": "post_session_simulator_objects", "snapshots": len(snapshots),
            "max_red_height_increase_cm": round(rise * 100, 3),
            "final_red_displacement_cm": round(math.dist(initial, final) * 100, 3),
            "final_red_height_increase_cm": round((final[2] - initial[2]) * 100, 3),
            "final_red_horizontal_displacement_cm": round(math.dist(initial[:2], final[:2]) * 100, 3),
            "final_gripper_openness": final_gripper,
            "criterion": "red cube rose at least 2cm above initial center height; object-elevation evidence alone does not prove a stable grasp",
            "placement_criterion": "prior rise >=2cm, final horizontal displacement >=2cm, final height within0.5cm of initial, final gripper openness >=70; evidence, not a collision-clearance guarantee",
        }
        metrics["automated_lift_verified"] = rise >= .02
        metrics["placed_evidence"] = (
            rise >= .02 and math.dist(initial[:2], final[:2]) >= .02
            and abs(final[2] - initial[2]) <= .005
            and final_gripper is not None and final_gripper >= 70
        )
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="pickup")
    parser.add_argument("--port", type=int, default=8012)
    parser.add_argument("--timeout", type=float, default=240)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.port == 8000:
        parser.error("port 8000 is reserved; use an isolated simulation port")
    scenarios = tomllib.loads((ROOT / "evals/native/scenarios.toml").read_text(encoding="utf-8"))
    if args.scenario not in scenarios:
        parser.error(f"unknown scenario {args.scenario!r}")
    output = args.output or ROOT / "artifacts/agent_usability/native" / time.strftime("%Y%m%d-%H%M%S")
    output.mkdir(parents=True, exist_ok=False)
    codex = shutil.which("codex")
    if not codex:
        parser.error("codex executable not found")
    # Fail rather than accidentally attaching to an existing sim or hardware server.
    with socket.socket() as check:
        check.bind(("127.0.0.1", args.port))
    prompt = scenarios[args.scenario]["prompt"]
    command = [codex, "exec", "--ignore-user-config", "--skip-git-repo-check", "--ephemeral", "--json",
               "--sandbox", "read-only", "-c", f'mcp_servers.so101.url="http://127.0.0.1:{args.port}/mcp"',
               "-c", 'mcp_servers.so101.default_tools_approval_mode="approve"',
               "-c", 'approval_policy="never"', "-o", str(output / "final.txt"), prompt]
    (output / "invocation.json").write_text(json.dumps(command, indent=2), encoding="utf-8")
    creation = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    started = time.monotonic()
    metadata = {"native": True, "backend": "sim", "scenario": args.scenario, "port": args.port}
    metadata["source_sha256"] = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((ROOT / "src").rglob("*.py"))
    }
    metadata["prompt_sha256"] = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    server = trial = None
    calls = []
    pending = {}
    try:
        with (output / "server.log").open("w", encoding="utf-8") as serverlog, (output / "stderr.log").open("w", encoding="utf-8") as errors:
            server = subprocess.Popen([str(ROOT / ".venv/Scripts/python.exe"), "-m", "so101_mcp.server",
                                       "--backend", "sim", "--port", str(args.port), "--mute", "--save-views", str(output / "views")],
                                      cwd=ROOT, stdout=serverlog, stderr=subprocess.STDOUT, creationflags=creation)
            ready_deadline = time.monotonic() + 60
            while time.monotonic() < ready_deadline:
                if server.poll() is not None:
                    raise RuntimeError(f"sim exited with {server.returncode}")
                try:
                    with socket.create_connection(("127.0.0.1", args.port), timeout=.5):
                        break
                except OSError:
                    time.sleep(.25)
            else:
                raise TimeoutError("sim startup exceeded 60 seconds")
            metadata["server_ready_s"] = round(time.monotonic() - started, 3)
            trial_started = time.monotonic()
            trial = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                     stderr=errors, text=True, encoding="utf-8", errors="replace", creationflags=creation)
            messages = queue.Queue()

            def read_events():
                for line in trial.stdout:
                    messages.put((time.monotonic(), line))
                messages.put(None)

            threading.Thread(target=read_events, daemon=True).start()
            print(json.dumps({"phase": "native_session_started", "output": str(output)}), flush=True)
            with (output / "events.jsonl").open("w", encoding="utf-8") as eventlog:
                while True:
                    if time.monotonic() - trial_started > args.timeout:
                        metadata["timed_out"] = True
                        break
                    try:
                        message = messages.get(timeout=.5)
                    except queue.Empty:
                        continue
                    if message is None:
                        break
                    at, line = message
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    event["elapsed_s"] = round(at - trial_started, 3)
                    clean = sanitize(event)
                    eventlog.write(json.dumps(clean) + "\n")
                    eventlog.flush()
                    item = clean.get("item", {})
                    if item.get("type") == "mcp_tool_call":
                        if clean.get("type") == "item.started":
                            pending[item["id"]] = at
                        elif clean.get("type") == "item.completed":
                            item["latency_s"] = round(at - pending.pop(item["id"], at), 3)
                            item["elapsed_s"] = clean["elapsed_s"]
                            calls.append(item)
                            print(json.dumps({"tool": item.get("tool"), "latency_s": item["latency_s"], "elapsed_s": item["elapsed_s"]}), flush=True)
                    if clean.get("type") == "turn.completed":
                        metadata["usage"] = clean.get("usage")
            if trial.poll() is None and not metadata.get("timed_out"):
                trial.wait(timeout=10)
            metadata["codex_exit_code"] = trial.poll()
            metadata["session_s"] = round(time.monotonic() - trial_started, 3)
    except BaseException as exc:
        metadata["error"] = str(exc)
        raise
    finally:
        shutdown(trial)
        shutdown(server)
        metadata["wall_time_s"] = round(time.monotonic() - started, 3)
        metadata["native_tool_calls"] = len(calls)
        metadata["metrics"] = collect_metrics(calls, output)
        (output / "timing.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        (output / "tool-summary.json").write_text(json.dumps(calls, indent=2), encoding="utf-8")
        print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()
