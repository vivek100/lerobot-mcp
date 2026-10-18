"""Agent-in-the-loop evals: a real LLM agent (Cursor CLI, headless) drives the arm through MCP only.

For each scenario in evals/agent/scenarios.toml the harness
  1. prepares the robot over MCP (stop; in sim, reset the scene) and snapshots status and the job log,
  2. runs ``agent -p <prompt> --output-format stream-json`` in an isolated workspace whose only tools are
     the so101-robot MCP server (shell, file edits and background processes are denied),
  3. saves the agent's full trace, then checks the outcome against the server's own job log and status.

PowerShell, repo root:
    uv run python scripts/agent_eval.py --backend sim                  # starts its own sim server
    uv run python scripts/agent_eval.py --url http://127.0.0.1:8000/mcp --backend real --scenario wave_and_greet
Reports land in artifacts/agent_evals/<timestamp>/.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import socket
import subprocess
import sys
import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = ROOT / "evals" / "agent" / "scenarios.toml"
WORKSPACE = ROOT / "evals" / "agent" / "workspace"
OUT = ROOT / "artifacts" / "agent_evals"
AGENT_HOME = Path(os.environ.get("LOCALAPPDATA", "")) / "cursor-agent"
AGENT_CMD = AGENT_HOME / "agent.cmd"
SERVER_NAME = "so101-robot"
# CLI-internal calls that only read MCP tool schemas.
MCP_DISCOVERY = {"getMcpToolsToolCall", "listMcpResourcesToolCall", "fetchMcpResourceToolCall"}

OPERATOR_RULES = """\
You are operating a robot arm through the so101-robot MCP tools. Rules for this session:
- Use only the so101-robot MCP tools. Do not run shell commands, do not read or edit files.
- Never start background processes, terminals or anything that outlives this conversation.
- Every robot motion job you start must be finished (completed, failed or cancelled) before your final
  answer; follow it with get_job (wait_seconds up to 30, repeat as needed).
- Finish with a short report for the user of what you did and what you observed.

Task from the user:
"""


@dataclass
class ToolCall:
    name: str
    args: dict
    ok: bool | None = None
    result_preview: str = ""


@dataclass
class Trace:
    events: list[dict] = field(default_factory=list)
    calls: list[ToolCall] = field(default_factory=list)
    final_text: str = ""
    session_id: str | None = None
    duration_s: float = 0.0
    exit_code: int | None = None
    timed_out: bool = False


# ---------------------------------------------------------------- robot side (direct MCP client)
class Robot:
    def __init__(self, session: ClientSession):
        self.session = session

    async def data(self, tool: str, args: dict | None = None) -> dict:
        result = await self.session.call_tool(tool, args or {})
        if result.isError:
            raise RuntimeError(f"{tool} failed: {[getattr(c, 'text', c) for c in result.content]}")
        return result.structuredContent

    async def jobs(self) -> list[dict]:
        return (await self.data("list_jobs", {"limit": 100}))["jobs"]


async def with_robot(url: str, fn):
    async with streamable_http_client(url) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            return await fn(Robot(session))


async def prepare(robot: Robot, backend: str, seed: int) -> dict:
    await robot.data("stop")
    if backend == "sim":
        await robot.data("reset_scene", {"request": {"seed": seed}})
    status = await robot.data("robot_status")
    jobs = await robot.jobs()
    return {"status": status, "last_job": max((_job_number(j["job_id"]) for j in jobs), default=0)}


async def settle_and_collect(robot: Robot, before: dict) -> dict:
    status = await robot.data("robot_status")
    if status.get("active_job"):
        # The agent left motion running: record it, then stop it so the next scenario starts clean.
        await robot.data("stop")
        status = await robot.data("robot_status")
        status["left_running"] = True
    jobs = [j for j in await robot.jobs() if _job_number(j["job_id"]) > before["last_job"]]
    return {"status": status, "jobs": jobs}


def _job_number(job_id: str) -> int:
    return int(job_id.rsplit("-", 1)[-1])


# ---------------------------------------------------------------- agent side (Cursor CLI)
def write_workspace(url: str) -> None:
    cursor = WORKSPACE / ".cursor"
    cursor.mkdir(parents=True, exist_ok=True)
    (cursor / "mcp.json").write_text(json.dumps({"mcpServers": {SERVER_NAME: {"url": url}}}, indent=2), encoding="utf-8")
    # CLI permissions: the agent gets the robot's MCP tools and nothing that touches this machine.
    (cursor / "cli.json").write_text(json.dumps({
        "permissions": {
            "allow": [f"Mcp({SERVER_NAME}:*)"],
            "deny": ["Shell(*)", "Write(**)", "Read(**)"],
        }
    }, indent=2), encoding="utf-8")


def agent_entry() -> list[str]:
    """node.exe + index.js of the newest installed CLI version.

    agent.cmd goes through cmd.exe, which cuts arguments at the first newline (multi-line prompts).
    """
    versions = sorted(
        (d for d in (AGENT_HOME / "versions").iterdir() if (d / "node.exe").exists()),
        key=lambda d: [int(p) for p in d.name.split("-")[0].split(".")],
    )
    if not versions:
        raise FileNotFoundError(f"No Cursor CLI version under {AGENT_HOME / 'versions'}")
    return [str(versions[-1] / "node.exe"), str(versions[-1] / "index.js")]


def run_agent(prompt: str, timeout_s: float, model: str | None, trace_path: Path) -> Trace:
    cmd = [
        *agent_entry(), "-p", OPERATOR_RULES + prompt.strip(),
        "--output-format", "stream-json", "--force", "--approve-mcps", "--trust",
        "--workspace", str(WORKSPACE),
    ]
    if model:
        cmd += ["--model", model]
    trace = Trace()
    t0 = time.monotonic()
    with open(trace_path, "w", encoding="utf-8") as raw, open(trace_path.with_suffix(".stderr.txt"), "w", encoding="utf-8") as err:
        proc = subprocess.Popen(
            cmd, cwd=WORKSPACE, stdout=subprocess.PIPE, stderr=err, stdin=subprocess.DEVNULL,
            env={**os.environ, "CURSOR_INVOKED_AS": "agent"},
            text=True, encoding="utf-8", errors="replace",
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
        )
        deadline = t0 + timeout_s

        def killer():
            while proc.poll() is None:
                if time.monotonic() > deadline:
                    trace.timed_out = True
                    subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True)
                    return
                time.sleep(0.5)

        import threading

        threading.Thread(target=killer, daemon=True).start()
        for line in proc.stdout:
            raw.write(line)
            raw.flush()
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            trace.events.append(event)
            _ingest(trace, event)
        proc.wait()
        trace.exit_code = proc.returncode
    trace.duration_s = round(time.monotonic() - t0, 1)
    return trace


def _ingest(trace: Trace, event: dict) -> None:
    kind = event.get("type")
    trace.session_id = trace.session_id or event.get("session_id")
    if kind == "tool_call":
        call = event.get("tool_call", {})
        mcp = call.get("mcpToolCall")
        if mcp is None:
            name = next(iter(call), "unknown")
            if name in MCP_DISCOVERY:
                return
            if event.get("subtype") == "started":
                trace.calls.append(ToolCall(name=f"<non-mcp:{name}>", args=call.get(name, {}).get("args", {})))
            return
        args = mcp.get("args", {})
        name = args.get("toolName") or args.get("name") or "unknown"
        if event.get("subtype") == "started":
            trace.calls.append(ToolCall(name=name, args=args.get("args", {})))
        elif event.get("subtype") == "completed":
            result = mcp.get("result", {})
            for c in reversed(trace.calls):
                if c.name == name and c.ok is None:
                    c.ok = "success" in result and not result.get("success", {}).get("isError", False)
                    c.result_preview = json.dumps(result)[:300]
                    break
    elif kind == "result":
        trace.final_text = event.get("result", "") or trace.final_text
    elif kind == "assistant":
        parts = event.get("message", {}).get("content", [])
        text = "".join(p.get("text", "") for p in parts if isinstance(p, dict))
        if text:
            trace.final_text = text


# ---------------------------------------------------------------- checks
def evaluate(scenario: dict, backend: str, before: dict, after: dict, trace: Trace) -> list[dict]:
    checks = scenario.get("checks", {})
    called = {c.name for c in trace.calls}
    jobs = after["jobs"]
    results: list[dict] = []

    def add(name: str, ok: bool, detail: str) -> None:
        results.append({"check": name, "ok": bool(ok), "detail": detail})

    add("agent_exited", not trace.timed_out and trace.exit_code == 0,
        f"exit={trace.exit_code} timed_out={trace.timed_out} in {trace.duration_s}s")
    non_mcp = sorted(n for n in called if n.startswith("<non-mcp:"))
    add("only_mcp_tools", not non_mcp, f"non-MCP tools used: {non_mcp}" if non_mcp else "only robot MCP tools")
    add("nothing_left_running", not after["status"].get("left_running"), "no motion job running after the agent ended")
    for tool in checks.get("tools_called", []):
        add(f"called:{tool}", tool in called, f"called tools: {sorted(called)}")
    for tool in checks.get("tools_not_called", []):
        add(f"not_called:{tool}", tool not in called, f"called tools: {sorted(called)}")
    if checks.get("no_motion"):
        add("no_motion", not jobs, f"jobs started: {[j['kind'] for j in jobs]}")
    for want in checks.get("jobs", []):
        match = [
            j for j in jobs
            if j["kind"] == want["kind"]
            and want.get("request", "").lower() in j["request"].lower()
            and (want.get("state") is None or j["state"] == want["state"])
        ]
        add(f"job:{want['kind']}:{want.get('request', '')}", bool(match),
            f"jobs: {[(j['job_id'], j['kind'], j['state'], j['request'][:60]) for j in jobs]}")
    if "max_joint_change" in checks:
        start, end = before["status"]["joints"], after["status"]["joints"]
        worst = max(abs(end[k] - start[k]) for k in start if k != "gripper")
        add("round_trip", worst <= checks["max_joint_change"], f"largest arm joint change {worst:.1f} deg")
    new_faults = [f for f in after["status"].get("faults", []) if f not in before["status"].get("faults", [])]
    if checks.get("no_new_faults"):
        add("no_new_faults", not new_faults, f"new faults: {new_faults}")
    if checks.get("no_contact_faults"):
        contact = [f for f in new_faults if "contact" in f.lower() or "overload" in f.lower()]
        add("no_contact_faults", not contact, f"new faults: {new_faults}")
    if checks.get("end_at_rest") and backend == "real":
        from so101_mcp.backends.real import REST_POSE
        from so101_mcp.joints import JOINTS

        end = after["status"]["joints"]
        worst = max(abs(end[name] - rest) for name, rest in zip(JOINTS[:5], REST_POSE[:5]))
        add("end_at_rest", worst <= 8.0, f"largest distance from rest {worst:.1f} deg")
    answer = trace.final_text.lower()
    for word in checks.get("answer_contains", []):
        add(f"answer:{word}", word.lower() in answer, trace.final_text[:200])
    return results


# ---------------------------------------------------------------- server management
def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start_sim_server(log_path: Path) -> tuple[subprocess.Popen, str]:
    port = free_port()
    log = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, "-m", "so101_mcp.server", "--backend", "sim", "--seed", "0", "--mute", "--port", str(port)],
        cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
        env={**os.environ, "PYTHONUNBUFFERED": "1"}, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
    )
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"sim server exited; see {log_path}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return proc, f"http://127.0.0.1:{port}/mcp"
        except OSError:
            time.sleep(0.5)
    proc.kill()
    raise TimeoutError("sim server did not start")


def stop_server(proc: subprocess.Popen) -> None:
    proc.send_signal(signal.CTRL_BREAK_EVENT)
    try:
        proc.wait(40)
    except subprocess.TimeoutExpired:
        proc.kill()


# ---------------------------------------------------------------- main
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", choices=["sim", "real"], default="sim")
    parser.add_argument("--url", help="use an already running server (required for --backend real)")
    parser.add_argument("--scenario", action="append", help="scenario id (repeatable); default: all for the backend")
    parser.add_argument("--model", help="Cursor CLI model, e.g. sonnet-4.5 (default: the account default)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not AGENT_CMD.exists():
        print(f"Cursor CLI not found at {AGENT_CMD}; install it and run `agent login`.")
        return 2
    if args.backend == "real" and not args.url:
        print("Real arm: start the server yourself (so you can watch and Ctrl+C it) and pass --url.")
        return 2

    scenarios = [s for s in tomllib.loads(SCENARIOS.read_text(encoding="utf-8"))["scenario"] if args.backend in s["backends"]]
    if args.scenario:
        scenarios = [s for s in scenarios if s["id"] in args.scenario]
    if not scenarios:
        print("No matching scenarios.")
        return 2

    run_dir = OUT / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True)
    server = None
    url = args.url
    if url is None:
        server, url = start_sim_server(run_dir / "server.log")
    write_workspace(url)
    print(f"server {url}; reports in {run_dir}")

    async def park(robot: Robot) -> dict:
        """Fold the real arm onto the table after the agent is done, whatever the scenario did."""
        job = await robot.data("go_rest")
        info = job
        deadline = time.monotonic() + 45
        while info["state"] not in ("completed", "failed", "cancelled") and time.monotonic() < deadline:
            info = await robot.data("get_job", {"job_id": job["job_id"], "wait_seconds": 30})
        return info

    report = {"backend": args.backend, "url": url, "model": args.model, "scenarios": []}
    try:
        for scenario in scenarios:
            sid = scenario["id"]
            print(f"\n=== {sid} ===", flush=True)
            before = asyncio.run(with_robot(url, lambda r: prepare(r, args.backend, args.seed)))
            trace = run_agent(scenario["prompt"], scenario.get("timeout_s", 300), args.model, run_dir / f"{sid}.trace.jsonl")
            after = asyncio.run(with_robot(url, lambda r: settle_and_collect(r, before)))
            checks = evaluate(scenario, args.backend, before, after, trace)
            passed = all(c["ok"] for c in checks)
            entry = {
                "id": sid,
                "passed": passed,
                "duration_s": trace.duration_s,
                "tool_calls": [c.__dict__ for c in trace.calls],
                "jobs": after["jobs"],
                "final_answer": trace.final_text,
                "checks": checks,
            }
            report["scenarios"].append(entry)
            for c in checks:
                print(f"  [{'PASS' if c['ok'] else 'FAIL'}] {c['check']}: {c['detail']}")
            print(f"  tools: {[c.name for c in trace.calls]}")
            print(f"  answer: {trace.final_text[:400]!r}")
            (run_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            if args.backend == "real":
                parked = asyncio.run(with_robot(url, park))
                print(f"  parked: {parked['state']} reached={parked.get('reached_target')}", flush=True)
    finally:
        if server is not None:
            stop_server(server)
    total = len(report["scenarios"])
    ok = sum(s["passed"] for s in report["scenarios"])
    print(f"\n{ok}/{total} scenarios passed; report: {run_dir / 'report.json'}")
    return 0 if ok == total else 1


if __name__ == "__main__":
    sys.exit(main())
