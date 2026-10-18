"""Local read-only demo pages, recorded MCP replies and live camera streaming.

Uses the running robot's camera HTTP endpoint only. Never sends MCP motion calls
and never opens the physical camera device or motor port.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
import time
from urllib.parse import unquote, urlparse
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web/demo"
CAPTURES = ROOT / "artifacts/agent_views"
EVIDENCE = ROOT / "artifacts/agent_usability/native"
STREAM_SLOTS = threading.BoundedSemaphore(4)


def latest_capture():
    if not (CAPTURES / "latest.txt").exists():
        return {"reply": None, "images": [], "history": [], "message": "No recorded MCP response yet."}
    session = (CAPTURES / "latest.txt").read_text(encoding="utf-8").strip()
    active_session = session
    folder = (CAPTURES / session).resolve()
    if not folder.is_relative_to(CAPTURES.resolve()):
        raise ValueError("Invalid capture session")
    replay = folder / "replay.jsonl"
    if not replay.exists():
        candidates = sorted((p for p in CAPTURES.glob("*/replay.jsonl") if p.stat().st_size), key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            return {"session": session, "reply": None, "images": [], "history": []}
        replay = candidates[0]
        folder = replay.parent
        session = folder.name
    records = []
    for line in replay.read_text(encoding="utf-8").splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            pass  # a concurrent append can briefly expose an incomplete last line
    if not records:
        return {"session": session, "reply": None, "images": [], "history": []}
    record = records[-1]
    step = f"{record['step']:03d}"
    return {
        "session": session, "step": record["step"], "tool": record["tool"],
        "captured_at": datetime.fromtimestamp((folder / step / "step.json").stat().st_mtime, timezone.utc).isoformat(),
        "from_previous_session": session != active_session,
        "request": record["request"], "reply": record["reply"],
        "images": [{"name": Path(name).stem, "url": f"/captures/{session}/{step}/{name}"} for name in record["images"]],
        "history": [{"step": r["step"], "tool": r["tool"], "result": r["reply"].get("result"),
                     "message": r["reply"].get("message")} for r in records[-12:]],
    }


class DemoHandler(BaseHTTPRequestHandler):
    robot_url = "http://127.0.0.1:8000"
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def respond(self, data, mime, status=200):
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(data)

    def frame(self, camera):
        with urlopen(f"{self.robot_url}/camera/frame/{camera}", timeout=3) as response:
            if response.headers.get_content_type() != "image/jpeg":
                raise ValueError("Camera returned a non-image response")
            return response.read(4 * 1024 * 1024)

    def stream(self, camera):
        if not STREAM_SLOTS.acquire(blocking=False):
            return self.respond(b"Too many camera viewers", "text/plain", 503)
        try:
            # Verify availability before sending a successful multipart response.
            try:
                data = self.frame(camera)
            except (OSError, TimeoutError, ValueError):
                return self.respond(b"Camera unavailable", "text/plain", 503)
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            while True:
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(data)).encode() + b"\r\n\r\n" + data + b"\r\n")
                self.wfile.flush()
                time.sleep(1 / 6)
                data = self.frame(camera)
        except (BrokenPipeError, ConnectionResetError, TimeoutError, OSError, ValueError):
            self.close_connection = True
        finally:
            STREAM_SLOTS.release()

    def do_GET(self):
        path = unquote(urlparse(self.path).path)
        try:
            if path == "/api/latest":
                return self.respond(json.dumps(latest_capture(), allow_nan=False).encode(), "application/json")
            if path.startswith("/api/stream/"):
                camera = path.rsplit("/", 1)[-1].removesuffix(".mjpg")
                if camera not in {"wrist", "front"}:
                    return self.respond(b"Unknown camera", "text/plain", 404)
                return self.stream(camera)
            if path.startswith("/api/frame/"):
                camera = path.rsplit("/", 1)[-1].removesuffix(".jpg")
                if camera not in {"wrist", "front"}:
                    return self.respond(b"Unknown camera", "text/plain", 404)
                return self.respond(self.frame(camera), "image/jpeg")
            base = CAPTURES if path.startswith("/captures/") else EVIDENCE if path.startswith("/evidence/") else WEB
            relative = path.removeprefix("/captures/") if base == CAPTURES else path.removeprefix("/evidence/") if base == EVIDENCE else (path.lstrip("/") or "index.html")
            file = (base / relative).resolve()
            if not file.is_relative_to(base.resolve()) or not file.is_file():
                return self.respond(b"Not found", "text/plain", 404)
            mime = mimetypes.guess_type(str(file))[0] or "application/octet-stream"
            if mime.startswith("text/"):
                mime += "; charset=utf-8"
            return self.respond(file.read_bytes(), mime)
        except (OSError, ValueError) as exc:
            self.respond(json.dumps({"error": "Camera or capture unavailable", "detail": str(exc)}).encode(), "application/json", 503)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--robot-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    DemoHandler.robot_url = args.robot_url.rstrip("/")
    server = ThreadingHTTPServer(("127.0.0.1", args.port), DemoHandler)
    server.daemon_threads = True
    print(f"Demo UI: http://127.0.0.1:{args.port}/index.html", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
