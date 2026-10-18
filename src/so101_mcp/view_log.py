"""Save each tool call for replay: the request, the reply, and every image the agent was sent.

Live frames (wrist, front) and the simulated angles (preview_front, preview_side, preview_top)
are stored beside step.json. replay.jsonl is one record per call.
"""

from __future__ import annotations

import html
import json
from datetime import datetime
from pathlib import Path

_LABELS = {
    "wrist": "wrist camera",
    "front": "laptop photo",
    "preview_front": "sim front",
    "preview_side": "sim side",
    "preview_top": "sim top",
}


class ViewLog:
    def __init__(self, directory: Path):
        base = Path(directory)
        base.mkdir(parents=True, exist_ok=True)
        self.root = base / datetime.now().strftime("%Y%m%d-%H%M%S")
        self.root.mkdir()
        (base / "latest.txt").write_text(self.root.name + "\n", encoding="utf-8")
        self.step = 0

    def write(self, tool: str, request: dict | None, payload_json: str, images: list[tuple[str, bytes]], verification: dict | None = None) -> None:
        self.step += 1
        folder = self.root / f"{self.step:03d}"
        folder.mkdir()
        names = []
        for name, data in images:
            safe = name.replace("/", "_")
            (folder / f"{safe}.jpg").write_bytes(data)
            names.append(safe)
        record = {
            "step": self.step,
            "tool": tool,
            "request": request or {},
            "reply": json.loads(payload_json),
            "images": [f"{name}.jpg" for name in names],
        }
        if verification is not None:
            record["verification"] = verification
        text = json.dumps(record)
        (folder / "step.json").write_text(text, encoding="utf-8")
        with (self.root / "replay.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(text + "\n")
        self._write_index()

    def _write_index(self) -> None:
        cards = []
        for folder in sorted(p for p in self.root.iterdir() if p.is_dir() and p.name.isdigit()):
            record = json.loads((folder / "step.json").read_text(encoding="utf-8"))
            cards.append(_card(folder.name, record))
        (self.root / "index.html").write_text(_page("\n".join(cards)), encoding="utf-8")


def _card(step: str, record: dict) -> str:
    payload = record.get("reply") or {}
    request = json.dumps(record.get("request") or {})
    requested = f"{record.get('tool') or ''} {request}"
    result = payload.get("result", "")
    message = payload.get("message", "")
    pose = payload.get("pose") or {}
    pose_text = (
        f"forward {pose.get('forward_cm')} cm, left {pose.get('left_cm')} cm, "
        f"up {pose.get('up_cm')} cm, tilt {pose.get('tilt_down_deg')}°, "
        f"roll {pose.get('roll_deg')}°, gripper {pose.get('gripper')}"
    )
    saved = [Path(name).stem for name in record.get("images") or []]
    order = list(payload.get("cameras") or [])
    for name in saved:
        if name not in order:
            order.append(name)
    live = [n for n in order if n in ("wrist", "front") and n in saved]
    other = [n for n in order if n not in ("wrist", "front") and n in saved]
    return f"""<section>
  <h2>{html.escape(step)}. {html.escape(str(result))}</h2>
  <p class="tool">request <code>{html.escape(str(requested))}</code> — {html.escape(str(message))}</p>
  <p>{html.escape(pose_text)}. Clearance {html.escape(str(payload.get("table_clearance_cm")))} cm.</p>
  <div class="views">{_figs(step, live)}</div>
  <div class="views sim">{_figs(step, other)}</div>
</section>"""


def _figs(step: str, names: list[str]) -> str:
    return "\n".join(
        f'<figure><img src="{html.escape(step)}/{html.escape(name)}.jpg" alt="{html.escape(name)}">'
        f"<figcaption>{html.escape(_LABELS.get(name, name))}</figcaption></figure>"
        for name in names
    )


def _page(body: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>What the agent received</title>
<style>
  body {{ font: 16px/1.45 system-ui, sans-serif; margin: 2rem auto; max-width: 980px; color: #1a1a1a; background: #f6f4ef; }}
  h1 {{ font-size: 1.4rem; margin-bottom: 0.3rem; }}
  .lead {{ color: #444; }}
  section {{ background: #fff; border: 1px solid #ddd; border-radius: 10px; padding: 1rem 1.2rem 1.2rem; margin: 1.2rem 0; }}
  h2 {{ font-size: 1.1rem; margin: 0 0 0.4rem; }}
  code {{ background: #f0eee8; padding: 0.05rem 0.3rem; border-radius: 4px; }}
  .views {{ display: grid; grid-template-columns: 1fr 1fr; gap: 0.6rem; margin-top: 0.8rem; }}
  .views.sim {{ grid-template-columns: repeat(3, 1fr); }}
  figure {{ margin: 0; }}
  img {{ width: 100%; border-radius: 6px; background: #eee; }}
  figcaption {{ font-size: 0.85rem; color: #555; text-align: center; }}
</style>
</head>
<body>
<h1>What the agent received</h1>
<p class="lead">Wrist camera and laptop photo are the live frames. The three smaller pictures are the simulated angles sent in the same reply.</p>
{body}
</body>
</html>
"""
