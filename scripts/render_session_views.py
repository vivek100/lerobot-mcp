"""Write the numbers-only page for the hand-drop replies.

Camera photos are not included. The joint numbers are the ones the tool replies returned.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from so101_mcp.planner import MotionPlanner

OUT = Path("artifacts/session_views")
VIEWS = ("preview_front", "preview_side", "preview_top")

POSES = [
    {
        "id": "over_palm",
        "title": "Held over the palm",
        "tool": "move",
        "request": {"left_cm": 8},
        "result": "moved",
        "joints": [-14.46, 33.76, -30.86, 88.92, 3.03, 6.7],
        "pose": "forward 28.9 cm, left 6.4 cm, up 3.3 cm, tilt 88.2°, roll 3.0°, gripper 6.7",
        "clearance": "5.9 cm",
        "room": "forward 1.9 · back 21.9 · left 5.3 · right 17.4 · up 2.4 · down 5.6",
        "note": "Jaws still closed on the Sharpie. Wrist camera showed the palm above the cap.",
    },
    {
        "id": "lifted",
        "title": "Lifted, still holding",
        "tool": "move",
        "request": {"up_cm": 2},
        "result": "moved",
        "joints": [-14.46, 33.76, -39.21, 96.31, 3.03, 6.7],
        "pose": "forward 29.0 cm, left 6.5 cm, up 5.3 cm, tilt 89.1°, roll 3.0°, gripper 6.7",
        "clearance": "7.8 cm",
        "room": "forward 0.0 (reach) · back 19.7 · left 1.5 · right 15.0 · up 0.0 (wrist) · down 7.5",
        "note": "Wrist flex was at its limit (96.3° of 96.5°), so this was as high as that reach allowed.",
    },
    {
        "id": "dropped",
        "title": "Jaws opened",
        "tool": "set_gripper",
        "request": {"openness": 100},
        "result": "moved",
        "joints": [-14.46, 33.76, -39.21, 96.31, 3.03, 99.1],
        "pose": "forward 29.0 cm, left 6.5 cm, up 5.3 cm, tilt 89.1°, roll 3.0°, gripper 99.1",
        "clearance": "7.9 cm",
        "room": "same reach as the lift · down 7.6, now limited by the open gripper",
        "note": "The marker landed on the table just in front of the palm. The arm did not lower.",
    },
    {
        "id": "rest",
        "title": "Folded at rest",
        "tool": "go_rest",
        "request": {},
        "result": "rested",
        "joints": [6.64, -100.57, 92.66, 73.89, 1.01, 1.66],
        "pose": "forward 16.1 cm, left -1.5 cm, up 0.7 cm, tilt 66.0°, roll 1.0°, gripper 1.7",
        "clearance": "2.3 cm",
        "room": "forward 21.9 · back 0.0 · left 6.2 · right 21.9 · up 11.1 · down 2.0",
        "note": "Shoulder lift and elbow are against their rest stops. Gripper closed.",
    },
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    planner = MotionPlanner()
    cards = []
    for pose in POSES:
        frames = planner.previews(np.array(pose["joints"], dtype=np.float64), width=480)
        for name in VIEWS:
            path = OUT / f"{pose['id']}_{name}.jpg"
            Image.fromarray(frames[name]).save(path, quality=85)
        cards.append(pose)
    (OUT / "index.html").write_text(_html(cards), encoding="utf-8")
    print(OUT / "index.html")


def _html(cards: list[dict]) -> str:
    blocks = []
    for pose in cards:
        request = ", ".join(f"{k}={v}" for k, v in pose["request"].items()) or "(none)"
        imgs = "\n".join(
            f'<figure><img src="{pose["id"]}_{name}.jpg" alt="{name}"><figcaption>{name.replace("preview_", "")}</figcaption></figure>'
            for name in VIEWS
        )
        blocks.append(
            f"""<section>
  <h2>{pose["title"]}</h2>
  <p class="tool"><code>{pose["tool"]}</code> request <code>{request}</code> → {pose["result"]}</p>
  <dl>
    <dt>Pose the agent got</dt><dd>{pose["pose"]}</dd>
    <dt>Clearance</dt><dd>{pose["clearance"]}</dd>
    <dt>Room left (cm)</dt><dd>{pose["room"]}</dd>
    <dt>Joints (deg)</dt><dd>{", ".join(str(v) for v in pose["joints"])}</dd>
  </dl>
  <p>{pose["note"]}</p>
  <div class="views">{imgs}</div>
</section>"""
        )
    body = "\n".join(blocks)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>What the agent saw — drop into the hand</title>
<style>
  body {{ font: 16px/1.45 system-ui, sans-serif; margin: 2rem auto; max-width: 980px; color: #1a1a1a; background: #f6f4ef; }}
  h1 {{ font-size: 1.4rem; margin-bottom: 0.3rem; }}
  .lead {{ color: #444; }}
  section {{ background: #fff; border: 1px solid #ddd; border-radius: 10px; padding: 1rem 1.2rem 1.2rem; margin: 1.2rem 0; }}
  h2 {{ font-size: 1.1rem; margin: 0 0 0.4rem; }}
  .tool {{ margin: 0 0 0.6rem; }}
  dl {{ display: grid; grid-template-columns: 11rem 1fr; gap: 0.2rem 0.8rem; margin: 0.4rem 0; }}
  dt {{ color: #555; }}
  dd {{ margin: 0; }}
  code {{ background: #f0eee8; padding: 0.05rem 0.3rem; border-radius: 4px; }}
  .views {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.6rem; margin-top: 0.8rem; }}
  figure {{ margin: 0; }}
  img {{ width: 100%; border-radius: 6px; background: #eee; }}
  figcaption {{ font-size: 0.85rem; color: #555; text-align: center; }}
</style>
</head>
<body>
<h1>Four poses from the drop</h1>
<p class="lead">Each live <code>move</code>, <code>set_gripper</code>, and <code>go_rest</code> already returns these three simulated angles plus the wrist and laptop photos. The photos were not saved. These pictures are redrawn from the joint numbers in those replies. The green bin and cube are the sim scene, not the real table.</p>
<p class="lead">Joint order: shoulder pan, shoulder lift, elbow, wrist flex, wrist roll, gripper (0 closed – 100 open).</p>
{body}
</body>
</html>
"""


if __name__ == "__main__":
    main()
