"""The view log stores a replay record: tool request, reply, live photos, and sim views."""

import json

from so101_mcp.view_log import ViewLog


def test_view_log_writes_replay_with_live_and_sim_frames(tmp_path):
    log = ViewLog(tmp_path)
    payload = (
        '{"result":"moved","message":"Move finished.","requested":"up cm=2",'
        '"table_clearance_cm":7.8,"cameras":["wrist","front","preview_front","preview_side","preview_top"],'
        '"pose":{"forward_cm":29.0,"left_cm":6.5,"up_cm":5.3,"tilt_down_deg":89.1,"roll_deg":3.0,"gripper":6.7}}'
    )
    log.write(
        "move",
        {"up_cm": 2},
        payload,
        [
            ("wrist", b"wrist-bytes"),
            ("front", b"front-bytes"),
            ("preview_front", b"front-sim"),
            ("preview_side", b"side-sim"),
            ("preview_top", b"top-sim"),
        ],
    )

    session = tmp_path / (tmp_path / "latest.txt").read_text(encoding="utf-8").strip()
    record = json.loads((session / "001" / "step.json").read_text(encoding="utf-8"))
    assert record["tool"] == "move" and record["request"] == {"up_cm": 2}
    assert record["reply"]["pose"]["up_cm"] == 5.3
    assert record["images"] == [
        "wrist.jpg", "front.jpg", "preview_front.jpg", "preview_side.jpg", "preview_top.jpg",
    ]
    assert (session / "001" / "wrist.jpg").read_bytes() == b"wrist-bytes"
    assert (session / "001" / "preview_top.jpg").read_bytes() == b"top-sim"
    lines = (session / "replay.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0])["tool"] == "move"
    page = (session / "index.html").read_text(encoding="utf-8")
    assert "wrist camera" in page and "laptop photo" in page and "sim top" in page
    assert "up_cm" in page
