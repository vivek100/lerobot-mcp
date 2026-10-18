# Continue the real Sharpie pickup

Project: D:\projectsnew\makesomethingagent

The real SO-101 server is already running at http://127.0.0.1:8000/mcp.
Do not launch another hardware owner. The arm was last parked. The yellow Sharpie is on
the table; its current position must be checked from fresh photos.

Project MCP configuration is in .codex/config.toml as `so101`, with automatic tool approvals.
Codex's user configuration contains a trust entry for this folder only; the MCP server itself
was not added globally. A fresh ordinary Codex session (without inline MCP URL overrides)
successfully discovered and called native `so101.robot_guide` after native tool discovery.
Some clients defer tools: use built-in tool discovery before declaring them unavailable.

The user wants the chat agent to interactively attempt a physical Sharpie pickup using
native robot MCP tools, deciding each action from camera feedback. Do not run a scripted
benchmark or hand the robot task to another agent/session. No recording/video work is wanted.

Read AGENT_USABILITY.md. Simulator pickup/place succeeded; the real hardware smoke test
only detected yellow, unparked/moved and parked. No physical Sharpie pickup has been attempted
in the current work. Physical camera calibration is absent: normalized target boxes are not
metric coordinates. Do not use the illustrative model camera cone as measured calibration.

Start by finding native so101 tools, calling robot_guide and observe, and checking faults.
Use slow bounded moves under existing local limits/table/contact guards. Inspect feedback
after each move. If alignment is uncertain, do not descend or close blindly. Stop on faults.
Only claim pickup after fresh camera evidence shows the marker held clear of the table.

The camera capture endpoint added during abandoned recording setup is read-only and is not
needed for this task. Do not spend more time on recording, setup or another benchmark.
