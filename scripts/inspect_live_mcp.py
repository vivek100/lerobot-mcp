"""Read-only inspection of the existing arm server; never sends motion commands."""
import asyncio
import base64
import json
from pathlib import Path

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client


async def main():
    out = Path("artifacts/agent_usability/live")
    out.mkdir(parents=True, exist_ok=True)
    async with streamable_http_client("http://127.0.0.1:8000/mcp") as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            names = [t.name for t in (await session.list_tools()).tools]
            result = await session.call_tool("observe", {"request": {"previews": True}})
            if result.isError:
                raise RuntimeError("Live observe failed")
            state = json.loads(next(c.text for c in result.content if c.type == "text"))
            images = [c for c in result.content if c.type == "image"]
            for name, img in zip(state["cameras"], images):
                (out / f"{name}.jpg").write_bytes(base64.b64decode(img.data))
            report = {"tools": names, "observation": state, "image_count": len(images)}
            (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(json.dumps({"tools": names, "mode": state["mode"], "images": state["cameras"]}))


if __name__ == "__main__":
    asyncio.run(main())
