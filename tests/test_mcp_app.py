"""Verify the app descriptor/resource/frame contract over the actual MCP protocol."""
import base64
import io

import numpy as np
import pytest
from PIL import Image

pytestmark = pytest.mark.anyio


async def test_wrist_app_is_discoverable_and_read_only(robot):
    tools = {tool.name: tool for tool in (await robot.session.list_tools()).tools}
    viewer = tools["show_wrist_camera"]
    assert viewer.meta["ui"]["resourceUri"] == "ui://so101/wrist-camera.html"
    assert tools["wrist_camera_frame"].meta["ui"]["visibility"] == ["app"]
    before = await robot.status()
    result = await robot.call("show_wrist_camera")
    assert result.structuredContent["read_only"] is True
    assert result.structuredContent["mode"] == "sim"
    resource = await robot.session.read_resource(viewer.meta["ui"]["resourceUri"])
    assert resource.contents[0].mimeType == "text/html;profile=mcp-app"
    assert "ui/initialize" in resource.contents[0].text
    assert "wrist_camera_frame" in resource.contents[0].text
    result = await robot.call("wrist_camera_frame")
    image = next(c for c in result.content if c.type == "image")
    decoded = Image.open(io.BytesIO(base64.b64decode(image.data)))
    assert decoded.size == (640, 480)
    after = await robot.status()
    assert np.allclose(list(before["joints"].values()), list(after["joints"].values()), atol=0.5)
