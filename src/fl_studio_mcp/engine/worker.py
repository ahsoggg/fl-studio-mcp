"""Engine worker process: hosts plugins and renders audio on behalf of the MCP server.

Launched by EngineClient as `python -m fl_studio_mcp.engine.worker <address>`.
Requests arrive over a multiprocessing connection as (operation, payload) and are
answered with (ok, result_or_error_message). The process's stdout/stderr go to a
log file because plugins write there.
"""

from __future__ import annotations

import os
import sys
import traceback
from multiprocessing.connection import Client
from typing import Any

from fl_studio_mcp.engine.model import ProjectError

KEY_ENV = "FL_MCP_ENGINE_KEY"


def handle(renderer: Any, operation: str, payload: dict[str, Any]) -> Any:
    if operation == "ping":
        return "pong"
    if operation == "plugin_info":
        return renderer.plugin_info(
            payload["path"], payload.get("plugin_name"), payload.get("parameters", True)
        )
    if operation == "render":
        return renderer.render(payload["project"], payload["options"])
    if operation == "analyze":
        return renderer.analyze_file(payload["path"])
    raise ValueError(f"Unknown engine operation: {operation}")


def main() -> None:
    from fl_studio_mcp.engine.render import Renderer, RenderError

    connection = Client(sys.argv[1], authkey=bytes.fromhex(os.environ.pop(KEY_ENV)))
    renderer = Renderer()
    while True:
        try:
            operation, payload = connection.recv()
        except (EOFError, OSError):
            return
        try:
            reply: tuple[bool, Any] = (True, handle(renderer, operation, payload))
        except (RenderError, ProjectError) as exc:
            reply = (False, str(exc))
        except Exception as exc:
            traceback.print_exc()
            reply = (False, f"{type(exc).__name__}: {exc}")
        connection.send(reply)


if __name__ == "__main__":
    main()
