from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from fl_studio_mcp.engine import client


@pytest.fixture(autouse=True)
def studio_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "studio"
    monkeypatch.setenv("FL_MCP_STUDIO_DIR", str(root))
    return root


@pytest.fixture(scope="session", autouse=True)
def stop_engine() -> Iterator[None]:
    yield
    if client._engine is not None:
        client._engine.stop()


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"
