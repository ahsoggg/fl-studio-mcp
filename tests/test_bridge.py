"""The MIDI + JSON bridge to FL Studio, exercised against a fake controller."""

from __future__ import annotations

import json
import re
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from fl_studio_mcp.utils import midi_connection
from fl_studio_mcp.utils.midi_connection import MIDIConnection

REPO = Path(__file__).resolve().parent.parent


class FakePort:
    """Stands in for the MIDI port: a trigger runs the fake controller in a thread."""

    def __init__(self, on_trigger: Callable[[], None]) -> None:
        self._on_trigger = on_trigger

    def send(self, message: Any) -> None:
        threading.Thread(target=self._on_trigger, daemon=True).start()


def make_connection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, controller: Callable[[MIDIConnection], None]
) -> MIDIConnection:
    monkeypatch.setattr(midi_connection, "_get_fl_hardware_dir", lambda: tmp_path)
    connection = MIDIConnection()
    connection._port = FakePort(lambda: controller(connection))  # type: ignore[assignment]
    connection._connected = True
    return connection


def read_command(connection: MIDIConnection) -> dict[str, Any]:
    return json.loads(connection._command_file.read_text())


def test_roundtrip_matches_request_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def controller(connection: MIDIConnection) -> None:
        command = read_command(connection)
        reply = {"success": True, "id": command["id"], "echo": command["action"]}
        connection._response_file.write_text(json.dumps(reply))

    connection = make_connection(tmp_path, monkeypatch, controller)
    result = connection.send_command("transport.start")
    assert result == {"success": True, "echo": "transport.start"}
    assert not connection._response_file.exists()


def test_half_written_response_is_read_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def controller(connection: MIDIConnection) -> None:
        command = read_command(connection)
        text = json.dumps({"success": True, "id": command["id"], "value": 7})
        connection._response_file.write_text(text[: len(text) // 2])
        time.sleep(0.15)
        connection._response_file.write_text(text)

    connection = make_connection(tmp_path, monkeypatch, controller)
    assert connection.send_command("mixer.getTrackCount") == {"success": True, "value": 7}


def test_stale_response_is_discarded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def controller(connection: MIDIConnection) -> None:
        command = read_command(connection)
        stale = {"success": True, "id": "old", "value": 1}
        connection._response_file.write_text(json.dumps(stale))
        time.sleep(0.15)
        fresh = {"success": True, "id": command["id"], "value": 2}
        connection._response_file.write_text(json.dumps(fresh))

    connection = make_connection(tmp_path, monkeypatch, controller)
    assert connection.send_command("mixer.getTrackCount") == {"success": True, "value": 2}


def test_controller_without_ids_still_works(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def controller(connection: MIDIConnection) -> None:
        connection._response_file.write_text(json.dumps({"success": True, "value": 3}))

    connection = make_connection(tmp_path, monkeypatch, controller)
    assert connection.send_command("mixer.getTrackCount") == {"success": True, "value": 3}


def test_timeout_reports_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    connection = make_connection(tmp_path, monkeypatch, lambda connection: None)
    result = connection.send_command("transport.start", timeout=0.2)
    assert result["success"] is False
    assert "Timeout" in result["error"]


def test_concurrent_commands_do_not_mix(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def controller(connection: MIDIConnection) -> None:
        command = read_command(connection)
        time.sleep(0.02)
        reply = {"success": True, "id": command["id"], "n": command["params"]["n"]}
        tmp = connection._response_file.with_name("reply.tmp")
        tmp.write_text(json.dumps(reply))
        tmp.replace(connection._response_file)

    connection = make_connection(tmp_path, monkeypatch, controller)
    results: dict[int, Any] = {}

    def worker(n: int) -> None:
        results[n] = connection.send_command("mixer.getTrackInfo", {"n": n})

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert results == {n: {"success": True, "n": n} for n in range(8)}


def test_server_and_controller_know_the_same_actions() -> None:
    pattern = re.compile(r'"((?:transport|mixer|channels|plugins)\.[A-Za-z]+)"')
    sent: set[str] = set()
    for path in (REPO / "src" / "fl_studio_mcp").rglob("*.py"):
        sent.update(pattern.findall(path.read_text(encoding="utf-8")))
    controller = (REPO / "fl_controller" / "device_FLStudioMCP.py").read_text(encoding="utf-8")
    handled = set(re.findall(r'action == "([^"]+)"', controller))
    assert sent == handled
