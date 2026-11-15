"""MIDI-based connection to FL Studio.

This module provides communication with FL Studio via MIDI messages and JSON files.
The approach is:
1. Write command data to a JSON file
2. Send a MIDI trigger note to FL Studio
3. FL Studio's MIDI controller script executes the command
4. Read the response from a JSON file

This is similar to how the piano_roll module works, but uses MIDI for triggering
instead of keystrokes.
"""

from __future__ import annotations

import itertools
import json
import os
import platform
import threading
import time
from pathlib import Path
from typing import Any


def _get_fl_hardware_dir() -> Path:
    """Get the FL Studio Hardware scripts directory."""
    system = platform.system()

    if system in ("Darwin", "Windows"):
        base = Path.home() / "Documents" / "Image-Line" / "FL Studio" / "Settings"
    else:
        # Linux fallback
        base = Path.home() / ".fl-studio" / "Settings"

    hardware_dir = base / "Hardware" / "FLStudioMCP"
    hardware_dir.mkdir(parents=True, exist_ok=True)
    return hardware_dir


class MIDIConnection:
    """MIDI-based connection to FL Studio.

    Communicates with FL Studio via:
    - JSON files for command/response data
    - MIDI trigger note to execute commands
    """

    # MIDI trigger note (same as in FL Studio controller script)
    TRIGGER_NOTE = 127

    def __init__(self) -> None:
        self._port = None
        self._port_name: str | None = None
        self._connected = False
        self._error: str | None = None
        self._lock = threading.Lock()
        self._request_ids = itertools.count(1)

        # File paths for JSON communication
        self._hardware_dir = _get_fl_hardware_dir()
        self._command_file = self._hardware_dir / "mcp_command.json"
        self._response_file = self._hardware_dir / "mcp_response.json"

    @property
    def is_connected(self) -> bool:
        """Check if connected to FL Studio."""
        return self._connected and self._port is not None

    @property
    def connection_error(self) -> str | None:
        """Get the last connection error, if any."""
        return self._error

    def connect(self) -> bool:
        """Attempt to connect to FL Studio via MIDI.

        Returns True if connection successful, False otherwise.
        """
        if self._connected and self._port is not None:
            return True

        try:
            import mido
        except ImportError:
            self._error = (
                "mido library not installed. Install with: pip install mido python-rtmidi"
            )
            return False

        # Find available MIDI output ports
        try:
            output_ports = mido.get_output_names()
        except Exception as e:
            self._error = f"Failed to get MIDI ports: {e}"
            return False

        if not output_ports:
            self._error = (
                "No MIDI output ports found. On Mac, enable IAC Driver in Audio MIDI Setup."
            )
            return False

        # Look for IAC Driver (Mac) or other virtual MIDI ports
        target_port = None
        for port_name in output_ports:
            # Prefer IAC Driver on Mac
            if "IAC" in port_name:
                target_port = port_name
                break
            # Also accept loopMIDI on Windows or any port with "FL" in name
            if "loopMIDI" in port_name or "FL" in port_name.upper():
                target_port = port_name
                break

        # If no specific port found, use the first available
        if target_port is None:
            target_port = output_ports[0]

        try:
            self._port = mido.open_output(target_port)
            self._port_name = target_port
            self._connected = True
            self._error = None
            return True
        except Exception as e:
            self._error = f"Failed to open MIDI port '{target_port}': {e}"
            return False

    def disconnect(self) -> None:
        """Close the MIDI connection."""
        if self._port is not None:
            try:
                self._port.close()
            except Exception:
                pass
            self._port = None
        self._connected = False
        self._port_name = None

    def ensure_connected(self) -> None:
        """Ensure connection to FL Studio is active. Raises RuntimeError if not."""
        if not self.is_connected:
            if not self.connect():
                raise RuntimeError(
                    self._error or "Failed to connect to FL Studio via MIDI"
                )

    def send_command(
        self,
        action: str,
        params: dict[str, Any] | None = None,
        timeout: float = 2.0,
    ) -> dict[str, Any]:
        """Send a command to FL Studio and wait for response.

        Args:
            action: The command action (e.g., "transport.start", "mixer.setTrackVolume")
            params: Optional parameters for the command
            timeout: Maximum time to wait for response in seconds

        Returns:
            Response dictionary from FL Studio

        Raises:
            RuntimeError: If not connected or command fails
        """
        # One command at a time: the bridge shares a single command/response file pair.
        with self._lock:
            self.ensure_connected()

            request_id = f"{os.getpid()}-{next(self._request_ids)}"
            command = {
                "id": request_id,
                "action": action,
                "params": params or {},
            }

            # Clear old response file
            try:
                self._response_file.unlink(missing_ok=True)
            except OSError:
                pass

            # Write the command atomically so FL Studio never reads half a file
            try:
                tmp_file = self._command_file.with_name(self._command_file.name + ".tmp")
                tmp_file.write_text(json.dumps(command, indent=2))
                os.replace(tmp_file, self._command_file)
            except Exception as e:
                return {"success": False, "error": f"Failed to write command file: {e}"}

            # Send MIDI trigger
            try:
                import mido
                trigger_msg = mido.Message("note_on", note=self.TRIGGER_NOTE, velocity=127)
                self._port.send(trigger_msg)
            except Exception as e:
                return {"success": False, "error": f"Failed to send MIDI trigger: {e}"}

            # Wait for response
            return self._wait_for_response(timeout, request_id)

    def _wait_for_response(self, timeout: float, request_id: str | None = None) -> dict[str, Any]:
        """Wait for the response file to appear and read it.

        A file that cannot be parsed yet is treated as still being written and is
        read again until the timeout. A response that carries the id of another
        request is stale and is discarded.

        Args:
            timeout: Maximum time to wait in seconds
            request_id: Id of the command this response must belong to

        Returns:
            Response dictionary or error dict if timeout
        """
        deadline = time.monotonic() + timeout
        poll_interval = 0.02  # 20ms between checks
        read_error: str | None = None

        while time.monotonic() < deadline:
            if self._response_file.exists():
                try:
                    response = json.loads(self._response_file.read_text())
                except (json.JSONDecodeError, OSError) as e:
                    read_error = str(e)
                    time.sleep(poll_interval)
                    continue

                # Clean up response file
                try:
                    self._response_file.unlink(missing_ok=True)
                except OSError:
                    pass

                if not isinstance(response, dict):
                    return {"success": False, "error": "Invalid response from FL Studio"}
                response_id = response.pop("id", None)
                if request_id is None or response_id is None or response_id == request_id:
                    return response
                time.sleep(poll_interval)
                continue

            time.sleep(poll_interval)

        detail = f" Last read error: {read_error}." if read_error else ""
        return {
            "success": False,
            "error": (
                f"Timeout waiting for FL Studio response after {timeout}s. "
                "Make sure FL Studio is running and the MCP controller is enabled "
                f"in MIDI Settings.{detail}"
            ),
        }

    def get_status(self) -> dict[str, Any]:
        """Get connection status information."""
        try:
            import mido
            output_ports = mido.get_output_names()
        except Exception:
            output_ports = []

        return {
            "connected": self.is_connected,
            "port_name": self._port_name,
            "available_ports": output_ports,
            "command_file": str(self._command_file),
            "response_file": str(self._response_file),
            "error": self._error,
        }


# Global connection instance
_connection: MIDIConnection | None = None


def get_connection() -> MIDIConnection:
    """Get the global MIDI connection instance."""
    global _connection
    if _connection is None:
        _connection = MIDIConnection()
    return _connection


def reset_connection() -> None:
    """Reset the connection state to allow reconnection attempts."""
    global _connection
    if _connection is not None:
        _connection.disconnect()
        _connection = None
