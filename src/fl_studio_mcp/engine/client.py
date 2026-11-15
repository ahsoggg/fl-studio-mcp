"""Client side of the engine worker: process lifecycle, requests, plugin probing."""

from __future__ import annotations

import atexit
import json
import os
import secrets
import subprocess
import sys
import threading
from multiprocessing.connection import Listener
from pathlib import Path
from typing import Any

from fl_studio_mcp.engine import store
from fl_studio_mcp.engine.worker import KEY_ENV

START_TIMEOUT = 60.0
PROBE_TIMEOUT = 120.0


class EngineError(RuntimeError):
    """The engine could not complete a request; the message is shown to the model."""


class EngineTimeout(EngineError):
    """The worker did not answer in time (says nothing about the plugin itself)."""


class EngineClient:
    """Owns one worker process and serialises requests to it."""

    def __init__(self, label: str = "worker") -> None:
        self._label = label
        self._lock = threading.Lock()
        self._process: subprocess.Popen[bytes] | None = None
        self._connection: Any = None

    @property
    def log_path(self) -> Path:
        return store.logs_dir() / f"engine_{self._label}.log"

    def _start(self) -> None:
        key = secrets.token_bytes(32)
        listener = Listener(authkey=key)
        accepted: list[Any] = []

        def accept() -> None:
            try:
                accepted.append(listener.accept())
            except Exception:
                pass

        env = dict(os.environ)
        env[KEY_ENV] = key.hex()
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        with open(self.log_path, "ab") as log:
            process = subprocess.Popen(
                [sys.executable, "-m", "fl_studio_mcp.engine.worker", str(listener.address)],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
                creationflags=flags,
            )
        thread = threading.Thread(target=accept, daemon=True)
        thread.start()
        waited = 0.0
        while thread.is_alive() and waited < START_TIMEOUT and process.poll() is None:
            thread.join(0.2)
            waited += 0.2
        listener.close()
        if not accepted:
            process.kill()
            raise EngineError(f"Engine worker failed to start; see {self.log_path}")
        self._process = process
        self._connection = accepted[0]

    def stop(self) -> None:
        with self._lock:
            self._stop()

    def _stop(self) -> None:
        if self._connection is not None:
            try:
                self._connection.close()
            except OSError:
                pass
            self._connection = None
        if self._process is not None:
            if self._process.poll() is None:
                self._process.kill()
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
            self._process = None

    def call(
        self, operation: str, payload: dict[str, Any] | None = None, timeout: float = 900.0
    ) -> Any:
        with self._lock:
            alive = self._process is not None and self._process.poll() is None
            if not alive or self._connection is None:
                self._stop()
                self._start()
            connection = self._connection
            assert connection is not None
            try:
                connection.send((operation, payload or {}))
                if not connection.poll(timeout):
                    self._stop()
                    raise EngineTimeout(
                        f"Engine did not answer '{operation}' within {timeout:.0f}s"
                    )
                ok, result = connection.recv()
            except (EOFError, OSError) as exc:
                self._stop()
                raise EngineError(
                    f"Engine worker crashed during '{operation}' (usually a plugin crash); "
                    f"see {self.log_path}"
                ) from exc
        if not ok:
            raise EngineError(str(result))
        return result


_engine: EngineClient | None = None
_engine_lock = threading.Lock()


def get_engine() -> EngineClient:
    global _engine
    with _engine_lock:
        if _engine is None:
            _engine = EngineClient()
            atexit.register(_engine.stop)
        return _engine


def _compat_path() -> Path:
    return store.studio_dir() / "plugin_compat.json"


def _read_compat() -> dict[str, Any]:
    try:
        data = json.loads(_compat_path().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def probe_plugin(
    path: str, plugin_name: str | None = None, refresh: bool = False
) -> dict[str, Any]:
    """Load a plugin once in a throwaway process and remember whether it works.

    Some plugins crash the host when loaded headless; probing them in their own
    process keeps the main worker (and its already loaded instruments) alive.
    """
    try:
        mtime = os.path.getmtime(path)
    except OSError as exc:
        raise EngineError(f"Plugin file not found: {path}") from exc
    cache_key = f"{path}::{plugin_name or ''}"
    with _engine_lock:
        cached = _read_compat().get(cache_key)
    if cached and cached.get("mtime") == mtime and not refresh:
        entry = cached
    else:
        prober = EngineClient(label="probe")
        request = {"path": path, "plugin_name": plugin_name, "parameters": False}
        try:
            info = prober.call("plugin_info", request, timeout=PROBE_TIMEOUT)
            try:
                request["parameters"] = True
                info = prober.call("plugin_info", request, timeout=PROBE_TIMEOUT)
                info["parameters_available"] = True
            except EngineTimeout:
                raise
            except EngineError:
                # The plugin loads and plays but crashes when its parameters are read.
                info["parameters"] = []
                info["parameters_available"] = False
            entry = {"mtime": mtime, "ok": True, "info": info}
        except EngineTimeout as exc:
            raise EngineError(f"Probing {Path(path).stem} timed out: {exc}") from exc
        except EngineError as exc:
            entry = {"mtime": mtime, "ok": False, "error": str(exc)}
        finally:
            prober.stop()
        with _engine_lock:
            compat = _read_compat()
            compat[cache_key] = entry
            store.write_text_atomic(_compat_path(), json.dumps(compat))
    if not entry["ok"]:
        raise EngineError(
            f"Plugin {Path(path).stem} cannot be used by the studio engine: {entry['error']}"
        )
    info = entry["info"]
    assert isinstance(info, dict)
    return info
