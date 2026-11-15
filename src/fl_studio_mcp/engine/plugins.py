"""Discovery of installed VST3 plugins (file-system only, nothing is loaded here)."""

from __future__ import annotations

import difflib
import os
import platform
from pathlib import Path

from fl_studio_mcp.engine.model import BUILTIN_INSTRUMENTS, ProjectError


def vst3_search_paths() -> list[Path]:
    paths: list[Path] = []
    extra = os.environ.get("FL_MCP_VST3_PATHS", "")
    paths.extend(Path(p) for p in extra.split(os.pathsep) if p)
    system = platform.system()
    if system == "Windows":
        common = os.environ.get("CommonProgramFiles", r"C:\Program Files\Common Files")
        paths.append(Path(common) / "VST3")
    elif system == "Darwin":
        paths.append(Path("/Library/Audio/Plug-Ins/VST3"))
        paths.append(Path.home() / "Library" / "Audio" / "Plug-Ins" / "VST3")
    else:
        paths.append(Path.home() / ".vst3")
        paths.append(Path("/usr/lib/vst3"))
        paths.append(Path("/usr/local/lib/vst3"))
    return paths


def _walk(root: Path, depth: int = 0) -> list[Path]:
    found: list[Path] = []
    try:
        entries = sorted(root.iterdir())
    except OSError:
        return found
    for entry in entries:
        if entry.suffix.lower() == ".vst3":
            found.append(entry)
        elif depth < 3 and entry.is_dir():
            found.extend(_walk(entry, depth + 1))
    return found


def scan_plugins() -> list[dict[str, str]]:
    """List installed VST3 plugins as {name, path}; the first path wins on name clashes."""
    seen: dict[str, dict[str, str]] = {}
    for root in vst3_search_paths():
        for path in _walk(root):
            seen.setdefault(path.stem.lower(), {"name": path.stem, "path": str(path)})
    return sorted(seen.values(), key=lambda p: p["name"].lower())


def is_builtin(spec: str) -> bool:
    return spec in BUILTIN_INSTRUMENTS


def resolve_plugin(spec: str) -> str:
    """Turn a plugin name or path into an absolute .vst3 path."""
    candidate = Path(spec)
    if candidate.suffix.lower() == ".vst3" and candidate.is_absolute():
        if not candidate.exists():
            raise ProjectError(f"Plugin file not found: {spec}")
        return str(candidate)
    plugins = scan_plugins()
    wanted = spec.lower().removesuffix(".vst3")
    for plugin in plugins:
        if plugin["name"].lower() == wanted:
            return plugin["path"]
    names = [p["name"] for p in plugins]
    close = difflib.get_close_matches(spec, names, n=5, cutoff=0.4)
    hint = f" Did you mean: {', '.join(close)}?" if close else ""
    raise ProjectError(
        f"Plugin {spec!r} is not installed.{hint} Use studio_list_plugins to see all names."
    )


def binary_path(path: str) -> str:
    """Path to hand to the plugin host: the module inside a .vst3 bundle folder on Windows."""
    bundle = Path(path)
    if platform.system() == "Windows" and bundle.is_dir():
        for arch in ("x86_64-win", "x86-win"):
            inner = bundle / "Contents" / arch / bundle.name
            if inner.is_file():
                return str(inner)
    return path
