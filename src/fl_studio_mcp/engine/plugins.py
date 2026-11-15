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


AUDIO_EXTENSIONS = (".wav", ".flac", ".aif", ".aiff", ".mp3", ".ogg")
PRESET_EXTENSIONS = (".vstpreset",)


def _is_network_or_device(path: Path) -> bool:
    text = str(path)
    return text.startswith(("\\\\", "//")) or path.drive.startswith("\\\\")


def local_file(path: str, extensions: tuple[str, ...], what: str) -> str:
    """Resolve a caller-supplied path to an existing local file of an expected type.

    Network shares and device paths are refused: opening one would send the
    user's credentials to whatever host the path names.
    """
    candidate = Path(path)
    if not candidate.is_absolute() or _is_network_or_device(candidate):
        raise ProjectError(f"{what} must be an absolute path to a local file, got {path!r}")
    if candidate.suffix.lower() not in extensions:
        raise ProjectError(f"{what} must be one of {', '.join(extensions)}, got {path!r}")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise ProjectError(f"{what} not found: {path}") from exc
    if _is_network_or_device(resolved) or not resolved.is_file():
        raise ProjectError(f"{what} must be a local file, got {path!r}")
    return str(resolved)


def allowed_plugin_path(path: str) -> str:
    """Accept a plugin only if it lives in a VST3 search folder: plugins are native code."""
    candidate = Path(path)
    if candidate.suffix.lower() != ".vst3" or _is_network_or_device(candidate):
        raise ProjectError(f"Not a local VST3 plugin path: {path!r}")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise ProjectError(f"Plugin file not found: {path}") from exc
    for root in vst3_search_paths():
        try:
            if not _is_network_or_device(resolved) and resolved.is_relative_to(root.resolve()):
                return str(resolved)
        except OSError:
            continue
    raise ProjectError(
        f"Plugin {path!r} is outside the VST3 folders. Install it there or add its "
        "folder to FL_MCP_VST3_PATHS."
    )


def is_builtin(spec: str) -> bool:
    return spec in BUILTIN_INSTRUMENTS


def resolve_plugin(spec: str) -> str:
    """Turn a plugin name or path into an absolute .vst3 path."""
    candidate = Path(spec)
    if candidate.suffix.lower() == ".vst3" and candidate.is_absolute():
        return allowed_plugin_path(spec)
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
