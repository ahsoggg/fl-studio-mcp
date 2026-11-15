"""Instrument profiles: playable ranges, keyswitches and writing tips per plugin."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fl_studio_mcp.engine.model import BUILTIN_INSTRUMENTS

PROFILES_DIR = Path(__file__).resolve().parent.parent / "profiles"


def find_profile(plugin: str) -> dict[str, Any] | None:
    """Return the profile for a plugin name, path or builtin id, if one exists."""
    if plugin in BUILTIN_INSTRUMENTS:
        builtins = json.loads((PROFILES_DIR / "builtin.json").read_text(encoding="utf-8"))
        profile = builtins.get(plugin)
        return profile if isinstance(profile, dict) else None
    wanted = Path(plugin).stem.lower()
    for path in sorted(PROFILES_DIR.glob("*.json")):
        if path.name == "builtin.json":
            continue
        profile = json.loads(path.read_text(encoding="utf-8"))
        if wanted in (name.lower() for name in profile.get("match", [])):
            return dict(profile)
    return None
