"""On-disk storage for studio projects, renders and exports."""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from fl_studio_mcp.engine.model import Project, ProjectError, validate_name

lock = threading.RLock()


def studio_dir() -> Path:
    """Root data directory (override with FL_MCP_STUDIO_DIR)."""
    override = os.environ.get("FL_MCP_STUDIO_DIR")
    root = Path(override) if override else Path.home() / "Documents" / "FL Studio MCP"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _subdir(*parts: str) -> Path:
    path = studio_dir().joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path


def project_path(name: str) -> Path:
    return _subdir("projects") / f"{validate_name(name)}.json"


def renders_dir(name: str) -> Path:
    return _subdir("renders", validate_name(name))


def midi_dir(name: str) -> Path:
    return _subdir("midi", validate_name(name))


def logs_dir() -> Path:
    return _subdir("logs")


def write_text_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def list_projects() -> list[str]:
    return sorted(p.stem for p in _subdir("projects").glob("*.json"))


def project_exists(name: str) -> bool:
    return project_path(name).exists()


def load_project(name: str) -> Project:
    path = project_path(name)
    if not path.exists():
        known = ", ".join(list_projects()) or "none"
        raise ProjectError(f"Project {name!r} not found. Existing projects: {known}")
    try:
        return Project.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ProjectError(f"Project file {path} is corrupted: {exc}") from exc


def save_project(project: Project) -> Path:
    path = project_path(project.name)
    write_text_atomic(path, json.dumps(project.to_dict(), indent=1))
    return path


@contextmanager
def edit_project(name: str) -> Generator[Project]:
    """Load a project, let the caller mutate it, and save it if no error was raised."""
    with lock:
        project = load_project(name)
        yield project
        save_project(project)
