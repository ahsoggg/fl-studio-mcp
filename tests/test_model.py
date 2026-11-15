from __future__ import annotations

from typing import Any

import pytest

from fl_studio_mcp.engine import store
from fl_studio_mcp.engine.model import Effect, Note, Project, ProjectError, Track


def test_project_roundtrips_through_storage() -> None:
    project = Project(name="Song 1", tempo=96, time_signature=[6, 8])
    track = Track(id="t1", name="Lead", plugin="builtin:sine")
    track.notes = [Note(60, 0.0, 1.0), Note(64, 1.5, 0.5, 0.4)]
    track.effects = [Effect("Reverb", {"room_size": 0.3})]
    project.tracks.append(track)
    store.save_project(project)

    loaded = store.load_project("Song 1")
    assert loaded.to_dict() == project.to_dict()
    assert loaded.beats_per_bar == 3.0
    assert loaded.length_beats() == 2.0
    assert store.list_projects() == ["Song 1"]


@pytest.mark.parametrize(
    "note",
    [
        {"midi": 128, "duration": 1},
        {"midi": 60.5, "duration": 1},
        {"midi": 60, "duration": 0},
        {"midi": 60, "duration": 1, "time": -1},
        {"midi": 60, "duration": 1, "velocity": 100},
        {"midi": 60, "duration": 1, "velocity": 0},
        {"midi": 60},
        {"midi": True, "duration": 1},
    ],
)
def test_invalid_notes_are_rejected(note: dict[str, Any]) -> None:
    with pytest.raises(ProjectError):
        Note.from_dict(note)


@pytest.mark.parametrize(
    "name",
    ["", "../evil", "a/b", "x" * 65, " lead", "con:1", "abc\n", "abc ", "CON", "nul", "Lpt1"],
)
def test_unsafe_names_are_rejected(name: str) -> None:
    with pytest.raises(ProjectError):
        store.project_path(name)


def test_effect_validation() -> None:
    with pytest.raises(ProjectError):
        Effect.from_dict({"type": "Nope"})
    with pytest.raises(ProjectError):
        Effect.from_dict({"type": "vst3"})
    with pytest.raises(ProjectError):
        Effect.from_dict({"type": "vst3", "plugin": "X", "params": {"mix": 2}})
    assert Effect.from_dict({"type": "Gain", "params": {"gain_db": -3}}).params == {"gain_db": -3}


def test_find_track_by_id_or_name() -> None:
    project = Project(name="p")
    project.tracks = [Track("t1", "Bass", "builtin:sine"), Track("t2", "Lead", "builtin:sine")]
    assert project.find_track("t2").name == "Lead"
    assert project.find_track("bass").id == "t1"
    assert project.next_track_id() == "t3"
    with pytest.raises(ProjectError):
        project.find_track("drums")
