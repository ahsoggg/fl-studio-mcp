"""Studio engine end to end: MCP tools -> worker process -> rendered audio."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import mido
import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError

from fl_studio_mcp.engine.model import Note, Project, Track
from fl_studio_mcp.engine.render import apply_mix, midi_messages, track_events
from fl_studio_mcp.server import mcp

pytestmark = pytest.mark.anyio


async def call(client: Client[Any], tool: str, **arguments: Any) -> Any:
    result = await client.call_tool(tool, arguments)
    return result.data


def beat_notes(pitches: list[int], step: float = 1.0) -> list[dict[str, Any]]:
    return [{"midi": p, "time": i * step, "duration": step * 0.9} for i, p in enumerate(pitches)]


def test_track_events_cut_notes_to_the_region() -> None:
    track = Track("t1", "Lead", "builtin:sine")
    track.notes = [Note(24, 0.0, 8.0), Note(60, 1.0, 1.0), Note(62, 4.0, 2.0), Note(64, 9.0, 1.0)]
    events = track_events(track, seconds_per_beat=0.5, start_beat=4.0, end_beat=8.0)
    assert events == [(0.0, 2.0, 24, 102), (0.0, 1.0, 62, 102)]


def test_midi_messages_put_note_offs_and_keyswitches_first() -> None:
    messages = midi_messages([(0.0, 1.0, 52, 100), (0.0, 1.0, 26, 100), (1.0, 2.0, 52, 90)])
    order = [(m.type, m.note) for m in messages]
    assert order[:2] == [("note_on", 26), ("note_on", 52)]
    assert order.index(("note_off", 52)) < len(order) - 2
    assert [m.time for m in messages] == sorted(m.time for m in messages)


def test_apply_mix_pan_and_gain() -> None:
    import numpy as np

    audio = np.ones((2, 4), dtype=np.float32)
    assert apply_mix(audio, 0.0, 0.0) == pytest.approx(audio, abs=1e-6)
    left = apply_mix(audio, -6.0, -1.0)
    assert left[1].max() == pytest.approx(0.0, abs=1e-6)
    assert left[0].max() == pytest.approx(10 ** (-6 / 20) * 2**0.5, abs=1e-4)


async def test_compose_render_and_export() -> None:
    async with Client(mcp) as client:
        await call(client, "studio_create_project", name="demo", tempo=120)
        drums = await call(client, "studio_add_track", project="demo", name="Drums",
                           plugin="builtin:drums")
        await call(client, "studio_add_track", project="demo", name="Bass", plugin="builtin:sine")
        assert drums["id"] == "t1"

        await call(client, "studio_write_notes", project="demo", track="Drums",
                   notes=beat_notes([36, 38, 36, 38]))
        copied = await call(client, "studio_copy_notes", project="demo", track="Drums",
                            source_start=0, source_end=4, dest_start=4, repeat=3)
        assert copied["total_notes"] == 16
        await call(client, "studio_write_notes", project="demo", track="t2",
                   notes=beat_notes([40, 43, 45, 47], step=4.0))
        await call(client, "studio_set_track_mix", project="demo", track="Bass",
                   volume_db=-6, pan=-0.5)
        await call(client, "studio_set_effects", project="demo", target="master",
                   effects=[{"type": "Limiter", "params": {"threshold_db": -1}}])

        overview = await call(client, "studio_get_project", project="demo")
        assert overview["length_bars"] == pytest.approx(15.9 / 4, abs=0.01)

        result = await call(client, "studio_render", project="demo", stems=True)
        analysis = result["analysis"]
        assert Path(result["path"]).stat().st_size > 100_000
        assert len(result["stems"]) == 2
        assert analysis["silent"] is False
        assert analysis["duration_seconds"] == pytest.approx(15.9 * 0.5 + 2.0, abs=0.05)
        assert -30 < analysis["lufs_integrated"] < -5
        assert [t["name"] for t in result["tracks"]] == ["Drums", "Bass"]
        assert result["warnings"] == []

        section = await call(client, "studio_render", project="demo", start_beat=4, end_beat=8,
                             tail_seconds=0)
        assert section["analysis"]["duration_seconds"] == pytest.approx(2.0, abs=0.01)

        again = await call(client, "studio_analyze_audio", path=result["path"])
        assert again["analysis"]["lufs_integrated"] == pytest.approx(
            analysis["lufs_integrated"], abs=0.3
        )

        exported = await call(client, "studio_export_midi", project="demo")
        song = mido.MidiFile(exported["song"])
        assert len(song.tracks) == 3
        note_ons = [m for m in song.tracks[1] if m.type == "note_on"]
        assert len(note_ons) == 16
        assert mido.tempo2bpm(song.tracks[0][0].tempo) == pytest.approx(120)


async def test_note_editing_tools() -> None:
    async with Client(mcp) as client:
        await call(client, "studio_create_project", name="edit")
        await call(client, "studio_add_track", project="edit", name="Lead", plugin="builtin:sine")
        await call(client, "studio_write_notes", project="edit", track="Lead",
                   notes=beat_notes([60, 62, 64, 65, 67, 69, 71, 72]))
        replaced = await call(client, "studio_write_notes", project="edit", track="Lead",
                              notes=[{"midi": 48, "time": 2, "duration": 2}],
                              mode="replace_range", range_start=2, range_end=4)
        assert (replaced["removed"], replaced["total_notes"]) == (2, 7)
        await call(client, "studio_delete_notes", project="edit", track="Lead", start=0, end=2,
                   midi=60)
        notes = await call(client, "studio_get_notes", project="edit", track="Lead", start=0,
                           end=4)
        assert [n["midi"] for n in notes["notes"]] == [62, 48]
        up = await call(client, "studio_copy_notes", project="edit", track="Lead",
                        source_start=4, source_end=8, dest_start=8, transpose=12)
        assert up["track_length_beats"] == pytest.approx(11.9)


async def test_bad_input_is_reported_as_tool_error() -> None:
    async with Client(mcp) as client:
        await call(client, "studio_create_project", name="errors")
        with pytest.raises(ToolError, match="already exists"):
            await call(client, "studio_create_project", name="errors")
        with pytest.raises(ToolError, match="not installed"):
            await call(client, "studio_add_track", project="errors", name="X",
                       plugin="No Such Plugin 123")
        await call(client, "studio_add_track", project="errors", name="Lead",
                   plugin="builtin:sine")
        with pytest.raises(ToolError, match="velocity"):
            await call(client, "studio_write_notes", project="errors", track="Lead",
                       notes=[{"midi": 60, "duration": 1, "velocity": 100}])
        with pytest.raises(ToolError, match="Nothing to render"):
            await call(client, "studio_render", project="errors")
        with pytest.raises(ToolError, match="start_beat"):
            await call(client, "studio_render", project="errors", start_beat=-4)
        with pytest.raises(ToolError, match="unknown type"):
            await call(client, "studio_set_effects", project="errors", target="Lead",
                       effects=[{"type": "Sparkle"}])
        with pytest.raises(ToolError, match="not found"):
            await call(client, "studio_get_project", project="missing")


async def test_sampler_plays_audio_files(tmp_path: Path) -> None:
    import numpy as np

    from fl_studio_mcp.engine.render import write_audio

    sample = tmp_path / "blip.wav"
    t = np.arange(4410) / 44100
    blip = (0.5 * np.sin(2 * np.pi * 880 * t)).astype(np.float32)
    write_audio(sample, np.stack([blip, blip]), 44100)

    async with Client(mcp) as client:
        await call(client, "studio_create_project", name="sampler")
        await call(client, "studio_add_track", project="sampler", name="Hits",
                   plugin="builtin:sampler", samples={"36": str(sample)})
        await call(client, "studio_write_notes", project="sampler", track="Hits",
                   notes=[{"midi": 36, "time": 0, "duration": 1},
                          {"midi": 40, "time": 1, "duration": 1}])
        result = await call(client, "studio_render", project="sampler", tail_seconds=0)
        assert result["tracks"][0]["peak_dbfs"] == pytest.approx(-6.0 - 1.94, abs=0.3)
        assert any("No sample mapped" in w for w in result["warnings"])


def test_profiles_exist_for_builtins_and_ample_guitar() -> None:
    from fl_studio_mcp.engine.profiles import find_profile

    guitar = find_profile(r"C:\Program Files\Common Files\VST3\AGM.vst3")
    assert guitar is not None
    assert (guitar["playable_range"]["low"], guitar["playable_range"]["high"]) == (40, 84)
    assert {"midi": 26, "name": "Palm Mute"}.items() <= guitar["keyswitches"][2].items()
    assert find_profile("builtin:drums") is not None
    assert find_profile("Some Unknown Synth") is None


def test_storage_never_touches_the_real_documents_folder(studio_dir: Path) -> None:
    from fl_studio_mcp.engine import store

    store.save_project(Project(name="isolated"))
    assert (studio_dir / "projects" / "isolated.json").exists()


@pytest.mark.skipif(
    not os.environ.get("FL_MCP_TEST_VST3"),
    reason="set FL_MCP_TEST_VST3 to an installed VST3 instrument name to run",
)
async def test_real_vst3_instrument_renders_sound() -> None:
    plugin = os.environ["FL_MCP_TEST_VST3"]
    async with Client(mcp) as client:
        info = await call(client, "studio_get_plugin_info", plugin=plugin, limit=5)
        assert info["is_instrument"] is True
        await call(client, "studio_create_project", name="vst")
        await call(client, "studio_add_track", project="vst", name="Inst", plugin=plugin)
        await call(client, "studio_write_notes", project="vst", track="Inst",
                   notes=beat_notes([52, 55, 59, 64]))
        result = await call(client, "studio_render", project="vst")
        assert result["analysis"]["silent"] is False
        assert result["warnings"] == []


def test_plugin_that_fails_to_initialise_is_not_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    from fl_studio_mcp.engine.render import Renderer, RenderError

    class FakePlugin:
        is_instrument = True
        parameters: dict[str, Any] = {}

    renderer = Renderer()
    monkeypatch.setattr(renderer, "_load", lambda path, plugin_name: FakePlugin())
    track = Track("t1", "Lead", "Fake.vst3", params={"cutoff": 0.5})
    key = ("instrument", "p", "t1")
    with pytest.raises(RenderError, match="no parameter 'cutoff'"):
        renderer._prepare_instrument(key, track)
    assert key not in renderer._loaded

    track.params = {}
    renderer._prepare_instrument(key, track)
    assert key in renderer._loaded
