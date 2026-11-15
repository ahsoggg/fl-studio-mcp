"""Studio engine tools: compose, render and analyse music without FL Studio's limits.

The studio engine hosts installed VST3 plugins itself, so a model can pick
instruments, arrange a whole song on a timeline, render it to audio and read the
analysis of the result. Finished parts can be exported to FL Studio as MIDI files
or sent straight into the piano roll.
"""

from __future__ import annotations

import functools
import time
from collections.abc import Callable
from typing import Any, TypeVar

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

from fl_studio_mcp.engine import store
from fl_studio_mcp.engine.client import EngineError, get_engine, probe_plugin
from fl_studio_mcp.engine.midi_export import export_midi
from fl_studio_mcp.engine.model import (
    BUILTIN_EFFECTS,
    BUILTIN_INSTRUMENTS,
    BUILTIN_SAMPLER,
    VST3_EFFECT,
    Effect,
    Note,
    Project,
    ProjectError,
    Track,
    validate_name,
)
from fl_studio_mcp.engine.plugins import (
    AUDIO_EXTENSIONS,
    PRESET_EXTENSIONS,
    is_builtin,
    local_file,
    resolve_plugin,
    scan_plugins,
)
from fl_studio_mcp.engine.profiles import find_profile

F = TypeVar("F", bound=Callable[..., Any])

_NO_PARAMETERS = (
    "{name} plays fine but crashes when its parameters are read without its window, "
    "so its parameters cannot be changed from here"
)


def _tool_errors(func: F) -> F:
    """Report bad input and engine failures to the client as tool errors."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except (ProjectError, EngineError) as exc:
            raise ToolError(str(exc)) from exc

    return wrapper  # type: ignore[return-value]


def _parse_notes(notes: list[dict[str, Any]]) -> list[Note]:
    if not isinstance(notes, list):
        raise ProjectError("notes must be a list of note objects")
    return [Note.from_dict(note, index) for index, note in enumerate(notes)]


def _note_dict(note: Note) -> dict[str, Any]:
    return {
        "midi": note.midi,
        "time": round(note.time, 4),
        "duration": round(note.duration, 4),
        "velocity": round(note.velocity, 3),
    }


def _track_summary(project: Project, track: Track) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "id": track.id,
        "name": track.name,
        "plugin": track.plugin,
        "notes": len(track.notes),
        "volume_db": track.volume_db,
        "pan": track.pan,
        "mute": track.mute,
        "effects": [
            {"type": e.type, "plugin": e.plugin, "params": e.params} for e in track.effects
        ],
    }
    if track.notes:
        summary["first_beat"] = round(min(n.time for n in track.notes), 3)
        summary["last_beat"] = round(track.length_beats(), 3)
        pitches = [n.midi for n in track.notes]
        summary["pitch_range"] = [min(pitches), max(pitches)]
    if track.params:
        summary["params"] = track.params
    if track.preset:
        summary["preset"] = track.preset
    if track.samples:
        summary["samples"] = track.samples
    return summary


def _project_summary(project: Project) -> dict[str, Any]:
    beats = project.length_beats()
    return {
        "name": project.name,
        "tempo": project.tempo,
        "time_signature": f"{project.time_signature[0]}/{project.time_signature[1]}",
        "beats_per_bar": project.beats_per_bar,
        "length_beats": round(beats, 3),
        "length_bars": round(beats / project.beats_per_bar, 2),
        "length_seconds": round(beats * project.seconds_per_beat, 2),
        "tracks": [_track_summary(project, t) for t in project.tracks],
        "master_effects": [
            {"type": e.type, "plugin": e.plugin, "params": e.params}
            for e in project.master_effects
        ],
    }


def _parse_time_signature(value: str) -> list[int]:
    try:
        numerator, denominator = (int(part) for part in value.split("/"))
    except ValueError as exc:
        raise ProjectError(f"time_signature must look like '4/4', got {value!r}") from exc
    return [numerator, denominator]


def _resolve_effects(effects: list[dict[str, Any]]) -> list[Effect]:
    if not isinstance(effects, list):
        raise ProjectError("effects must be a list")
    parsed = [Effect.from_dict(effect, index) for index, effect in enumerate(effects)]
    for effect in parsed:
        if effect.type == VST3_EFFECT:
            path = resolve_plugin(str(effect.plugin))
            info = probe_plugin(path)
            if not info["is_effect"]:
                raise ProjectError(f"{info['name']} is an instrument, not an effect")
            known = {p["name"] for p in info["parameters"]}
            unknown = sorted(set(effect.params) - known)
            if effect.params and not info.get("parameters_available", True):
                raise ProjectError(_NO_PARAMETERS.format(name=info["name"]))
            if unknown:
                raise ProjectError(
                    f"{info['name']} has no parameters {unknown}; "
                    "see studio_get_plugin_info for the names"
                )
            effect.plugin = path
    return parsed


def register_studio_tools(mcp: FastMCP) -> None:
    """Register studio engine tools with the MCP server."""

    @mcp.tool()
    @_tool_errors
    def studio_list_plugins(name_filter: str = "") -> dict:
        """List instruments and effects the studio engine can use.

        Returns the built-in instruments, the built-in effect types and every
        installed VST3 plugin (name only; nothing is loaded). Use
        studio_get_plugin_info to learn whether a VST3 is an instrument or an
        effect and to see its parameters.

        Args:
            name_filter: Optional case-insensitive substring to narrow the VST3 list
        """
        plugins = [p["name"] for p in scan_plugins() if name_filter.lower() in p["name"].lower()]
        return {
            "builtin_instruments": list(BUILTIN_INSTRUMENTS),
            "builtin_effects": list(BUILTIN_EFFECTS),
            "vst3_plugins": plugins,
        }

    @mcp.tool()
    @_tool_errors
    def studio_get_plugin_info(
        plugin: str, name_filter: str = "", limit: int = 60, refresh: bool = False
    ) -> dict:
        """Load a VST3 plugin and report its type and parameters.

        The first call for a plugin loads it in an isolated process (this can take
        10-30 seconds for sample libraries) and remembers the result. Plugins that
        crash or fail to load headless are reported as unusable.

        Parameter values are normalized 0.0-1.0; "text" is how the plugin displays
        the current value.

        Args:
            plugin: Plugin name from studio_list_plugins, or an absolute .vst3 path
            name_filter: Optional substring to narrow the parameter list
            limit: Maximum number of parameters to return
            refresh: Probe the plugin again instead of using the remembered result
        """
        info = probe_plugin(resolve_plugin(plugin), refresh=refresh)
        matching = [p for p in info["parameters"] if name_filter.lower() in p["name"].lower()]
        return {
            "name": info["name"],
            "manufacturer": info["manufacturer"],
            "is_instrument": info["is_instrument"],
            "is_effect": info["is_effect"],
            "parameters_available": info.get("parameters_available", True),
            "parameter_count": len(info["parameters"]),
            "parameters": matching[: max(0, limit)],
            "has_profile": find_profile(plugin) is not None,
        }

    @mcp.tool()
    @_tool_errors
    def studio_get_instrument_profile(plugin: str) -> dict:
        """Get the playing guide for an instrument: range, keyswitches, drum map, tips.

        Read this before writing notes for an instrument that has a profile
        (for example Ample Guitar M articulations are selected with keyswitch notes).

        Args:
            plugin: Plugin name, path, or a builtin id such as "builtin:drums"
        """
        profile = find_profile(plugin)
        if profile is None:
            return {
                "profile": None,
                "note": "No profile for this plugin. Render a short test phrase to find "
                "its playable range.",
            }
        return {"profile": profile}

    @mcp.tool()
    @_tool_errors
    def studio_list_projects() -> list[str]:
        """List the names of saved studio projects."""
        return store.list_projects()

    @mcp.tool()
    @_tool_errors
    def studio_create_project(
        name: str, tempo: float = 120.0, time_signature: str = "4/4", overwrite: bool = False
    ) -> dict:
        """Create an empty studio project.

        Positions and lengths everywhere in the studio are in beats (quarter
        notes), starting at 0. In 4/4, bar N starts at beat (N - 1) * 4.

        Args:
            name: Project name (letters, digits, spaces, '-' and '_')
            tempo: Tempo in BPM
            time_signature: For example "4/4", "3/4" or "6/8"
            overwrite: Replace an existing project with the same name
        """
        validate_name(name)
        signature = _parse_time_signature(time_signature)
        project = Project(name=name, tempo=tempo, time_signature=signature)
        with store.lock:
            if store.project_exists(name) and not overwrite:
                raise ProjectError(
                    f"Project {name!r} already exists; pass overwrite=true to replace it"
                )
            store.save_project(project)
        return _project_summary(project)

    @mcp.tool()
    @_tool_errors
    def studio_get_project(project: str) -> dict:
        """Get a project overview: tempo, length, tracks, note counts, mix and effects."""
        return _project_summary(store.load_project(project))

    @mcp.tool()
    @_tool_errors
    def studio_set_project_settings(
        project: str, tempo: float | None = None, time_signature: str | None = None
    ) -> dict:
        """Change the tempo and/or time signature of a project.

        Args:
            project: Project name
            tempo: New tempo in BPM
            time_signature: New time signature such as "3/4"
        """
        with store.edit_project(project) as current:
            checked = Project(
                name=current.name,
                tempo=current.tempo if tempo is None else tempo,
                time_signature=current.time_signature
                if time_signature is None
                else _parse_time_signature(time_signature),
            )
            current.tempo = checked.tempo
            current.time_signature = checked.time_signature
            return _project_summary(current)

    @mcp.tool()
    @_tool_errors
    def studio_add_track(
        project: str,
        name: str,
        plugin: str,
        preset: str | None = None,
        samples: dict[str, str] | None = None,
    ) -> dict:
        """Add an instrument track to a project.

        Args:
            project: Project name
            name: Track name, for example "Acoustic Guitar"
            plugin: An installed VST3 instrument (name or path), or a built-in:
                    "builtin:drums", "builtin:sampler", "builtin:sine"
            preset: Optional absolute path to a .vstpreset file for a VST3 instrument
            samples: For "builtin:sampler" only: MIDI note to audio file,
                     for example {"36": "C:/samples/kick.wav"}

        Returns the new track, including its id.
        """
        validate_name(name)
        spec = plugin
        if preset is not None:
            preset = local_file(preset, PRESET_EXTENSIONS, "Preset")
        checked_samples: dict[str, str] = {}
        for note, sample in (samples or {}).items():
            if not note.isdigit() or not 0 <= int(note) <= 127:
                raise ProjectError(f"Sample keys must be MIDI note numbers 0-127, got {note!r}")
            checked_samples[str(int(note))] = local_file(sample, AUDIO_EXTENSIONS, "Sample")
        if is_builtin(plugin):
            if samples and plugin != BUILTIN_SAMPLER:
                raise ProjectError("samples are only used by builtin:sampler")
            if plugin == BUILTIN_SAMPLER and not samples:
                raise ProjectError("builtin:sampler needs a samples mapping")
        else:
            spec = resolve_plugin(plugin)
            info = probe_plugin(spec)
            if not info["is_instrument"]:
                raise ProjectError(
                    f"{info['name']} is an effect; add it with studio_set_effects instead"
                )
        with store.edit_project(project) as current:
            if any(t.name.lower() == name.lower() for t in current.tracks):
                raise ProjectError(f"Project {project!r} already has a track named {name!r}")
            track = Track(
                id=current.next_track_id(),
                name=name,
                plugin=spec,
                preset=preset,
                samples=checked_samples,
            )
            current.tracks.append(track)
            summary = _track_summary(current, track)
        summary["has_profile"] = find_profile(plugin) is not None
        return summary

    @mcp.tool()
    @_tool_errors
    def studio_remove_track(project: str, track: str) -> str:
        """Remove a track (by id or name) and all its notes from a project."""
        with store.edit_project(project) as current:
            target = current.find_track(track)
            current.tracks.remove(target)
        return f"Removed track {target.id} ({target.name})"

    @mcp.tool()
    @_tool_errors
    def studio_write_notes(
        project: str,
        track: str,
        notes: list[dict],
        mode: str = "add",
        range_start: float | None = None,
        range_end: float | None = None,
    ) -> dict:
        """Write notes to a track.

        Each note: {"midi": 60, "time": 0.0, "duration": 1.0, "velocity": 0.8}
          - midi: MIDI note number 0-127 (60 = middle C)
          - time: start in beats from the beginning of the song
          - duration: length in beats
          - velocity: 0.0-1.0 (optional, default 0.8)
        This is the same note format as fl_send_notes.

        Args:
            project: Project name
            track: Track id or name
            notes: Notes to write
            mode: "add" keeps existing notes; "replace" clears the whole track first;
                  "replace_range" clears notes starting in [range_start, range_end) first
            range_start: Start of the cleared range in beats (replace_range only)
            range_end: End of the cleared range in beats (replace_range only)
        """
        parsed = _parse_notes(notes)
        if mode not in ("add", "replace", "replace_range"):
            raise ProjectError("mode must be 'add', 'replace' or 'replace_range'")
        if mode == "replace_range" and (range_start is None or range_end is None):
            raise ProjectError("replace_range needs range_start and range_end")
        with store.edit_project(project) as current:
            target = current.find_track(track)
            before = len(target.notes)
            if mode == "replace":
                target.notes = []
            elif mode == "replace_range":
                assert range_start is not None and range_end is not None
                target.notes = [n for n in target.notes if not range_start <= n.time < range_end]
            removed = before - len(target.notes)
            target.notes.extend(parsed)
            target.sort_notes()
            return {
                "track": target.id,
                "added": len(parsed),
                "removed": removed,
                "total_notes": len(target.notes),
                "track_length_beats": round(target.length_beats(), 3),
            }

    @mcp.tool()
    @_tool_errors
    def studio_get_notes(
        project: str,
        track: str,
        start: float = 0.0,
        end: float | None = None,
        limit: int = 400,
    ) -> dict:
        """Read the notes of a track that start in [start, end) beats.

        Args:
            project: Project name
            track: Track id or name
            start: Range start in beats
            end: Range end in beats (default: end of track)
            limit: Maximum number of notes to return
        """
        target = store.load_project(project).find_track(track)
        selected = [n for n in target.notes if n.time >= start and (end is None or n.time < end)]
        return {
            "track": target.id,
            "count": len(selected),
            "truncated": len(selected) > limit,
            "notes": [_note_dict(n) for n in selected[: max(0, limit)]],
        }

    @mcp.tool()
    @_tool_errors
    def studio_delete_notes(
        project: str, track: str, start: float, end: float, midi: int | None = None
    ) -> str:
        """Delete notes of a track that start in [start, end) beats.

        Args:
            project: Project name
            track: Track id or name
            start: Range start in beats
            end: Range end in beats
            midi: Only delete notes with this MIDI number
        """
        with store.edit_project(project) as current:
            target = current.find_track(track)
            before = len(target.notes)
            target.notes = [
                n
                for n in target.notes
                if not (start <= n.time < end and (midi is None or n.midi == midi))
            ]
            removed = before - len(target.notes)
        return f"Deleted {removed} note(s) from {target.id}"

    @mcp.tool()
    @_tool_errors
    def studio_copy_notes(
        project: str,
        track: str,
        source_start: float,
        source_end: float,
        dest_start: float,
        repeat: int = 1,
        transpose: int = 0,
        dest_track: str | None = None,
        replace: bool = True,
    ) -> dict:
        """Copy a section of notes to another position, optionally repeated and transposed.

        Use this to build song structure (repeat a 4-bar pattern, reuse a verse)
        without sending the same notes again.

        Args:
            project: Project name
            track: Source track id or name
            source_start: Start of the source section in beats
            source_end: End of the source section in beats (exclusive)
            dest_start: Where the first copy begins, in beats
            repeat: Number of consecutive copies
            transpose: Semitones to shift the copies
            dest_track: Destination track (default: the source track)
            replace: Clear the destination range before pasting
        """
        length = source_end - source_start
        if length <= 0:
            raise ProjectError("source_end must be greater than source_start")
        if not 1 <= repeat <= 256:
            raise ProjectError("repeat must be between 1 and 256")
        if dest_start < 0:
            raise ProjectError("dest_start must be >= 0")
        with store.edit_project(project) as current:
            source = current.find_track(track)
            destination = current.find_track(dest_track) if dest_track else source
            section = [n for n in source.notes if source_start <= n.time < source_end]
            if not section:
                raise ProjectError(f"No notes start in beats {source_start}-{source_end}")
            copies: list[Note] = []
            for index in range(repeat):
                offset = dest_start + index * length - source_start
                for note in section:
                    pitch = note.midi + transpose
                    if not 0 <= pitch <= 127:
                        raise ProjectError(f"Transposing puts a note outside 0-127 ({pitch})")
                    copies.append(Note(pitch, note.time + offset, note.duration, note.velocity))
            if replace:
                dest_end = dest_start + repeat * length
                destination.notes = [
                    n for n in destination.notes if not dest_start <= n.time < dest_end
                ]
            destination.notes.extend(copies)
            destination.sort_notes()
            return {
                "track": destination.id,
                "added": len(copies),
                "total_notes": len(destination.notes),
                "track_length_beats": round(destination.length_beats(), 3),
            }

    @mcp.tool()
    @_tool_errors
    def studio_set_track_mix(
        project: str,
        track: str,
        volume_db: float | None = None,
        pan: float | None = None,
        mute: bool | None = None,
    ) -> dict:
        """Set a track's volume, pan and mute.

        Args:
            project: Project name
            track: Track id or name
            volume_db: Gain in dB (0 = unchanged, -6 = half amplitude)
            pan: -1.0 (left) to 1.0 (right)
            mute: Exclude the track from renders
        """
        with store.edit_project(project) as current:
            target = current.find_track(track)
            if volume_db is not None:
                if not -60 <= volume_db <= 24:
                    raise ProjectError("volume_db must be between -60 and 24")
                target.volume_db = float(volume_db)
            if pan is not None:
                if not -1 <= pan <= 1:
                    raise ProjectError("pan must be between -1 and 1")
                target.pan = float(pan)
            if mute is not None:
                target.mute = mute
            return _track_summary(current, target)

    @mcp.tool()
    @_tool_errors
    def studio_set_plugin_params(
        project: str, track: str, params: dict[str, float], replace: bool = False
    ) -> dict:
        """Set parameters of a track's VST3 instrument.

        Values are normalized 0.0-1.0. Find parameter names with
        studio_get_plugin_info. The values are applied on every render.

        Args:
            project: Project name
            track: Track id or name
            params: Parameter name to normalized value
            replace: Forget previously set parameters instead of merging
        """
        plugin = store.load_project(project).find_track(track).plugin
        if is_builtin(plugin):
            raise ProjectError("Built-in instruments have no parameters")
        info = probe_plugin(plugin)
        if not info.get("parameters_available", True):
            raise ProjectError(_NO_PARAMETERS.format(name=info["name"]))
        unknown = sorted(set(params) - {p["name"] for p in info["parameters"]})
        if unknown:
            raise ProjectError(
                f"Unknown parameters {unknown}; see studio_get_plugin_info for the names"
            )
        with store.edit_project(project) as current:
            target = current.find_track(track)
            if target.plugin != plugin:
                raise ProjectError(f"Track {track!r} changed while its plugin was checked")
            merged = {} if replace else dict(target.params)
            merged.update(params)
            checked = Track.from_dict({"id": "x", "name": "x", "plugin": "x", "params": merged})
            target.params = checked.params
            return {"track": target.id, "params": target.params}

    @mcp.tool()
    @_tool_errors
    def studio_set_effects(project: str, target: str, effects: list[dict]) -> dict:
        """Set the whole effect chain of a track or of the master bus (in order).

        Built-in effects take real-world values, for example:
          {"type": "HighpassFilter", "params": {"cutoff_frequency_hz": 90}}
          {"type": "Compressor", "params": {"threshold_db": -18, "ratio": 3,
                                            "attack_ms": 10, "release_ms": 120}}
          {"type": "Reverb", "params": {"room_size": 0.4, "wet_level": 0.2, "dry_level": 0.8}}
          {"type": "Delay", "params": {"delay_seconds": 0.375, "feedback": 0.3, "mix": 0.2}}
          {"type": "Gain", "params": {"gain_db": -3}}
          {"type": "Limiter", "params": {"threshold_db": -1, "release_ms": 100}}
        Other built-ins: LowpassFilter, HighShelfFilter, LowShelfFilter, PeakFilter
        (cutoff_frequency_hz, gain_db, q), Chorus, Phaser, Distortion (drive_db), NoiseGate.

        Installed VST3 effects take normalized 0.0-1.0 parameters:
          {"type": "vst3", "plugin": "ValhallaSupermassive", "params": {"mix": 0.3}}

        Args:
            project: Project name
            target: Track id or name, or "master" for the master bus
            effects: The new chain; an empty list removes all effects
        """
        chain = _resolve_effects(effects)
        with store.edit_project(project) as current:
            if target.lower() == "master":
                current.master_effects = chain
                owner = "master"
            else:
                track = current.find_track(target)
                track.effects = chain
                owner = track.id
        return {
            "target": owner,
            "effects": [{"type": e.type, "plugin": e.plugin, "params": e.params} for e in chain],
        }

    @mcp.tool()
    @_tool_errors
    def studio_render(
        project: str,
        stems: bool = False,
        start_beat: float | None = None,
        end_beat: float | None = None,
        tail_seconds: float = 2.0,
    ) -> dict:
        """Render a project to a WAV file and analyse the result.

        This is how to "listen": the analysis reports integrated loudness (LUFS),
        peak level, clipping, loudness over time, energy per frequency band and
        stereo correlation for the mix, plus level numbers for every track. Use
        them to balance the mix, then render again.

        The first render after adding a sample-based VST3 instrument loads its
        samples and can take up to a minute; later renders reuse the loaded plugin.

        Args:
            project: Project name
            stems: Also write one WAV per track
            start_beat: Render from this beat (default: the beginning)
            end_beat: Render up to this beat (default: the end)
            tail_seconds: Extra time after the last note for release and reverb tails
        """
        current = store.load_project(project)
        if not 0 <= tail_seconds <= 30:
            raise ProjectError("tail_seconds must be between 0 and 30")
        for label, beat in (("start_beat", start_beat), ("end_beat", end_beat)):
            if beat is not None and not 0 <= beat < float("inf"):
                raise ProjectError(f"{label} must be a finite number >= 0")
        options = {
            "out_dir": str(store.renders_dir(project)),
            "stamp": f"{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns() // 1_000_000 % 1000:03d}",
            "stems": stems,
            "start_beat": start_beat,
            "end_beat": end_beat,
            "tail_seconds": tail_seconds,
        }
        result = get_engine().call("render", {"project": current.to_dict(), "options": options})
        assert isinstance(result, dict)
        return result

    @mcp.tool()
    @_tool_errors
    def studio_analyze_audio(path: str) -> dict:
        """Analyse any audio file (WAV, FLAC, MP3...) with the same metrics as studio_render.

        Args:
            path: Absolute path to the audio file
        """
        checked = local_file(path, AUDIO_EXTENSIONS, "Audio file")
        result = get_engine().call("analyze", {"path": checked})
        assert isinstance(result, dict)
        return result

    @mcp.tool()
    @_tool_errors
    def studio_export_midi(project: str) -> dict:
        """Export a project as MIDI files: one for the whole song and one per track.

        The files can be dragged into FL Studio (or any DAW) to continue working
        there with the same notes and tempo.
        """
        current = store.load_project(project)
        return export_midi(current, store.midi_dir(project))

    @mcp.tool()
    @_tool_errors
    def studio_send_track_to_fl(
        project: str, track: str, mode: str = "replace", auto_trigger: bool = True
    ) -> str:
        """Send a track's notes into the piano roll that is currently open in FL Studio.

        Open the piano roll of the target channel in FL Studio first. Uses the same
        bridge as fl_send_notes (ComposeWithLLM piano roll script).

        Args:
            project: Project name
            track: Track id or name
            mode: "replace" clears the piano roll first, "add" keeps existing notes
            auto_trigger: Send the keystroke that runs the piano roll script
        """
        from fl_studio_mcp.tools.piano_roll import _get_trigger_info, _write_request

        if mode not in ("add", "replace"):
            raise ProjectError("mode must be 'add' or 'replace'")
        target = store.load_project(project).find_track(track)
        if not target.notes:
            raise ProjectError(f"Track {target.name!r} has no notes")
        requests: list[dict[str, Any]] = [{"action": "clear"}] if mode == "replace" else []
        requests.append({"action": "add_notes", "notes": [_note_dict(n) for n in target.notes]})
        _write_request(requests)
        trigger_info = _get_trigger_info(auto_trigger)
        return f"Queued {len(target.notes)} note(s) from {target.name}.{trigger_info}"
