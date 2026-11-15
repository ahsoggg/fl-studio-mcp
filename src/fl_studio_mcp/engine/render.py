"""Offline renderer: turns a studio project into audio by hosting VST3 plugins.

Runs only inside the engine worker process (see worker.py): plugins print to
stdout and can crash, so they must never be loaded in the MCP server process.
"""

from __future__ import annotations

import difflib
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from fl_studio_mcp.engine import builtin
from fl_studio_mcp.engine.analysis import analyze, level_summary
from fl_studio_mcp.engine.builtin import Event
from fl_studio_mcp.engine.model import (
    BUILTIN_DRUMS,
    BUILTIN_SAMPLER,
    BUILTIN_SINE,
    VST3_EFFECT,
    Effect,
    Project,
    Track,
)
from fl_studio_mcp.engine.plugins import (
    AUDIO_EXTENSIONS,
    PRESET_EXTENSIONS,
    allowed_plugin_path,
    binary_path,
    local_file,
)

Audio = NDArray[np.float32]

WARMUP_SECONDS = 45.0
_SILENT_PEAK = 1e-5


class RenderError(RuntimeError):
    """A render request that cannot be fulfilled; the message is shown to the model."""


@dataclass
class _Loaded:
    plugin: Any
    initial: dict[str, float] = field(default_factory=dict)


def track_events(
    track: Track, seconds_per_beat: float, start_beat: float, end_beat: float
) -> list[Event]:
    """Notes of a track inside [start_beat, end_beat) as events relative to the region start.

    Notes already sounding at the region start are cut to begin there, so held
    keyswitches and long pads still apply when only a section is rendered.
    """
    events: list[Event] = []
    for note in track.notes:
        if note.time >= end_beat or note.end <= start_beat:
            continue
        begin = max(note.time, start_beat) - start_beat
        finish = min(note.end, end_beat) - start_beat
        velocity = min(127, max(1, round(note.velocity * 127)))
        events.append((begin * seconds_per_beat, finish * seconds_per_beat, note.midi, velocity))
    return events


def midi_messages(events: list[Event]) -> list[Any]:
    """Note on/off messages with absolute times in seconds, ordered for a sampler.

    At equal times note-offs go first and lower pitches lead, so low keyswitch
    notes take effect before the notes they modify.
    """
    from mido import Message

    messages = []
    for start, end, midi, velocity in events:
        messages.append(Message("note_on", note=midi, velocity=velocity, time=start))
        messages.append(Message("note_off", note=midi, velocity=0, time=max(end, start + 0.001)))
    messages.sort(key=lambda m: (m.time, m.type == "note_on", m.note))
    return messages


def apply_mix(audio: Audio, volume_db: float, pan: float) -> Audio:
    gain = 10.0 ** (volume_db / 20.0)
    angle = (min(1.0, max(-1.0, pan)) + 1.0) * math.pi / 4.0
    left = gain * math.cos(angle) * math.sqrt(2.0)
    right = gain * math.sin(angle) * math.sqrt(2.0)
    return (audio * np.array([[left], [right]], dtype=np.float32)).astype(np.float32)


def _fit(audio: NDArray[Any], n: int) -> Audio:
    data = np.asarray(audio, dtype=np.float32)
    if data.ndim == 1:
        data = data[np.newaxis, :]
    if data.shape[0] == 1:
        data = np.repeat(data, 2, axis=0)
    data = data[:2, :n]
    if data.shape[1] < n:
        data = np.pad(data, ((0, 0), (0, n - data.shape[1])))
    return data


def read_audio(path: str, sample_rate: int) -> NDArray[np.float32]:
    from pedalboard.io import AudioFile

    with AudioFile(local_file(path, AUDIO_EXTENSIONS, "Sample")).resampled_to(
        sample_rate
    ) as handle:
        return handle.read(handle.frames)


def write_audio(path: Path, audio: Audio, sample_rate: int) -> None:
    from pedalboard.io import AudioFile

    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(audio, -1.0, 1.0)
    if path.suffix.lower() == ".mp3":
        with AudioFile(str(path), "w", sample_rate, audio.shape[0], quality=320) as handle:
            handle.write(clipped)
        return
    with AudioFile(str(path), "w", sample_rate, audio.shape[0], bit_depth=24) as handle:
        handle.write(clipped)


def _parameter_text(parameter: Any) -> str:
    try:
        return str(parameter.string_value)
    except Exception:
        return ""


class Renderer:
    """Keeps loaded plugins between renders so sample libraries load only once."""

    def __init__(self) -> None:
        self._loaded: dict[tuple[Any, ...], _Loaded] = {}
        self._probe: tuple[tuple[str, str | None], Any] | None = None

    def _load(self, path: str, plugin_name: str | None) -> Any:
        from pedalboard import load_plugin

        try:
            module = binary_path(allowed_plugin_path(path))
            return load_plugin(module, plugin_name=plugin_name, initialization_timeout=30.0)
        except Exception as exc:
            raise RenderError(f"Could not load plugin {path}: {exc}") from exc

    def plugin_info(
        self, path: str, plugin_name: str | None = None, parameters: bool = True
    ) -> dict[str, Any]:
        """Describe a plugin. Reading parameters crashes some plugins, so it is optional."""
        key = (path, plugin_name)
        if self._probe is None or self._probe[0] != key:
            self._probe = (key, self._load(path, plugin_name))
        plugin = self._probe[1]
        info: dict[str, Any] = {
            "name": str(plugin.name),
            "path": path,
            "manufacturer": str(getattr(plugin, "manufacturer_name", "")),
            "is_instrument": bool(plugin.is_instrument),
            "is_effect": bool(plugin.is_effect),
        }
        if parameters:
            info["parameters"] = [
                {
                    "name": name,
                    "value": round(float(parameter.raw_value), 4),
                    "text": _parameter_text(parameter),
                }
                for name, parameter in plugin.parameters.items()
            ]
        return info

    def _get(
        self, key: tuple[Any, ...], path: str, plugin_name: str | None
    ) -> tuple[_Loaded, bool]:
        fresh = key not in self._loaded
        if fresh:
            self._loaded[key] = _Loaded(self._load(path, plugin_name))
        return self._loaded[key], fresh

    @staticmethod
    def _apply_params(loaded: _Loaded, params: dict[str, Any], what: str) -> None:
        if not params and not loaded.initial:
            return
        parameters = loaded.plugin.parameters
        for name, value in loaded.initial.items():
            if name not in params:
                parameters[name].raw_value = value
        for name, value in params.items():
            if name not in parameters:
                close = difflib.get_close_matches(name, list(parameters), n=5, cutoff=0.4)
                hint = f" Did you mean: {', '.join(close)}?" if close else ""
                raise RenderError(f"{what} has no parameter {name!r}.{hint}")
            loaded.initial.setdefault(name, float(parameters[name].raw_value))
            parameters[name].raw_value = float(value)

    def _warm_up(self, plugin: Any, events: list[Event], sample_rate: int) -> bool:
        """Sample libraries load in the background and play silence until ready."""
        pitches = sorted(event[2] for event in events)
        probe = [(0.05, 0.6, pitches[len(pitches) // 2], 100)]
        deadline = time.monotonic() + WARMUP_SECONDS
        while True:
            audio = plugin(midi_messages(probe), duration=1.0, sample_rate=sample_rate)
            if float(np.abs(audio).max()) > _SILENT_PEAK:
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(1.0)

    def _render_instrument(
        self,
        project: Project,
        track: Track,
        events: list[Event],
        n: int,
        used: set[tuple[Any, ...]],
        warnings: list[str],
    ) -> Audio:
        rate = project.sample_rate
        if track.plugin == BUILTIN_SINE:
            return builtin.render_sine(events, n, rate)
        if track.plugin == BUILTIN_DRUMS:
            audio, notes = builtin.render_drums(events, n, rate)
            warnings.extend(f"{track.name}: {w}" for w in notes)
            return audio
        if track.plugin == BUILTIN_SAMPLER:
            audio, notes = builtin.render_sampler(events, n, rate, track.samples, read_audio)
            warnings.extend(f"{track.name}: {w}" for w in notes)
            return audio

        key = ("instrument", project.name, track.id, track.plugin, track.plugin_name, track.preset)
        used.add(key)
        warmed = key not in self._loaded
        plugin = self._prepare_instrument(key, track)
        if warmed and not self._warm_up(plugin, events, rate):
            warnings.append(
                f"{track.name}: the instrument stayed silent for {WARMUP_SECONDS:.0f}s of "
                "warm-up; its samples may be missing or the probe note is out of range"
            )
        audio = plugin(midi_messages(events), duration=n / rate, sample_rate=rate, reset=True)
        return _fit(audio, n)

    def _prepare_instrument(self, key: tuple[Any, ...], track: Track) -> Any:
        loaded, fresh = self._get(key, track.plugin, track.plugin_name)
        plugin = loaded.plugin
        try:
            if not plugin.is_instrument:
                raise RenderError(
                    f"Track {track.name!r}: {track.plugin} is an effect, not an instrument"
                )
            if fresh and track.preset:
                try:
                    preset = local_file(track.preset, PRESET_EXTENSIONS, "Preset")
                    plugin.load_preset(preset)
                except Exception as exc:
                    raise RenderError(
                        f"Track {track.name!r}: cannot load preset: {exc}"
                    ) from exc
            self._apply_params(loaded, track.params, f"Track {track.name!r} instrument")
        except RenderError:
            if fresh:
                del self._loaded[key]
            raise
        return plugin

    def _apply_effects(
        self,
        project: Project,
        owner: str,
        effects: list[Effect],
        audio: Audio,
        used: set[tuple[Any, ...]],
    ) -> Audio:
        import pedalboard

        rate = project.sample_rate
        for index, effect in enumerate(effects):
            what = f"{owner} effect {index} ({effect.type})"
            if effect.type == VST3_EFFECT:
                key = ("effect", project.name, owner, index, effect.plugin)
                loaded, fresh = self._get(key, str(effect.plugin), None)
                used.add(key)
                try:
                    if not loaded.plugin.is_effect:
                        raise RenderError(f"{what}: {effect.plugin} is not an effect plugin")
                    self._apply_params(loaded, effect.params, what)
                except RenderError:
                    if fresh:
                        del self._loaded[key]
                    raise
                processor = loaded.plugin
            else:
                try:
                    processor = getattr(pedalboard, effect.type)(**effect.params)
                except (TypeError, ValueError) as exc:
                    raise RenderError(f"{what}: {exc}") from exc
            audio = _fit(processor(audio, rate, reset=True), audio.shape[1])
        return audio

    def render(self, project_data: dict[str, Any], options: dict[str, Any]) -> dict[str, Any]:
        started = time.monotonic()
        project = Project.from_dict(project_data)
        rate = project.sample_rate
        spb = project.seconds_per_beat

        start_beat = float(options.get("start_beat") or 0.0)
        length = project.length_beats()
        end_option = options.get("end_beat")
        end_beat = length if end_option is None else min(float(end_option), length)
        if end_beat <= start_beat:
            raise RenderError(
                f"Nothing to render: region {start_beat}-{end_beat} beats, "
                f"project length {length} beats"
            )
        tail = float(options.get("tail_seconds", 2.0))
        n = int(((end_beat - start_beat) * spb + tail) * rate)

        out_dir = Path(options["out_dir"])
        stamp = str(options["stamp"])
        used: set[tuple[Any, ...]] = set()
        warnings: list[str] = []
        mix = np.zeros((2, n), dtype=np.float32)
        tracks_report: list[dict[str, Any]] = []
        stems: list[str] = []

        for index, track in enumerate(project.tracks):
            events = track_events(track, spb, start_beat, end_beat)
            if track.mute or not events:
                continue
            audio = self._render_instrument(project, track, events, n, used, warnings)
            audio = self._apply_effects(project, f"track {track.id}", track.effects, audio, used)
            audio = apply_mix(audio, track.volume_db, track.pan)
            mix += audio
            report: dict[str, Any] = {"id": track.id, "name": track.name}
            report.update(level_summary(audio, rate))
            if report["peak_dbfs"] <= -100:
                warnings.append(f"{track.name}: rendered silence (notes outside playable range?)")
            tracks_report.append(report)
            if options.get("stems"):
                safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in track.name)
                stem_path = out_dir / f"{stamp}_stems" / f"{index + 1:02d}_{safe}.wav"
                write_audio(stem_path, audio, rate)
                stems.append(str(stem_path))

        mix = self._apply_effects(project, "master", project.master_effects, mix, used)
        for key in [k for k in self._loaded if k not in used]:
            del self._loaded[key]

        extension = "mp3" if options.get("format") == "mp3" else "wav"
        path = out_dir / f"{project.name}_{stamp}.{extension}"
        write_audio(path, mix, rate)
        return {
            "path": str(path),
            "stems": stems,
            "region_beats": [start_beat, end_beat],
            "analysis": analyze(mix, rate),
            "tracks": tracks_report,
            "warnings": warnings,
            "render_seconds": round(time.monotonic() - started, 1),
        }

    def analyze_file(self, path: str) -> dict[str, Any]:
        from pedalboard.io import AudioFile

        with AudioFile(local_file(path, AUDIO_EXTENSIONS, "Audio file")) as handle:
            audio = handle.read(handle.frames)
            rate = int(handle.samplerate)
        return {"path": path, "analysis": analyze(audio, rate)}
