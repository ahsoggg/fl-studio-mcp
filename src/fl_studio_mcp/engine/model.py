"""Data model for studio projects: plain dataclasses that round-trip through JSON."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

BUILTIN_SINE = "builtin:sine"
BUILTIN_DRUMS = "builtin:drums"
BUILTIN_SAMPLER = "builtin:sampler"
BUILTIN_INSTRUMENTS = (BUILTIN_SINE, BUILTIN_DRUMS, BUILTIN_SAMPLER)

BUILTIN_EFFECTS = (
    "Gain",
    "Reverb",
    "Compressor",
    "Limiter",
    "HighpassFilter",
    "LowpassFilter",
    "HighShelfFilter",
    "LowShelfFilter",
    "PeakFilter",
    "Delay",
    "Chorus",
    "Phaser",
    "Distortion",
    "NoiseGate",
)
VST3_EFFECT = "vst3"

_NAME_RE = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9 _-]{0,62}[A-Za-z0-9_-])?")
_RESERVED_NAMES = frozenset(
    ["con", "prn", "aux", "nul"]
    + [f"com{i}" for i in range(1, 10)]
    + [f"lpt{i}" for i in range(1, 10)]
)


class ProjectError(ValueError):
    """Invalid project data supplied by the caller."""


def validate_name(name: str) -> str:
    if (
        not isinstance(name, str)
        or not _NAME_RE.fullmatch(name)
        or name.lower() in _RESERVED_NAMES
    ):
        raise ProjectError(
            f"Invalid name {name!r}: use 1-64 letters, digits, spaces, '_' or '-', "
            "starting with a letter or digit and not ending with a space"
        )
    return name


def _number(value: Any, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProjectError(f"{what} must be a number, got {value!r}")
    result = float(value)
    if result != result or result in (float("inf"), float("-inf")):
        raise ProjectError(f"{what} must be finite, got {value!r}")
    return result


@dataclass
class Note:
    midi: int
    time: float
    duration: float
    velocity: float = 0.8

    @classmethod
    def from_dict(cls, data: Any, index: int = 0) -> Note:
        if not isinstance(data, dict):
            raise ProjectError(f"Note {index} must be an object")
        for key in ("midi", "duration"):
            if key not in data:
                raise ProjectError(f"Note {index} is missing '{key}'")
        midi = data["midi"]
        if isinstance(midi, bool) or not isinstance(midi, int) or not 0 <= midi <= 127:
            raise ProjectError(f"Note {index}: midi must be an integer 0-127, got {midi!r}")
        time = _number(data.get("time", 0), f"Note {index}: time")
        duration = _number(data["duration"], f"Note {index}: duration")
        velocity = _number(data.get("velocity", 0.8), f"Note {index}: velocity")
        if time < 0:
            raise ProjectError(f"Note {index}: time must be >= 0, got {time}")
        if duration <= 0:
            raise ProjectError(f"Note {index}: duration must be > 0, got {duration}")
        if not 0 < velocity <= 1:
            raise ProjectError(
                f"Note {index}: velocity must be in (0, 1], got {velocity} "
                "(divide MIDI velocity by 127)"
            )
        return cls(midi=midi, time=time, duration=duration, velocity=velocity)

    @property
    def end(self) -> float:
        return self.time + self.duration


@dataclass
class Effect:
    type: str
    params: dict[str, Any] = field(default_factory=dict)
    plugin: str | None = None

    @classmethod
    def from_dict(cls, data: Any, index: int = 0) -> Effect:
        if not isinstance(data, dict) or "type" not in data:
            raise ProjectError(f"Effect {index} must be an object with a 'type'")
        kind = data["type"]
        params = data.get("params") or {}
        plugin = data.get("plugin")
        if not isinstance(params, dict):
            raise ProjectError(f"Effect {index}: params must be an object")
        if kind == VST3_EFFECT:
            if not isinstance(plugin, str) or not plugin:
                raise ProjectError(f"Effect {index}: type 'vst3' needs a 'plugin' name or path")
            _validate_raw_params(params, f"Effect {index}")
        elif kind in BUILTIN_EFFECTS:
            plugin = None
            for key, value in params.items():
                _number(value, f"Effect {index}: param {key}")
        else:
            raise ProjectError(
                f"Effect {index}: unknown type {kind!r}. "
                f"Use 'vst3' or one of: {', '.join(BUILTIN_EFFECTS)}"
            )
        return cls(type=kind, params=dict(params), plugin=plugin)


def _validate_raw_params(params: dict[str, Any], what: str) -> None:
    for key, value in params.items():
        raw = _number(value, f"{what}: param {key}")
        if not 0 <= raw <= 1:
            raise ProjectError(
                f"{what}: param {key} must be a normalized value 0.0-1.0, got {raw}"
            )


@dataclass
class Track:
    id: str
    name: str
    plugin: str
    plugin_name: str | None = None
    preset: str | None = None
    params: dict[str, float] = field(default_factory=dict)
    samples: dict[str, str] = field(default_factory=dict)
    notes: list[Note] = field(default_factory=list)
    effects: list[Effect] = field(default_factory=list)
    volume_db: float = 0.0
    pan: float = 0.0
    mute: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Track:
        params = data.get("params") or {}
        _validate_raw_params(params, f"Track {data.get('name')!r}")
        return cls(
            id=str(data["id"]),
            name=str(data["name"]),
            plugin=str(data["plugin"]),
            plugin_name=data.get("plugin_name"),
            preset=data.get("preset"),
            params={k: float(v) for k, v in params.items()},
            samples={str(k): str(v) for k, v in (data.get("samples") or {}).items()},
            notes=[Note.from_dict(n, i) for i, n in enumerate(data.get("notes") or [])],
            effects=[Effect.from_dict(e, i) for i, e in enumerate(data.get("effects") or [])],
            volume_db=_number(data.get("volume_db", 0.0), "volume_db"),
            pan=_number(data.get("pan", 0.0), "pan"),
            mute=bool(data.get("mute", False)),
        )

    def sort_notes(self) -> None:
        self.notes.sort(key=lambda n: (n.time, n.midi))

    def length_beats(self) -> float:
        return max((n.end for n in self.notes), default=0.0)


@dataclass
class Project:
    name: str
    tempo: float = 120.0
    time_signature: list[int] = field(default_factory=lambda: [4, 4])
    sample_rate: int = 44100
    tracks: list[Track] = field(default_factory=list)
    master_effects: list[Effect] = field(default_factory=list)

    def __post_init__(self) -> None:
        validate_name(self.name)
        self.tempo = _number(self.tempo, "tempo")
        if not 20 <= self.tempo <= 400:
            raise ProjectError(f"tempo must be 20-400 BPM, got {self.tempo}")
        sig = self.time_signature
        if (
            not isinstance(sig, (list, tuple))
            or len(sig) != 2
            or any(isinstance(v, bool) or not isinstance(v, int) for v in sig)
            or not 1 <= sig[0] <= 32
            or sig[1] not in (1, 2, 4, 8, 16)
        ):
            raise ProjectError(f"time_signature must look like [4, 4], got {sig!r}")
        self.time_signature = [sig[0], sig[1]]
        if self.sample_rate not in (44100, 48000, 88200, 96000):
            raise ProjectError(f"Unsupported sample_rate {self.sample_rate}")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Project:
        return cls(
            name=data["name"],
            tempo=data.get("tempo", 120.0),
            time_signature=list(data.get("time_signature", [4, 4])),
            sample_rate=int(data.get("sample_rate", 44100)),
            tracks=[Track.from_dict(t) for t in data.get("tracks") or []],
            master_effects=[
                Effect.from_dict(e, i) for i, e in enumerate(data.get("master_effects") or [])
            ],
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def beats_per_bar(self) -> float:
        return self.time_signature[0] * 4.0 / self.time_signature[1]

    @property
    def seconds_per_beat(self) -> float:
        return 60.0 / self.tempo

    def length_beats(self) -> float:
        return max((t.length_beats() for t in self.tracks), default=0.0)

    def find_track(self, ref: str) -> Track:
        for track in self.tracks:
            if track.id == ref:
                return track
        matches = [t for t in self.tracks if t.name.lower() == str(ref).lower()]
        if len(matches) == 1:
            return matches[0]
        known = ", ".join(f"{t.id} ({t.name})" for t in self.tracks) or "none"
        raise ProjectError(f"Track {ref!r} not found in project {self.name!r}. Tracks: {known}")

    def next_track_id(self) -> str:
        used = {t.id for t in self.tracks}
        index = 1
        while f"t{index}" in used:
            index += 1
        return f"t{index}"
