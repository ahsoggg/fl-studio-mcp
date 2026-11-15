"""Built-in instruments that need no plugin: a sine synth, synthesized drums, a sampler."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

Audio = NDArray[np.float32]
# (start_seconds, end_seconds, midi, velocity 1-127)
Event = tuple[float, float, int, int]

# General MIDI drum notes handled by builtin:drums
DRUM_NAMES: dict[int, str] = {
    35: "kick",
    36: "kick",
    37: "rimshot",
    38: "snare",
    39: "clap",
    40: "snare",
    41: "tom_low",
    42: "hihat_closed",
    43: "tom_low",
    44: "hihat_closed",
    45: "tom_mid",
    46: "hihat_open",
    47: "tom_mid",
    48: "tom_high",
    49: "crash",
    50: "tom_high",
    51: "ride",
    57: "crash",
}


def _mix_in(out: NDArray[np.float64], start: int, sound: NDArray[np.float64]) -> None:
    if start >= out.shape[-1]:
        return
    length = min(sound.shape[-1], out.shape[-1] - start)
    out[..., start : start + length] += sound[..., :length]


def render_sine(events: list[Event], n: int, sample_rate: int) -> Audio:
    out = np.zeros(n)
    attack = max(1, int(0.005 * sample_rate))
    release = max(1, int(0.08 * sample_rate))
    for start, end, midi, velocity in events:
        length = max(1, int((end - start) * sample_rate)) + release
        t = np.arange(length) / sample_rate
        envelope = np.ones(length)
        envelope[:attack] = np.linspace(0.0, 1.0, attack)
        envelope[-release:] = np.linspace(1.0, 0.0, release)
        freq = 440.0 * 2.0 ** ((midi - 69) / 12.0)
        tone = np.sin(2 * np.pi * freq * t) + 0.3 * np.sin(4 * np.pi * freq * t)
        _mix_in(out, int(start * sample_rate), 0.2 * (velocity / 127.0) * tone * envelope)
    return np.stack([out, out]).astype(np.float32)


def _decay(length: int, sample_rate: int, seconds: float) -> NDArray[np.float64]:
    return np.exp(-np.arange(length) / (seconds * sample_rate))


def _highpassed_noise(rng: np.random.Generator, length: int, amount: float) -> NDArray[np.float64]:
    noise = rng.standard_normal(length)
    return noise - amount * np.concatenate(([0.0], noise[:-1]))


def _kick(sr: int, rng: np.random.Generator) -> NDArray[np.float64]:
    n = int(0.45 * sr)
    t = np.arange(n) / sr
    phase = 2 * np.pi * (45.0 * t + (110.0 / 28.0) * (1 - np.exp(-28.0 * t)))
    return np.sin(phase) * _decay(n, sr, 0.12) * 0.95


def _snare(sr: int, rng: np.random.Generator) -> NDArray[np.float64]:
    n = int(0.3 * sr)
    t = np.arange(n) / sr
    tone = np.sin(2 * np.pi * 190.0 * t) * _decay(n, sr, 0.05)
    return 0.5 * tone + 0.45 * _highpassed_noise(rng, n, 0.6) * _decay(n, sr, 0.07)


def _hat(seconds: float) -> Callable[[int, np.random.Generator], NDArray[np.float64]]:
    def make(sr: int, rng: np.random.Generator) -> NDArray[np.float64]:
        n = int(seconds * 6 * sr)
        return 0.22 * _highpassed_noise(rng, n, 0.95) * _decay(n, sr, seconds)

    return make


def _tom(freq: float) -> Callable[[int, np.random.Generator], NDArray[np.float64]]:
    def make(sr: int, rng: np.random.Generator) -> NDArray[np.float64]:
        n = int(0.4 * sr)
        t = np.arange(n) / sr
        phase = 2 * np.pi * (freq * t + (freq / 30.0) * (1 - np.exp(-20.0 * t)))
        return 0.7 * np.sin(phase) * _decay(n, sr, 0.13)

    return make


def _clap(sr: int, rng: np.random.Generator) -> NDArray[np.float64]:
    n = int(0.25 * sr)
    out = np.zeros(n)
    for offset in (0.0, 0.012, 0.024):
        start = int(offset * sr)
        burst = _highpassed_noise(rng, n - start, 0.8) * _decay(n - start, sr, 0.04)
        out[start:] += 0.3 * burst
    return out


def _rimshot(sr: int, rng: np.random.Generator) -> NDArray[np.float64]:
    n = int(0.08 * sr)
    t = np.arange(n) / sr
    return 0.6 * np.sin(2 * np.pi * 820.0 * t) * _decay(n, sr, 0.015)


_DRUM_SYNTHS: dict[str, Callable[[int, np.random.Generator], NDArray[np.float64]]] = {
    "kick": _kick,
    "snare": _snare,
    "rimshot": _rimshot,
    "clap": _clap,
    "hihat_closed": _hat(0.035),
    "hihat_open": _hat(0.2),
    "tom_low": _tom(90.0),
    "tom_mid": _tom(130.0),
    "tom_high": _tom(180.0),
    "crash": _hat(0.55),
    "ride": _hat(0.3),
}


def render_drums(events: list[Event], n: int, sample_rate: int) -> tuple[Audio, list[str]]:
    out = np.zeros(n)
    cache: dict[str, NDArray[np.float64]] = {}
    unmapped: set[int] = set()
    for start, _end, midi, velocity in events:
        name = DRUM_NAMES.get(midi)
        if name is None:
            unmapped.add(midi)
            continue
        if name not in cache:
            cache[name] = _DRUM_SYNTHS[name](sample_rate, np.random.default_rng(midi))
        _mix_in(out, int(start * sample_rate), 0.5 * cache[name] * (velocity / 127.0) ** 1.5)
    warnings = (
        [f"builtin:drums has no sound for MIDI notes {sorted(unmapped)}"] if unmapped else []
    )
    return np.stack([out, out]).astype(np.float32), warnings


def render_sampler(
    events: list[Event],
    n: int,
    sample_rate: int,
    samples: dict[str, str],
    read_audio: Callable[[str, int], NDArray[Any]],
) -> tuple[Audio, list[str]]:
    """Play one-shot audio files mapped to MIDI notes (samples: {"36": "kick.wav"})."""
    out = np.zeros((2, n))
    cache: dict[int, NDArray[np.float64]] = {}
    unmapped: set[int] = set()
    for start, _end, midi, velocity in events:
        path = samples.get(str(midi))
        if path is None:
            unmapped.add(midi)
            continue
        if midi not in cache:
            sound = np.asarray(read_audio(path, sample_rate), dtype=np.float64)
            if sound.ndim == 1:
                sound = sound[np.newaxis, :]
            cache[midi] = sound[:2] if sound.shape[0] >= 2 else np.repeat(sound, 2, axis=0)
        _mix_in(out, int(start * sample_rate), cache[midi] * (velocity / 127.0))
    warnings = [f"No sample mapped for MIDI notes {sorted(unmapped)}"] if unmapped else []
    return out.astype(np.float32), warnings
