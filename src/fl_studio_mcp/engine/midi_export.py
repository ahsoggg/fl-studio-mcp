"""Export studio projects as Standard MIDI Files."""

from __future__ import annotations

from pathlib import Path

import mido

from fl_studio_mcp.engine.model import Project, Track

TICKS_PER_BEAT = 480


def _header(project: Project) -> list[mido.MetaMessage]:
    numerator, denominator = project.time_signature
    return [
        mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(project.tempo), time=0),
        mido.MetaMessage(
            "time_signature", numerator=numerator, denominator=denominator, time=0
        ),
    ]


def _track(track: Track, channel: int) -> mido.MidiTrack:
    events: list[tuple[int, int, mido.Message]] = []
    for note in track.notes:
        start = round(note.time * TICKS_PER_BEAT)
        end = max(start + 1, round(note.end * TICKS_PER_BEAT))
        velocity = min(127, max(1, round(note.velocity * 127)))
        events.append(
            (start, 1, mido.Message("note_on", note=note.midi, velocity=velocity, channel=channel))
        )
        events.append(
            (end, 0, mido.Message("note_off", note=note.midi, velocity=0, channel=channel))
        )
    events.sort(key=lambda e: (e[0], e[1], e[2].note))
    result = mido.MidiTrack()
    result.append(mido.MetaMessage("track_name", name=track.name, time=0))
    position = 0
    for tick, _order, message in events:
        result.append(message.copy(time=tick - position))
        position = tick
    return result


def _safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)


def export_midi(project: Project, out_dir: Path) -> dict[str, object]:
    """Write one multi-track file for the song plus one file per track."""
    out_dir.mkdir(parents=True, exist_ok=True)
    song = mido.MidiFile(type=1, ticks_per_beat=TICKS_PER_BEAT)
    conductor = mido.MidiTrack()
    conductor.extend(_header(project))
    song.tracks.append(conductor)

    files: list[dict[str, str]] = []
    for index, track in enumerate(project.tracks):
        if not track.notes:
            continue
        song.tracks.append(_track(track, 0))
        single = mido.MidiFile(type=1, ticks_per_beat=TICKS_PER_BEAT)
        header = mido.MidiTrack()
        header.extend(_header(project))
        single.tracks.append(header)
        single.tracks.append(_track(track, 0))
        path = out_dir / f"{index + 1:02d}_{_safe(track.name)}.mid"
        single.save(str(path))
        files.append({"track": track.id, "name": track.name, "path": str(path)})

    song_path = out_dir / f"{_safe(project.name)}.mid"
    song.save(str(song_path))
    return {"song": str(song_path), "tracks": files}
