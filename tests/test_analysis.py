from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from fl_studio_mcp.engine.analysis import analyze, level_summary

RATE = 44100


def sine(freq: float, dbfs: float, seconds: float = 3.0) -> NDArray[Any]:
    t = np.arange(int(seconds * RATE)) / RATE
    wave = 10 ** (dbfs / 20) * np.sin(2 * np.pi * freq * t)
    return np.stack([wave, wave])


def test_stereo_1khz_sine_reads_its_own_level_in_lufs() -> None:
    # BS.1770: a 1 kHz sine at -20 dBFS in both channels measures -20 LUFS.
    result = analyze(sine(1000.0, -20.0), RATE)
    assert result["lufs_integrated"] == pytest.approx(-20.0, abs=0.2)
    assert result["peak_dbfs"] == pytest.approx(-20.0, abs=0.05)
    assert result["rms_dbfs"] == pytest.approx(-23.01, abs=0.05)
    assert result["clipped_samples"] == 0
    assert result["stereo_correlation"] == pytest.approx(1.0)
    assert result["hints"] == []


def test_k_weighting_attenuates_low_frequencies() -> None:
    low = analyze(sine(50.0, -20.0), RATE)["lufs_integrated"]
    mid = analyze(sine(1000.0, -20.0), RATE)["lufs_integrated"]
    assert low < mid - 3


def test_band_energy_follows_the_signal() -> None:
    bands = analyze(sine(100.0, -12.0), RATE)["band_energy_db"]
    assert max(bands, key=bands.get) == "bass_60_250"
    bands = analyze(sine(3000.0, -12.0), RATE)["band_energy_db"]
    assert max(bands, key=bands.get) == "high_mid_2k_6k"


def test_clipping_and_phase_problems_are_reported() -> None:
    wave = np.clip(sine(220.0, 6.0)[0], -1.0, 1.0)
    result = analyze(np.stack([wave, -wave]), RATE)
    assert result["clipped_samples"] > 0
    assert result["stereo_correlation"] == pytest.approx(-1.0)
    assert len(result["hints"]) >= 2


def test_silence_and_gaps() -> None:
    silent = analyze(np.zeros((2, RATE)), RATE)
    assert silent["silent"] is True
    assert silent["lufs_integrated"] is None

    padded = np.concatenate(
        [np.zeros((2, RATE)), sine(440.0, -10.0, 1.0), np.zeros((2, RATE // 2))], axis=1
    )
    result = analyze(padded, RATE)
    assert result["leading_silence_seconds"] == pytest.approx(1.0, abs=0.01)
    assert result["trailing_silence_seconds"] == pytest.approx(0.5, abs=0.01)
    assert len(result["loudness_timeline"]["lufs"]) <= 32


def test_level_summary_handles_short_and_mono_audio() -> None:
    summary = level_summary(sine(440.0, -6.0, 0.1)[0], RATE)
    assert summary["peak_dbfs"] == pytest.approx(-6.0, abs=0.1)
