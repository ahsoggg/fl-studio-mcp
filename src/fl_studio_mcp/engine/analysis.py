"""Audio analysis that lets a model "hear" a render through numbers.

Loudness follows ITU-R BS.1770 (K-weighting, 400 ms blocks, absolute and relative
gating). The K-weighting is applied in the frequency domain per block, so the
result can differ from a time-domain meter by a few hundredths of a LU.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating[Any]]

BANDS: tuple[tuple[str, float, float], ...] = (
    ("sub_20_60", 20.0, 60.0),
    ("bass_60_250", 60.0, 250.0),
    ("low_mid_250_500", 250.0, 500.0),
    ("mid_500_2k", 500.0, 2000.0),
    ("high_mid_2k_6k", 2000.0, 6000.0),
    ("high_6k_20k", 6000.0, 20000.0),
)
_SILENCE_DB = -60.0
_MAX_SEGMENTS = 32
_FFT_CHUNK = 128


def _db(value: float, floor: float = -120.0) -> float:
    return max(floor, 20.0 * math.log10(value)) if value > 0 else floor


def _power_db(value: float, floor: float = -120.0) -> float:
    return max(floor, 10.0 * math.log10(value)) if value > 0 else floor


def k_weight_power(freqs: FloatArray) -> FloatArray:
    """Squared magnitude of the BS.1770 K-weighting filter at the given frequencies."""
    w = 2.0 * np.pi * np.minimum(freqs, 24000.0) / 48000.0
    z = np.exp(-1j * w)
    shelf = (1.53512485958697 - 2.69169618940638 * z + 1.19839281085285 * z**2) / (
        1.0 - 1.69065929318241 * z + 0.73248077421585 * z**2
    )
    highpass = (1.0 - 2.0 * z + z**2) / (1.0 - 1.99004745483398 * z + 0.99007225036621 * z**2)
    return np.abs(shelf * highpass) ** 2


def _block_spectra(audio: FloatArray, sample_rate: int) -> tuple[FloatArray, FloatArray, float]:
    """Return (K-weighted mean square per channel per block, mean power spectrum, hop seconds)."""
    channels, n = audio.shape
    block = min(n, int(0.4 * sample_rate))
    hop = max(1, int(0.1 * sample_rate))
    freqs = np.fft.rfftfreq(block, 1.0 / sample_rate)
    weight = k_weight_power(freqs)
    fold = np.full(freqs.shape, 2.0)
    fold[0] = 1.0
    if block % 2 == 0:
        fold[-1] = 1.0
    starts = np.arange(0, n - block + 1, hop)
    weighted = np.zeros((channels, len(starts)))
    spectrum = np.zeros(freqs.shape)
    for ch in range(channels):
        windows = np.lib.stride_tricks.sliding_window_view(audio[ch], block)[::hop]
        for i in range(0, len(windows), _FFT_CHUNK):
            power = np.abs(np.fft.rfft(windows[i : i + _FFT_CHUNK], axis=1)) ** 2 * fold
            weighted[ch, i : i + _FFT_CHUNK] = (power * weight).sum(axis=1) / block**2
            spectrum += power.sum(axis=0)
    spectrum /= max(1, len(starts)) * block**2
    return weighted, spectrum, hop / sample_rate


def _integrated_lufs(block_power: FloatArray) -> float | None:
    """Gated integrated loudness from summed-channel block powers."""
    if block_power.size == 0:
        return None
    loudness = -0.691 + 10.0 * np.log10(np.maximum(block_power, 1e-12))
    gated = block_power[loudness > -70.0]
    if gated.size == 0:
        return None
    relative = -0.691 + 10.0 * math.log10(float(gated.mean())) - 10.0
    final = block_power[(loudness > -70.0) & (loudness > relative)]
    if final.size == 0:
        return None
    return -0.691 + 10.0 * math.log10(float(final.mean()))


def _as_2d(audio: FloatArray) -> FloatArray:
    data = np.asarray(audio, dtype=np.float64)
    if data.ndim == 1:
        data = data[np.newaxis, :]
    if data.ndim != 2:
        raise ValueError("audio must be shaped (channels, samples)")
    return data


def level_summary(audio: FloatArray, sample_rate: int) -> dict[str, Any]:
    """Peak, RMS and integrated loudness: the cheap per-track numbers."""
    data = _as_2d(audio)
    if data.shape[1] == 0:
        return {"peak_dbfs": -120.0, "rms_dbfs": -120.0, "lufs": None}
    weighted, _, _ = _block_spectra(data, sample_rate)
    lufs = _integrated_lufs(weighted.sum(axis=0))
    return {
        "peak_dbfs": round(_db(float(np.abs(data).max())), 2),
        "rms_dbfs": round(_db(float(np.sqrt((data**2).mean()))), 2),
        "lufs": None if lufs is None else round(lufs, 2),
    }


def analyze(audio: FloatArray, sample_rate: int) -> dict[str, Any]:
    """Full analysis of a mix: levels, loudness over time, spectrum balance, stereo image."""
    data = _as_2d(audio)
    channels, n = data.shape
    if n == 0:
        return {"duration_seconds": 0.0, "silent": True, "hints": ["The audio is empty."]}

    magnitude = np.abs(data)
    peak = float(magnitude.max())
    rms = float(np.sqrt((data**2).mean()))
    weighted, spectrum, hop_seconds = _block_spectra(data, sample_rate)
    block_power = weighted.sum(axis=0)
    lufs = _integrated_lufs(block_power)

    loud = np.flatnonzero(magnitude.max(axis=0) > 10 ** (_SILENCE_DB / 20.0))
    leading = float(loud[0] / sample_rate) if loud.size else n / sample_rate
    trailing = float((n - 1 - loud[-1]) / sample_rate) if loud.size else n / sample_rate

    freqs = np.fft.rfftfreq(min(n, int(0.4 * sample_rate)), 1.0 / sample_rate)
    total = float(spectrum[freqs >= 20.0].sum())
    bands = {
        name: round(_power_db(float(spectrum[(freqs >= lo) & (freqs < hi)].sum()) / total), 1)
        if total > 0
        else -120.0
        for name, lo, hi in BANDS
    }

    count = min(_MAX_SEGMENTS, max(1, len(block_power)))
    timeline = {
        "segment_seconds": round(n / sample_rate / count, 2),
        "lufs": [
            round(-0.691 + _power_db(float(block_power[chunk].mean())), 1)
            for chunk in np.array_split(np.arange(len(block_power)), count)
            if chunk.size
        ],
    }

    result: dict[str, Any] = {
        "duration_seconds": round(n / sample_rate, 3),
        "sample_rate": sample_rate,
        "channels": channels,
        "silent": peak < 10 ** (_SILENCE_DB / 20.0),
        "peak_dbfs": round(_db(peak), 2),
        "rms_dbfs": round(_db(rms), 2),
        "crest_factor_db": round(_db(peak) - _db(rms), 2),
        "lufs_integrated": None if lufs is None else round(lufs, 2),
        "clipped_samples": int((magnitude >= 1.0).sum()),
        "dc_offset": round(float(np.abs(data.mean(axis=1)).max()), 5),
        "leading_silence_seconds": round(leading, 3),
        "trailing_silence_seconds": round(trailing, 3),
        "band_energy_db": bands,
        "loudness_timeline": timeline,
    }
    if channels >= 2:
        left, right = data[0], data[1]
        denom = float(np.sqrt((left**2).sum() * (right**2).sum()))
        correlation = float((left * right).sum() / denom) if denom > 0 else 1.0
        mid = float((((left + right) / 2) ** 2).mean())
        side = float((((left - right) / 2) ** 2).mean())
        result["stereo_correlation"] = round(correlation, 3)
        result["side_to_mid_db"] = round(_power_db(side / mid), 1) if mid > 0 else None
    result["hints"] = _hints(result)
    return result


def _hints(result: dict[str, Any]) -> list[str]:
    hints: list[str] = []
    if result["silent"]:
        return ["The render is silent: check that tracks have notes inside the playable range."]
    if result["clipped_samples"] > 0:
        hints.append(
            f"{result['clipped_samples']} samples reach or exceed 0 dBFS: lower track volumes "
            "or put a Limiter last in the master chain."
        )
    lufs = result["lufs_integrated"]
    if lufs is not None and lufs < -20:
        hints.append(f"Quiet mix ({lufs} LUFS); streaming masters usually sit around -14 LUFS.")
    if lufs is not None and lufs > -8:
        hints.append(f"Very loud mix ({lufs} LUFS); expect audible limiting or distortion.")
    if result.get("stereo_correlation", 1.0) < 0:
        hints.append("Negative stereo correlation: the mix will lose level in mono.")
    if result["dc_offset"] > 0.01:
        hints.append("Noticeable DC offset: add a HighpassFilter around 20-30 Hz.")
    return hints
