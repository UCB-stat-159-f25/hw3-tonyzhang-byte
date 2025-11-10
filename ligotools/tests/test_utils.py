from pathlib import Path

import numpy as np
import pytest
from scipy.interpolate import interp1d
from scipy.io import wavfile

from ligotools.utils import reqshift, whiten, write_wavfile


def test_whiten_matches_reference_values():
    strain = np.array([1.0, 2.0, 3.0, 4.0])
    dt = 0.1
    freqs = np.fft.rfftfreq(len(strain), dt)
    psd_values = np.array([1.0, 2.0, 3.0])
    interp_psd = interp1d(freqs, psd_values, fill_value="extrapolate")

    whitened = whiten(strain, interp_psd, dt)

    expected = np.array([
        0.6727067778594763,
        0.9309056676066377,
        1.3051623098931524,
        1.5633611996403134,
    ])
    assert whitened == pytest.approx(expected)


def test_write_wavfile_scales_signal_and_creates_file(tmp_path):
    data = np.linspace(-1.0, 1.0, 4096)
    output = tmp_path / "test.wav"

    write_wavfile(output, 4096, data)

    assert output.exists()
    rate, samples = wavfile.read(output)
    assert rate == 4096
    assert samples.dtype == np.int16
    assert np.max(np.abs(samples)) <= 32767


def test_reqshift_moves_frequency_peak():
    fs = 1024
    duration = 1.0
    t = np.arange(0, int(fs * duration)) / fs
    base_freq = 50.0
    signal = np.sin(2 * np.pi * base_freq * t)

    shifted = reqshift(signal, fshift=100.0, sample_rate=fs)

    freqs = np.fft.rfftfreq(len(shifted), 1 / fs)
    amplitude = np.abs(np.fft.rfft(shifted))
    peak_frequency = freqs[np.argmax(amplitude)]

    assert peak_frequency == pytest.approx(base_freq + 100.0, abs=1.0)
