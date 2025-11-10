"""Utility functions for the ligotools package."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile


def whiten(strain: np.ndarray, interp_psd, dt: float) -> np.ndarray:
    """Whiten a strain time series using an interpolated PSD."""
    nt = len(strain)
    freqs = np.fft.rfftfreq(nt, dt)

    hf = np.fft.rfft(strain)
    norm = 1.0 / np.sqrt(1.0 / (dt * 2))
    white_hf = hf / np.sqrt(interp_psd(freqs)) * norm
    white_ht = np.fft.irfft(white_hf, n=nt)
    return white_ht


def write_wavfile(filename: str | Path, fs: int, data: np.ndarray) -> None:
    """Scale a signal to 16-bit range and write it to a WAV file."""
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    max_val = np.max(np.abs(data))
    if max_val == 0:
        scaled = np.zeros_like(data, dtype=np.int16)
    else:
        scaled = np.int16(data / max_val * 32767 * 0.9)
    wavfile.write(path, int(fs), scaled)


def reqshift(data: np.ndarray, fshift: float = 100, sample_rate: float = 4096) -> np.ndarray:
    """Shift the frequency of a band-passed signal by a constant amount."""
    spectrum = np.fft.rfft(data)
    duration = len(data) / float(sample_rate)
    df = 1.0 / duration
    nbins = int(fshift / df)
    shifted = np.roll(spectrum.real, nbins) + 1j * np.roll(spectrum.imag, nbins)
    shifted[0:nbins] = 0.0
    return np.fft.irfft(shifted)


def plot_matched_filter_results(
    time: np.ndarray,
    tevent: float,
    timemax: float,
    eventname: str,
    det: str,
    snr: np.ndarray,
    strain_whitenbp: np.ndarray,
    template_match: np.ndarray,
    template_fft: np.ndarray,
    datafreq: np.ndarray,
    data_psd: np.ndarray,
    freqs: np.ndarray,
    fs: float,
    plottype: str,
    color: str,
    d_eff: float,
    figures_dir: str | Path = "figures",
) -> None:
    """Create diagnostic plots for matched-filter results."""
    figures_path = Path(figures_dir)
    figures_path.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)
    plt.plot(time - timemax, snr, color, label=f"{det} SNR(t)")
    plt.grid("on")
    plt.ylabel("SNR")
    plt.xlabel(f"Time since {timemax:.4f}")
    plt.legend(loc="upper left")
    plt.title(f"{det} matched filter SNR around event")

    plt.subplot(2, 1, 2)
    plt.plot(time - timemax, snr, color, label=f"{det} SNR(t)")
    plt.grid("on")
    plt.ylabel("SNR")
    plt.xlim([-0.15, 0.05])
    plt.xlabel(f"Time since {timemax:.4f}")
    plt.legend(loc="upper left")
    plt.savefig(figures_path / f"{eventname}_{det}_SNR.{plottype}")

    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)
    plt.plot(time - tevent, strain_whitenbp, color, label=f"{det} whitened h(t)")
    plt.plot(time - tevent, template_match, "k", label="Template(t)")
    plt.ylim([-10, 10])
    plt.xlim([-0.15, 0.05])
    plt.grid("on")
    plt.xlabel(f"Time since {timemax:.4f}")
    plt.ylabel("whitened strain (units of noise stdev)")
    plt.legend(loc="upper left")
    plt.title(f"{det} whitened data around event")

    plt.subplot(2, 1, 2)
    plt.plot(
        time - tevent,
        strain_whitenbp - template_match,
        color,
        label=f"{det} resid",
    )
    plt.ylim([-10, 10])
    plt.xlim([-0.15, 0.05])
    plt.grid("on")
    plt.xlabel(f"Time since {timemax:.4f}")
    plt.ylabel("whitened strain (units of noise stdev)")
    plt.legend(loc="upper left")
    plt.title(f"{det} Residual whitened data after subtracting template around event")
    plt.savefig(figures_path / f"{eventname}_{det}_matchtime.{plottype}")

    plt.figure(figsize=(10, 6))
    template_f = np.abs(template_fft) * np.sqrt(np.abs(datafreq)) / d_eff
    plt.loglog(datafreq, template_f, "k", label="template(f)*sqrt(f)")
    plt.loglog(freqs, np.sqrt(data_psd), color, label=f"{det} ASD")
    plt.xlim(20, fs / 2)
    plt.ylim(1e-24, 1e-20)
    plt.grid()
    plt.xlabel("frequency (Hz)")
    plt.ylabel("strain noise ASD (strain/rtHz), template h(f)*rt(f)")
    plt.legend(loc="upper left")
    plt.title(f"{det} ASD and template around event")
    plt.savefig(figures_path / f"{eventname}_{det}_matchfreq.{plottype}")

    plt.close("all")
