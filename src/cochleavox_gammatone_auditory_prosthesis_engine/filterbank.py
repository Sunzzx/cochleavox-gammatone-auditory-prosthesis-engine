"""Glasberg & Moore ERB-Scale Gammatone Filterbank.

Implements 4th-order Gammatone filters based on Patterson, Holdsworth, and Glasberg & Moore (1990).
Transfer function approximation via digital bilinear/impulse-invariant 4th-order IIR cascade.
"""

import numpy as np
from scipy import signal
from dataclasses import dataclass
from typing import Tuple, List, Optional


def hz_to_erb(freq_hz: np.ndarray | float) -> np.ndarray | float:
    """Convert frequency in Hz to Equivalent Rectangular Bandwidth (ERB) rate (Glasberg & Moore 1990).
    
    ERBS = 21.4 * log10(4.37 * 1e-3 * f + 1)
    """
    f = np.asarray(freq_hz, dtype=np.float64)
    erb_rate = 21.4 * np.log10(0.00437 * f + 1.0)
    if np.ndim(freq_hz) == 0:
        return float(erb_rate.item())
    return erb_rate


def erb_to_hz(erb_rate: np.ndarray | float) -> np.ndarray | float:
    """Convert ERB rate to frequency in Hz.
    
    f = (10^(ERBS / 21.4) - 1) / 0.00437
    """
    e = np.asarray(erb_rate, dtype=np.float64)
    freq = (10.0 ** (e / 21.4) - 1.0) / 0.00437
    if np.ndim(erb_rate) == 0:
        return float(freq.item())
    return freq


def erb_bandwidth(cf_hz: float) -> float:
    """Equivalent Rectangular Bandwidth at center frequency cf (Glasberg & Moore 1990).
    
    b(f) = 24.7 * (4.37 * 1e-3 * f + 1)
    """
    return 24.7 * (4.37e-3 * float(cf_hz) + 1.0)


@dataclass
class GammatoneFilterbank:
    """Auditory Gammatone Filterbank.
    
    Attributes:
        num_channels: Number of frequency channels
        f_min: Lowest center frequency in Hz
        f_max: Highest center frequency in Hz
        sample_rate: Sampling frequency in Hz
    """
    num_channels: int = 32
    f_min: float = 80.0
    f_max: float = 8000.0
    sample_rate: float = 44100.0

    def __post_init__(self):
        if self.f_max > self.sample_rate / 2.0:
            raise ValueError(f"f_max ({self.f_max}) cannot exceed Nyquist ({self.sample_rate / 2.0})")
        if self.f_min <= 0 or self.f_max <= self.f_min:
            raise ValueError("Invalid frequency limits: f_min must be > 0 and < f_max")
        
        # Design center frequencies linearly spaced on ERB-number scale
        erb_min = hz_to_erb(self.f_min)
        erb_max = hz_to_erb(self.f_max)
        erb_points = np.linspace(erb_min, erb_max, self.num_channels)
        self.center_frequencies = erb_to_hz(erb_points)
        self._design_filters()

    def _design_filters(self):
        """Design 4th-order Gammatone filters via IIR pole-zero decomposition.
        
        An nth-order gammatone impulse response is g(t) = a * t^(n-1) * cos(2*pi*f*t + phi) * exp(-2*pi*b*t)
        For n=4, this can be implemented as 4 cascaded 1st-order complex poles or 4 cascaded 2nd-order IIR filters.
        """
        dt = 1.0 / self.sample_rate
        self.sos_filters = []

        cf_arr = np.atleast_1d(self.center_frequencies)
        for fc in cf_arr:
            # Bandwidth according to Glasberg & Moore (1.019 * ERB)
            b = 1.019 * erb_bandwidth(fc)
            # Complex pole: p = exp(-2*pi*b*dt + 1j*2*pi*fc*dt)
            p = np.exp(-2.0 * np.pi * b * dt + 1j * 2.0 * np.pi * fc * dt)
            
            # 1-pole filter in s-domain corresponds to 1-pole in z-domain: H(z) = 1 / (1 - p*z^-1)
            # Cascading 4 identical 1-pole complex filters: H(z)^4
            # We implement real-valued equivalent by cascading second-order sections (SOS)
            # Each complex conjugate pole pair gives: (1 - p*z^-1)(1 - p* z^-1) = 1 - 2*Re(p)*z^-1 + |p|^2*z^-2
            # For 4th order, repeat 4 times in cascade
            r = np.abs(p)
            theta = np.angle(p)
            
            # Second order section for each stage
            a1 = -2.0 * r * np.cos(theta)
            a2 = r ** 2
            
            # Normalization gain to make peak magnitude at fc close to unity
            # At z = exp(1j*theta), |1 - p*z^-1| = 1 - r
            gain_per_stage = 1.0 - r
            
            # 4 identical cascaded stages: SOS matrix has shape (4, 6)
            sos = np.zeros((4, 6), dtype=np.float64)
            for stage in range(4):
                sos[stage, :] = [gain_per_stage, 0.0, 0.0, 1.0, a1, a2]
            
            self.sos_filters.append(sos)

    def process(self, audio: np.ndarray) -> np.ndarray:
        """Filter input audio through the Gammatone filterbank.
        
        Args:
            audio: 1D array of audio samples
            
        Returns:
            2D array of subband signals of shape (num_channels, len(audio))
        """
        audio = np.asarray(audio, dtype=np.float64)
        if audio.ndim != 1:
            raise ValueError("Audio input must be a 1D array")
        
        subbands = np.zeros((self.num_channels, len(audio)), dtype=np.float64)
        for ch_idx, sos in enumerate(self.sos_filters):
            subbands[ch_idx, :] = signal.sosfilt(sos, audio)
            
        return subbands

    def extract_envelope(self, subbands: np.ndarray) -> np.ndarray:
        """Extract instantaneous envelope for each subband via Hilbert transform.
        
        Args:
            subbands: 2D array of shape (num_channels, num_samples)
            
        Returns:
            2D array of instantaneous envelopes, same shape
        """
        analytic = signal.hilbert(subbands, axis=1)
        return np.abs(analytic)
