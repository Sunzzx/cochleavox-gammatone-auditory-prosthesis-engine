"""Continuous Interleaved Sampling (CIS) Cochlear Implant Speech Processor.

Simulates clinical cochlear implant electrical stimulation protocols:
- Pre-emphasis filtering (high-frequency boost to match speech spectrum)
- Bandpass analysis filterbank
- Full-wave / half-wave rectification & low-pass envelope detection
- Non-linear logarithmic loudness mapping (acoustic-to-electric dynamic range compression)
- Interleaved biphasic charge-balanced pulsatile electrostimulation generation
- Direct nerve excitation electrogram synthesis.
"""

import numpy as np
from scipy import signal
from dataclasses import dataclass
from typing import Tuple, List, Optional


@dataclass
class CISParams:
    """Parameters for Continuous Interleaved Sampling (CIS) processor."""
    num_electrodes: int = 16
    pulse_rate_per_channel: float = 900.0  # Pulses per second per electrode
    pulse_width_us: float = 25.0           # Phase duration in microseconds
    interphase_gap_us: float = 5.0         # Gap between biphasic phases in microseconds
    envelope_cutoff_hz: float = 400.0      # Envelope lowpass filter cutoff
    dynamic_range_db: float = 30.0         # Acoustic dynamic range in dB
    electric_dynamic_range_cu: float = 200.0  # Clinical current units range (T to C level)
    t_level: float = 50.0                  # Threshold current units
    c_level: float = 250.0                 # Comfort current units
    compression_c: float = 400.0           # Logarithmic compression curvature constant


class CochlearImplantCISProcessor:
    """CIS Speech Processor and Biphasic Pulse Synthesizer."""
    
    def __init__(
        self,
        sample_rate: float = 44100.0,
        params: CISParams | None = None
    ):
        self.sample_rate = sample_rate
        self.dt = 1.0 / sample_rate
        self.params = params or CISParams()
        self._design_envelope_filter()

    def _design_envelope_filter(self):
        """Design 2nd-order Butterworth low-pass filter for envelope extraction."""
        nyquist = self.sample_rate / 2.0
        norm_cutoff = min(self.params.envelope_cutoff_hz / nyquist, 0.95)
        self.b_env, self.a_env = signal.butter(2, norm_cutoff, btype='low')

    def pre_emphasis(self, audio: np.ndarray, alpha: float = 0.95) -> np.ndarray:
        """Pre-emphasis filter: y[n] = x[n] - alpha * x[n-1] to boost high frequencies."""
        return signal.lfilter([1.0, -alpha], [1.0], audio)

    def extract_channel_envelopes(self, subbands: np.ndarray) -> np.ndarray:
        """Full-wave rectify and low-pass filter to obtain smoothed subband envelopes."""
        rectified = np.abs(subbands)
        envelopes = signal.lfilter(self.b_env, self.a_env, rectified, axis=1)
        return np.maximum(0.0, envelopes)

    def logarithmic_compression(self, envelope: np.ndarray) -> np.ndarray:
        """Compress acoustic amplitude to electrical stimulation current units (T-to-C mapping).
        
        E = T + (C - T) * log(1 + c * A / A_max) / log(1 + c)
        """
        p = self.params
        eps = 1e-12
        env_max = np.max(envelope) if np.max(envelope) > eps else 1.0
        norm_env = np.clip(envelope / env_max, 0.0, 1.0)
        
        compressed = np.log1p(p.compression_c * norm_env) / np.log1p(p.compression_c)
        current_units = p.t_level + (p.c_level - p.t_level) * compressed
        return current_units

    def synthesize_electrogram(
        self,
        subbands: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate interleaved biphasic electrical stimulation pulses across electrodes.
        
        Returns:
            electrogram: 2D array (num_electrodes, num_samples) of biphasic current pulses
            current_levels: 2D array of modulating amplitudes
        """
        n_ch, n_samples = subbands.shape
        envelopes = self.extract_channel_envelopes(subbands)
        current_levels = self.logarithmic_compression(envelopes)
        
        electrogram = np.zeros((n_ch, n_samples), dtype=np.float64)
        
        # Pulse period in samples per channel
        channel_period_samples = int(np.round(self.sample_rate / self.params.pulse_rate_per_channel))
        if channel_period_samples < n_ch * 2:
            channel_period_samples = max(n_ch * 2, 1)
            
        # Pulse width in samples
        phase_width_samples = max(1, int(np.round((self.params.pulse_width_us * 1e-6) / self.dt)))
        
        # Interleave channels: each channel is staggered by channel_period / n_ch
        stagger_step = channel_period_samples // n_ch
        
        for ch in range(n_ch):
            offset = ch * stagger_step
            pulse_indices = np.arange(offset, n_samples - 2 * phase_width_samples, channel_period_samples)
            
            for p_idx in pulse_indices:
                amplitude = current_levels[ch, p_idx]
                # Anodic phase
                electrogram[ch, p_idx : p_idx + phase_width_samples] = amplitude
                # Cathodic phase (charge balancing)
                cath_start = p_idx + phase_width_samples
                electrogram[ch, cath_start : cath_start + phase_width_samples] = -amplitude
                
        return electrogram, current_levels
