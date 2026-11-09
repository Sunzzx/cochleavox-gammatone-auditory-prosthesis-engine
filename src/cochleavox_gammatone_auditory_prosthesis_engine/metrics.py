"""Auditory Model Metrics: PSTH, Phase-Locking Vector Strength, and Neurogram Similarity.
"""

import numpy as np
from typing import Tuple, Dict


def compute_psth(
    spike_trains: np.ndarray,
    sample_rate: float,
    bin_width_ms: float = 2.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Post-Stimulus Time Histogram (PSTH) from population spike trains.
    
    Args:
        spike_trains: 2D binary array of shape (num_fibers, num_samples)
        sample_rate: Sampling frequency in Hz
        bin_width_ms: Histogram bin width in milliseconds
        
    Returns:
        bin_centers: Time values for bin centers in seconds
        firing_rates: Average firing rate per fiber in spikes/second
    """
    num_fibers, num_samples = spike_trains.shape
    bin_samples = max(1, int(np.round((bin_width_ms * 1e-3) * sample_rate)))
    num_bins = num_samples // bin_samples
    
    if num_bins == 0:
        return np.array([0.0]), np.array([0.0])
        
    trimmed_spikes = spike_trains[:, :num_bins * bin_samples]
    binned = trimmed_spikes.reshape(num_fibers, num_bins, bin_samples)
    # Sum across bin samples and average across fibers
    spike_counts = np.sum(binned, axis=(0, 2)) / num_fibers
    bin_duration_sec = bin_samples / sample_rate
    firing_rates = spike_counts / bin_duration_sec
    
    bin_centers = (np.arange(num_bins) + 0.5) * bin_duration_sec
    return bin_centers, firing_rates


def vector_strength(spike_times_sec: np.ndarray, stim_freq_hz: float) -> float:
    """Goldberg & Brown (1969) Vector Strength (VS) phase-locking metric.
    
    VS = (1 / N) * sqrt((sum(cos(theta_i)))^2 + (sum(sin(theta_i)))^2)
    where theta_i = 2 * pi * f_stim * t_i mod 2*pi
    """
    if len(spike_times_sec) == 0:
        return 0.0
        
    theta = 2.0 * np.pi * stim_freq_hz * spike_times_sec
    r = np.sqrt(np.sum(np.cos(theta)) ** 2 + np.sum(np.sin(theta)) ** 2) / len(spike_times_sec)
    return float(r)


def neurogram_similarity(
    neurogram_ref: np.ndarray,
    neurogram_test: np.ndarray
) -> float:
    """Compute 2D Normalized Cross-Correlation Neurogram Similarity Metric (NSIM).
    
    Quantifies preservation of spectro-temporal auditory neural excitation patterns.
    """
    if neurogram_ref.shape != neurogram_test.shape:
        raise ValueError(f"Shape mismatch: {neurogram_ref.shape} vs {neurogram_test.shape}")
        
    ref_z = neurogram_ref - np.mean(neurogram_ref)
    test_z = neurogram_test - np.mean(neurogram_test)
    
    norm_ref = np.linalg.norm(ref_z)
    norm_test = np.linalg.norm(test_z)
    
    if norm_ref == 0 or norm_test == 0:
        return 1.0 if norm_ref == norm_test else 0.0
        
    correlation = np.sum(ref_z * test_z) / (norm_ref * norm_test)
    return float(np.clip(correlation, -1.0, 1.0))
