"""Comprehensive test suite for CochleaVox Gammatone Auditory Prosthesis Engine."""

import pytest
import numpy as np
from cochleavox_gammatone_auditory_prosthesis_engine import (
    GammatoneFilterbank,
    hz_to_erb,
    erb_to_hz,
    erb_bandwidth,
    InnerHairCell,
    AuditoryNerveFiberPopulation,
    CochlearImplantCISProcessor,
    CISParams,
    compute_psth,
    vector_strength,
    neurogram_similarity,
)


def test_erb_conversions():
    freq = 1000.0
    erb = hz_to_erb(freq)
    recon_freq = erb_to_hz(erb)
    assert np.isclose(freq, recon_freq, rtol=1e-4)
    assert erb_bandwidth(1000.0) > 0.0


def test_filterbank_generation_and_filtering():
    fs = 16000.0
    num_channels = 16
    fb = GammatoneFilterbank(num_channels=num_channels, f_min=100.0, f_max=6000.0, sample_rate=fs)
    assert len(fb.center_frequencies) == num_channels
    
    # Generate 50ms of a 1kHz tone
    t = np.linspace(0, 0.05, int(fs * 0.05), endpoint=False)
    sig = np.sin(2 * np.pi * 1000.0 * t)
    filtered = fb.process(sig)
    
    assert filtered.shape == (num_channels, len(sig))
    channel_energies = np.sum(filtered ** 2, axis=1)
    closest_idx = np.argmin(np.abs(fb.center_frequencies - 1000.0))
    assert np.argmax(channel_energies) == closest_idx


def test_inner_hair_cell_dynamics():
    fs = 16000.0
    ihc = InnerHairCell(sample_rate=fs)
    t = np.linspace(0, 0.05, int(fs * 0.05), endpoint=False)
    bm_displacement = np.sin(2 * np.pi * 500.0 * t) * 1e-6
    subbands = np.array([bm_displacement, bm_displacement])
    
    cleft_profiles = ihc.transduce_multichannel(subbands)
    assert cleft_profiles.shape == subbands.shape
    assert np.all(cleft_profiles >= 0.0)


def test_auditory_nerve_fiber_population():
    fs = 16000.0
    anf = AuditoryNerveFiberPopulation(sample_rate=fs, fibers_per_channel=5, seed=42)
    n_samples = int(fs * 0.04)
    cleft_release = np.full((4, n_samples), 0.05)
    
    neurogram = anf.generate_neurogram(cleft_release)
    assert neurogram.shape == (4, n_samples)
    
    single_ch_spikes = anf.simulate_channel_population(cleft_release[0])
    assert single_ch_spikes.shape == (5, n_samples)
    assert np.all((single_ch_spikes == 0) | (single_ch_spikes == 1))


def test_cochlear_implant_cis_processor():
    fs = 16000.0
    n_channels = 8
    cis = CochlearImplantCISProcessor(sample_rate=fs, params=CISParams(num_electrodes=n_channels))
    
    subbands = np.random.randn(n_channels, int(fs * 0.04))
    electrogram, current_levels = cis.synthesize_electrogram(subbands)
    assert electrogram.shape == subbands.shape
    assert current_levels.shape == subbands.shape


def test_metrics_psth_and_similarity():
    fs = 16000.0
    n_fibers = 10
    n_samples = int(fs * 0.05)
    rng = np.random.default_rng(42)
    spike_trains = (rng.random((n_fibers, n_samples)) > 0.95).astype(np.int32)
    
    psth_bins, psth_rates = compute_psth(spike_trains, sample_rate=fs, bin_width_ms=2.0)
    assert len(psth_bins) == len(psth_rates)
    assert np.all(psth_rates >= 0.0)
    
    spike_indices = np.where(spike_trains == 1)[1]
    spike_times = spike_indices / fs
    vs = vector_strength(spike_times, 250.0)
    assert 0.0 <= vs <= 1.0
    
    # Neurogram similarity
    n1 = rng.random((8, 100))
    sim_self = neurogram_similarity(n1, n1)
    assert np.isclose(sim_self, 1.0, atol=1e-3)
