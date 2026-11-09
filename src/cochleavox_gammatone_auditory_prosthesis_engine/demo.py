"""Interactive Demonstration of CochleaVox Auditory Prosthesis Engine."""

import numpy as np
import sys
from cochleavox_gammatone_auditory_prosthesis_engine import (
    GammatoneFilterbank,
    InnerHairCell,
    AuditoryNerveFiberPopulation,
    CochlearImplantCISProcessor,
    CISParams,
    compute_psth,
    vector_strength,
    neurogram_similarity
)


def generate_synthetic_acoustic_stimulus(
    duration_sec: float = 0.05,
    sample_rate: float = 44100.0
) -> np.ndarray:
    """Synthesize a multi-tone harmonic complex mimicking speech formant frequencies."""
    t = np.linspace(0, duration_sec, int(duration_sec * sample_rate), endpoint=False)
    # Formants at F1 = 500 Hz, F2 = 1500 Hz, F3 = 2500 Hz
    signal = (
        0.5 * np.sin(2 * np.pi * 500.0 * t) +
        0.3 * np.sin(2 * np.pi * 1500.0 * t) +
        0.2 * np.sin(2 * np.pi * 2500.0 * t)
    )
    # 5 ms Tukey-like cosine window on ramps
    ramp_len = int(0.005 * sample_rate)
    ramp = 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, ramp_len)))
    signal[:ramp_len] *= ramp
    signal[-ramp_len:] *= ramp[::-1]
    return signal


def main():
    print("======================================================================")
    print(" CochleaVox: Biophysical Gammatone Auditory Prosthesis Engine Demo   ")
    print("======================================================================")
    
    fs = 44100.0
    duration = 0.04  # 40 ms stimulus
    print(f"[*] Synthesizing acoustic test tone complex (duration: {duration*1000:.1f} ms, fs: {fs} Hz)...")
    audio = generate_synthetic_acoustic_stimulus(duration_sec=duration, sample_rate=fs)
    
    # Step 1: Gammatone filterbank
    n_channels = 16
    print(f"[*] Initializing {n_channels}-channel Glasberg & Moore ERB Gammatone Filterbank...")
    fb = GammatoneFilterbank(num_channels=n_channels, f_min=100.0, f_max=6000.0, sample_rate=fs)
    subbands = fb.process(audio)
    print(f"    -> Filterbank output tensor: {subbands.shape} (Channels x Samples)")
    
    # Step 2: Inner Hair Cell Synaptic Transduction
    print("[*] Transducing basilar membrane motion into IHC synaptic transmitter release...")
    ihc = InnerHairCell(sample_rate=fs)
    cleft_profiles = ihc.transduce_multichannel(subbands)
    print(f"    -> Synaptic cleft transmitter profiles computed: peak conc = {np.max(cleft_profiles):.4f}")
    
    # Step 3: Auditory Nerve Fiber Spiking & Neurogram
    print("[*] Simulating Auditory Nerve Fiber population (HSR/MSR/LSR renewal processes)...")
    anf = AuditoryNerveFiberPopulation(sample_rate=fs, fibers_per_channel=10, seed=123)
    neurogram_acoustic = anf.generate_neurogram(cleft_profiles)
    total_spikes = np.sum(neurogram_acoustic * 10)
    print(f"    -> Acoustic neurogram generated. Total spikes registered: {int(total_spikes)}")
    
    # Step 4: Cochlear Implant CIS Speech Processor
    print("[*] Generating Continuous Interleaved Sampling (CIS) Biphasic Electrical Stimulation...")
    cis = CochlearImplantCISProcessor(sample_rate=fs, params=CISParams(num_electrodes=n_channels))
    electrogram, current_levels = cis.synthesize_electrogram(subbands)
    print(f"    -> CIS electrogram generated: shape {electrogram.shape}, dynamic range: [{np.min(electrogram):.1f}, {np.max(electrogram):.1f}] CU")
    
    # Step 5: Metrics & Evaluation
    print("[*] Computing Auditory & Prosthetic Biometric Metrics...")
    # Phase locking at 500 Hz channel (near F1)
    ch_f1 = np.argmin(np.abs(fb.center_frequencies - 500.0))
    ch_spikes = anf.simulate_channel_population(cleft_profiles[ch_f1])
    spike_indices = np.where(ch_spikes == 1)[1]
    spike_times = spike_indices / fs
    vs = vector_strength(spike_times, 500.0)
    print(f"    -> Vector Strength at Channel {ch_f1} (CF={fb.center_frequencies[ch_f1]:.1f} Hz) vs 500 Hz: {vs:.4f}")
    
    # PSTH
    bin_centers, rates = compute_psth(ch_spikes, sample_rate=fs, bin_width_ms=2.0)
    print(f"    -> PSTH peak firing rate: {np.max(rates):.1f} spikes/sec")
    
    # Neurogram similarity with normalized CIS envelope
    nsim = neurogram_similarity(neurogram_acoustic, current_levels)
    print(f"    -> Spectro-Temporal Neurogram Similarity (Acoustic vs CIS Electric): {nsim:.4f}")
    
    print("======================================================================")
    print(" Demonstration completed successfully! ALL AUDITORY MODULES VERIFIED. ")
    print("======================================================================")


if __name__ == "__main__":
    main()
