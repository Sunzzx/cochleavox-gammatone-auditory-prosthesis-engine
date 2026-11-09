"""Auditory Nerve Fiber (ANF) Population and Spiking Dynamics.

Implements inhomogeneous Poisson point process with absolute and relative refractory periods
across three distinct spontaneous rate (SR) classes:
- High SR (HSR): > 18 spikes/sec, low threshold, high sensitivity
- Medium SR (MSR): 0.5 - 18 spikes/sec, intermediate threshold
- Low SR (LSR): < 0.5 spikes/sec, high threshold, wide dynamic range.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple


@dataclass
class FiberTypeParams:
    name: str
    spontaneous_rate: float  # spikes / sec
    saturation_rate: float   # spikes / sec
    threshold_cleft: float   # cleft transmitter concentration threshold
    refractory_abs_ms: float = 0.75  # ms absolute refractory period
    refractory_rel_ms: float = 1.25  # ms relative refractory time constant


FIBER_PROFILES = {
    "HSR": FiberTypeParams(
        name="High-SR",
        spontaneous_rate=50.0,
        saturation_rate=250.0,
        threshold_cleft=0.01,
        refractory_abs_ms=0.75,
        refractory_rel_ms=1.0
    ),
    "MSR": FiberTypeParams(
        name="Medium-SR",
        spontaneous_rate=10.0,
        saturation_rate=220.0,
        threshold_cleft=0.08,
        refractory_abs_ms=0.85,
        refractory_rel_ms=1.5
    ),
    "LSR": FiberTypeParams(
        name="Low-SR",
        spontaneous_rate=1.0,
        saturation_rate=180.0,
        threshold_cleft=0.25,
        refractory_abs_ms=1.0,
        refractory_rel_ms=2.0
    )
}


class AuditoryNerveFiberPopulation:
    """Auditory nerve fiber population simulation with refractoriness."""
    
    def __init__(
        self,
        sample_rate: float = 44100.0,
        hsr_ratio: float = 0.60,
        msr_ratio: float = 0.25,
        lsr_ratio: float = 0.15,
        fibers_per_channel: int = 20,
        seed: int | None = 42
    ):
        self.sample_rate = sample_rate
        self.dt = 1.0 / sample_rate
        self.fibers_per_channel = fibers_per_channel
        self.rng = np.random.default_rng(seed)
        
        # Assign fiber types according to physiological ratios (Liberman 1978)
        n_hsr = int(np.round(fibers_per_channel * hsr_ratio))
        n_msr = int(np.round(fibers_per_channel * msr_ratio))
        n_lsr = fibers_per_channel - n_hsr - n_msr
        
        self.fiber_types: List[FiberTypeParams] = (
            [FIBER_PROFILES["HSR"]] * n_hsr +
            [FIBER_PROFILES["MSR"]] * n_msr +
            [FIBER_PROFILES["LSR"]] * n_lsr
        )

    def generate_spikes(
        self,
        cleft_profile: np.ndarray,
        fiber_param: FiberTypeParams
    ) -> np.ndarray:
        """Generate spike train for a single fiber using renewal process with refractoriness.
        
        Args:
            cleft_profile: Instantaneous cleft neurotransmitter concentration
            fiber_param: Fiber physiological parameters
            
        Returns:
            Binary 1D array where 1 indicates an action potential spike at that sample.
        """
        n_samples = len(cleft_profile)
        spikes = np.zeros(n_samples, dtype=np.uint8)
        
        # Synaptic driving rate: R(t) = R_spont + (R_sat - R_spont) * (c / (c + c_thresh))
        c = np.maximum(0.0, cleft_profile)
        rel_drive = c / (c + fiber_param.threshold_cleft + 1e-12)
        instant_rate = fiber_param.spontaneous_rate + (
            fiber_param.saturation_rate - fiber_param.spontaneous_rate
        ) * rel_drive
        
        # Refractory dynamics
        abs_refract_samples = int(np.round((fiber_param.refractory_abs_ms * 1e-3) / self.dt))
        rel_refract_tau_samples = (fiber_param.refractory_rel_ms * 1e-3) / self.dt
        
        last_spike_idx = -100000
        
        # Vector of uniform random numbers for Poisson discharge generation
        rand_vals = self.rng.random(n_samples)
        
        for t in range(n_samples):
            delta_t_samples = t - last_spike_idx
            
            if delta_t_samples <= abs_refract_samples:
                # Within absolute refractory period: probability of spike is 0
                refract_mult = 0.0
            else:
                # Relative refractory recovery: 1 - exp(-(t - t_abs) / tau_rel)
                recovery = 1.0 - np.exp(-(delta_t_samples - abs_refract_samples) / rel_refract_tau_samples)
                refract_mult = recovery
                
            p_spike = 1.0 - np.exp(-instant_rate[t] * refract_mult * self.dt)
            
            if rand_vals[t] < p_spike:
                spikes[t] = 1
                last_spike_idx = t
                
        return spikes

    def simulate_channel_population(
        self,
        cleft_profile: np.ndarray
    ) -> np.ndarray:
        """Simulate all fibers in a single channel pool.
        
        Returns:
            2D binary array of shape (fibers_per_channel, n_samples)
        """
        n_samples = len(cleft_profile)
        pop_spikes = np.zeros((self.fibers_per_channel, n_samples), dtype=np.uint8)
        
        for i, fiber_param in enumerate(self.fiber_types):
            pop_spikes[i] = self.generate_spikes(cleft_profile, fiber_param)
            
        return pop_spikes

    def generate_neurogram(
        self,
        cleft_multichannel: np.ndarray
    ) -> np.ndarray:
        """Generate full auditory nerve neurogram (PSTH across tonotopic channels).
        
        Args:
            cleft_multichannel: 2D array of shape (num_channels, n_samples)
            
        Returns:
            2D array of shape (num_channels, n_samples) representing summed spike activity.
        """
        num_channels, n_samples = cleft_multichannel.shape
        neurogram = np.zeros((num_channels, n_samples), dtype=np.float64)
        
        for ch in range(num_channels):
            spikes = self.simulate_channel_population(cleft_multichannel[ch])
            neurogram[ch] = np.sum(spikes, axis=0) / self.fibers_per_channel
            
        return neurogram
