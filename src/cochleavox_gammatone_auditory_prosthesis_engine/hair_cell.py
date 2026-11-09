"""Inner Hair Cell (IHC) and Synaptic Vesicle Dynamics Model.

Based on Meddis (1986, 1988) inner hair cell model with transmitter release and replenishing kinetics:
- Hair cell receptor potential from basilar membrane mechanical displacement
- Presynaptic calcium channel activation: [Ca2+] entry
- Available transmitter pool (q), cleft pool (c), and reprocessing pool (w)
- Immediate release, cleft clearance, and endocytosis recycling kinetics.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class HairCellParams:
    """Physiological parameters for Meddis IHC synaptic model."""
    # Mechanical-to-receptor potential rectification and saturation
    a: float = 0.1
    b: float = 2000.0
    g_max: float = 2.5e-9  # Maximum conductances
    
    # Permeability / calcium kinetics
    perm_max: float = 600.0
    ca_thresh: float = 0.05
    
    # Synaptic vesicle pool capacities and rates
    m: float = 1.0        # Max transmitter pool capacity per active zone
    k_replenish: float = 130.0  # Replenishment rate (1/s) from factory
    l_loss: float = 250.0       # Loss / degradation rate from cleft (1/s)
    r_reuptake: float = 6580.0  # Cleft reuptake back into processing store (1/s)
    x_reprocess: float = 66.3   # Reprocessing rate into available pool (1/s)


class InnerHairCell:
    """Meddis-style Inner Hair Cell Transduction and Synapse Model."""
    
    def __init__(self, sample_rate: float = 44100.0, params: HairCellParams | None = None):
        self.sample_rate = sample_rate
        self.dt = 1.0 / sample_rate
        self.params = params or HairCellParams()

    def receptor_potential(self, bm_motion: np.ndarray) -> np.ndarray:
        """Compute IHC receptor potential from basilar membrane displacement.
        
        Applies asymmetric half-wave rectification and compressive logarithmic saturation:
        V_ihc(t) = sign(x) * log(1 + a * |x|) / (1 + b * |x|)
        """
        x = np.asarray(bm_motion, dtype=np.float64)
        # Asymmetric transduction: hair bundle deflection opens MET channels mainly in depolarizing direction
        # Depolarization occurs when stereocilia shear towards kinocilium (positive displacement)
        v = np.zeros_like(x)
        pos_mask = x > 0
        neg_mask = ~pos_mask
        
        # Positive half: compressive saturation
        v[pos_mask] = np.log1p(self.params.a * x[pos_mask]) / (1.0 + self.params.a * x[pos_mask] * 0.1)
        # Negative half: hyperpolarization cut-off with small leakage
        v[neg_mask] = -0.1 * np.log1p(np.abs(x[neg_mask]))
        
        # Normalize into range [-0.2, 1.0]
        return v

    def synaptic_transmitter_release(self, v_ihc: np.ndarray) -> np.ndarray:
        """Simulate neurotransmitter release across the IHC-auditory nerve synapse.
        
        Solves the Meddis coupled differential equations for pool stores:
        dq/dt = y * (M - q) + x * w - k(t) * q
        dc/dt = k(t) * q - (l + r) * c
        dw/dt = r * c - x * w
        
        where k(t) is time-varying permeability driven by calcium channel opening.
        
        Returns:
            c(t): Neurotransmitter concentration in synaptic cleft (proportional to instantaneous firing rate).
        """
        p = self.params
        dt = self.dt
        n_samples = len(v_ihc)
        
        # Permeability k(t) driven by depolarization above threshold
        # k(t) = perm_max * (v_ihc(t) - thresh)^2 / ((v_ihc(t) - thresh)^2 + 0.1) if v > thresh else 0
        v_eff = np.maximum(0.0, v_ihc - p.ca_thresh)
        k_t = p.perm_max * (v_eff ** 2) / (v_eff ** 2 + 0.05 + 1e-12)
        
        # State variables for pools
        q = np.zeros(n_samples, dtype=np.float64)  # Free transmitter store
        c = np.zeros(n_samples, dtype=np.float64)  # Cleft transmitter
        w = np.zeros(n_samples, dtype=np.float64)  # Reprocessing pool
        
        # Initial resting equilibrium state
        q_val = p.m
        c_val = 0.0
        w_val = 0.0
        
        # Numerical integration using 1st-order Euler with sub-stepping for numerical stiffness
        substeps = 2
        dt_sub = dt / substeps
        
        for t in range(n_samples):
            kt = k_t[t]
            for _ in range(substeps):
                # Transmitter release: rate = kt * q
                rel = kt * q_val
                
                dq = p.k_replenish * (p.m - q_val) + p.x_reprocess * w_val - rel
                dc = rel - (p.l_loss + p.r_reuptake) * c_val
                dw = p.r_reuptake * c_val - p.x_reprocess * w_val
                
                q_val = np.clip(q_val + dq * dt_sub, 0.0, p.m)
                c_val = np.maximum(0.0, c_val + dc * dt_sub)
                w_val = np.maximum(0.0, w_val + dw * dt_sub)
                
            q[t] = q_val
            c[t] = c_val
            w[t] = w_val
            
        return c

    def transduce_multichannel(self, subbands: np.ndarray) -> np.ndarray:
        """Process all subbands from Gammatone filterbank into cleft neurotransmitter profiles.
        
        Args:
            subbands: 2D array of shape (num_channels, num_samples)
            
        Returns:
            2D array of shape (num_channels, num_samples) of transmitter concentration in synaptic cleft.
        """
        num_channels, num_samples = subbands.shape
        cleft_rates = np.zeros_like(subbands)
        
        for ch in range(num_channels):
            v = self.receptor_potential(subbands[ch])
            c = self.synaptic_transmitter_release(v)
            cleft_rates[ch] = c
            
        return cleft_rates
