"""CochleaVox: Biophysical Gammatone Auditory Perceptual & Neural Prosthesis Engine."""

from .filterbank import GammatoneFilterbank, hz_to_erb, erb_to_hz, erb_bandwidth
from .hair_cell import InnerHairCell, HairCellParams
from .nerve_fiber import AuditoryNerveFiberPopulation, FiberTypeParams, FIBER_PROFILES
from .prosthesis import CochlearImplantCISProcessor, CISParams
from .metrics import compute_psth, vector_strength, neurogram_similarity

__version__ = "0.1.0"
__all__ = [
    "GammatoneFilterbank",
    "hz_to_erb",
    "erb_to_hz",
    "erb_bandwidth",
    "InnerHairCell",
    "HairCellParams",
    "AuditoryNerveFiberPopulation",
    "FiberTypeParams",
    "FIBER_PROFILES",
    "CochlearImplantCISProcessor",
    "CISParams",
    "compute_psth",
    "vector_strength",
    "neurogram_similarity",
]
