# src/spec_normalizer/__init__.py
from .t5_client import T5Client
from .spec_fusion import SpecFusion
from .spec_normalizer_interface import SpecNormalizerModelInterface
from .diffusion_spec_normalizer import DiffusionSpecNormalizer # Add this

__all__ = [
    "T5Client",
    "SpecFusion",
    "SpecNormalizerModelInterface",
    "DiffusionSpecNormalizer" # Add this
]
