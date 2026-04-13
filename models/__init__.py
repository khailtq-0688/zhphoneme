"""
ViWordFormer Pretraining Architecture Module
Registers the model with the architecture registry
"""

from builders.registry import META_ARCHITECTURE
from models.viwordformer import ViWordFormer

__all__ = ['ViWordFormer']

