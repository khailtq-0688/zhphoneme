"""
ViWordFormer Pretraining Architecture Module
Registers the model with the architecture registry
"""

from builders.registry import META_ARCHITECTURE
from .viwordformer import ViWordFormer

# Register model
META_ARCHITECTURE.register(ViWordFormer)

__all__ = ['ViWordFormer']

