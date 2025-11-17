"""
Small Language Model (SLM) Package
"""

from .model import SmallLanguageModel
from .data_loader import TextDataLoader

__version__ = "1.0.0"
__all__ = ["SmallLanguageModel", "TextDataLoader"]
