"""
SRB Adapters Package

This module exposes the officially supported model adapter interfaces for SRB.
Adapters translate backend model outputs into SRB-compatible TokenStep traces.
"""

from __future__ import annotations

from .base import ModelAdapter
from .openai import OpenAIChatAdapter
from .local import LocalTransformersAdapter

__all__ = [
    "ModelAdapter",
    "OpenAIChatAdapter",
    "LocalTransformersAdapter",
]