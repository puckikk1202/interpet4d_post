"""Compatibility export for the VQ bottleneck used by current models."""

from archived.bottleneck import Bottleneck, NoBottleneck

__all__ = ["Bottleneck", "NoBottleneck"]
