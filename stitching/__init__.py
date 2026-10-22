"""Image stitching API. Calling stitch never writes files."""
from .core import StitchConfig, StitchingError, StitchResult, stitch

__all__ = ['StitchConfig', 'StitchingError', 'StitchResult', 'stitch']
