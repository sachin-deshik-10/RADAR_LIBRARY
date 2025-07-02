"""
Radar Perception Library

A comprehensive collection of radar datasets, detection algorithms, 
tracking methods, and sensor fusion techniques for autonomous systems 
and robotics applications.
"""

__version__ = "1.0.0"
__author__ = "Radar Perception Library Contributors"
__email__ = "contact@radarperception.dev"
__license__ = "MIT"

from . import signal_processing
from . import detection
from . import tracking
from . import fusion
from . import datasets
from . import utils
from . import visualization

__all__ = [
    "signal_processing",
    "detection", 
    "tracking",
    "fusion",
    "datasets",
    "utils",
    "visualization",
]
