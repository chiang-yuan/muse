"""Muse — build amorphous solids and liquid mixtures from Materials Project structures.

Muse provides tools for generating amorphous and liquid mixture structures
using Packmol packing and equilibrating them with machine learning
interatomic potentials via ASE molecular dynamics.

Example:
    >>> from muse.transforms.mixture import mix_number
    >>> atoms = mix_number(recipe={"NaCl": 3, "KCl": 1}, seed=42)
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("muse-xtal")
except PackageNotFoundError:
    __version__ = "0.2.0"  # fallback for editable installs without metadata

from muse.calcs.density import DensityCalc
from muse.calcs.utils import TrajectoryObserver
from muse.transforms.mixture import mix_cell, mix_number

__all__ = [
    "__version__",
    "DensityCalc",
    "TrajectoryObserver",
    "mix_cell",
    "mix_number",
]
