"""I/O utilities for converting trajectory formats.

Provides :func:`pmgtraj_to_extxyz` for converting pymatgen
:class:`~pymatgen.core.trajectory.Trajectory` objects to extended XYZ files.
"""

from muse.io.mptrj import pmgtraj_to_extxyz

__all__ = ["pmgtraj_to_extxyz"]
