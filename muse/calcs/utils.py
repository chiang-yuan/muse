"""Trajectory observation utilities for MD simulations.

Provides :class:`TrajectoryObserver`, a callback hook that records
energies, forces, stresses, positions, and cell parameters during
ASE relaxations and molecular dynamics runs.
"""

from __future__ import annotations

import logging
import pickle
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np
    from ase import Atoms


logger = logging.getLogger(__name__)


class TrajectoryObserver:
    """Trajectory observer that records simulation data at each step.

    Attach this observer to an ASE optimizer or dynamics object to
    capture per-step energies, forces, stresses, positions, and cell
    matrices. Data can be serialized to a pickle file for post-processing.

    This class is adapted from
    `matcalc <https://github.com/materialsvirtuallab/matcalc>`_.

    Args:
        atoms: The ASE Atoms object to observe.
    """

    def __init__(self, atoms: Atoms) -> None:
        self.atoms = atoms
        self.energies: list[float] = []
        self.forces: list[np.ndarray] = []
        self.stresses: list[np.ndarray] = []
        self.atom_positions: list[np.ndarray] = []
        self.cells: list[np.ndarray] = []

    def __call__(self) -> None:
        """Record the current state of the Atoms object.

        Captures potential energy, forces, stress tensor (including ideal
        gas contribution when available), positions, and cell matrix.
        """
        self.energies.append(float(self.atoms.get_potential_energy()))
        self.forces.append(self.atoms.get_forces())
        # Stress tensor should include the contribution from the momenta, otherwise
        # during MD simulation the stress tensor ignores the effect of kinetic part,
        # leading to the discrepancy between applied pressure and the stress tensor.
        # For more details, see: https://gitlab.com/ase/ase/-/merge_requests/1500
        try:
            stress = self.atoms.get_stress(include_ideal_gas=True)
        except TypeError:
            stress = self.atoms.get_stress()
        self.stresses.append(stress)

        self.atom_positions.append(self.atoms.get_positions())
        self.cells.append(self.atoms.get_cell()[:])

    def save(self, filename: str | Path) -> None:
        """Save the recorded trajectory data to a pickle file.

        The output dictionary contains:
            - ``energy``: List of potential energies (eV).
            - ``forces``: List of force arrays (eV/Å).
            - ``stresses``: List of stress tensors (eV/ų Voigt).
            - ``atom_positions``: List of position arrays (Å).
            - ``cell``: List of cell matrices (Å).
            - ``atomic_number``: Array of atomic numbers.

        Args:
            filename: Path to the output pickle file.
        """
        out = {
            "energy": self.energies,
            "forces": self.forces,
            "stresses": self.stresses,
            "atom_positions": self.atom_positions,
            "cell": self.cells,
            "atomic_number": self.atoms.get_atomic_numbers(),
        }
        with open(filename, "wb") as file:
            pickle.dump(out, file)
