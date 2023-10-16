from __future__ import annotations

import pickle
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np
    from ase import Atoms
    from ase.calculators.calculator import Calculator

class TrajectoryObserver:
    """Trajectory observer is a hook in the relaxation process that saves the
    intermediate structures. This is a class modified from matcalc 
    https://github.com/materialsvirtuallab/matcalc
    """

    def __init__(self, atoms: Atoms) -> None:
        """
        Init the Trajectory Observer from a Atoms.

        Args:
            atoms (Atoms): Structure to observe.
        """
        self.atoms = atoms
        self.energies: list[float] = []
        self.forces: list[np.ndarray] = []
        self.stresses: list[np.ndarray] = []
        self.atom_positions: list[np.ndarray] = []
        self.cells: list[np.ndarray] = []

    def __call__(self) -> None:
        """The logic for saving the properties of an Atoms during the relaxation."""
        self.energies.append(float(self.atoms.get_potential_energy()))
        self.forces.append(self.atoms.get_forces())
        # Stress tensor should include the contribution from the momenta, otherwise
        # during MD simulattion the stress tensor ignores the effect of kinetic part,
        # leanding to the discrepancy between applied pressure and the stress tensor.
        # For more details, see: https://gitlab.com/ase/ase/-/merge_requests/1500
        try:
            stress = self.atoms.get_stress(include_ideal_gas=True)
        except Exception:
            stress = self.atoms.get_stress()
        self.stresses.append(stress)

        self.atom_positions.append(self.atoms.get_positions())
        self.cells.append(self.atoms.get_cell()[:])

    def save(self, filename: str) -> None:
        """Save the trajectory to file.

        Args:
            filename (str): filename to save the trajectory.
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

