"""Convert pymatgen trajectory data to extended XYZ format.

Supports writing energies, forces, stresses, charges, magnetic moments,
and dipoles from Materials Project trajectory (MPtrj) data.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

    from pymatgen.core.trajectory import Trajectory

__author__ = "Yuan Chiang"
__date__ = "2023-08-02"


def pmgtraj_to_extxyz(pmgtraj: Trajectory, fname: str | Path) -> None:
    """Convert a pymatgen Trajectory to an extended XYZ file.

    Writes each frame of the trajectory in the extended XYZ format,
    including lattice vectors, per-atom properties (species, positions,
    forces, charges, magnetic moments, dipoles), and frame-level
    properties (energies, stresses).

    Args:
        pmgtraj: A pymatgen Trajectory object, typically from MPtrj data.
        fname: Output file path for the extended XYZ file.
    """
    with open(fname, "w") as f:
        for iframe, structure in enumerate(pmgtraj):
            # Write the extxyz format for each snapshot
            f.write(f"{len(structure)}\n")
            f.write(
                f'Lattice="{" ".join(map(str, structure.lattice.matrix.ravel()))}" '
            )
            f.write("Properties=species:S:1:pos:R:3")

            properties = pmgtraj.frame_properties[iframe]
            if "forces" in properties:
                f.write(":forces:R:3")
            if "charges" in properties:
                f.write(":charges:R:1")
            if "magmoms" in properties:
                f.write(":magmoms:R:3")
            if "dipoles" in properties:
                f.write(":dipoles:R:3")

            if "e_0_energy" in properties:
                f.write(f" energy={properties['e_0_energy']}")
            if "e_fr_energy" in properties:
                f.write(f" free_energy={properties['e_fr_energy']}")
            # e_wo_entrp: energy without electronic entropy (0K electron)
            if "e_wo_entrp" in properties:
                f.write(f" e_wo_entrp={properties['e_wo_entrp']}")

            if "stresses" in properties:
                f.write(
                    f' stresses="{" ".join(map(str, np.array(properties["stresses"]).ravel()))}"'
                )

            f.write("\n")

            for idx, site in enumerate(structure):
                line = f"{site.species_string} {' '.join(map(str, site.coords))}"

                if "forces" in properties:
                    line += f" {' '.join(map(str, properties['forces'][idx]))}"
                if "charges" in properties:
                    line += f" {' '.join(map(str, properties['charges'][idx]))}"
                if "magmoms" in properties:
                    line += f" {' '.join(map(str, properties['magmoms'][idx]))}"
                if "dipoles" in properties:
                    line += f" {' '.join(map(str, properties['dipoles'][idx]))}"

                line += "\n"
                f.write(line)
