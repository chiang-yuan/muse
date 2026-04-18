"""Structure generation for amorphous solid and liquid mixtures.

Provides :func:`mix_number` and :func:`mix_cell` for building
multicomponent mixtures using Packmol packing with crystal structure
prototypes retrieved from the Materials Project.
"""

from __future__ import annotations

import logging
import math

import numpy as np
from ase import Atoms
from ase.build import sort
from ase.cell import Cell
from ase.formula import Formula
from ase.io import read
from monty.tempfile import ScratchDir
from mp_api.client import MPRester
from pymatgen.analysis.structure_analyzer import SpacegroupAnalyzer
from pymatgen.core import Molecule
from pymatgen.io.packmol import PackmolBoxGen

from muse.utils import MP_API_KEY

logger = logging.getLogger(__name__)


def mix_number(
    recipe: dict[str, int],
    density: float | None = None,
    tolerance: float = 2.0,
    rattle: float = 0.5,
    scale: float = 1.0,
    shuffle: bool = False,
    seed: int = 1,
    timeout: int = 30,
    log: bool = False,
    mp_api_key: str | None = MP_API_KEY,
    retry: int = 1000,
    retry_scale: float = 1.5,
) -> Atoms:
    """Build a mixture structure by specifying formula unit counts.

    Retrieves primitive crystal structures from Materials Project for each
    component, then uses Packmol to pack the specified number of formula
    units into a cubic simulation cell. The cell size is estimated from
    the sum of solid-state primitive cell volumes.

    Args:
        recipe: Mapping of chemical formula strings to the desired number
            of formula units (e.g., ``{"NaCl": 3, "KCl": 1}``).
        density: Target mass density in amu/ų. If provided, the cell is
            rescaled after packing. Defaults to None (use solid-state volume).
        tolerance: Minimum distance between packed molecules in Å.
            Defaults to 2.0.
        rattle: Standard deviation of Gaussian noise added to atomic
            positions in Å. Defaults to 0.5.
        scale: Multiplicative factor for the estimated cubic cell edge.
            Defaults to 1.0.
        shuffle: If True, randomly permute atomic species numbers.
            Defaults to False.
        seed: Random seed for Packmol and numpy. Defaults to 1.
        timeout: Packmol timeout in seconds. Defaults to 30.
        log: If True, print diagnostic information. Defaults to False.
        mp_api_key: Materials Project API key. Defaults to the
            ``MP_API_KEY`` environment variable.
        retry: Number of Packmol attempts before enlarging the box.
            Defaults to 1000.
        retry_scale: Factor by which to enlarge the box after ``retry``
            failures. Defaults to 1.5.

    Returns:
        Sorted ASE Atoms object with periodic boundary conditions.
    """
    mpr = MPRester(mp_api_key)
    rng = np.random.default_rng(seed)

    molecules = []

    for formula, units in recipe.items():
        if units == 0:
            continue

        reduced_formula, input_mult = Formula(formula).reduce()

        docs = mpr.materials.summary.search(
            formula=str(reduced_formula),
            is_stable=True,
            fields=["material_id", "formula_pretty", "structure"],
        )

        sga = SpacegroupAnalyzer(docs[0].structure)
        primitive_structure = sga.get_primitive_standard_structure()

        primitive_formula = Formula(primitive_structure.composition.to_pretty_string())

        molecule = Molecule(
            species=primitive_structure.species, coords=primitive_structure.cart_coords
        )
        _, primitive_mult = primitive_formula.reduce()

        number: float = 0.0
        count: int = 0
        while number == 0 or not math.isclose(number, round(number), rel_tol=1e-3):
            if count > 0:
                for key, value in recipe.items():
                    recipe[key] = value / count * (count + 1)

                for d in molecules:
                    d["number"] = d["number"] / count * (count + 1)

            number = input_mult * recipe[formula] / primitive_mult
            if log:
                print(recipe[formula], Formula(formula), number, primitive_formula)

            count += 1

        molecules.append(
            {
                "name": primitive_structure.composition.to_pretty_string(),
                "number": int(number),
                "coords": molecule,
                "volume": primitive_structure.volume,
            }
        )

    if log:
        print(molecules)

    total_volume = 0
    for molecule in molecules:
        total_volume += molecule["volume"] * molecule["number"]

    a = total_volume ** (1.0 / 3.0) * scale

    with ScratchDir("."):
        while True:
            try:
                input_gen = PackmolBoxGen(
                    tolerance=tolerance,
                    seed=seed,
                )

                margin = 0.5 * tolerance
                packmol_set = input_gen.get_input_set(
                    molecules=molecules,
                    box=[margin, margin, margin, a - margin, a - margin, a - margin],
                )
                packmol_set.write_input(".")
                packmol_set.run(".", timeout=timeout)

                atoms = read("packmol_out.xyz", format="xyz")
                break
            except Exception as e:
                if log:
                    print(e)
                seed += 1

                if a > total_volume ** (1.0 / 3.0) * scale * 2:
                    if log:
                        logger.warning(
                            "Box size was increased by more than 2x. "
                            "Generating random structure."
                        )

                    a = total_volume ** (1.0 / 3.0) * scale

                    symbols = ""
                    for molecule in molecules:
                        symbols += Formula(molecule["name"]) * int(molecule["number"])

                    atoms = Atoms(symbols=symbols, cell=[a, a, a], pbc=True)
                    atoms = sort(atoms)

                    atoms.set_scaled_positions(rng.random(size=atoms.positions.shape))
                    break

                if seed % retry == 0:
                    a *= retry_scale
                    if log:
                        logger.warning(
                            "Packmol failed %d times. Trying again with larger box. "
                            "New box size: %s",
                            retry,
                            a,
                        )

    atoms.set_cell([a, a, a])
    atoms.set_pbc(True)

    if a != total_volume ** (1.0 / 3.0) * scale:
        if log:
            logger.warning("Box size was increased. Shrinking to the designated size.")
        scaled_positions = atoms.get_scaled_positions()
        a = total_volume ** (1.0 / 3.0) * scale
        atoms.set_cell([a, a, a])
        atoms.set_scaled_positions(scaled_positions)

    if rattle > 0:
        atoms.positions += rng.normal(0, rattle, size=atoms.positions.shape)

    if shuffle:
        atoms.numbers = rng.permutation(atoms.numbers)

    if density is not None:
        cellpar = atoms.cell.cellpar()

        volume = atoms.get_masses().sum() / density

        factor = (volume / atoms.get_volume()) ** (1.0 / 3.0)
        atoms.set_cell(
            Cell.fromcellpar(
                [
                    cellpar[0] * factor,
                    cellpar[1] * factor,
                    cellpar[2] * factor,
                    cellpar[3],
                    cellpar[4],
                    cellpar[5],
                ]
            ),
            scale_atoms=True,
        )

    return sort(atoms)


def mix_cell(
    recipe: dict[str, float],
    cell: Cell,
    tolerance: float = 2.0,
    rattle: float = 0.5,
    scale: float = 1.0,
    shuffle: bool = True,
    seed: int = 1,
    log: bool = False,
    mp_api_key: str | None = MP_API_KEY,
    retry_scale: float = 1.5,
) -> Atoms:
    """Build a mixture structure to fill a given simulation cell.

    Similar to :func:`mix_number`, but instead of specifying absolute
    formula unit counts, the recipe specifies molar fractions and the
    total number of atoms is determined by the target cell volume.

    The function scales the number of molecules to fill the provided
    cell based on the ratio of cell volume to the total solid-state
    volume of the components.

    Args:
        recipe: Mapping of chemical formula strings to molar ratios
            (e.g., ``{"NaCl": 0.7, "KCl": 0.3}``).
        cell: Target ASE Cell object defining the simulation box shape.
        tolerance: Minimum distance between packed molecules in Å.
            Defaults to 2.0.
        rattle: Standard deviation of Gaussian noise added to atomic
            positions in Å. Defaults to 0.5.
        scale: Multiplicative factor for cell dimensions during packing.
            Defaults to 1.0.
        shuffle: If True, randomly permute atomic species numbers.
            Defaults to True.
        seed: Random seed for Packmol and numpy. Defaults to 1.
        log: If True, print diagnostic information. Defaults to False.
        mp_api_key: Materials Project API key. Defaults to the
            ``MP_API_KEY`` environment variable.
        retry_scale: Factor by which to enlarge the box after 1000
            Packmol failures. Defaults to 1.5.

    Returns:
        Sorted ASE Atoms object with periodic boundary conditions
        matching the target cell.
    """
    mpr = MPRester(mp_api_key)
    rng = np.random.default_rng(seed)

    molecules = []

    for formula, units in recipe.items():
        if units == 0:
            continue

        reduced_formula, input_mult = Formula(formula).reduce()

        docs = mpr.materials.summary.search(
            formula=str(reduced_formula),
            is_stable=True,
            fields=["material_id", "formula_pretty", "structure"],
        )

        sga = SpacegroupAnalyzer(docs[0].structure)
        primitive_structure = sga.get_primitive_standard_structure()

        primitive_formula = Formula(primitive_structure.composition.to_pretty_string())

        molecule = Molecule(
            species=primitive_structure.species, coords=primitive_structure.cart_coords
        )
        _, primitive_mult = primitive_formula.reduce()

        number: float = 0.0
        count: int = 0
        while number == 0 or not number.is_integer():
            if count > 0:
                for key, value in recipe.items():
                    recipe[key] = value / count * (count + 1)

                for d in molecules:
                    d["number"] = d["number"] / count * (count + 1)

            number = input_mult * recipe[formula] / primitive_mult
            if log:
                print(recipe[formula], Formula(formula), number, primitive_formula)

            count += 1

        molecules.append(
            {
                "name": primitive_structure.composition.to_pretty_string(),
                "number": int(number),
                "coords": molecule,
                "volume": primitive_structure.volume,
            }
        )

    total_volume = 0
    for molecule in molecules:
        total_volume += molecule["volume"] * molecule["number"]

    nfactor = cell.volume / total_volume

    for molecule in molecules:
        molecule["number"] = int(molecule["number"] * nfactor)

    if log:
        print(molecules)

    a, b, c, alpha, beta, gamma = cell.cellpar()

    with ScratchDir("."):
        while True:
            try:
                input_gen = PackmolBoxGen(
                    tolerance=tolerance,
                    seed=seed,
                )

                margin = 0.5 * tolerance
                packmol_set = input_gen.get_input_set(
                    molecules=molecules,
                    box=[margin, margin, margin, a - margin, b - margin, c - margin],
                )
                packmol_set.write_input(".")
                packmol_set.run(".")

                atoms = read("packmol_out.xyz", format="xyz")
                break
            except Exception as e:
                if log:
                    print(e)
                seed += 1

                if a > cell.volume ** (1.0 / 3.0) * scale * 2:
                    if log:
                        logger.warning(
                            "Box size was increased by more than 2x. "
                            "Generating random structure."
                        )

                    a, b, c, alpha, beta, gamma = cell.cellpar()
                    atoms = Atoms(cell=cell, pbc=True)
                    atoms.set_scaled_positions(rng.random(size=atoms.positions.shape))
                    break

                if seed % 1000 == 0:
                    a *= retry_scale
                    if log:
                        logger.warning(
                            "Packmol failed 1000 times. Trying again with larger box. "
                            "New box size: %s",
                            a,
                        )

    atoms.set_cell(cell)
    atoms.set_pbc(True)

    if rattle > 0:
        atoms.positions += rng.normal(0, rattle, size=atoms.positions.shape)

    if shuffle:
        atoms.numbers = rng.permutation(atoms.numbers)

    return sort(atoms)
