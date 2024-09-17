import numpy as np
from ase import Atoms
from ase.build import sort
from ase.cell import Cell
from ase.formula import Formula
from ase.io import read
from monty.tempfile import ScratchDir
from mp_api.client import MPRester
from pymatgen.analysis.structure_analyzer import SpacegroupAnalyzer
from pymatgen.core import Composition, Molecule
from pymatgen.io.packmol import PackmolBoxGen

from muse.utils import MP_API_KEY


def mix_number(
    recipe: dict[str, int],
    density: float | None = None,
    tolerance: float = 2.0,
    rattle: float = 0.5,
    scale: float = 1.0,
    shuffle: bool = True,
    seed: int = 1,
    timeout: int = 30,
    log: bool = False,
    mp_api_key: str = MP_API_KEY,
    retry: int = 1000,
    retry_scale: float = 1.5,
) -> Atoms:
    """Mixes a set of molecules according to a recipe.

    Args:
        recipe (dict[str, int]): a dictionary of molecules and their desired ratios.
        density (float | None, optional): assinged mass density in atomic units.
            Defaults to None to calculate from solid state 0K density.
        tolerance (float, optional): tolerance for packmol in Angstrom. Defaults to 2.0.
        rattle (float, optional): position rattle in Angstrom. Defaults to 0.5.
        scale (float, optional): scale factor for box size. Defaults to 1.0.
        shuffle (bool, optional): shuffle atomic species. Defaults to True.
            Otherwise, the neighbor environment will be similar to solid state.
        seed (int, optional): Defaults to 1.
        log (bool, optional): Defaults to False.
        mp_api_key (str, optional): Defaults to MP_API_KEY.
        retry_scale (float, optional): factor to scale box after 1000 times of Packmol failure. Defaults to 1.5.

    Returns:
        Atoms: generated structure as ASE Atoms object.
    """

    mpr = MPRester(mp_api_key)
    np.random.seed(seed)

    molecules = []

    for formula, units in recipe.items():
        if units == 0:
            continue

        reduced_formula, input_mult = Formula(formula).reduce()

        docs = mpr.materials.summary.search(
            formula=str(reduced_formula),
            is_stable=True,
            fields=["material_id", "pretty_formula", "structure"],
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
                        print(
                            "WARNING: Box size was increased by more than 2x. Generate random strcture."
                        )

                    a = total_volume ** (1.0 / 3.0) * scale

                    symbols = ""
                    for molecule in molecules:
                        symbols += Formula(molecule["name"]) * int(molecule["number"])

                    atoms = Atoms(symbols=symbols, cell=[a, a, a], pbc=True)
                    atoms = sort(atoms)

                    atoms.set_scaled_positions(
                        np.random.random(size=atoms.positions.shape)
                    )
                    break

                if seed % retry == 0:
                    a *= retry_scale
                    if log:
                        print(
                            f"WARNING: Packmol failed {retry} times. Trying again with larger box. New box size: {a}"
                        )

    atoms.set_cell([a, a, a])
    atoms.set_pbc(True)

    if a != total_volume ** (1.0 / 3.0) * scale:
        if log:
            print("WARNING: Box size was increased. Shrinking to the designated size.")
        scaled_positions = atoms.get_scaled_positions()
        a = total_volume ** (1.0 / 3.0) * scale
        atoms.set_cell([a, a, a])
        atoms.set_scaled_positions(scaled_positions)

    if rattle > 0:
        atoms.positions += np.random.normal(0, rattle, size=atoms.positions.shape)

    if shuffle:
        atoms.numbers = np.random.permutation(atoms.numbers)

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
    cell: tuple[float, float],
    density: float | None = None,
    tolerance: float = 2.0,
    rattle: float = 0.5,
    scale: float = 1.0,
    shuffle: bool = True,
    seed: int = 1,
    log: bool = False,
    mp_api_key: str = MP_API_KEY,
    retry_scale: float = 1.5,
) -> Atoms:


    mpr = MPRester(mp_api_key)
    np.random.seed(seed)

    molecules = []

    for formula, units in recipe.items():
        if units == 0:
            continue

        reduced_formula, input_mult = Formula(formula).reduce()

        docs = mpr.materials.summary.search(
            formula=str(reduced_formula),
            is_stable=True,
            fields=["material_id", "pretty_formula", "structure"],
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
