import numpy as np
from ase import Atoms
from ase.build import sort
from ase.formula import Formula
from ase.io import read
from monty.tempfile import ScratchDir
from mp_api.client import MPRester
from pymatgen.analysis.structure_analyzer import SpacegroupAnalyzer
from pymatgen.core import Composition, Molecule
from pymatgen.io.packmol import PackmolBoxGen

from muse.utils import MP_API_KEY

mpr = MPRester(MP_API_KEY)

def mix(
        recipe: dict[str, int], 
        tolerance=2.0,
        rattle=0.5, 
        scale=1.0,
        shuffle=True,
        seed=1,
        log=False
        ) -> Atoms:
    """
    Mixes a set of molecules according to a recipe.
    """

    np.random.seed(seed)

    molecules = []
    
    for formula, units in recipe.items():

        reduced_formula, input_mult = Formula(formula).reduce()

        entries = mpr.get_entries(
            chemsys_formula_mpids=str(reduced_formula),
            conventional_unit_cell=False,
            sort_by_e_above_hull=True
        )

        sga = SpacegroupAnalyzer(entries[0].structure)
        primitive_structure = sga.get_primitive_standard_structure()

        primitive_formula = Formula(primitive_structure.composition.to_pretty_string())

        molecule = Molecule(
            species=primitive_structure.species,
            coords=primitive_structure.cart_coords
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
            # number = float(Formula(formula) * int(recipe[formula])) // primitive_formula
            if log:
                print(recipe[formula], Formula(formula), number, primitive_formula)

            count += 1

        molecules.append({
            "name": primitive_structure.composition.to_pretty_string(),
            "number": int(number),
            "coords": molecule,
            "volume": primitive_structure.volume,
        })
    
    if log:
        print(molecules)

    total_volume = 0
    for molecule in molecules:
        total_volume += molecule["volume"] * molecule["number"]

    a = total_volume**(1.0/3.0) * scale
    
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
                    box=[
                        margin, margin, margin, 
                        a-margin, a-margin, a-margin
                        ]
                )
                packmol_set.write_input(".")
                packmol_set.run(".")

                atoms = read("packmol_out.xyz", format="xyz")
                break
            except Exception as e:
                if log:
                    print(e)
                seed += 1
                if seed % int(1e3) == 0:
                    if log:
                        print("WARNING: Packmol failed 1000 times. Trying again with larger box.")
                    a *= 1.05

                if a > total_volume**(1.0/3.0) * scale * 2:
                    if log:
                        print("WARNING: Box size was increased by more than 2x. Generate random strcture.")

                    a = total_volume**(1.0/3.0) * scale

                    symbols = ""
                    for molecule in molecules:
                        symbols += molecule["name"] * molecule["number"]

                    atoms = Atoms(
                        symbols=symbols, 
                        cell=[a, a, a], 
                        pbc=True
                        )
                    atoms.set_scaled_positions(np.random.random(size=atoms.positions.shape))
    
    atoms.set_cell([a, a, a])
    atoms.set_pbc(True)

    if a != total_volume**(1.0/3.0) * scale:
        if log:
            print("WARNING: Box size was increased. Shrinking to designated size.")
        scaled_positions = atoms.get_scaled_positions()
        a = total_volume**(1.0/3.0) * scale
        atoms.set_cell([a, a, a])
        atoms.set_scaled_positions(scaled_positions)

    if rattle > 0:
        atoms.positions += np.random.normal(0, rattle, size=atoms.positions.shape)
    
    if shuffle:
        atoms.numbers = np.random.permutation(atoms.numbers)

    return sort(atoms)








