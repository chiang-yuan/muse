"""Relax an arbitrary binary mixture using MACE.

Example usage:
    python binary-mixture.py '{"NaCl": 3, "KCl": 1}' 1100 0.0 \
        --calculator /path/to/mace/model \
        --log --seed 42
"""

import argparse
import json
import os
import os.path as osp
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
from ase import units
from ase.calculators.lj import LennardJones
from ase.io import write
from ase.optimize import FIRE
from mace.calculators import MACECalculator

from muse.calcs.density import DensityCalc
from muse.transforms.mixture import mix_number

os.environ["OMP_NUM_THREADS"] = "1"


def main(args):
    parent_dir = osp.join(args.root, "-".join(args.recipe.keys()))
    os.makedirs(parent_dir, exist_ok=True)

    # 0. Generate initial configuration for the mixture
    atoms = mix_number(
        recipe=args.recipe,
        tolerance=args.tolerance,
        rattle=args.rattle,
        scale=args.scale,
        shuffle=args.shuffle,
        seed=args.seed,
        log=args.log,
    )
    print(atoms)

    out_dir = osp.join(parent_dir, str(atoms.symbols))
    os.makedirs(out_dir, exist_ok=True)

    write(osp.join(out_dir, "0-packmol-mixture.xyz"), atoms)

    # 1. Pre-relax with Lennard-Jones potential
    sigma = atoms.get_volume() / len(atoms) ** (1 / 3)
    atoms.calc = LennardJones(sigma=1.5 * sigma, epsilon=1.0, rc=None, smooth=True)
    optimizer = FIRE(atoms)
    optimizer.run(fmax=0.1)

    write(osp.join(out_dir, "1-lj-relaxed.xyz"), atoms)

    # 2. Density equilibration with MACE
    dt = 5 * units.fs
    steps = 200  # 1 ps for demo

    temp = args.temperature  # K
    pressure = args.pressure  # eV/A^3

    # thermostat
    ttime = 25 * units.fs

    # barostat
    B = 23 * units.GPa  # bulk modulus of solid NaCl (MP)
    ptime = 75 * units.fs  # 75 fs suggested by ase
    pfactor = ptime**2 * B

    out_stem = str(osp.join(out_dir, f"T_{temp}-P_{pressure}-seed_{args.seed}"))
    os.makedirs(out_stem, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    calculator = MACECalculator(
        model_paths=args.calculator,
        device=device,
    )

    density_calc = DensityCalc(
        calculator=calculator,
        optimizer=FIRE,
        steps=steps,
        mask=np.eye(3),
        rtol=1e-3,
        atol=5e-4,
        out_stem=out_stem,
    )

    results = density_calc.calc(
        atoms=atoms,
        temperature=temp,
        externalstress=pressure,
        timestep=dt,
        ttime=ttime,
        pfactor=pfactor,
        annealing=1.2,
    )

    print(results)

    write(osp.join(out_stem, "2-mace-relaxed.xyz"), atoms)

    return "success"


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description="Run binary mixture density equilibration with MACE."
    )
    argparser.add_argument("recipe", type=json.loads, help="JSON dict of formula:count")
    argparser.add_argument("temperature", type=float, help="Temperature in K")
    argparser.add_argument("pressure", type=float, help="Pressure in eV/A^3")
    argparser.add_argument(
        "--calculator", type=str, required=True, help="Path to MACE model"
    )
    argparser.add_argument("--tolerance", type=float, default=2.0)
    argparser.add_argument("--rattle", type=float, default=0.1)
    argparser.add_argument("--scale", type=float, default=1.05)
    argparser.add_argument("--no-shuffle", dest="shuffle", action="store_false")
    argparser.add_argument("--seed", type=int, default=1)
    argparser.add_argument("--log", action="store_true")
    argparser.add_argument("--root", type=Path, default=Path.cwd())

    args = argparser.parse_args()

    start_time = time.time()
    print(main(args))
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total time: {timedelta(seconds=elapsed_time)} s")
