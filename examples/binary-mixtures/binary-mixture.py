"""Relax an arbitrary binary mixture"""

import argparse
import json
import os
import os.path as osp
import shutil
import sys
import tempfile
import time
from math import sqrt
from pathlib import Path

import numpy as np
import psutil
from ase import units
from ase.calculators.lj import LennardJones
from ase.calculators.socketio import SocketIOCalculator
from ase.io import write
from ase.optimize import BFGS, FIRE
from pymatgen.core.periodic_table import Element
from vasp_interactive import VaspInteractive

from muse.calcs.density import DensityCalc
from muse.transforms.mixture import mix

ncpus = psutil.cpu_count(logical=False)

os.environ["OMP_NUM_THREADS"] = "1"
os.environ[
    "ASE_VASP_COMMAND"
] = "ulimit -s unlimited; module load intelmpi/20.4-intel20.4; mpirun -np 32 /jet/home/ychiang4/.local/bin/vasp_std"


def main(args):
    parent_dir = osp.join(os.getcwd(), "-".join(args.recipe.keys()))
    os.makedirs(parent_dir, exist_ok=True)

    # 1. Generate initial configuration for the mixture

    atoms = mix(
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

    njobs = max(min(ncpus, len(atoms)), 8)

    # os.environ["ASE_VASP_COMMAND"] = f"ulimit -s unlimited; mpirun -np {njobs} /jet/home/ychiang4/.local/bin/vasp_std"

    # 2. Relax the mixture using Lennard-Jones potential

    atoms.calc = LennardJones(
        sigma=1.5 * args.tolerance, epsilon=1.0, rc=None, smooth=True
    )
    optimizer = FIRE(atoms)
    optimizer.run(fmax=0.1)

    write(osp.join(out_dir, "1-lj-relaxed.xyz"), atoms)

    # 3. Relax the mixture using density workflow

    dt = 5 * units.fs
    steps = 200  # 1 ps for demo

    temp = args.temperature  # K
    # pressure = 6.32420912e-7  # eV/A^3 of 1 atm
    pressure = args.pressure  # eV/A^3

    # thermostat
    ttime = 25 * units.fs

    # barostat
    B = 23 * units.GPa  # bulk modulus of solid NaCl (MP)
    ptime = 75 * units.fs  # 75 fs suggested by ase
    pfactor = ptime**2 * B

    params = dict(
        xc="pbe",
        # metagga="R2SCAN",
        algo="All",
        ediff=1e-4,  # MPR2SCANRelaxSet: 1e-5
        ediffg=0,
        enaug=1360,
        encut=680,
        ibrion=-1,
        isif=3,
        ismear=0,
        sigma=0.05,  # 8.615e-5*temp for consistent electronic temperature
        kpts=1,
        # kspacing=0.22,
        laechg=True,  # AECCARs
        lasph=True,  # aspherical charge density
        lcharg=True,  # CHGCAR
        lelf=False,  # ELFCAR
        lmaxmix=max(map(lambda a: "spdf".index(Element(a.symbol).block) * 2, atoms)),
        lmixtau=True,  # send kinetic energy through the density mixer
        lorbit=11,  # lm-decomposed DOSCAR
        lreal="Auto",
        lvtot=True,  # LOCPOT
        lwave=True,  # WAVECAR
        istart=1,
        nelm=200,
        nsw=int(steps * 2),
        prec="Accurate",
        # Massively parallel machines (Cray)
        lplane=False,
        npar=int(sqrt(ncpus)),
        nsim=1
        # Multicore modern linux machines
        # lplane=True,
        # npar=2,
        # lscalu=False,
        # nsim=4
    )

    out_stem = str(osp.join(out_dir, f"T_{temp}-P_{pressure}-seed_{args.seed}"))
    os.makedirs(out_stem, exist_ok=True)

    # with tempfile.TemporaryDirectory(dir=out_dir) as tmpdir:

    cloned = atoms.copy()
    calc = VaspInteractive(**params, directory=out_stem)

    with calc:
        cloned.calc = calc
        optimizer = BFGS(cloned)
        optimizer.run(fmax=0.1, steps=3)

    # shutil.copy(Path(tmpdir) / "*", out_stem)

    # density_calc = DensityCalc(
    #     calculator=calc,
    #     optimizer=BFGS,
    #     steps=steps,
    #     mask=np.eye(3),
    #     rtol=1e-2,
    #     atol=5e-4,
    #     out_stem=str(osp.join(out_dir, f'T_{temp}-P_{pressure}')),
    # )

    # density_calc.calc(
    #     atoms=atoms,
    #     temperature=temp,
    #     externalstress=pressure, # equivalent to (-p, -p, -p, 0, 0, 0)
    #     timestep=dt,
    #     ttime=ttime,
    #     pfactor=pfactor,
    #     annealing=1.0,
    #     calc=calc
    # )

    write(osp.join(out_dir, f"2-{calc.name}-relaxed.xyz"), atoms)

    return "success"


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("recipe", type=json.loads)
    argparser.add_argument("temperature", type=float)
    argparser.add_argument("pressure", type=float)
    argparser.add_argument("--tolerance", type=float, default=2.0)
    argparser.add_argument("--rattle", type=float, default=0.1)
    argparser.add_argument("--scale", type=float, default=1.05)
    argparser.add_argument("--no-shuffle", dest="shuffle", action="store_false")
    argparser.add_argument("--seed", type=int, default=1)
    argparser.add_argument("--log", action="store_true")

    args = argparser.parse_args()

    start_time = time.time()
    print(main(args))
    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f} s")
