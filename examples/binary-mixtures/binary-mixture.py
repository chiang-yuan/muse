"""Relax an arbitrary binary mixture"""

import argparse
import json
import os
import os.path as osp
import tempfile
import time
from datetime import timedelta
from pathlib import Path
from typing import Union

import numpy as np
import psutil
import torch
from ase import units
from ase.calculators.lj import LennardJones
from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.md.npt import NPT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.optimize import BFGS, FIRE
from ase.optimize.optimize import Optimizer
from mace.calculator import MACECalculator
from matcalc.base import PropCalc
from pymatgen.core.periodic_table import Element
from vasp_interactive import VaspInteractive

from muse.calcs.density import DensityCalc
from muse.transforms.mixture import mix

os.environ["OMP_NUM_THREADS"] = "1"

def main(args):
    parent_dir = osp.join(args.root, "-".join(args.recipe.keys()))
    os.makedirs(parent_dir, exist_ok=True)

    # 0. Generate initial configuration for the mixture

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

    ncpus = psutil.cpu_count(logical=False)
    njobs = min(ncpus, len(atoms))

    # os.environ["ASE_VASP_COMMAND"] = f"ulimit -s unlimited; mpirun -np {njobs} /jet/home/ychiang4/.local/bin/vasp_std"
    os.environ[
        "ASE_VASP_COMMAND"
    ] = f"ulimit -s unlimited; module load intelmpi/20.4-intel20.4; mpirun -np {njobs} /jet/home/ychiang4/.local/bin/vasp_std"

    # 1. Relax the mixture using Lennard-Jones potential

    atoms.calc = LennardJones(
        sigma=1.5 * args.tolerance, epsilon=1.0, rc=None, smooth=True
    )
    optimizer = FIRE(atoms)
    optimizer.run(fmax=0.1)

    write(osp.join(out_dir, "1-lj-relaxed.xyz"), atoms)

    # 2. Relax the mixture using density workflow

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

    out_stem = str(osp.join(out_dir, f"T_{temp}-P_{pressure}-seed_{args.seed}"))
    os.makedirs(out_stem, exist_ok=True)

    if "vasp" in args.calculator.lower():
        raise NotImplementedError("Running density pipeline using VASP is not supported yet.")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        calculator = MACECalculator(
            # model_paths='/ocean/projects/dmr110014p/ychiang4/2023-08-14-mace-universal.model', 
            model_paths=args.calculator,
            device=device
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
        externalstress=pressure, # equivalent to (-p, -p, -p, 0, 0, 0)
        timestep=dt,
        ttime=ttime,
        pfactor=pfactor,
        annealing=1.2
    )

    print(results)

    write(osp.join(out_stem, f"2-{calculator.name}-relaxed.xyz"), atoms)

    # 3. Run several single-point calculations for the relaxation trajectory

    params = dict(
        xc="pbe",
        gga=None,
        metagga="R2SCAN",
        algo="All",
        ediff=1e-4,  # MPR2SCANRelaxSet: 1e-5
        ediffg=0,
        enaug=1360,
        encut=680,
        ibrion=-1,
        isif=3,
        ismear=0,
        sigma=0.05,  # 8.615e-5*temp for consistent electronic temperature
        # kpts=1,
        kspacing=0.22,
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
        # lplane=False,
        # npar=int(sqrt(ncpus)),
        # nsim=1
        # Multicore modern linux machines
        lplane=True,
        npar=2,
        lscalu=False,
        nsim=4
    )

    traj_file = osp.join(out_stem, density_calc.final_traj_fpath)

    traj = read(traj_file, index=":")

    for i, cloned in enumerate(traj):

        if i % args.nsamples != 0:
            continue

        run_dir = os.makedirs(osp.join(out_stem, "3-vasp", f"{i:03d}"), exist_ok=True)

        # with tempfile.TemporaryDirectory(dir=run_dir) as tmpdir:
        calc = VaspInteractive(**params, directory=run_dir)

        with calc:
            cloned.calc = calc
            # cloned.get_potential_energy()
            print(f"VASP calculator frame {i}: {cloned.get_potential_energy()}")

            cloned = calc.get_atoms()

            write(osp.join(out_stem, f"3-{calc.name}-vasp-{i:03d}.xyz"), cloned)
            
            # npt = NPT(
            #     cloned,
            #     timestep=dt,
            #     temperature_K=temp,
            #     externalstress=pressure,
            #     ttime=ttime,
            #     pfactor=pfactor,
            #     mask=np.eye(3),
            # )
            # npt.set_fraction_traceless(0.0)  # fix shape

            # traj_fpath = osp.join(out_stem, f"3-{calc.name}-vasp.traj")
            # traj = Trajectory(traj_fpath, "w", atoms)
            # npt.attach(traj.write, interval=1)
            # npt.run(steps=args.nsamples)
            # traj.close()

    # traj = read(traj_fpath, index=":", append=True)
    
    # for atoms in traj:
    #     write(osp.join(out_dir, f"3-{calc.name}-vasp.xyz"), atoms, append=True)

    # shutil.copy(Path(tmpdir) / "*", out_stem)

    return "success"


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("recipe", type=json.loads)
    argparser.add_argument("temperature", type=float)
    argparser.add_argument("pressure", type=float)
    argparser.add_argument("--calculator", type=str, default="vasp_std")
    argparser.add_argument("--tolerance", type=float, default=2.0)
    argparser.add_argument("--rattle", type=float, default=0.1)
    argparser.add_argument("--scale", type=float, default=1.05)
    argparser.add_argument("--no-shuffle", dest="shuffle", action="store_false")
    argparser.add_argument("--seed", type=int, default=1)
    argparser.add_argument("--log", action="store_true")
    argparser.add_argument("--root", type=Path, default=Path.cwd())
    argparser.add_argument("--nsamples", type=int, default=5)

    args = argparser.parse_args()

    start_time = time.time()
    print(main(args))
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total time: {str(timedelta(seconds=elapsed_time))} s")
