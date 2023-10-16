"""Relax an arbitrary binary mixture"""

import argparse
import json
import os
import os.path as osp

import numpy as np
from ase import units
from ase.calculators.lj import LennardJones
from ase.calculators.socketio import SocketIOCalculator
from ase.io import write
from ase.optimize import BFGS, FIRE
from pymatgen.core.periodic_table import Element
from vasp_interactive import VaspInteractive

from muse.calcs.density import DensityCalc
from muse.transforms.mixture import mix

os.environ["ASE_VASP_COMMAND"] = "mpirun -np 64 /jet/home/ychiang4/.local/bin/vasp_gam"

recipe = {"NaCl": 3, "CsCl": 1}

parent_dir = osp.join(os.getcwd(), '-'.join(recipe.keys()))
os.makedirs(parent_dir, exist_ok=True)

atoms = mix(
    recipe=recipe,
    tolerance=2.0,
    rattle=0.1,
    scale=1.05,
    shuffle=True,
    seed=1,
    log=True,
)
print(atoms)

out_dir = osp.join(parent_dir, str(atoms.symbols))
os.makedirs(out_dir, exist_ok=True)

write(osp.join(out_dir, "0-packmol-mixture.xyz"), atoms)

atoms.calc = LennardJones(sigma=1.5 * 2.0, epsilon=1.0, rc=None, smooth=True)
optimizer = FIRE(atoms)
optimizer.run(fmax=0.1)

write(osp.join(out_dir, "1-lj-relaxed.xyz"), atoms)

dt = 5 * units.fs
steps = 2000

temp = 1100 # K
# pressure = 6.32420912e-7 # eV/A^3 of 1 atm
pressure = 0 # eV/A^3

# thermostat
ttime = 25 * units.fs

# barostat
B = 23 * units.GPa  # bulk modulus of solid NaCl (MP)
ptime = 75 * units.fs # 75 fs suggested by ase
ptime**2 * B
pfactor =  ptime**2 * B

params = dict(
    xc="pbe",
    # metagga="R2SCAN",
    algo="VeryFast",
    ediff=1e-4,
    ediffg=0,
    enaug=1360,
    encut=680,
    ibrion=-1,
    isif=3,
    ismear=-1,
    sigma=8.615e-5*temp,
    kpts=1,
    # kspacing=0.22,
    laechg=True, # AECCARs
    lasph=True,  # aspherical charge density
    lcharg=True, # CHGCAR
    lelf=False,  # ELFCAR
    lmaxmix=max(map(lambda a: 'spdf'.index(Element(a.symbol).block)*2, atoms)),
    lmixtau=True,# send kinetic energy through the density mixer 
    lorbit=11,   # lm-decomposed DOSCAR 
    lreal="Auto",
    lvtot=True,  # LOCPOT
    lwave=True, # WAVECAR
    istart=1,
    nelm=200,
    nsw=2000,
    prec="Accurate", 
)

vpi = VaspInteractive(
    **params,
    directory=out_dir
)


# with VaspInteractive(
#     **params,
#     directory=out_dir
# ) as calc:
with SocketIOCalculator(vpi) as calc:
    str(osp.join(out_dir, f'T_{temp}-P_{pressure}'))
    atoms.calc = calc

    density_calc = DensityCalc(
        calculator=calc,
        optimizer=BFGS,
        steps=steps,
        mask=np.eye(3),
        rtol=1e-3,
        atol=5e-4,
        out_stem=str(osp.join(out_dir, f'T_{temp}-P_{pressure}')),
    )

    density_calc.calc(
        atoms=atoms, 
        temperature=temp, 
        externalstress=pressure, # equivalent to (-p, -p, -p, 0, 0, 0)
        timestep=dt,
        ttime=ttime,
        pfactor=pfactor,
        annealing=1.2
    )

    write(osp.join(out_dir, f"2-{calc.name}-relaxed.xyz"), atoms)
