"""Calculator for density-related properties via NPT molecular dynamics."""

from __future__ import annotations

import contextlib
import io
import logging
from inspect import isclass
from typing import TYPE_CHECKING

import numpy as np
from ase import optimize, units
from ase.geometry.cell import Cell
from ase.io.trajectory import Trajectory
from ase.md.npt import NPT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.optimize.optimize import Optimizer
from matcalc import PropCalc

from muse.calcs.utils import TrajectoryObserver

if TYPE_CHECKING:
    from pathlib import Path

    from ase import Atoms
    from ase.calculators.calculator import Calculator

__author__ = "Yuan Chiang"
__date__ = "2023-08-02"

logger = logging.getLogger(__name__)


class DensityCalc(PropCalc):
    """Relax and run NPT simulations to compute the equilibrium density.

    This calculator performs a three-stage molecular dynamics workflow:

    1. **0 K relaxation** — Minimize forces with the chosen optimizer.
    2. **NVT equilibration** — Thermalize at the target temperature with
       fixed volume until energy converges.
    3. **NPT production** — Allow both temperature and pressure to
       equilibrate, then compute density from the final volume.

    Args:
        calculator: ASE calculator to use for energy/force evaluation.
        optimizer: ASE optimizer class or name string. Defaults to ``"FIRE"``.
        steps: Number of MD steps per convergence window. Defaults to 500.
        interval: Trajectory save interval in steps. Defaults to 1.
        fmax: Maximum force for structural relaxation (eV/Å). Defaults to 0.1.
        mask: 3×3 mask array controlling which cell degrees of freedom
            are relaxed in the NPT barostat. Defaults to None (all free).
        rtol: Relative tolerance for energy convergence between windows.
            Defaults to 1e-4.
        atol: Absolute tolerance for stress convergence (eV/ų).
            Defaults to 1e-4.
        out_stem: Path stem for saving trajectory and observer files.
            Defaults to ``"."``.
    """

    def __init__(
        self,
        calculator: Calculator,
        optimizer: Optimizer | str = "FIRE",
        steps: int = 500,
        interval: int = 1,
        fmax: float = 0.1,
        mask: list | np.ndarray | None = None,
        rtol: float = 1e-4,
        atol: float = 1e-4,
        out_stem: str | Path = ".",
    ):
        self.calculator = calculator

        # check str is valid optimizer key
        def is_ase_optimizer(key):
            return isclass(obj := getattr(optimize, key)) and issubclass(obj, Optimizer)

        valid_keys = [key for key in dir(optimize) if is_ase_optimizer(key)]
        if isinstance(optimizer, str) and optimizer not in valid_keys:
            raise ValueError(f"Unknown {optimizer=}, must be one of {valid_keys}")

        self.optimizer: Optimizer = (
            getattr(optimize, optimizer) if isinstance(optimizer, str) else optimizer
        )
        self.steps = steps
        self.interval = interval
        self.fmax = fmax
        self.mask = mask
        self.rtol = rtol
        self.atol = atol
        self.out_stem = out_stem

    def calc(
        self,
        atoms: Atoms,
        temperature: float,
        externalstress: float | np.ndarray,
        timestep: float = 2.0 * units.fs,
        ttime: float = 25.0 * units.fs,
        pfactor: float = (75 * units.fs) ** 2 * units.GPa,
        annealing: float = 1.0,
        momentum: float = 0.9,
    ) -> dict:
        """Relax the structure and run NPT simulations to compute the density.

        Args:
            atoms: Structure to relax and equilibrate.
            temperature: Temperature of the simulation in Kelvin.
            externalstress: External pressure in eV/ų (scalar for isotropic,
                or Voigt 6-vector).
            timestep: MD timestep in ASE internal units. Defaults to 2.0 fs.
            ttime: Thermostat characteristic timescale in ASE internal units.
                Defaults to 25.0 fs.
            pfactor: Barostat constant in ASE internal units.
                Defaults to (75 fs)² × 1 GPa.
            annealing: Temperature scaling factor for NVT initialization.
                Values > 1 start hotter to aid equilibration. Defaults to 1.0.
            momentum: Exponential moving average factor for energy convergence
                tracking between NVT/NPT windows. Defaults to 0.9.

        Returns:
            Dictionary with keys:
                - ``volume_avg``: Mean cell volume (ų).
                - ``volume_std``: Standard deviation of cell volume.
                - ``atomic_density``: Number density (atoms/ų).
                - ``mass_density``: Mass density (amu/ų).
                - ``energy_avg``: Mean potential energy (eV).
                - ``energy_std``: Standard deviation of potential energy.
        """
        # relax the structure
        atoms.calc = self.calculator

        stream = io.StringIO()

        # step 0: relax at 0 K
        with contextlib.redirect_stdout(stream):
            optimizer = self.optimizer(atoms)

            if self.out_stem is not None:
                traj = Trajectory(f"{self.out_stem}-relax.traj", "w", atoms)
                optimizer.attach(traj.write, interval=self.interval)

            obs = TrajectoryObserver(atoms)
            optimizer.attach(obs, interval=self.interval)
            optimizer.run(fmax=self.fmax, steps=self.steps)

        if self.out_stem is not None:
            traj.close()
            obs()
            obs.save(f"{self.out_stem}-relax.pkl")
            del obs

        # step 1: run NVT simulation
        MaxwellBoltzmannDistribution(atoms, temperature_K=temperature * annealing)
        Stationary(atoms, preserve_temperature=True)

        nvt = NPT(
            atoms,
            timestep=timestep,
            temperature_K=temperature * annealing,
            externalstress=externalstress,
            ttime=ttime,
            pfactor=None,
        )
        nvt.set_fraction_traceless(0.0)

        converged, erg_converged, str_converged = False, False, False
        restart = 0
        last_erg_avg, first_erg_avg = None, None
        while not converged:
            if self.out_stem is not None:
                traj = Trajectory(f"{self.out_stem}-nvt-{restart}.traj", "w", atoms)
                nvt.attach(traj.write, interval=self.interval)

            obs = TrajectoryObserver(atoms)
            nvt.attach(obs, interval=self.interval)
            nvt.run(steps=self.steps)

            erg_avg, erg_std = np.mean(obs.energies), np.std(obs.energies)

            if last_erg_avg is None or first_erg_avg is None:
                last_erg_avg = erg_avg
                first_erg_avg = erg_avg
                erg_converged = False
            else:
                erg_converged = (
                    abs(erg_avg - last_erg_avg) / last_erg_avg < self.rtol
                    and np.sign(erg_avg - first_erg_avg) * (erg_avg - last_erg_avg) < 0
                )

            stress = np.mean(np.stack(obs.stresses, axis=0), axis=0)
            str_converged = np.allclose(
                nvt.externalstress, stress, atol=self.atol, rtol=self.rtol
            )

            converged = erg_converged  # and str_converged

            if self.out_stem is not None:
                traj.close()
                obs()
                obs.save(f"{self.out_stem}-nvt-{restart}.pkl")
                del obs

            if not converged:
                logger.info(
                    "NVT - %d: Energy or stress not converged, restarting simulation.",
                    restart,
                )
                if not erg_converged:
                    logger.info(
                        "Current relative energy deviation: %.4f %%. Target: %.4f %%.",
                        (erg_avg - last_erg_avg) / last_erg_avg * 100,
                        self.rtol * 100,
                    )
                else:
                    logger.info("Energy converged.")
                if not str_converged:
                    logger.info("Current pressure: %s eV/ų.", stress)
                    logger.info("Target pressure: %s eV/ų.", nvt.externalstress)
                else:
                    logger.info("Pressure converged.")

                nvt.observers.clear()
                last_erg_avg = momentum * erg_avg + (1 - momentum) * last_erg_avg
                restart += 1

        # step 2: run NPT simulation
        npt = NPT(
            atoms,
            timestep=timestep,
            temperature_K=temperature,
            externalstress=externalstress,
            ttime=ttime,
            pfactor=pfactor,
            mask=self.mask,
        )
        if np.array_equal(self.mask, np.eye(3)):
            npt.set_fraction_traceless(0.0)  # fix shape

        converged, erg_converged, str_converged = False, False, False
        restart = 0
        last_erg_avg, first_erg_avg = None, None
        while not converged:
            if self.out_stem is not None:
                self.final_traj_fpath = f"{self.out_stem}-npt-{restart}.traj"
                traj = Trajectory(self.final_traj_fpath, "w", atoms)
                npt.attach(traj.write, interval=self.interval)

            obs = TrajectoryObserver(atoms)
            npt.attach(obs, interval=self.interval)
            npt.run(steps=self.steps)

            erg_avg, erg_std = np.mean(obs.energies), np.std(obs.energies)

            if last_erg_avg is None or first_erg_avg is None:
                last_erg_avg = erg_avg
                first_erg_avg = erg_avg
                erg_converged = False
            else:
                erg_converged = (
                    abs(erg_avg - last_erg_avg) / last_erg_avg < self.rtol
                    and np.sign(erg_avg - first_erg_avg) * (erg_avg - last_erg_avg) < 0
                )

            stress = np.mean(np.stack(obs.stresses, axis=0), axis=0)
            str_converged = np.allclose(
                npt.externalstress, stress, atol=self.atol, rtol=self.rtol
            )

            converged = erg_converged and str_converged

            if self.out_stem is not None:
                traj.close()
                obs()
                obs.save(f"{self.out_stem}-npt-{restart}.pkl")

            if not converged:
                logger.info(
                    "NPT - %d: Energy or stress not converged, restarting simulation.",
                    restart,
                )
                if not erg_converged:
                    logger.info(
                        "Current relative energy deviation: %.4f %%. Target: %.4f %%.",
                        (erg_avg - last_erg_avg) / last_erg_avg * 100,
                        self.rtol * 100,
                    )
                else:
                    logger.info("Energy converged.")
                if not str_converged:
                    logger.info("Current pressure: %s eV/ų.", stress)
                    logger.info("Target pressure: %s eV/ų.", npt.externalstress)
                else:
                    logger.info("Pressure converged.")

                npt.observers.clear()
                last_erg_avg = momentum * erg_avg + (1 - momentum) * last_erg_avg
                restart += 1

        volumes = [Cell(matrix).volume for matrix in obs.cells]
        vol_avg, vol_std = np.mean(volumes), np.std(volumes)
        erg_avg, erg_std = np.mean(obs.energies), np.std(obs.energies)

        return {
            "volume_avg": vol_avg,
            "volume_std": vol_std,
            "atomic_density": atoms.get_global_number_of_atoms() / vol_avg,
            "mass_density": atoms.get_masses().sum() / vol_avg,
            "energy_avg": erg_avg,
            "energy_std": erg_std,
        }
