"""Calculator for density related properties."""
from __future__ import annotations

import contextlib
import io
import re
from collections.abc import Callable
from inspect import isclass
from typing import TYPE_CHECKING

import numpy as np
from ase import optimize, units
from ase.geometry.cell import Cell
from ase.io.trajectory import Trajectory
from ase.md.npt import NPT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.optimize.optimize import Optimizer
from matcalc.base import PropCalc

from muse.calcs.utils import TrajectoryObserver

if TYPE_CHECKING:
    from pathlib import Path

    from ase import Atoms
    from ase.calculators.calculator import Calculator

__author__ = "Yuan Chiang"
__date__ = "2023-08-02"


class DensityCalc(PropCalc):
    """Relaxes and run NPT simulations to compute the density of structures."""

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
        """Initialize the Density Calculator.

        Args:
            calculator (Calculator): Calculator to use.
            optimizer (Optimizer | str): Optimizer to use. Defaults to "FIRE".
            steps (int, optional): Number of steps to run the relaxation. Defaults to 500.
            interval (int, optional): Interval to save the trajectory. Defaults to 1.
            fmax (float, optional): Maximum force to stop the relaxation. Defaults to 0.1.
            mask (list | np.ndarray | None, optional): Mask allowing cell parameter relaxation. Defaults to None.
            rtol (float, optional): Relative tolerance for the NPT simulation, in the unit of eV/A^3. Defaults to 1e-5.
            atol (float, optional): Absolute tolerance for the NPT simulation. Defaults to 1e-5.
            out_stem (str | None, optional): Filename to save the trajectory. Defaults to None.
        """
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
            atoms (Atoms): Structure to relax.
            temperature (float): Temperature of the simulation in Kelvin.
            externalstress: External pressure of the simulation in eV/A^3.
            timestep (float, optional): Timestep of the simulation in ASE internal units. Defaults to 2.0 fs.
            ttime (float | None, optional): Characteristic timescale of thermostat in ASE internal units.
                                            Defaults to 25.0 fs.
            pfactor (float | None, optional): Constant factor in barastat differential equation in ASE interel units.
                                              Defaults to (75 fs)^2 * 1 GPa.
            annealing (float, optional): Temperature factor for the nvt velocities. Defaults to 1.0.

        Returns:
            Atoms: Relaxed structure.
        """
        # relax the structure

        atoms.calc = self.calculator

        stream = io.StringIO()

        # step 0: relax at 0 K

        with contextlib.redirect_stdout(stream):
            # assert isinstance(self.optimizer, Callable)
            optimizer = self.optimizer(atoms)
            # assert isinstance(self.optimizer, Optimizer)

            # if self.mask is not None:
            #     ecf = ExpCellFilter(atoms, mask=self.mask)

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

        # print("Relaxation done.")

        # if self.mask is not None:
        #     atoms = ecf.atoms

        # step 1: run nvt simulation

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
                print(
                    f"NVT - {restart}: Energy or stress not converged, restarting simulation."
                )
                print(
                    f"Current relative energy deviation: {(erg_avg - last_erg_avg)/last_erg_avg*100} %."
                    if not erg_converged
                    else "Energy converged."
                )
                print(
                    f"Target relative energy deviation: {self.rtol*100} %."
                    if not erg_converged
                    else "\r"
                )
                print(
                    f"Current pressure: {stress} eV/A^3."
                    if not str_converged
                    else "Pressure converged."
                )
                print(
                    f"Target pressure: {nvt.externalstress} eV/A^3."
                    if not str_converged
                    else "\r"
                )
                nvt.observers.clear()
                # npt.zero_center_of_mass_momentum()
                # alpha = np.exp(-restart / 10)
                last_erg_avg = momentum * erg_avg + (1 - momentum) * last_erg_avg
                restart += 1

        # step 3: run NPT simulation

        # MaxwellBoltzmannDistribution(atoms, temperature_K=temperature)
        # Stationary(atoms, preserve_temperature=True)

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
                print(
                    f"NPT - {restart}: Energy or stress not converged, restarting simulation."
                )
                print(
                    f"Current relative energy deviation: {(erg_avg - last_erg_avg)/last_erg_avg*100} %."
                    if not erg_converged
                    else "Energy converged."
                )
                print(
                    f"Target relative energy deviation: {self.rtol*100} %."
                    if not erg_converged
                    else "\r"
                )
                print(
                    f"Current pressure: {stress} eV/A^3."
                    if not str_converged
                    else "Pressure converged."
                )
                print(
                    f"Target pressure: {npt.externalstress} eV/A^3."
                    if not str_converged
                    else "\r"
                )
                npt.observers.clear()
                # npt.zero_center_of_mass_momentum()
                # alpha = np.exp(-restart / 10)
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
