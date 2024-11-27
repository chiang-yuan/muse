from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy.optimize import curve_fit
from ase import Atoms, units
from ase.build import sort
from ase.formula import Formula
from matplotlib.axes._axes import Axes
from matplotlib.axes._base import _AxesBase
from matplotlib.figure import Figure

__author__ = "Yuan Chiang"
__date__ = "2023-11-06"

eps = 1e-10

def redlich_kister_model(x, T, *params):
    """
    Redlich-Kister expansion for excess property rho_ex.
    
    x  : mole fraction of one component
    T  : temperature (assumed constant for each data point)
    *params : array of A_n and B_n parameters for the Redlich-Kister expansion
    """
    N = len(params) // 2  # Number of terms N
    rho_ex = 0
    for n in range(1, N + 1):
        A_n = params[2 * (n - 1)]
        B_n = params[2 * (n - 1) + 1]
        L_n = A_n + B_n * T  # Linear temperature-dependent term L_n
        rho_ex += L_n * (2 * x - 1) ** (n - 1)  # Redlich-Kister term
    return x * (1 - x) * rho_ex

class BinaryDXDiagram(Axes):
    """Binary density-composition diagram plotter."""

    def __init__(
        self,
        fig: Figure,
        *args: Any,
        facecolor=None,
        frameon: bool = True,
        sharex: Axes | None = None,
        sharey: Axes | None = None,
        label: str = "",
        xscale: float | None = None,
        yscale: float | None = None,
        box_aspect: float | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            fig,
            args,
            facecolor=facecolor,
            frameon=frameon,
            sharex=sharex,
            sharey=sharey,
            label=label,
            xscale=xscale,
            yscale=yscale,
            box_aspect=box_aspect,
            **kwargs,
        )

    def process(
        self,
        trajectories: Sequence[Sequence[Atoms]],
        phases: Sequence[str | Formula],
    ):
        # Change strings to Formula objects and sort symbols
        for phase in phases:
            phase = Formula.from_list(phase) if isinstance(phase, str) else sort(phase)

        x, denavgs, denstds, volavgs, volstds = [], [], [], [], []
        for traj in trajectories:
            atoms = traj[0]
            formula = atoms.symbols.formula

            portions = {}
            for phase in phases:
                portions[phase] = formula // phase

            total_units = sum(portions.values())

            fractions = {}
            for phase in phases:
                fractions[phase] = portions[phase] / total_units

            x.append(fractions[phases[-1]])

            densities = []
            volumes = []
            for atoms in traj:
                densities.append(
                    atoms.get_masses().sum()
                    * 1.66054e-24
                    / (atoms.get_volume() * 1e-24)
                )
                volumes.append(atoms.get_volume() / total_units)

            denavgs.append(np.mean(densities))
            denstds.append(np.std(densities))
            volavgs.append(np.mean(volumes))
            volstds.append(np.std(volumes))

        x = np.array(x)
        idx = np.argsort(x)
        self.x = x[idx]

        self.y = {} if getattr(self, "y", None) is None else self.y
        self.y["density.avg"] = np.array(denavgs)[idx]
        self.y["density.std"] = np.array(denstds)[idx]
        self.y["volume.avg"] = np.array(volavgs)[idx]
        self.y["volume.std"] = np.array(volstds)[idx]

    def from_trajectories(
        self,
        trajectories: Sequence[Sequence[Atoms]],
        phases: Sequence[str | Formula],
        temperature: float | None = None,
        label: str | None = None,
        rk: int = 2,
        **kwargs,
    ) -> None:
        """Plot a binary phase diagram from a list of trajectories."""
        assert len(phases) == 2

        self.process(trajectories, phases)

        color = kwargs.pop("color", 'k')

        self.errorbar(
            self.x,
            self.y["density.avg"],
            yerr=self.y["density.std"],
            label=f"{label}: $\\rho_m$" if label else "$\\rho_m$",
            color=color,
            fmt='o',
            **kwargs,
        )
        
        T = temperature or 1000

        # Fitting the Redlich-Kister expansion model to Delta H (dH)
        initial_guess = [0.0] * (2 * rk)  # Initial guess for [A1, B1, A2, B2, ..., AN, BN]
        
        y = self.y["density.avg"]
        
        params_opt, params_cov = curve_fit(
            lambda x_T, *params: redlich_kister_model(x_T[0], x_T[1], *params),
            (self.x, np.ones_like(self.x)*T), y - (y[0] + self.x * (y[-1] - y[0])), p0=initial_guess
        )

        # Extract fitted parameters for Redlich-Kister expansion
        fitted_params = params_opt.reshape(-1, 2)
        print("Fitted Redlich-Kister parameters (A_n, B_n):", fitted_params)

        # Calculate fitted curve for Delta H using the fitted parameters
        xs = np.linspace(self.x.min(), self.x.max(), int(1e3))
        ys = redlich_kister_model(xs, T, *params_opt)
        
        ys += (y[0] + xs * (y[-1] - y[0]))

        # Plotting the fitted Redlich-Kister curve
        self.plot(
            xs,
            ys,
            label=f"{label}: Redlich-Kister Fit" if label else "Redlich-Kister Fit",
            linestyle="--",
            lw=kwargs.get("lw", 1),
            color=color
        )


    def plot_volume(
        self,
        label: str | None = None,
        **kwargs,
    ):
        self.vol_ax = self.twinx()
        self.vol_ax.errorbar(
            self.x,
            self.y["volume.avg"],
            yerr=self.y["volume.std"],
            label="$\\bar{V}$" if label else "$\\bar{V}$",
            **kwargs,
        )
        return self.vol_ax