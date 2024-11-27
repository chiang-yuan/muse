from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy.optimize import curve_fit
from ase import Atoms, units
from ase.build import sort
from ase.formula import Formula
from matplotlib.axes._axes import Axes
from matplotlib.figure import Figure

import numpy as np

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


class BinaryGXDiagram(Axes):
    """Binary G-x diagram plotter."""

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

        # Change strings to Formula objects and sort symbols
        for phase in phases:
            phase = Formula.from_list(phase) if isinstance(phase, str) else sort(phase)

        x, ergavgs, ergstds = [], [], []
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

            energies = []
            for atoms in traj:
                energies.append(atoms.get_potential_energy() / total_units)
            ergavgs.append(np.mean(energies))
            ergstds.append(np.std(energies))

        x = np.array(x)
        idx = np.argsort(x)
        x = x[idx]

        ergavgs = np.array(ergavgs)[idx]
        ergstds = np.array(ergstds)[idx]

        dH = ergavgs - (ergavgs[0] + x * (ergavgs[-1] - ergavgs[0]))
        dS = -units.kB * (x * np.log(x + eps) + (1 - x) * np.log(1 - x + eps))
        
        color = kwargs.pop("color", 'k')

        self.errorbar(
            x,
            dH,
            yerr=ergstds,
            label=f"{label}: $\\Delta H$" if label else "$\\Delta H$",
            color=color,
            fmt='o',
            **kwargs,
        )

        # Fitting the Redlich-Kister expansion model to Delta H (dH)
        initial_guess = [0.0] * (2 * rk)  # Initial guess for [A1, B1, A2, B2, ..., AN, BN]
        params_opt, params_cov = curve_fit(
            lambda x_T, *params: redlich_kister_model(x_T[0], x_T[1], *params),
            (x, np.ones_like(x)*(temperature or 1000)), dH, p0=initial_guess
        )

        # Extract fitted parameters for Redlich-Kister expansion
        fitted_params = params_opt.reshape(-1, 2)
        print("Fitted Redlich-Kister parameters (A_n, B_n):", fitted_params)

        # Calculate fitted curve for Delta H using the fitted parameters
        xs = np.linspace(x.min(), x.max(), int(1e3))
        dH_fitted = redlich_kister_model(xs, temperature, *params_opt)

        # Plotting the fitted Redlich-Kister curve
        self.plot(
            xs,
            dH_fitted,
            label=f"{label}: Redlich-Kister Fit" if label else "Redlich-Kister Fit",
            linestyle="--",
            lw=kwargs.get("lw", 1),
            color=color
        )