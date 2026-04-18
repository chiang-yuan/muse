"""Binary Gibbs energy–composition (G–x) diagram with Redlich–Kister curve fitting."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
from ase import units
from ase.build import sort
from ase.formula import Formula
from matplotlib.axes._axes import Axes
from scipy.optimize import curve_fit

from muse.plots._utils import EPS, redlich_kister_model

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ase import Atoms
    from matplotlib.figure import Figure

logger = logging.getLogger(__name__)


class BinaryGXDiagram(Axes):
    """Custom Matplotlib Axes for binary Gibbs energy–composition (G–x) diagrams.

    Computes mixing enthalpy ΔH and ideal entropy of mixing ΔS from MD
    trajectories, then fits ΔH with a Redlich–Kister polynomial.
    """

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
        """Plot a binary G–x diagram from MD trajectories.

        Computes mixing enthalpy ΔH = E(x) - [E(0) + x*(E(1) - E(0))]
        and ideal entropy of mixing, then fits ΔH with a Redlich–Kister
        polynomial expansion.

        Args:
            trajectories: List of MD trajectories at different compositions.
            phases: Two-element list of phase formulas defining the binary system.
            temperature: Temperature in Kelvin for the Redlich–Kister fit.
                Defaults to 1000 K if not provided.
            label: Label prefix for the legend entries.
            rk: Number of Redlich–Kister terms. Defaults to 2.
            **kwargs: Additional keyword arguments passed to ``errorbar``.
        """
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
        _dS = -units.kB * (x * np.log(x + EPS) + (1 - x) * np.log(1 - x + EPS))  # noqa: F841

        color = kwargs.pop("color", "k")

        self.errorbar(
            x,
            dH,
            yerr=ergstds,
            label=f"{label}: $\\Delta H$" if label else "$\\Delta H$",
            color=color,
            fmt="o",
            **kwargs,
        )

        # Fitting the Redlich-Kister expansion model to Delta H
        initial_guess = [0.0] * (2 * rk)
        params_opt, params_cov = curve_fit(
            lambda x_T, *params: redlich_kister_model(x_T[0], x_T[1], *params),
            (x, np.ones_like(x) * (temperature or 1000)),
            dH,
            p0=initial_guess,
        )

        fitted_params = params_opt.reshape(-1, 2)
        logger.info("Fitted Redlich-Kister parameters (A_n, B_n): %s", fitted_params)

        xs = np.linspace(x.min(), x.max(), int(1e3))
        dH_fitted = redlich_kister_model(xs, temperature, *params_opt)

        self.plot(
            xs,
            dH_fitted,
            label=f"{label}: Redlich-Kister Fit" if label else "Redlich-Kister Fit",
            linestyle="--",
            lw=kwargs.get("lw", 1),
            color=color,
        )