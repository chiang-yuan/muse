"""Binary mixing volume diagram with Redlich–Kister curve fitting."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
from ase.build import sort
from ase.formula import Formula
from matplotlib.axes._axes import Axes
from scipy.optimize import curve_fit

from muse.plots._utils import redlich_kister_model

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ase import Atoms
    from matplotlib.figure import Figure

__author__ = "Yuan Chiang"
__date__ = "2023-12-11"

logger = logging.getLogger(__name__)


class MixingVolumeDiagram(Axes):
    """Custom Matplotlib Axes for binary excess mixing volume diagrams.

    Computes the deviation of molar volume from ideal mixing
    (Vegard's law) and fits the excess volume with a Redlich–Kister
    polynomial.
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

    def process(
        self,
        trajectories: Sequence[Sequence[Atoms]],
        phases: Sequence[str | Formula],
    ) -> None:
        """Process MD trajectories to extract density, volume, and excess volume.

        Computes mass density (g/cm³), molar volume (ų/formula unit),
        and volume deviation from ideal (Vegard's law) mixing for each
        trajectory, storing results sorted by composition.

        Args:
            trajectories: List of MD trajectories, each a sequence of Atoms.
            phases: Two-element list of phase formulas defining the binary system.
        """
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
        self.y["volume.deviation"] = self.y["volume.avg"] - (
            self.y["volume.avg"][0]
            + self.x * (self.y["volume.avg"][-1] - self.y["volume.avg"][0])
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
        """Plot a binary excess volume diagram from MD trajectories.

        Computes the volume deviation from ideal mixing, plots it with
        error bars, and overlays a Redlich–Kister polynomial fit.

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

        self.process(trajectories, phases)

        color = kwargs.pop("color", "k")

        self.errorbar(
            self.x,
            self.y["volume.deviation"],
            yerr=self.y["volume.std"],
            label="$\\Delta \\overline{V}$ " + label
            if label
            else "$\\Delta \\overline{V}$",
            color=color,
            fmt="o",
            **kwargs,
        )

        # Fitting the Redlich-Kister expansion model to excess volume
        initial_guess = [0.0] * (2 * rk)
        params_opt, params_cov = curve_fit(
            lambda x_T, *params: redlich_kister_model(x_T[0], x_T[1], *params),
            (self.x, np.ones_like(self.x) * (temperature or 1000)),
            self.y["volume.deviation"],
            p0=initial_guess,
        )

        fitted_params = params_opt.reshape(-1, 2)
        logger.info("Fitted Redlich-Kister parameters (A_n, B_n): %s", fitted_params)

        xs = np.linspace(self.x.min(), self.x.max(), int(1e3))
        dV_fitted = redlich_kister_model(xs, temperature, *params_opt)

        self.plot(
            xs,
            dV_fitted,
            label=f"{label}: Redlich-Kister Fit" if label else "Redlich-Kister Fit",
            linestyle="--",
            lw=kwargs.get("lw", 1),
            color=color,
        )
