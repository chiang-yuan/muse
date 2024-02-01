from collections.abc import Sequence
from typing import Any

import numpy as np
from ase import Atoms, units
from ase.build import sort
from ase.formula import Formula
from matplotlib.axes._axes import Axes
from matplotlib.axes._base import _AxesBase
from matplotlib.figure import Figure

__author__ = "Yuan Chiang"
__date__ = "2023-12-11"

eps = 1e-10


class MixingVolumeDiagram(Axes):
    """Binary mixing volume diagram plotter."""

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
        # volavgs =
        self.y["volume.deviation"] = self.y["volume.avg"] - (
            self.y["volume.avg"][0]
            + self.x * (self.y["volume.avg"][-1] - self.y["volume.avg"][0])
        )

    def from_trajectories(
        self,
        trajectories: Sequence[Sequence[Atoms]],
        phases: Sequence[str | Formula],
        label: str | None = None,
        **kwargs,
    ) -> None:
        """Plot a binary phase diagram from a list of trajectories."""
        assert len(phases) == 2

        self.process(trajectories, phases)

        self.errorbar(
            self.x,
            self.y["volume.deviation"],
            yerr=self.y["volume.std"],
            label="$\\Delta \\overline{V}$ " + label
            if label
            else "$\\Delta \\overline{V}$",
            **kwargs,
        )
