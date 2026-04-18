# Muse

**Muse** (**M**ixture b**u**ilder for **s**imulation **e**nvironments) is a Python package for rapidly building amorphous solids and liquid mixtures from relaxed solid-state structures on [Materials Project](https://materialsproject.org/).

It uses [Packmol](http://m3g.iqm.unicamp.br/packmol/home.shtml) for packing molecules into simulation cells and supports density equilibration through molecular dynamics with machine learning interatomic potentials (MLIPs), especially universal interatomic potentials (UIPs) such as [MACE](https://github.com/ACEsuit/mace) and [CHGNet](https://github.com/CederGroupHub/chgnet).

## Features

- **Structure generation** — Build binary/multicomponent amorphous mixtures from Materials Project crystal structures via `mix_number` and `mix_cell`
- **Density equilibration** — Run NVT → NPT molecular dynamics workflows to compute equilibrium densities with `DensityCalc`
- **Thermodynamic analysis** — Plot binary mixing enthalpy (G–x), density–composition, and excess volume diagrams with Redlich–Kister fits
- **Trajectory I/O** — Convert pymatgen trajectories to extended XYZ format
- **HPC integration** — Submit SLURM batch jobs programmatically

## Installation

```bash
pip install muse-xtal
```

Or with [uv](https://github.com/astral-sh/uv):

```bash
uv pip install muse-xtal
```

### Prerequisites

Muse requires [Packmol](http://m3g.iqm.unicamp.br/packmol/home.shtml) to be installed and available on your `PATH`.

You also need a [Materials Project API key](https://materialsproject.org/api) set as the `MP_API_KEY` environment variable (or in a `.env` file).

```{tableofcontents}
```
