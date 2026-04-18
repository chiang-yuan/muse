<p align="center">
  <img src="docs/logo.png" alt="Muse logo" width="200">
</p>

# Muse

[![PyPI version](https://img.shields.io/pypi/v/muse-xtal.svg)](https://pypi.org/project/muse-xtal/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build and Test](https://github.com/chiang-yuan/muse/actions/workflows/test.yml/badge.svg)](https://github.com/chiang-yuan/muse/actions/workflows/test.yml)

**Muse** (**M**ixture b**u**ilder for **s**imulation **e**nvironments) is a Python package for rapidly building amorphous solids and liquid mixtures from relaxed solid-state structures on [Materials Project](https://materialsproject.org/). It uses [Packmol](http://m3g.iqm.unicamp.br/packmol/home.shtml) for packing molecules into simulation cells and supports density equilibration through molecular dynamics with machine learning interatomic potentials (MLIPs), especially universal interatomic potentials (UIPs) such as [MACE](https://github.com/ACEsuit/mace) and [CHGNet](https://github.com/CederGroupHub/chgnet).

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

### Optional extras

```bash
# MACE calculator support
pip install "muse-xtal[mace]"

# Development tools (ruff, pytest)
pip install "muse-xtal[dev]"

# Documentation building
pip install "muse-xtal[docs]"
```

### Prerequisites

Muse requires [Packmol](http://m3g.iqm.unicamp.br/packmol/home.shtml) to be installed and available on your `PATH`. You can compile it from source:

```bash
bash scripts/install-packmol.sh
```

You also need a [Materials Project API key](https://materialsproject.org/api) set as the `MP_API_KEY` environment variable (or in a `.env` file).

## Quick Start

```python
from muse.transforms.mixture import mix_number

# Build a NaCl–KCl mixture (3:1 ratio, ~20 atoms)
atoms = mix_number(
    recipe={"NaCl": 3, "KCl": 1},
    tolerance=2.0,
    scale=1.05,
    seed=42,
)
print(atoms)  # Atoms object ready for simulation
```

### Density equilibration with MACE

```python
import numpy as np
from ase import units
from mace.calculators import MACECalculator
from muse.calcs.density import DensityCalc

calc = MACECalculator(model_paths="path/to/model", device="cpu")

density_calc = DensityCalc(
    calculator=calc,
    optimizer="FIRE",
    steps=500,
    mask=np.eye(3),
    rtol=1e-3,
    atol=5e-4,
)

results = density_calc.calc(
    atoms=atoms,
    temperature=1100,  # K
    externalstress=0.0,  # eV/Å³
)
print(f"Density: {results['mass_density']:.4f} amu/ų")
```

## Documentation

Full documentation is available at [chiang-yuan.github.io/muse](https://chiang-yuan.github.io/muse).

## Citation

If you use Muse in your research, please cite:

```bibtex
@software{chiang2023muse,
  author    = {Chiang, Yuan},
  title     = {muse-xtal},
  version   = {0.2.0},
  year      = {2023},
  doi       = {10.5281/zenodo.10369245},
  url       = {https://github.com/chiang-yuan/muse}
}
```

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
