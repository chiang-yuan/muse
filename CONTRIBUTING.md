# Contributing

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given. You can contribute in the ways listed below.

## Report Bugs

Report bugs using GitHub issues.

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

## Fix Bugs

Look through the GitHub issues for bugs. Anything tagged with "bug" and "help
wanted" is open to whoever wants to implement it.

## Implement Features

Look through the GitHub issues for features. Anything tagged with "enhancement"
and "help wanted" is open to whoever wants to implement it.

## Write Documentation

muse could always use more documentation, whether as part of the
official muse docs, in docstrings, or even on the web in blog posts,
articles, and such.

## Submit Feedback

The best way to send feedback is to file an issue on GitHub.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions
  are welcome :)

## Get Started

Ready to contribute? Here's how to set up `muse` for local development.

1. Fork the repo on GitHub.
2. Clone your fork locally:
   ```bash
   git clone https://github.com/<your-username>/muse.git
   cd muse
   ```
3. Install the package in development mode with dev dependencies:
   ```bash
   pip install -e ".[dev]"
   ```
4. Install [Packmol](http://m3g.iqm.unicamp.br/packmol/home.shtml) (required for structure generation):
   ```bash
   bash scripts/install-packmol.sh
   ```
5. Create a branch for local development and make changes locally:
   ```bash
   git checkout -b feature/my-new-feature
   ```
6. Check your code style with ruff:
   ```bash
   ruff check muse/ tests/
   ruff format --check muse/ tests/
   ```
7. Run the test suite with pytest:
   ```bash
   pytest tests/ -v
   ```
8. Commit your changes and push your branch to GitHub.
9. Submit a pull request through the GitHub website.

## Code Style

This project uses [ruff](https://github.com/astral-sh/ruff) for linting and formatting.
Configuration is in `pyproject.toml`. Run formatting before committing:

```bash
ruff check --fix muse/ tests/
ruff format muse/ tests/
```

## Code of Conduct

Please note that the muse project is released with a [Contributor Code of Conduct](CONDUCT.md). By contributing to this project you agree to abide by its terms.
