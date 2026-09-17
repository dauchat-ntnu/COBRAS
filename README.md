# COBRAS

COBRAS is a Python framework for analysing radial power grids in market and
state studies. It provides a common workflow to load and orient radial network
data, run a convex AC optimal-power-flow (ACOPF) reference, run modular
estimators, and compare dispatch, electrical-state, economic-state, accuracy,
and runtime results.

The built-in methods are:

- an SOCP radial branch-flow ACOPF relaxation implemented with Pyomo;
- a merit-order dispatch estimator;
- a backward-forward sweep algorithm (BFSA) physical-state estimator; and
- a BFSA-based distribution locational marginal price (DLMP) estimator.

COBRAS is intended for research workflows that need repeated radial-grid
analysis, including multi-timestep studies and benchmarking fast estimators
against an OPF reference.

## Installation

COBRAS runs from a GitHub clone with Python 3.10 or later. Clone this repository,
open a terminal in its root directory, then create and activate the environment:

```bash
git clone https://github.com/dauchat-ntnu/COBRAS.git
cd COBRAS
conda env create -f environment.yml
conda activate cobras
```

The SOCP method also needs an installed Pyomo-compatible conic solver. The
examples default to MOSEK, which must be installed and licensed separately.
Users may select another compatible solver when it supports the required conic
model and dual-value extraction.

## Input data

The standard loader expects a case folder containing:

- `mpc_base_mva`
- `mpc_bus.csv`
- `mpc_branch.csv`
- `p_load.csv`
- `q_load.csv`
- `generators.xlsx`

The loader reads MATPOWER-style CSV tables, not `.m` files. Load units depend
on the selected mode; generator limits are already in per-unit. See the
[user guide](docs/documentation.tex) for the complete specification.

## Run a study

Launch the graphical runner:

```bash
python examples/run_framework.py
```

Or run the estimator pipeline on the bundled 80-bus case:

```bash
python examples/run_framework.py --preset estimators --load-index 0
```

To run both the SOCP reference and the estimator pipeline, provide a configured
solver (MOSEK is the default):

```bash
python examples/run_framework.py --preset comparison --case-folder path/to/case --opf-solver mosek
```

The project also includes scripts for sensitivity studies, load-profile analysis,
plot generation, and speed analysis in [`examples/`](examples).

## Documentation

The extended user guide is [docs/documentation.tex](docs/documentation.tex).
It includes installation, data units, Python examples, range studies, plotting,
method extension, and limitations. 

This repository is released under the [BSD 3-Clause License](LICENSE).
