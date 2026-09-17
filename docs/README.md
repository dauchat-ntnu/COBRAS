# COBRAS user guide

COBRAS supports radial power-grid market and state analysis. A study can run a
convex AC optimal-power-flow (ACOPF) reference, an estimator pipeline, or both,
then store and compare electrical, dispatch, and economic results.

## Requirements

- Python 3.10 or later
- The dependencies declared in `environment.yml`
- For SOCP runs, a Pyomo-compatible conic solver. COBRAS examples default to
   MOSEK, which must be installed and licensed separately.

Clone the GitHub repository, open its root directory, then create the environment:

```bash
git clone https://github.com/dauchat-ntnu/COBRAS.git
cd COBRAS
conda env create -f environment.yml
conda activate cobras
```

## Case-folder format

The standard loader expects the following files in one case folder:

| File | Purpose |
| --- | --- |
| `mpc_base_mva` | Plain-text base power in MVA. |
| `mpc_bus.csv` | Bus identifiers, voltage limits, and optional slack-bus information. |
| `mpc_branch.csv` | Branch endpoints, resistance, reactance, and optional thermal limits. |
| `p_load.csv` | Active-power time series, with `Date` as the first column. |
| `q_load.csv` | Reactive-power time series, with `Date` as the first column. |
| `generators.xlsx` | Generator limits and cost information. |

In explicit `absolute` load mode, demands are converted to per-unit values using
`mpc_base_mva`; generator limits must already be per-unit. COBRAS
expects its built-in OPF and estimator methods to operate on a connected,
acyclic radial network.

## Running COBRAS

Start the graphical runner:

```bash
python examples/run_framework.py
```

Run only the estimator pipeline:

```bash
python examples/run_framework.py --preset estimators --case-folder path/to/case
```

Compare the SOCP reference with the estimator pipeline:

```bash
python examples/run_framework.py --preset comparison --case-folder path/to/case --opf-solver mosek
```

The `--preset` option accepts `opf`, `estimators`, and `comparison`. Use
`python examples/run_framework.py --help` to view the available solver,
estimator, and BFSA options.

## Methods and outputs

The built-in SOCP model returns dispatch, voltages, branch flows, currents, and
selected dual quantities. The estimator pipeline combines merit-order dispatch,
BFSA electrical-state estimation, and BFSA-based DLMP estimation. Multi-timestep
workflows can persist results in SQLite databases, which can be explored with
the plotting tools in `examples/generate_plots.py`.

## Further documentation

The detailed technical manual is maintained in
[`docs/documentation.tex`](documentation.tex). It describes
the data model, radial orientation, method-family interfaces, result objects,
database persistence, and plotting workflows.

Compile the self-contained guide from this directory with two passes of:

```bash
pdflatex -interaction=nonstopmode -halt-on-error documentation.tex
```

## Important limitations

- SOCP exactness is not guaranteed; inspect constraints, residuals, and
   comparison outputs for the cases you study.
- Dual-value availability depends on the selected solver and its Pyomo support.
- The built-in merit-order estimator does not enforce network constraints during
   clearing.
- Generator capacities are read as per-unit values. Explicit `absolute` load
   mode divides MW/MVAr by base MVA; legacy `auto` mode can instead pass values
   through as per-unit. Choose the mode deliberately.
