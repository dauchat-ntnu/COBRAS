"""Sensitivity sweep example

This script runs a sensitivity sweep over the generator reactive-to-active
cost ratio (c_q / c_p) for a single generator and a small set of snapshot
labels (e.g. 'average', 'high', 'low').

Behavior:
- Loads a case snapshot for each requested `timestep_label` using
  `load_case_from_folder(load_at=...)` so the loader's label-indexing is used.
- For each ratio value, updates the target generator's `c_q` to
  `c_q = ratio * c_p` (keeps `c_p` unchanged) and solves the case with
  the configured solvers.
- Stores results in a dedicated output folder and a single SQLite database
  per sweep run. The `scenario_label` stored in the DB encodes the
  sensitivity name, ratio value and timestep label to make later filtering
  simple.

notes:
- This is a standalone example. It reuses the existing loaders, solvers and
  `TimestepDatabase` schema. It writes one DB per sweep run and a small
  `meta.json` describing the sweep for provenance.
"""

from pathlib import Path
import sys
import json
from datetime import datetime
from typing import List, Union
import pandas as pd

import numpy as np

# Ensure project root is on sys.path for local imports (matches other example scripts)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from libs.shared import load_case_from_folder, orient_radial_network
from libs.shared.timestep_database import TimestepDatabase
from libs.shared.multi_timestep_solver import MultiTimestepSolver
from libs.methods import EstimatorPipelineSolver
from libs.methods.opf.socp import SOCPSolver

from examples import plot_sensitivity_lmp as pslmp

def run_cq_cp_sweep(
    *,
    network_folder: Path,
    output_root: Path,
    gen_id: int,
    ratios: List[float],
    timestep_labels: List[str],
    case_config: dict,
    solvers: dict,
    sensitivity_name: str = "cq_cp",
):
    """Run the c_q / c_p sensitivity sweep.

    Args:
        network_folder: Path to network data folder (e.g. data/MV_solar)
        output_root: Base output folder where sweep run folder will be created
        gen_id: Generator index (the `gen_id` field assigned by the loader)
        ratios: List of c_q/c_p ratio values to sweep
        timestep_labels: Snapshot labels to load (passed as `load_at` to loader)
        case_config: Case loader config (keys used by load_case_from_folder)
        solvers: Dict[str, solver_instance]
        sensitivity_name: Short name used in output folder naming
    """

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    run_dir = Path(output_root) / f"sensitivity_{sensitivity_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    db_dir = run_dir / "db"
    db_dir.mkdir(parents=True, exist_ok=True)

    # Persist metadata for the run
    meta = {
        "network": str(network_folder),
        "gen_id": int(gen_id),
        "ratios": list(ratios),
        "timestep_labels": list(timestep_labels),
        "sensitivity_name": sensitivity_name,
        "timestamp": timestamp,
    }
    with open(run_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    db_path = db_dir / "results.db"
    if db_path.exists():
        print(f"Note: removing existing DB at {db_path}")
        db_path.unlink()

    # Resolve timestep labels to loader-friendly values (either index ints or existing labels)
    def _resolve_labels(folder: Path, labels: List[str], p_load_fname: str) -> List[Union[str, int]]:
        p_path = folder / p_load_fname
        if not p_path.exists():
            return labels
        df = pd.read_csv(p_path, sep=None, engine="python", index_col=0)
        resolved = []
        sums = None
        try:
            sums = df.apply(pd.to_numeric, errors="coerce").sum(axis=1)
        except Exception:
            sums = None

        for lab in labels:
            if lab is None:
                resolved.append(None)
                continue
            # If label matches an index entry, keep as string
            if str(lab) in df.index.astype(str).tolist():
                resolved.append(str(lab))
                continue
            # Map descriptive keywords to indices if possible
            key = str(lab).strip().lower()
            if sums is not None and key in ("high", "low", "average"):
                if key == "high":
                    lbl = sums.idxmax()
                    resolved.append(int(sums.index.get_loc(lbl)))
                elif key == "low":
                    lbl = sums.idxmin()
                    resolved.append(int(sums.index.get_loc(lbl)))
                else:
                    mean_val = sums.mean()
                    lbl = (sums - mean_val).abs().idxmin()
                    resolved.append(int(sums.index.get_loc(lbl)))
                continue
            # As last resort, try parsing as integer index
            try:
                resolved.append(int(lab))
                continue
            except Exception:
                pass
            # Unknown label -> pass through (loader may raise)
            resolved.append(lab)

        return resolved

    p_load_fname = case_config.get("p_load_filename", "p_load.csv")
    resolved_labels = _resolve_labels(network_folder, timestep_labels, p_load_fname)

    # Load an initial case to create DB schema (use first resolved label)
    sample_label = resolved_labels[0] if resolved_labels else None
    sample_case = load_case_from_folder(
        network_folder,
        load_at=sample_label if isinstance(sample_label, str) else None,
        load_index=sample_label if isinstance(sample_label, int) else None,
        generators_filename=case_config.get("generators_filename", "generators.xlsx"),
        p_load_filename=case_config.get("p_load_filename", "p_load.csv"),
        q_load_filename=case_config.get("q_load_filename", "q_load.csv"),
        mpc_bus_filename=case_config.get("mpc_bus_filename", "mpc_bus.csv"),
        mpc_branch_filename=case_config.get("mpc_branch_filename", "mpc_branch.csv"),
    )
    sample_case = orient_radial_network(sample_case)

    # Initialize DB
    db = TimestepDatabase(str(db_path))
    db.create_tables(sample_case)
    db.open_session()

    # Prepare a MultiTimestepSolver instance for convenience (for exports/prints)
    multi_solver = MultiTimestepSolver(
        input_folder=network_folder,
        output_db=db_path,
        case_config=case_config,
        solvers=solvers,
    )
    # Bind the live DB instance so helper methods can use it
    multi_solver.db = db

    timestep_counter = 0
    solver_outputs = {name: [] for name in solvers.keys()}

    try:
        for orig_label, resolved_label in zip(timestep_labels, resolved_labels):
            for ratio in ratios:
                print(f"Running sensitivity: label={orig_label} (resolved={resolved_label}), ratio={ratio:.4f} -> timestep #{timestep_counter}")

                # Pass either load_at (string label) or load_index (int position)
                load_at = resolved_label if isinstance(resolved_label, str) else None
                load_index = resolved_label if isinstance(resolved_label, int) else None

                case = load_case_from_folder(
                    network_folder,
                    load_at=load_at,
                    load_index=load_index,
                    generators_filename=case_config.get("generators_filename", "generators.xlsx"),
                    p_load_filename=case_config.get("p_load_filename", "p_load.csv"),
                    q_load_filename=case_config.get("q_load_filename", "q_load.csv"),
                    mpc_bus_filename=case_config.get("mpc_bus_filename", "mpc_bus.csv"),
                    mpc_branch_filename=case_config.get("mpc_branch_filename", "mpc_branch.csv"),
                )
                case = orient_radial_network(case)

                # Find target generator
                target_gen = None
                for g in case.generators:
                    if int(g.gen_id) == int(gen_id):
                        target_gen = g
                        break
                if target_gen is None:
                    raise ValueError(f"Generator with gen_id={gen_id} not found in case")

                # Apply ratio: keep c_p, set c_q = ratio * c_p
                orig_cp = float(target_gen.c_p)
                print(f"  Original c_p={orig_cp:.2f}, c_q={target_gen.c_q:.2f} -> new c_q={float(ratio) * orig_cp:.2f}")
                target_gen.c_p = orig_cp
                target_gen.c_q = float(ratio) * orig_cp

                for solver_name, solver in solvers.items():
                    scenario_label = f"sensitivity:{sensitivity_name};ratio:{ratio:.6f};label:{orig_label};gen:{gen_id}"
                    try:
                        result = solver.solve(case)
                    except Exception as ex:
                        # store failure marker
                        db.store_solver_failure(
                            timestep=timestep_counter,
                            scenario_label=scenario_label,
                            solver_name=solver_name,
                            convergence_status="failed",
                            error_message=str(ex),
                        )
                        print(f"  {solver_name}: FAILED ({ex})")
                        continue

                    # Store result
                    db.store_timestep_result(
                        timestep=timestep_counter,
                        scenario_label=scenario_label,
                        solver_name=solver_name,
                        result=result,
                        case=case,
                    )
                    solver_outputs[solver_name].append(result)
                    print(f"  {solver_name}: OK cost={result.cost:.2f} time={result.solve_time:.3f}s")

                timestep_counter += 1
    finally:
        db.close_session()

    # Compute aggregates for entire pseudo-range
    if timestep_counter > 0:
        db.compute_aggregates(0, timestep_counter - 1)

    # Export CSV per solver
    for solver_name in solvers.keys():
        export_dir = run_dir / solver_name
        multi_solver.export_results_to_csv(0, max(0, timestep_counter - 1), solver_name, export_dir, verbose=True)

    print(f"[OK] Sensitivity scan complete. Results in: {run_dir}")


def main():
    ROOT = Path(__file__).resolve().parents[1]
    data_folder = ROOT / "data"

    # User-configurable area
    network_name = "80_bus_SOLE"
    network_folder = data_folder / network_name
    output_root = data_folder / network_name / "sensitivity"
    print(f"Sensitivity outputs will be written to: {output_root}")

    # Which generator (gen_id assigned by the loader, typically row index in generators file)
    gen_id = 0

    # Build ratios: either explicit list or numpy linspace
    ratios = list(np.linspace(0.0, 2.0, 9))  # 0.0,0.25,...,2.0 by default

    # Snapshot labels to run (passed to load_case_from_folder as `load_at`)
    timestep_labels = ["average", "high", "low"]

    case_config = {
        "mpc_base_mva_filename": "mpc_base_mva",
        "mpc_bus_filename": "mpc_bus.csv",
        "mpc_branch_filename": "mpc_branch.csv",
        "p_load_filename": "p_load.csv",
        "q_load_filename": "q_load.csv",
        "bus_mapping_filename": None,
        "generators_filename": "generators.xlsx",
    }

    # Initialize solvers
    socp_solver = SOCPSolver(solver_name="mosek", tee=False)
    bfsa_solver = EstimatorPipelineSolver()
    solvers = {"socp": socp_solver, "bfsa": bfsa_solver}

    run_cq_cp_sweep(
        network_folder=network_folder,
        output_root=output_root,
        gen_id=gen_id,
        ratios=ratios,
        timestep_labels=timestep_labels,
        case_config=case_config,
        solvers=solvers,
        sensitivity_name="cq_cp",
    )
    
    pslmp.main()


if __name__ == "__main__":
    main()
