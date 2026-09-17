"""
Run repeated solver timing analysis.

Measures BFSA and SOCP build and solve times across repeated runs, then exports
per-run timings and solver-level runtime comparisons.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import inspect
import sys
import time
from pathlib import Path
from typing import Any, Callable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from libs.methods import EstimatorPipelineSolver
from libs.shared import load_case_from_folder, orient_radial_network


#======================================= EDIT this =================================
DEFAULT_INPUT_FOLDER = ROOT / "data" / "Methodology Paper" / "80_bus_base"
DEFAULT_RUNS         = 30
#===================================================================================


def _repo_relative_path(path: Path) -> Path:
    """Resolve relative paths from the repository root."""
    if path.is_absolute():
        return path
    return ROOT / path


def _default_output_dir(input_folder: Path) -> Path:
    return input_folder / "results" / "speedanalysis"


def _summary_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    timing_columns = ["solve_time_s", "solve_wall_time_s"]

    summary_rows = []
    for solver, group in df.groupby("solver", sort=True):
        for column in timing_columns:
            values = pd.to_numeric(group[column], errors="coerce").dropna()
            summary_rows.append(
                {
                    "solver": solver,
                    "metric": column,
                    "count": int(values.count()),
                    "mean": float(values.mean()) if not values.empty else None,
                    "median": float(values.median()) if not values.empty else None,
                    "min": float(values.min()) if not values.empty else None,
                    "max": float(values.max()) if not values.empty else None,
                    "std": float(values.std()) if len(values) > 1 else 0.0,
                }
            )

    return pd.DataFrame(summary_rows)


def _comparison_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    timing_columns = ["solve_time_s", "solve_wall_time_s"]
    df = pd.DataFrame(rows)

    grouped = list(df.groupby("solver", sort=True))
  
    mean_wt_by_solver = {
        solver: float(pd.to_numeric(group["solve_wall_time_s"], errors="coerce").dropna().mean())
        for solver, group in grouped
    }
    bfsa_mean_total = mean_wt_by_solver.get("BFSA", None)

    summary_rows = []
    for solver, group in grouped:
        row: dict[str, Any] = {
            "solver": solver,
            "runs": int(len(group)),
            "horizon_start": int(group["horizon_start"].iloc[0]),
            "horizon_end": int(group["horizon_end"].iloc[0]),
            "horizon_steps": int(group["horizon_steps"].iloc[0]),
        }
        for column in timing_columns:
            values = pd.to_numeric(group[column], errors="coerce").dropna()
            row[f"{column}_mean"] = float(values.mean()) if not values.empty else None
            row[f"{column}_median"] = float(values.median()) if not values.empty else None
            row[f"{column}_min"] = float(values.min()) if not values.empty else None
            row[f"{column}_max"] = float(values.max()) if not values.empty else None
            row[f"{column}_std"] = float(values.std()) if len(values) > 1 else 0.0
        mean_total = row.get("solve_wall_time_s_mean")
        if bfsa_mean_total and mean_total:
            row["runtime_ratio_vs_bfsa"] = float(mean_total / bfsa_mean_total)
            row["bfsa_speedup_vs_solver"] = float(mean_total / bfsa_mean_total)
        else:
            row["runtime_ratio_vs_bfsa"] = None
            row["bfsa_speedup_vs_solver"] = None
        summary_rows.append(row)

    return pd.DataFrame(summary_rows)


def _solver_options_for_cores(solver_name: str, cores: int) -> dict[str, Any]:
    if cores < 1:
        raise ValueError("--socp-cores must be at least 1")
    if cores == 1:
        return {}

    solver_key = solver_name.strip().lower()
    if solver_key == "mosek":
        return {"MSK_IPAR_NUM_THREADS": int(cores)}
    if solver_key in {"gurobi", "cplex"}:
        return {"threads": int(cores)}
    return {}


def _load_indices(load_index: int, start_index: int | None, end_index: int | None) -> list[int]:
    if start_index is None and end_index is None:
        if load_index < 0:
            raise ValueError("--load-index must be non-negative")
        return [load_index]

    if start_index is None or end_index is None:
        raise ValueError("--start-index and --end-index must be provided together")
    if start_index < 0 or end_index < 0:
        raise ValueError("--start-index and --end-index must be non-negative")
    if end_index < start_index:
        raise ValueError("--end-index must be greater than or equal to --start-index")

    return list(range(start_index, end_index + 1))


def _solve_case(solver: Any, case) -> Any:
    """Solve an oriented case using either legacy or refactored solver APIs."""
    if hasattr(solver, "solve"):
        solve: Callable[..., Any] = solver.solve
        kwargs: dict[str, Any] = {}
        signature = inspect.signature(solve)
        if "orient" in signature.parameters:
            kwargs["orient"] = False
        result = solve(case, **kwargs)
    elif hasattr(solver, "solve_optimization"):
        result = solver.solve_optimization(case, orient=False)
    else:
        raise TypeError(f"Solver {solver!r} has neither solve() nor solve_optimization()")

    return result


def _result_diagnostics(result: Any) -> dict[str, Any]:
    diagnostics = getattr(result, "convergence_info", None)
    if diagnostics is None:
        diagnostics = getattr(result, "diagnostics", None)
    return dict(diagnostics or {})


def _result_cost(result: Any) -> float:
    cost = getattr(result, "cost", None)
    if cost is None:
        cost = getattr(result, "objective", None)
    return float(cost or 0.0)


def _time_solver_over_horizon(
    *,
    solver_name: str,
    solver_factory,
    input_folder: Path,
    load_indices: list[int],
    run_idx: int,
    horizon_start: int,
    horizon_end: int,
    horizon_steps: int,
) -> dict[str, Any]:

    solve_time = 0.0
    solve_wall_time = 0.0
    internal_build_time = 0.0
    internal_extract_time = 0.0
    internal_total_time = 0.0
    total_cost = 0.0
    converged = True
    iteration_values = []

    solver = solver_factory()

    for load_index in load_indices:
        with contextlib.redirect_stdout(io.StringIO()):
            case = load_case_from_folder(input_folder, load_index=load_index)
        case = orient_radial_network(case)

        solve_start = time.perf_counter()
        result = _solve_case(solver, case)
        step_solve_wall_time = time.perf_counter() - solve_start

        convergence_info = _result_diagnostics(result)
        timing = convergence_info.get("timing", {}) or {}

        step_internal_build_time = float(timing.get("build", 0.0) or 0.0)
        internal_build_time += step_internal_build_time
        internal_extract_time += float(timing.get("extract", 0.0) or 0.0)
        internal_total_time += float(timing.get("total", 0.0) or 0.0)

        solve_wall_time += step_solve_wall_time
        solve_time += float(getattr(result, "solve_time", None) or 0.0)

        if convergence_info.get("converged") is False:
            converged = False
        if convergence_info.get("iterations") is not None:
            iteration_values.append(int(convergence_info["iterations"]))
        total_cost += _result_cost(result)

    return {
        "run": run_idx,
        "solver": solver_name,
        "input_folder": str(input_folder),
        "horizon_start": horizon_start,
        "horizon_end": horizon_end,
        "horizon_steps": horizon_steps,
        "solve_time_s": solve_time,
        "solve_wall_time_s": solve_wall_time,
        "internal_build_time_s": internal_build_time or None,
        "internal_extract_time_s": internal_extract_time or None,
        "internal_total_time_s": internal_total_time or None,
        "converged": converged,
        "iterations": sum(iteration_values) if iteration_values else None,
        "cost": total_cost,
    }


def run_speed_analysis(
    input_folder: Path,
    output_dir: Path,
    runs: int,
    load_indices: list[int],
    *,
    include_bfsa: bool = True,
    include_socp: bool = True,
    socp_solver_name: str = "mosek",
    socp_cores: int = 1,
    socp_tee: bool = False,
    socp_allow_load_shedding: bool = False,
) -> tuple[Path, Path, Path]:
    if runs < 1:
        raise ValueError("--runs must be at least 1")
    if not load_indices:
        raise ValueError("At least one load index is required")
    if not input_folder.exists():
        raise FileNotFoundError(f"Input folder not found: {input_folder}")
    if not include_bfsa and not include_socp:
        raise ValueError("At least one solver must be enabled")

    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    horizon_start = load_indices[0]
    horizon_end = load_indices[-1]
    horizon_steps = len(load_indices)
    solver_factories = []

    if include_bfsa:
        solver_factories.append(
            (
                "BFSA",
                lambda: EstimatorPipelineSolver(
                    physical_options={
                        "compute_losses": True,
                        "tol": 1e-4,
                        "max_iterations": 1000,
                    }
                ),
            )
        )

    if include_socp:
        from libs.methods.opf.socp import SOCPSolver

        socp_options = _solver_options_for_cores(socp_solver_name, socp_cores)
        solver_factories.append(
            (
                "SOCP",
                lambda: SOCPSolver(
                    solver_name=socp_solver_name,
                    tee=socp_tee,
                    solver_options=socp_options,
                    allow_load_shedding=socp_allow_load_shedding,
                ),
            )
        )

    for run_idx in range(1, runs + 1):
        for solver_name, solver_factory in solver_factories:
            row = _time_solver_over_horizon(
                solver_name=solver_name,
                solver_factory=solver_factory,
                input_folder=input_folder,
                load_indices=load_indices,
                run_idx=run_idx,
                horizon_start=horizon_start,
                horizon_end=horizon_end,
                horizon_steps=horizon_steps,
            )
            rows.append(row)

            print(
                f"Run {run_idx}/{runs}, solver={solver_name}, "
                f"horizon={horizon_start}..{horizon_end} ({horizon_steps} steps): "
                f"solve={row['solve_time_s']:.6f}s, "
                f"wall={row['solve_wall_time_s']:.6f}s"
            )

    runs_path = output_dir / "solver_speed_runs.csv"
    summary_path = output_dir / "solver_speed_summary.csv"
    comparison_path = output_dir / "solver_speed_comparison.csv"

    pd.DataFrame(rows).to_csv(runs_path, index=False)
    _summary_frame(rows).to_csv(summary_path, index=False)
    _comparison_frame(rows).to_csv(comparison_path, index=False)

    return runs_path, summary_path, comparison_path


def run_bfsa_speed_analysis(
    input_folder: Path,
    output_dir: Path,
    runs: int,
    load_indices: list[int],
) -> tuple[Path, Path, Path]:
    return run_speed_analysis(
        input_folder=input_folder,
        output_dir=output_dir,
        runs=runs,
        load_indices=load_indices,
        include_bfsa=True,
        include_socp=False,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run repeated BFSA/SOCP speed analysis.")
    parser.add_argument(
        "--input-folder",
        type=Path,
        default=DEFAULT_INPUT_FOLDER,
        help="Network input folder. Relative paths are resolved from the repository root.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=DEFAULT_RUNS,
        help="Number of timing runs per solver.",
    )
    parser.add_argument(
        "--load-index",
        type=int,
        default=4829,
        help="Single load profile row index to solve when no range is provided.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=None,
        help="First load profile row index for range timing, inclusive.",
    )
    parser.add_argument(
        "--end-index",
        type=int,
        default=None,
        help="Last load profile row index for range timing, inclusive.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <input-folder>/results/speedanalysis.",
    )
    parser.add_argument(
        "--include-socp",
        action="store_true",
        help="Compatibility flag; SOCP is included by default.",
    )
    parser.add_argument(
        "--bfsa-only",
        action="store_true",
        help="Only time BFSA.",
    )
    parser.add_argument(
        "--socp-only",
        action="store_true",
        help="Only time SOCP.",
    )
    parser.add_argument(
        "--socp-solver",
        default="mosek",
        help="Pyomo solver name for SOCP timing.",
    )
    parser.add_argument(
        "--socp-cores",
        type=int,
        default=1,
        help="SOCP solver thread/core count when supported.",
    )
    parser.add_argument(
        "--socp-tee",
        action="store_true",
        help="Stream SOCP solver output.",
    )
    parser.add_argument(
        "--socp-allow-load-shedding",
        action="store_true",
        help="Enable SOCP load shedding for infeasible snapshots.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_folder = _repo_relative_path(args.input_folder)
    output_dir = _repo_relative_path(args.output_dir) if args.output_dir else _default_output_dir(input_folder)
    output_dir = output_dir / f"time_steps_{args.start_index}_{args.end_index}" if args.start_index is not None and args.end_index is not None else output_dir / f"time_step_{args.load_index}"
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)


    load_indices = _load_indices(args.load_index, args.start_index, args.end_index)
    if args.bfsa_only and args.socp_only:
        raise ValueError("--bfsa-only and --socp-only cannot be used together")

    runs_path, summary_path, comparison_path = run_speed_analysis(
        input_folder=input_folder,
        output_dir=output_dir,
        runs=args.runs,
        load_indices=load_indices,
        include_bfsa=not args.socp_only,
        include_socp=not args.bfsa_only,
        socp_solver_name=args.socp_solver,
        socp_cores=args.socp_cores,
        socp_tee=args.socp_tee,
        socp_allow_load_shedding=args.socp_allow_load_shedding,
    )

    print("[OK] Solver speed analysis complete")
    print(f"Load indices: {load_indices[0]}..{load_indices[-1]} ({len(load_indices)} total)")
    print(f"Runs: {runs_path}")
    print(f"Summary: {summary_path}")
    print(f"Comparison: {comparison_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
