"""
Multi-timestep solver orchestrator for COBRAS.

Coordinates sequential execution of optimization solvers and estimators across timestep ranges
and persists results to SQLite database.
"""

from pathlib import Path
import inspect
from typing import Dict, List, Optional, Any
import time

from tqdm import tqdm

from libs.shared.data import OPFResult
from libs.shared.io import build_case_loader_from_folder
from libs.shared.network import orient_radial_network
from libs.shared.timestep_database import TimestepDatabase
from libs.methods import EstimatorPipelineSolver


def _is_acopf_infeasible_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return (
        "infeasibleorunbounded" in text
        or "termination_condition=infeasible" in text
        or "infeasible for the current snapshot" in text
    )


def _solver_supports_warm_start(solver: Any) -> bool:
    capabilities = getattr(solver, "capabilities", None)
    if capabilities is not None and bool(getattr(capabilities, "supports_warm_start", False)):
        return True

    solve = getattr(solver, "solve", None)
    if solve is None:
        return False
    try:
        return "warm_start" in inspect.signature(solve).parameters
    except (TypeError, ValueError):
        return False


def _method_label(solver_name: str, solver: Any) -> str:
    name = getattr(solver, "name", None)
    if name:
        return str(name)
    return str(solver_name)


class MultiTimestepSolver:
    """
    Execute configured solvers sequentially across a timestep range.
    
    Loads cases for each timestep, solves them, and stores results to SQLite.
    """

    def __init__(
        self,
        input_folder: Path,
        output_db: Path,
        case_config: Dict[str, str],
        solvers: Dict[str, Any],
    ):
        """
        Initialize orchestrator.
        
        Args:
            input_folder: Path to network data folder (e.g., data/MV_solar/)
            output_db: Path to SQLite database file (will be created)
            case_config: Dict with keys: mpc_bus, mpc_branch, p_load, q_load, 
                         bus_mapping, generators, base_mva
            solvers: Dict[solver_name] -> solver instance. Solver keys are user/config
                     method names; ACOPF solvers are detected by warm-start support.
        """
        self.input_folder = Path(input_folder)
        self.output_db = Path(output_db)
        self.case_config = case_config
        self.solvers = solvers
        self.db: Optional[TimestepDatabase] = None

    def run_range(
        self,
        start_idx: int,
        end_idx: int,
        solver_names: Optional[List[str]] = None,
        scenario_label: str = "Range",
        verbose: bool = True,
        kill_flag: Optional[List[bool]] = None,
        continue_on_socp_infeasible: bool = True,
        warm_start_prev: bool = False,
        warm_start_bfsa: bool = False,
        warm_start_mode: Optional[str] = None,
        warm_start_solver_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute solvers sequentially for timestep range.
        
        Args:
            start_idx: Start timestep index (inclusive, 0-based)
            end_idx: End timestep index (inclusive, 0-based)
            solver_names: List of solver names to run.
                         If None, uses all solvers provided in init.
            scenario_label: User-facing label for results.
            verbose: Print progress messages
            kill_flag: List[bool] to allow external interruption (checked each iteration)
            
        Returns:
            Dict with keys:
                - timesteps_completed: Number of timesteps processed
                - solver_outputs: Dict[solver_name] -> List[OPFResult]
                - db_path: Path to SQLite database
                - stats_by_solver: Dict[solver_name] -> Dict with summary stats
                
        Raises:
            ValueError: If timestep range is invalid or computation was killed
            IOError: If database file already exists or output dir can't be created
        """
        if kill_flag is None:
            kill_flag = [False]
            
        if solver_names is None:
            solver_names = list(self.solvers.keys())

        if not solver_names:
            raise ValueError("No solvers specified")

        if start_idx < 0 or end_idx < start_idx:
            raise ValueError(f"Invalid timestep range: [{start_idx}, {end_idx}]")

        selected_warm_start_mode = str(warm_start_mode or "None").strip()
        selected_warm_start_mode_lower = selected_warm_start_mode.lower()
        if selected_warm_start_mode_lower == "none":
            if warm_start_bfsa:
                selected_warm_start_mode = "Estimator t"
            elif warm_start_prev:
                selected_warm_start_mode = "ACOPF t-1"
        elif selected_warm_start_mode_lower in {"socp t-1", "acopf t-1"}:
            warm_start_prev = True
        elif selected_warm_start_mode_lower in {"bfsa t", "estimator t"}:
            warm_start_bfsa = True
        else:
            raise ValueError(f"Unknown warm_start_mode: {selected_warm_start_mode}")

        # Ensure output directory exists
        self.output_db.parent.mkdir(parents=True, exist_ok=True)

        # Build fast per-timestep loader: static inputs are parsed once.
        case_at = build_case_loader_from_folder(
            self.input_folder,
            mpc_base_mva_filename=self.case_config.get("mpc_base_mva_filename", "mpc_base_mva"),
            mpc_bus_filename=self.case_config.get("mpc_bus_filename", "mpc_bus.csv"),
            mpc_branch_filename=self.case_config.get("mpc_branch_filename", "mpc_branch.csv"),
            p_load_filename=self.case_config.get("p_load_filename", "p_load.csv"),
            q_load_filename=self.case_config.get("q_load_filename", "q_load.csv"),
            bus_mapping_filename=self.case_config.get("bus_mapping_filename", None),
            generators_filename=self.case_config.get("generators_filename", "generators.xlsx"),
            profiles_filename=self.case_config.get("profiles_filename", "profiles.csv"),
            override_voltage_bounds=bool(self.case_config.get("override_voltage_bounds", False)),
            default_vmin=float(self.case_config.get("default_vmin", 0.9)),
            default_vmax=float(self.case_config.get("default_vmax", 1.1)),
            load_interpretation_mode=self.case_config.get("load_interpretation_mode", "auto"),
        )
        snapshot_count = getattr(case_at, "snapshot_count", None)
        if snapshot_count is not None and end_idx >= int(snapshot_count):
            last_idx = int(snapshot_count) - 1
            raise ValueError(
                f"Invalid timestep range: end_idx={end_idx} is out of bounds for "
                f"{snapshot_count} available snapshots. Valid timestep indices are "
                f"0-{last_idx}; the end timestep is inclusive."
            )

        # Initialize database with grid metadata from first case
        if verbose:
            print(f"[*] Loading initial case from timestep {start_idx}...")
        
        case = case_at(start_idx)
        case = orient_radial_network(case)

        if verbose:
            print(f"[*] Initializing database: {self.output_db}")
        else:
            print(f"Initializing database and loading timesteps {start_idx}-{end_idx}...")
        
        self.db = TimestepDatabase(str(self.output_db))
        self.db.create_tables(case)
        self.db.open_session()

        results_by_solver = {s: [] for s in solver_names}
        stats_by_solver = {}
        solver_wall_time = {s: 0.0 for s in solver_names}
        solver_internal_timing: Dict[str, Dict[str, float]] = {s: {} for s in solver_names}
        warm_start_mode_by_solver: Dict[str, str] = {s: "none" for s in solver_names}
        timing_breakdown = {
            "load_case": 0.0,
            "orient_network": 0.0,
            "db_write": 0.0,
            "compute_aggregates": 0.0,
            "loop_total": 0.0,
        }

        # Main loop: iterate timesteps with progress bar
        timesteps = range(start_idx, end_idx + 1)
        iterator = tqdm(
            timesteps,
            desc="Executing timesteps",
            disable=verbose,  # Show progress bar when not verbose
            unit="timestep",
        )

        try:
            for t in iterator:
                t_loop_start = time.perf_counter()

                # Check kill flag
                if kill_flag[0]:
                    if not verbose:
                        print(f"Computation killed at timestep {t}. Completed {t - start_idx} of {end_idx - start_idx + 1} timesteps.")
                    raise ValueError(f"Computation killed by user at timestep {t}")

                t_load_start = time.perf_counter()
                case = case_at(t)
                timing_breakdown["load_case"] += time.perf_counter() - t_load_start

                t_orient_start = time.perf_counter()
                case = orient_radial_network(case)
                timing_breakdown["orient_network"] += time.perf_counter() - t_orient_start

                if verbose:
                    print(f"\n[{t:04d}] Loaded timestep {t}")
                    print(f"[{t:04d}]   Buses: {len(case.buses)}, Branches: {len(case.branches)}, "
                          f"Generators: {len(case.generators)}, Loads: {len(case.loads)}")

                # Solve with each specified solver
                timestep_results = {}
                for solver_name in solver_names:
                    if solver_name not in self.solvers:
                        if verbose:
                            print(f"[{t:04d}]   WARN: Solver '{solver_name}' not provided, skipping")
                        continue

                    solver = self.solvers[solver_name]
                    
                    if verbose:
                        print(f"[{t:04d}]   Solving with {solver_name.upper()}...")

                    t_solver_start = time.perf_counter()
                    try:
                        solver_is_acopf = _solver_supports_warm_start(solver)
                        if solver_is_acopf:
                            warm_start = None
                            warm_start_source = "none"
                            if warm_start_bfsa:
                                estimator_solver_name = warm_start_solver_name
                                estimator_solver = None
                                if estimator_solver_name:
                                    estimator_solver = self.solvers.get(estimator_solver_name)
                                    if estimator_solver is None:
                                        raise ValueError(
                                            f"Warm-start estimator '{estimator_solver_name}' was not provided. "
                                            f"Available solvers: {', '.join(sorted(self.solvers))}"
                                        )
                                else:
                                    for candidate_name, candidate_solver in self.solvers.items():
                                        if candidate_name == solver_name:
                                            continue
                                        if not _solver_supports_warm_start(candidate_solver):
                                            estimator_solver_name = candidate_name
                                            estimator_solver = candidate_solver
                                            break
                                if estimator_solver is None:
                                    estimator_solver_name = "estimator_pipeline"
                                    estimator_solver = EstimatorPipelineSolver()
                                if estimator_solver_name not in timestep_results:
                                    timestep_results[estimator_solver_name] = estimator_solver.solve(case)
                                warm_start = timestep_results[estimator_solver_name]
                                warm_start_source = str(estimator_solver_name)
                            elif warm_start_prev and results_by_solver.get(solver_name):
                                warm_start = results_by_solver[solver_name][-1]
                                warm_start_source = "previous"

                            if warm_start is not None:
                                result = solver.solve(case, warm_start=warm_start)
                            else:
                                result = solver.solve(case)

                            if isinstance(result.convergence_info, dict):
                                result.convergence_info["warm_start_source"] = warm_start_source
                            else:
                                result.convergence_info = {"warm_start_source": warm_start_source}
                            warm_start_mode_by_solver[solver_name] = warm_start_source
                        else:
                            if solver_name not in timestep_results:
                                timestep_results[solver_name] = solver.solve(case)
                            result = timestep_results[solver_name]
                    except Exception as ex:
                        solver_is_acopf = _solver_supports_warm_start(solver)
                        allow_shedding = bool(getattr(solver, "allow_load_shedding", False))

                        if (
                            continue_on_socp_infeasible
                            and solver_is_acopf
                            and (not allow_shedding)
                            and _is_acopf_infeasible_error(ex)
                        ):
                            solver_wall_time[solver_name] += time.perf_counter() - t_solver_start

                            t_db_start = time.perf_counter()
                            self.db.store_solver_failure(
                                timestep=t,
                                scenario_label=scenario_label,
                                solver_name=solver_name,
                                convergence_status="infeasible_no_shedding",
                                error_message=str(ex),
                                load_interpretation_mode=self.case_config.get("load_interpretation_mode", "auto"),
                                allow_load_shedding=bool(getattr(solver, "allow_load_shedding", False)),
                                warm_start_mode=selected_warm_start_mode,
                            )
                            timing_breakdown["db_write"] += time.perf_counter() - t_db_start
                            continue

                        raise

                    solver_wall_time[solver_name] += time.perf_counter() - t_solver_start

                    timing_info = result.convergence_info.get("timing", {}) if isinstance(result.convergence_info, dict) else {}
                    for key, value in timing_info.items():
                        if isinstance(value, (int, float)):
                            solver_internal_timing[solver_name][key] = solver_internal_timing[solver_name].get(key, 0.0) + float(value)

                    # Store to database
                    t_db_start = time.perf_counter()
                    self.db.store_timestep_result(
                        timestep=t,
                        scenario_label=scenario_label,
                        solver_name=solver_name,
                        result=result,
                        case=case,
                        load_interpretation_mode=self.case_config.get("load_interpretation_mode", "auto"),
                        allow_load_shedding=bool(getattr(solver, "allow_load_shedding", False)),
                        warm_start_mode=selected_warm_start_mode if _solver_supports_warm_start(solver) else None,
                    )
                    timing_breakdown["db_write"] += time.perf_counter() - t_db_start
                    
                    results_by_solver[solver_name].append(result)

                    if verbose:
                        print(f"[{t:04d}]   {_method_label(solver_name, solver)} done: "
                              f"cost=${result.cost:.2f}, time={result.solve_time:.3f}s")

                timing_breakdown["loop_total"] += time.perf_counter() - t_loop_start
        finally:
            self.db.close_session()

        # Compute aggregates
        if verbose:
            print(f"\n[*] Computing aggregates for range [{start_idx}, {end_idx}]...")
        else:
            print("Computing aggregates...")

        t_agg_start = time.perf_counter()
        self.db.compute_aggregates(start_idx, end_idx)
        timing_breakdown["compute_aggregates"] += time.perf_counter() - t_agg_start

        # Gather statistics
        for solver_name in solver_names:
            stats_by_solver[solver_name] = self._gather_solver_stats(
                start_idx, end_idx, solver_name, results_by_solver[solver_name]
            )
            if self.db is not None:
                self.db.store_solver_timing_summary(
                    start_timestep=start_idx,
                    end_timestep=end_idx,
                    solver_name=solver_name,
                    total_time=stats_by_solver[solver_name].get("total_time"),
                    wall_time=solver_wall_time.get(solver_name),
                    internal_timing=solver_internal_timing.get(solver_name, {}),
                )

        if not verbose:
            print(f"[OK] Complete! Results stored in {self.output_db}")

        return {
            "timesteps_completed": end_idx - start_idx + 1,
            "timestep_range": [start_idx, end_idx],
            "solvers_run": solver_names,
            "solver_outputs": results_by_solver,
            "stats_by_solver": stats_by_solver,
            "timing_breakdown": timing_breakdown,
            "solver_wall_time": solver_wall_time,
            "solver_internal_timing": solver_internal_timing,
            "warm_start_mode_by_solver": warm_start_mode_by_solver,
            "selected_warm_start_mode": selected_warm_start_mode,
            "db_path": str(self.output_db),
        }

    def _gather_solver_stats(
        self,
        start_idx: int,
        end_idx: int,
        solver_name: str,
        results: List[OPFResult],
    ) -> Dict[str, Any]:
        """Gather summary statistics for a solver across range."""
        if not results:
            return {}

        solve_times = [r.solve_time for r in results if r.solve_time is not None]
        costs = [r.cost for r in results if r.cost is not None]

        return {
            "num_solves": len(results),
            "total_time": sum(solve_times),
            "avg_time": sum(solve_times) / len(solve_times) if solve_times else 0.0,
            "min_time": min(solve_times) if solve_times else 0.0,
            "max_time": max(solve_times) if solve_times else 0.0,
            "total_cost": sum(costs),
            "avg_cost": sum(costs) / len(costs) if costs else 0.0,
            "min_cost": min(costs) if costs else 0.0,
            "max_cost": max(costs) if costs else 0.0,
        }

    def print_summary(
        self,
        result: Dict[str, Any],
    ) -> None:
        """
        Pretty-print summary of multi-timestep run.
        
        Args:
            result: Dict returned from run_range()
        """
        print("\n" + "=" * 70)
        print("MULTI-TIMESTEP EXECUTION SUMMARY")
        print("=" * 70)

        print(f"\nRange: [{result['timestep_range'][0]}, {result['timestep_range'][1]}]")
        print(f"Timesteps completed: {result['timesteps_completed']}")
        print(f"Database: {result['db_path']}")

        timing = result.get("timing_breakdown", {})
        if timing:
            n_steps = max(result["timesteps_completed"], 1)
            print("\nTiming Breakdown:")
            print(f"  Load case: {timing.get('load_case', 0.0):.3f}s total ({timing.get('load_case', 0.0)/n_steps:.4f}s/step)")
            print(f"  Orient network: {timing.get('orient_network', 0.0):.3f}s total ({timing.get('orient_network', 0.0)/n_steps:.4f}s/step)")
            print(f"  DB write: {timing.get('db_write', 0.0):.3f}s total ({timing.get('db_write', 0.0)/n_steps:.4f}s/step)")
            print(f"  Aggregate computation: {timing.get('compute_aggregates', 0.0):.3f}s")
            print(f"  Loop total: {timing.get('loop_total', 0.0):.3f}s total ({timing.get('loop_total', 0.0)/n_steps:.4f}s/step)")

        for solver_name in result["solvers_run"]:
            stats = result["stats_by_solver"].get(solver_name, {})
            if not stats:
                continue

            solver = self.solvers.get(solver_name)
            print(f"\n{_method_label(solver_name, solver).upper()} Results:")
            if solver is not None and _solver_supports_warm_start(solver):
                warm_start_mode = str(result.get("selected_warm_start_mode", "none")).strip()
                if warm_start_mode.lower() == "none":
                    warm_start_mode = str(result.get("warm_start_mode_by_solver", {}).get(solver_name, "none")).strip()
                label = {
                    "bfsa": "Estimator t",
                    "bfsa t": "BFSA t",
                    "estimator t": "Estimator t",
                    "estimator_pipeline": "Estimator t",
                    "previous": "ACOPF t-1",
                    "acopf t-1": "ACOPF t-1",
                    "socp t-1": "SOCP t-1",
                    "none": "None",
                }.get(str(warm_start_mode).lower(), warm_start_mode or "None")
                print(f"  Warm-start: {label}")
            print(f"  Solves: {stats['num_solves']}")
            print(f"  Total time: {stats['total_time']:.3f}s")
            print(f"  Avg time: {stats['avg_time']:.3f}s (min: {stats['min_time']:.3f}s, "
                  f"max: {stats['max_time']:.3f}s)")

            solver_wall = result.get("solver_wall_time", {}).get(solver_name, 0.0)
            if solver_wall > 0.0:
                print(f"  Wall time in solver call: {solver_wall:.3f}s ({solver_wall / max(stats['num_solves'], 1):.4f}s/solve)")

            internal = result.get("solver_internal_timing", {}).get(solver_name, {})
            if internal:
                print("  Internal timing:")
                for key in sorted(internal.keys()):
                    value = internal[key]
                    print(f"    {key}: {value:.3f}s total ({value / max(stats['num_solves'], 1):.4f}s/solve)")

            print(f"  Total cost: ${stats['total_cost']:.2f}")
            print(f"  Avg cost: ${stats['avg_cost']:.2f} (min: ${stats['min_cost']:.2f}, "
                  f"max: ${stats['max_cost']:.2f})")

        print("\n" + "=" * 70 + "\n")

    def export_results_to_csv(
        self,
        start_idx: int,
        end_idx: int,
        solver_name: str,
        output_dir: Path,
        verbose: bool = True,
    ) -> None:
        """
        Export timestep results to CSV files.
        
        Args:
            start_idx: Start timestep
            end_idx: End timestep
            solver_name: Solver name
            output_dir: Output directory
            verbose: Print progress
        """
        if self.db is None:
            raise RuntimeError("Database not initialized; run_range() must be called first")

        if verbose:
            print(f"Exporting {solver_name} results to {output_dir}...")

        self.db.export_to_csv(start_idx, end_idx, solver_name, output_dir)

        if verbose:
            print(f"[OK] Exported to {output_dir}")

    def query_metric_stats(
        self,
        start_idx: int,
        end_idx: int,
        solver_name: str,
        metric: str = "voltage",
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Query aggregate statistics for a metric.
        
        Args:
            start_idx: Start timestep
            end_idx: End timestep
            solver_name: Solver name
            metric: Metric type ("voltage", "lmp_p", "lmp_q", "flow_p", "flow_q")
            verbose: Print results
            
        Returns:
            Dict with aggregate stats
        """
        if self.db is None:
            raise RuntimeError("Database not initialized; run_range() must be called first")

        stats = self.db.get_timestep_range_stats(start_idx, end_idx, solver_name, metric)

        if verbose and "error" not in stats:
            print(f"\n{solver_name.upper()} - {metric.upper()} Statistics [{start_idx}, {end_idx}]:")
            print(f"  Mean: {stats['mean']:.6f}")
            print(f"  Max: {stats['max']:.6f}")
            print(f"  Min: {stats['min']:.6f}")
            print(f"  Entities: {stats['entities']}\n")

        return stats
