"""
Phase 5: Comparison Framework
==============================

This module provides generic benchmark tools for comparing
any two OPF solvers (SOCP, BFSA, Pandapower, etc.).
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from pathlib import Path
from scipy import stats as scipy_stats

from libs.shared import PowerFlowCase, OPFResult


class SolverComparison:
    """Result container for comparing two solver outputs."""

    FLOW_DEVIATION_THRESHOLD = 1e-2
    
    def __init__(
        self,
        case: PowerFlowCase,
        result1: OPFResult,
        result2: OPFResult,
        names: Tuple[str, str] = ("Solver 1", "Solver 2"),
        flow_deviation_threshold: float = FLOW_DEVIATION_THRESHOLD,
    ):
        """
        Args:
            case: Original PowerFlowCase
            result1, result2: OPFResult from two solvers
            names: Names of solvers for display
        """
        self.case = case
        self.result1 = result1
        self.result2 = result2
        self.names = names
        self.flow_deviation_threshold = float(flow_deviation_threshold)
        self.deviations = self._compute_deviations()

    def _ordered_results_for_deviation(self) -> Tuple[OPFResult, OPFResult]:
        """Return numerator/reference order for percent deviations."""
        normalized_names = [str(name).strip().lower() for name in self.names]
        if "bfsa" in normalized_names and "socp" in normalized_names:
            bfsa_index = normalized_names.index("bfsa")
            socp_index = normalized_names.index("socp")
            results = (self.result1, self.result2)
            return results[bfsa_index], results[socp_index]
        return self.result1, self.result2

    def _filter_small_flow_deviation(self, s1: float, s2: float) -> bool:
        """Return True when both flows are too small to produce a meaningful deviation."""
        threshold = self.flow_deviation_threshold
        return max(abs(float(s1)), abs(float(s2))) < threshold
    
    def _compute_deviations(self) -> Dict:
        """Compute % deviations in key outputs."""
        numerator_result, reference_result = self._ordered_results_for_deviation()
        
        # Voltage deviations
        v_dev = {}
        for bus_id in numerator_result.voltages:
            if bus_id in reference_result.voltages:
                numerator = numerator_result.voltages[bus_id]
                reference = reference_result.voltages[bus_id]
                if reference != 0:
                    v_dev[bus_id] = 100 * (numerator - reference) / abs(reference)
                else:
                    v_dev[bus_id] = 0.0
        
        # DLMP Active deviations
        lmp_p_dev = {}
        for bus_id in numerator_result.duals_p:
            if bus_id in reference_result.duals_p:
                numerator = numerator_result.duals_p[bus_id]
                reference = reference_result.duals_p[bus_id]
                if reference != 0:
                    lmp_p_dev[bus_id] = 100 * (numerator - reference) / abs(reference)
                else:
                    lmp_p_dev[bus_id] = 0.0
        
        # DLMP Reactive deviations
        lmp_q_dev = {}
        if hasattr(numerator_result, 'duals_q') and hasattr(reference_result, 'duals_q'):
            for bus_id in numerator_result.duals_q:
                if bus_id in reference_result.duals_q:
                    numerator = numerator_result.duals_q[bus_id]
                    reference = reference_result.duals_q[bus_id]
                    if reference != 0:
                        lmp_q_dev[bus_id] = 100 * (numerator - reference) / abs(reference)
                    else:
                        lmp_q_dev[bus_id] = 0.0
        
        # Flow deviations
        flow_dev = {}
        flow_p_dev = {}
        flow_q_dev = {}
        for edge in numerator_result.flows:
            if edge in reference_result.flows:
                p_num, q_num = numerator_result.flows[edge]
                p_ref, q_ref = reference_result.flows[edge]
                s_num = (p_num**2 + q_num**2)**0.5
                s_ref = (p_ref**2 + q_ref**2)**0.5
                if self._filter_small_flow_deviation(s_num, s_ref):
                    flow_dev[edge] = 0.0
                elif s_ref != 0:
                    flow_dev[edge] = 100 * (s_num - s_ref) / abs(s_ref)
                else:
                    flow_dev[edge] = 0.0

                if self._filter_small_flow_deviation(p_num, p_ref):
                    flow_p_dev[edge] = 0.0
                elif p_ref != 0:
                    flow_p_dev[edge] = 100 * (p_num - p_ref) / abs(p_ref)
                else:
                    flow_p_dev[edge] = 0.0

                if self._filter_small_flow_deviation(q_num, q_ref):
                    flow_q_dev[edge] = 0.0
                elif q_ref != 0:
                    flow_q_dev[edge] = 100 * (q_num - q_ref) / abs(q_ref)
                else:
                    flow_q_dev[edge] = 0.0
        
        return {
            "voltage_dev": v_dev,
            "lmp_p_dev": lmp_p_dev,
            "lmp_q_dev": lmp_q_dev,
            "flow_dev": flow_dev,
            "flow_p_dev": flow_p_dev,
            "flow_q_dev": flow_q_dev,
        }
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert deviations to DataFrame for export."""
        
        data = []
        for bus_id in self.deviations["lmp_p_dev"]:
            data.append({
                "bus": bus_id,
                "v_dev_%": self.deviations["voltage_dev"].get(bus_id, None),
                "lmp_p_dev_%": self.deviations["lmp_p_dev"].get(bus_id, None),
            })
        
        return pd.DataFrame(data)
    
    def _compute_comprehensive_stats(self, devs: List[float]) -> Dict:
        """
        Compute comprehensive statistics for a list of deviations.
        
        Args:
            devs: List of deviation values
        
        Returns:
            Dict with mean, median, std, min, max, abs_mean, and percentiles
        """
        if not devs:
            return {
                "count": 0,
                "mean": 0,
                "median": 0,
                "std": 0,
                "min": 0,
                "max": 0,
                "abs_mean": 0,
                "p25": 0,
                "p75": 0,
                "p95": 0,
            }
        
        devs_array = np.array(devs)
        abs_devs = np.abs(devs_array)
        
        return {
            "count": len(devs),
            "mean": float(np.mean(devs_array)),
            "median": float(np.median(devs_array)),
            "std": float(np.std(devs_array)),
            "min": float(np.min(devs_array)),
            "max": float(np.max(devs_array)),
            "abs_mean": float(np.mean(abs_devs)),
            "abs_median": float(np.median(abs_devs)),
            "abs_max": float(np.max(abs_devs)),
            "p25": float(np.percentile(devs_array, 25)),
            "p75": float(np.percentile(devs_array, 75)),
            "p95": float(np.percentile(devs_array, 95)),
        }
    
    def summary_stats(self) -> Dict:
        """Compute comprehensive summary statistics."""
        
        v_devs = list(self.deviations["voltage_dev"].values())
        lmp_p_devs = list(self.deviations["lmp_p_dev"].values())
        lmp_q_devs = list(self.deviations["lmp_q_dev"].values())
        flow_devs = list(self.deviations["flow_dev"].values())
        flow_p_devs = list(self.deviations["flow_p_dev"].values())
        flow_q_devs = list(self.deviations["flow_q_dev"].values())
        
        return {
            "voltage": self._compute_comprehensive_stats(v_devs),
            "lmp_p": self._compute_comprehensive_stats(lmp_p_devs),
            "lmp_q": self._compute_comprehensive_stats(lmp_q_devs),
            "flow": self._compute_comprehensive_stats(flow_devs),
            "flow_p": self._compute_comprehensive_stats(flow_p_devs),
            "flow_q": self._compute_comprehensive_stats(flow_q_devs),
            "objectives": {
                self.names[0]: self.result1.cost,
                self.names[1]: self.result2.cost,
                "gap": (
                    (self.result2.cost - self.result1.cost)
                    if self.result1.cost is not None and self.result2.cost is not None
                    else None
                ),
            },
            "solve_times": {
                self.names[0]: self.result1.solve_time,
                self.names[1]: self.result2.solve_time,
            },
            "speedup": (
                (self.result1.solve_time / self.result2.solve_time)
                if self.result2.solve_time not in (None, 0)
                else (float("inf") if self.result2.solve_time == 0 and self.result1.solve_time is not None else None)
            ),
        }


class Benchmark:
    """Generic benchmark framework for comparing any two OPF solvers."""
    
    ERROR_TOLERANCE = 1e-3  # Errors below this threshold display as 0
    
    def __init__(self, solver1, solver2, names: Tuple[str, str] = ("Solver 1", "Solver 2")):
        """
        Args:
            solver1, solver2: Solver objects with .solve(case) -> OPFResult
            names: Display names for solvers
        """
        self.solver1 = solver1
        self.solver2 = solver2
        self.names = names
        self.comparisons = []

        self.metrics = ["voltage", "lmp_p", "lmp_q", "flow", "flow_p", "flow_q"]
        self.stats = {metric: [] for metric in self.metrics}
    
    def run_on_case(self, case: PowerFlowCase) -> SolverComparison:
        """Run both solvers on single case and compare."""
        
        result1 = self.solver1.solve(case)
        result2 = self.solver2.solve(case)
        
        comparison = SolverComparison(case, result1, result2, self.names)
        self.comparisons.append(comparison)
        
        return comparison
    
    def run_sweep(
        self,
        network_paths: List[str],
        load_indices: Optional[List[int]] = None,
    ) -> List[SolverComparison]:
        """
        Run comparison over multiple networks/scenarios.
        
        Args:
            network_paths: List of network data folder paths
            load_indices: Time indices to test (default: [0])
        
        Returns:
            List of SolverComparison objects
        """
        
        from libs.shared import load_case_from_folder, orient_radial_network
        
        load_indices = load_indices or [0]
        results = []
        
        for net_path in network_paths:
            for load_idx in load_indices:
                case = load_case_from_folder(
                    Path(net_path),
                    load_index=load_idx,
                )
                case = orient_radial_network(case)
                
                comparison = self.run_on_case(case)
                results.append(comparison)
        
        return results
    
    def _apply_tolerance(self, value: float) -> float:
        """Apply tolerance threshold: errors < 1e-3 display as 0."""
        if value is None or np.isnan(value) or np.isinf(value):
            return value
        return 0.0 if abs(value) < self.ERROR_TOLERANCE else value
    
    def _format_value(self, value: float, decimals: int = 3) -> float:
        """Format numeric value to specified decimal places, preserving None/NaN/inf."""
        if value is None or np.isnan(value) or np.isinf(value):
            return value
        return round(float(value), decimals)

    def _solver_timing_fields(self, solver_name: str, result: OPFResult) -> Dict[str, Any]:
        """Return optional wall/internal timing columns for a solver result."""
        fields: Dict[str, Any] = {}
        convergence_info = result.convergence_info if isinstance(result.convergence_info, dict) else {}

        wall_time = convergence_info.get("wall_time")
        if wall_time is not None:
            fields[f"{solver_name}_wall_time_s"] = self._format_value(wall_time)

        timing = convergence_info.get("timing", {})
        if isinstance(timing, dict):
            for key in ("build", "extract", "optimize", "total"):
                value = timing.get(key)
                if value is not None:
                    fields[f"{solver_name}_internal_{key}_time_s"] = self._format_value(value)

        return fields
    
    def export_results(self, output_dir: str) -> None:
        """
        Export comparison results to CSV files (errors and times separated).
        
        Args:
            output_dir: Directory to write files
        """
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Export individual comparisons
        all_data = []
        for i, comp in enumerate(self.comparisons):
            df = comp.to_dataframe()
            df["case"] = i
            all_data.append(df)
        
        if all_data:
            combined = pd.concat(all_data, ignore_index=True)
            combined.to_csv(output_dir / "deviations.csv", index=False)
        
        # Export error metrics and times separately
        error_df, times_df = self.generate_summary_tables_split()
        error_df.to_csv(output_dir / "summary_errors.csv", index=False)
        times_df.to_csv(output_dir / "summary_times.csv", index=False)
        
        print(f"[OK] Results exported to {output_dir}")
    
    def generate_summary_table(self) -> pd.DataFrame:
        """
        Generate comprehensive summary table with statistics (all numeric values to 3 decimals).
        
        Returns:
            DataFrame with columns: metric, count, mean, median, std, min, max, 
                                    abs_mean, p25, p75, p95, solve_time_s1, solve_time_s2, speedup
        """
        summary_rows = []
        
        for i, comp in enumerate(self.comparisons):
            stats = comp.summary_stats()
            
            for metric_name in self.metrics:
                if metric_name not in stats:
                    continue
                    
                metric_stats = stats[metric_name]
                if metric_stats["count"] == 0:
                    continue
                
                row = {
                    "case_id": i,
                    "metric": metric_name,
                    "count": metric_stats["count"],
                    "mean_%": self._format_value(metric_stats["mean"]),
                    "median_%": self._format_value(metric_stats["median"]),
                    "std_%": self._format_value(metric_stats["std"]),
                    "min_%": self._format_value(metric_stats["min"]),
                    "max_%": self._format_value(metric_stats["max"]),
                    "abs_mean_%": self._format_value(metric_stats["abs_mean"]),
                    "abs_median_%": self._format_value(metric_stats["abs_median"]),
                    "abs_max_%": self._format_value(metric_stats["abs_max"]),
                    "p25_%": self._format_value(metric_stats["p25"]),
                    "p75_%": self._format_value(metric_stats["p75"]),
                    "p95_%": self._format_value(metric_stats["p95"]),
                }
                
                # Add solver times on first metric only
                if metric_name == "voltage":
                    row[f"{comp.names[0]}_time_s"] = self._format_value(stats["solve_times"].get(comp.names[0]))
                    row[f"{comp.names[1]}_time_s"] = self._format_value(stats["solve_times"].get(comp.names[1]))
                    row[f"{comp.names[0]}_total_time_s"] = self._format_value(stats["solve_times"].get(comp.names[0]))
                    row[f"{comp.names[1]}_total_time_s"] = self._format_value(stats["solve_times"].get(comp.names[1]))
                    row.update(self._solver_timing_fields(comp.names[0], comp.result1))
                    row.update(self._solver_timing_fields(comp.names[1], comp.result2))
                    row[f"{comp.names[0]}_objective"] = self._format_value(stats["objectives"].get(comp.names[0]))
                    row[f"{comp.names[1]}_objective"] = self._format_value(stats["objectives"].get(comp.names[1]))
                    row["objective_gap"] = self._format_value(stats["objectives"].get("gap"))
                    row["speedup"] = self._format_value(stats["speedup"])
                
                summary_rows.append(row)
        
        return pd.DataFrame(summary_rows)
    
    def generate_summary_tables_split(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate separate summary tables for error metrics and times with tolerance applied.
        All numeric values formatted to 3 decimal places.
        
        Returns:
            Tuple of (error_metrics_df, times_df)
        """
        error_rows = []
        time_rows = []
        
        for i, comp in enumerate(self.comparisons):
            stats = comp.summary_stats()
            
            # Error metrics table
            for metric_name in self.metrics:
                if metric_name not in stats:
                    continue
                    
                metric_stats = stats[metric_name]
                if metric_stats["count"] == 0:
                    continue
                
                error_row = {
                    "case_id": i,
                    "metric": metric_name,
                    "count": metric_stats["count"],
                    "mean_%": self._format_value(self._apply_tolerance(metric_stats["mean"])),
                    "median_%": self._format_value(self._apply_tolerance(metric_stats["median"])),
                    "std_%": self._format_value(self._apply_tolerance(metric_stats["std"])),
                    "min_%": self._format_value(self._apply_tolerance(metric_stats["min"])),
                    "max_%": self._format_value(self._apply_tolerance(metric_stats["max"])),
                    "abs_mean_%": self._format_value(self._apply_tolerance(metric_stats["abs_mean"])),
                    "abs_median_%": self._format_value(self._apply_tolerance(metric_stats["abs_median"])),
                    "abs_max_%": self._format_value(self._apply_tolerance(metric_stats["abs_max"])),
                    "p25_%": self._format_value(self._apply_tolerance(metric_stats["p25"])),
                    "p75_%": self._format_value(self._apply_tolerance(metric_stats["p75"])),
                    "p95_%": self._format_value(self._apply_tolerance(metric_stats["p95"])),
                }
                error_rows.append(error_row)
            
            # Times table (only once per case)
            time_row = {
                "case_id": i,
                f"{comp.names[0]}_time_s": self._format_value(stats["solve_times"].get(comp.names[0])),
                f"{comp.names[1]}_time_s": self._format_value(stats["solve_times"].get(comp.names[1])),
                f"{comp.names[0]}_total_time_s": self._format_value(stats["solve_times"].get(comp.names[0])),
                f"{comp.names[1]}_total_time_s": self._format_value(stats["solve_times"].get(comp.names[1])),
                f"{comp.names[0]}_objective": self._format_value(stats["objectives"].get(comp.names[0])),
                f"{comp.names[1]}_objective": self._format_value(stats["objectives"].get(comp.names[1])),
                "objective_gap": self._format_value(stats["objectives"].get("gap")),
                "speedup": self._format_value(stats["speedup"]),
            }
            time_row.update(self._solver_timing_fields(comp.names[0], comp.result1))
            time_row.update(self._solver_timing_fields(comp.names[1], comp.result2))
            time_rows.append(time_row)
        
        error_df = pd.DataFrame(error_rows)
        time_df = pd.DataFrame(time_rows)
        
        return error_df, time_df
    
    def generate_aggregated_comparison_table(self) -> pd.DataFrame:
        """
        Generate aggregated comparison table across all cases.
        Useful for publication showing overall performance.
        
        Returns:
            DataFrame with metrics and aggregated statistics
        """
        all_stats = {metric: [] for metric in self.metrics}
        
        solve_times = {self.names[0]: [], self.names[1]: []}
        
        for comp in self.comparisons:
            stats = comp.summary_stats()
            
            for metric in all_stats:
                if stats[metric]["count"] > 0:
                    # Store raw deviations for aggregation
                    devs = list(comp.deviations[f"{metric}_dev"].values())
                    all_stats[metric].extend(devs)
            
            solve_times[self.names[0]].append(stats["solve_times"].get(self.names[0]))
            solve_times[self.names[1]].append(stats["solve_times"].get(self.names[1]))
        
        # Compute aggregate stats
        result_rows = []
        for metric_name in self.metrics:
            if not all_stats[metric_name]:
                continue
            
            comp_obj = self.comparisons[0]  # Use first comparison for helper function
            metric_stats = comp_obj._compute_comprehensive_stats(all_stats[metric_name])
            
            result_rows.append({
                "metric": metric_name,
                "count": metric_stats["count"],
                "mean_%": self._format_value(metric_stats["mean"]),
                "median_%": self._format_value(metric_stats["median"]),
                "std_%": self._format_value(metric_stats["std"]),
                "min_%": self._format_value(metric_stats["min"]),
                "max_%": self._format_value(metric_stats["max"]),
                "abs_mean_%": self._format_value(metric_stats["abs_mean"]),
                "abs_median_%": self._format_value(metric_stats["abs_median"]),
                "abs_max_%": self._format_value(metric_stats["abs_max"]),
                "p25_%": self._format_value(metric_stats["p25"]),
                "p75_%": self._format_value(metric_stats["p75"]),
                "p95_%": self._format_value(metric_stats["p95"]),
                f"{self.names[0]}_objective": self._format_value(np.mean([c.result1.cost for c in self.comparisons if c.result1.cost is not None])),
                f"{self.names[1]}_objective": self._format_value(np.mean([c.result2.cost for c in self.comparisons if c.result2.cost is not None])),
            })
        
        # Add compute times
        times_1 = [t for t in solve_times[self.names[0]] if t is not None]
        times_2 = [t for t in solve_times[self.names[1]] if t is not None]
        
        if times_1 and times_2:
            speedup = np.mean(times_1) / np.mean(times_2) if np.mean(times_2) > 0 else float("inf")
            result_rows.append({
                "metric": "compute_time",
                "count": len(times_1),
                "mean_%": self._format_value(np.mean(times_1)),
                "median_%": self._format_value(np.median(times_1)),
                "std_%": self._format_value(np.std(times_1)),
                "min_%": self._format_value(np.min(times_1)),
                "max_%": self._format_value(np.max(times_1)),
                "abs_mean_%": self._format_value(np.mean(times_2)),
                "abs_median_%": self._format_value(np.median(times_2)),
                "abs_max_%": self._format_value(np.max(times_2)),
                "p25_%": self._format_value(speedup),
                "p75_%": None,
                "p95_%": None,
            })
        
        return pd.DataFrame(result_rows)
    
    def generate_aggregated_comparison_tables_split(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate separate aggregated comparison tables for error metrics and times with tolerance applied.
        All numeric values formatted to 3 decimal places.
        Useful for publication showing overall performance across all cases.
        
        Returns:
            Tuple of (error_metrics_df, times_df)
        """
        all_stats = {metric: [] for metric in self.metrics}
        
        solve_times = {self.names[0]: [], self.names[1]: []}
        
        for comp in self.comparisons:
            stats = comp.summary_stats()
            
            for metric in all_stats:
                if stats[metric]["count"] > 0:
                    # Store raw deviations for aggregation
                    devs = list(comp.deviations[f"{metric}_dev"].values())
                    all_stats[metric].extend(devs)
            
            solve_times[self.names[0]].append(stats["solve_times"].get(self.names[0]))
            solve_times[self.names[1]].append(stats["solve_times"].get(self.names[1]))
        
        # Error metrics table
        error_rows = []
        for metric_name in self.metrics:
            if not all_stats[metric_name]:
                continue
            
            comp_obj = self.comparisons[0]  # Use first comparison for helper function
            metric_stats = comp_obj._compute_comprehensive_stats(all_stats[metric_name])
            
            error_rows.append({
                "metric": metric_name,
                "count": metric_stats["count"],
                "mean_%": self._format_value(self._apply_tolerance(metric_stats["mean"])),
                "median_%": self._format_value(self._apply_tolerance(metric_stats["median"])),
                "std_%": self._format_value(self._apply_tolerance(metric_stats["std"])),
                "min_%": self._format_value(self._apply_tolerance(metric_stats["min"])),
                "max_%": self._format_value(self._apply_tolerance(metric_stats["max"])),
                "abs_mean_%": self._format_value(self._apply_tolerance(metric_stats["abs_mean"])),
                "abs_median_%": self._format_value(self._apply_tolerance(metric_stats["abs_median"])),
                "abs_max_%": self._format_value(self._apply_tolerance(metric_stats["abs_max"])),
                "p25_%": self._format_value(self._apply_tolerance(metric_stats["p25"])),
                "p75_%": self._format_value(self._apply_tolerance(metric_stats["p75"])),
                "p95_%": self._format_value(self._apply_tolerance(metric_stats["p95"])),
            })
        
        error_df = pd.DataFrame(error_rows)
        
        # Times table
        time_rows = []
        times_1 = [t for t in solve_times[self.names[0]] if t is not None]
        times_2 = [t for t in solve_times[self.names[1]] if t is not None]
        
        if times_1 and times_2:
            speedup = np.mean(times_1) / np.mean(times_2) if np.mean(times_2) > 0 else float("inf")
            time_row = {
                "metric": "compute_time",
                "count": len(times_1),
                f"{self.names[0]}_mean_s": self._format_value(np.mean(times_1)),
                f"{self.names[0]}_median_s": self._format_value(np.median(times_1)),
                f"{self.names[0]}_std_s": self._format_value(np.std(times_1)),
                f"{self.names[0]}_min_s": self._format_value(np.min(times_1)),
                f"{self.names[0]}_max_s": self._format_value(np.max(times_1)),
                f"{self.names[1]}_mean_s": self._format_value(np.mean(times_2)),
                f"{self.names[1]}_median_s": self._format_value(np.median(times_2)),
                f"{self.names[1]}_std_s": self._format_value(np.std(times_2)),
                f"{self.names[1]}_min_s": self._format_value(np.min(times_2)),
                f"{self.names[1]}_max_s": self._format_value(np.max(times_2)),
                f"{self.names[0]}_objective": self._format_value(np.mean([c.result1.cost for c in self.comparisons if c.result1.cost is not None])),
                f"{self.names[1]}_objective": self._format_value(np.mean([c.result2.cost for c in self.comparisons if c.result2.cost is not None])),
                "speedup": self._format_value(speedup),
            }
            first_comp = self.comparisons[0]
            time_row.update(self._solver_timing_fields(self.names[0], first_comp.result1))
            time_row.update(self._solver_timing_fields(self.names[1], first_comp.result2))
            time_rows.append(time_row)
        
        time_df = pd.DataFrame(time_rows)
        
        return error_df, time_df
