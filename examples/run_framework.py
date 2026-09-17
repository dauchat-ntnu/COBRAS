"""Run the framework by selecting OPF and estimator methods.

With no arguments this opens the shared GUI. With arguments it runs a simple
framework pipeline from the command line.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from comparison import OptimizationComparison
from examples.custom_methods import (
    build_dispatch_estimator,
    build_economic_estimator,
    build_opf_solver,
    build_physical_estimator,
)
from libs.shared import DispatchResult, OptimizationResult, load_case_from_folder, orient_radial_network


PRESETS = {
    "opf": {"opf": "socp", "dispatch": "none", "physical": "none", "economic": "none"},
    "estimators": {
        "opf": "none",
        "dispatch": "dummy_merit_order",
        "physical": "bfsa",
        "economic": "bfsa_dlmp",
    },
    "comparison": {
        "opf": "socp",
        "dispatch": "dummy_merit_order",
        "physical": "bfsa",
        "economic": "bfsa_dlmp",
    },
}


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if not argv:
        from libs.shared.gui_runner import build_solver_gui

        build_solver_gui()
        return 0

    args = _parse_args(argv)
    selected = _selected_methods(args)

    case_folder = Path(args.case_folder)
    if not case_folder.exists():
        print(f"[WARN] Case folder not found: {case_folder}")
        return 1

    case = orient_radial_network(load_case_from_folder(
        case_folder, load_index=args.load_index,
        load_interpretation_mode=args.load_mode,
    ))
    print_case_summary(case)

    dispatch_estimator = build_dispatch_estimator(
        selected["dispatch"],
        merit_order_mode=args.merit_order_mode,
        mcp_active_price=args.mcp_active_price,
        mcp_reactive_price=args.mcp_reactive_price,
    )
    physical_estimator = build_physical_estimator(
        selected["physical"],
        compute_losses=True,
        tol=args.bfsa_tol,
        max_iterations=args.bfsa_max_iter,
        convergence_check=args.bfsa_convergence_check,
        merit_order_mode=args.merit_order_mode,
        mcp_active_price=args.mcp_active_price,
        mcp_reactive_price=args.mcp_reactive_price,
    )
    economic_estimator = build_economic_estimator(
        selected["economic"],
        merit_order_mode=args.merit_order_mode,
        mcp_active_price=args.mcp_active_price,
        mcp_reactive_price=args.mcp_reactive_price,
    )
    opf_solver = build_opf_solver(
        selected["opf"],
        solver_name=args.opf_solver,
        verbose=args.verbose,
        allow_load_shedding=args.allow_load_shedding,
    )

    dispatch = None
    physical = None
    economic = None
    estimator_result = None
    opf_result = None

    if dispatch_estimator is not None:
        print(f"Running dispatch estimator: {dispatch_estimator.name}")
        dispatch = dispatch_estimator.estimate(case)

    if physical_estimator is not None:
        print(f"Running physical estimator: {physical_estimator.name}")
        physical = physical_estimator.estimate(case, dispatch=dispatch)

    if economic_estimator is not None:
        if physical is None:
            raise ValueError("An economic estimator requires a physical state estimate.")
        print(f"Running economic estimator: {economic_estimator.name}")
        economic = economic_estimator.estimate(case, physical, dispatch=dispatch)

    if physical is not None:
        estimator_result = OptimizationResult(
            physical_state=physical,
            dispatch=dispatch or DispatchResult(),
            economic_state=economic,
            method_name="estimator_pipeline",
        )
        print_result_summary("Estimator pipeline", estimator_result)

    if opf_solver is not None:
        print(f"Running OPF solver: {opf_solver.name}")
        opf_result = opf_solver.solve_optimization(case, warm_start=physical)
        print_result_summary("OPF", opf_result)

    if opf_result is not None and estimator_result is not None:
        print_comparison_summary(OptimizationComparison(candidate=estimator_result, reference=opf_result))

    return 0


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="comparison")
    parser.add_argument("--case-folder", default=str(ROOT / "data" / "Methodology Paper" / "80_bus_base"))
    parser.add_argument("--load-index", type=int, default=0)
    parser.add_argument("--load-mode", choices=["auto", "normalized", "absolute"], default="absolute")
    parser.add_argument("--opf", choices=["none", "socp"], default=None)
    parser.add_argument("--dispatch", choices=["none", "dummy_merit_order"], default=None)
    parser.add_argument("--physical", choices=["none", "bfsa"], default=None)
    parser.add_argument("--economic", choices=["none", "bfsa_dlmp"], default=None)
    parser.add_argument("--opf-solver", default="mosek")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--allow-load-shedding", action="store_true")
    parser.add_argument("--bfsa-tol", type=float, default=1e-3)
    parser.add_argument("--bfsa-max-iter", type=int, default=100)
    parser.add_argument("--bfsa-convergence-check", choices=["voltage", "current", "both"], default="both")
    parser.add_argument("--merit-order-mode", choices=["dummy", "mcp"], default="dummy")
    parser.add_argument("--mcp-active-price", type=float, default=None)
    parser.add_argument("--mcp-reactive-price", type=float, default=None)
    return parser.parse_args(argv)


def _selected_methods(args: argparse.Namespace) -> dict[str, str]:
    selected = dict(PRESETS[args.preset])
    for family in ("opf", "dispatch", "physical", "economic"):
        override = getattr(args, family)
        if override is not None:
            selected[family] = override
    return selected


def print_case_summary(case) -> None:
    print("Loaded radial case:")
    print(f"  Buses: {len(case.buses)}")
    print(f"  Branches: {len(case.branches)}")
    print(f"  Generators: {len(case.generators)}")
    print(f"  Loads: {len(case.loads)}")


def print_result_summary(label: str, result: OptimizationResult) -> None:
    print(f"{label} result:")
    print(f"  Method: {result.method_name}")
    if result.solve_time is not None:
        print(f"  Solve time: {result.solve_time:.3f} s")
    if result.dispatch.objective is not None:
        print(f"  Objective: {result.dispatch.objective:.3f}")
    print(f"  Voltage entries: {len(result.physical_state.voltages)}")
    print(f"  Flow entries: {len(result.physical_state.flows)}")
    if result.economic_state is not None:
        print(f"  Lambda_p entries: {len(result.economic_state.lambda_p)}")


def print_comparison_summary(comparison: OptimizationComparison) -> None:
    print("Estimator vs OPF comparison:")
    for group, metrics in comparison.summary_stats().items():
        print(f"  {group}:")
        for metric, stats in metrics.items():
            print(f"    {metric}: count={stats['count']}, abs_mean={stats['abs_mean']}, max_abs={stats['max_abs']}")


if __name__ == "__main__":
    raise SystemExit(main())
