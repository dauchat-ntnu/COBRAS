"""Unified GUI runner for SOCP, BFSA, and comparison examples."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import tempfile
from pathlib import Path
import sys
import traceback
import webbrowser
import re
import pandas as pd
import threading
import subprocess
import os 

from .io import load_case_from_folder
from .network import orient_radial_network
from .multi_timestep_solver import MultiTimestepSolver
from .gui_method_config import methods_config_from_legacy_job
from .data import OPFResult
from .results import OptimizationResult, optimization_result_from_opf_result
from libs.methods import EstimatorPipelineSolver, build_method, list_method_specs
from libs.methods.execution import run_estimator_stages
from libs.shared.results import opf_result_from_optimization
from libs.visualization import (
    build_plot_inputs,
    verify_load_balance,
    create_combined_plot,
    create_deviation_plot,
    create_merit_order_plot,
    create_white_network_plot,
    export_network_plot,
    export_side_by_side_dashboard,
    set_line_color_mode,
    set_visual_theme,
)
from comparison import OptimizationComparison

default_BFSA_MCPA = 20
default_BFSA_MCPR = 0.15


ROOT = Path(__file__).resolve().parents[2]

data_path = Path(os.environ.get("COBRAS_DATA_DIR", ROOT / "data" / "Methodology Paper"))


def SOCPSolver(*args, **kwargs):
    """Lazily instantiate the SOCP solver so GUI imports do not require Pyomo."""
    from libs.methods.opf.socp import SOCPSolver as _SOCPSolver

    return _SOCPSolver(*args, **kwargs)


class _TeeStream:
    def __init__(self, *streams) -> None:
        self._streams = streams

    def write(self, text: str) -> int:
        written = 0
        for stream in self._streams:
            written = stream.write(text)
            stream.flush()
        return written

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


def _launch_plot_gui_async() -> None:
    """Launch the plot generator GUI in a separate process."""
    def _run_plot_gui() -> None:
        try:
            # Launch as subprocess to avoid Tkinter threading issues
            python_exe = sys.executable
            script = ROOT / "examples" / "generate_plots.py"
            
            if script.exists():
                print(f"Launching plot generator GUI from {script}...", file=sys.stderr)
                subprocess.Popen(
                    [python_exe, str(script)],
                    cwd=str(ROOT),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            else:
                print(f"Warning: Plot generator script not found at {script}", file=sys.stderr)
        except Exception as ex:
            print(f"Warning: Could not launch plot GUI: {ex}", file=sys.stderr)
    
    # Run in daemon thread so it doesn't block the main solver runner
    plot_thread = threading.Thread(target=_run_plot_gui, daemon=True)
    plot_thread.start()

MODE_SOCP = "socp"
MODE_BFSA = "bfsa"
MODE_COMPARISON = "comparison"

MODE_LABEL_TO_KEY = {
    "OPF": MODE_SOCP,
    "Estimator pipeline": MODE_BFSA,
    "Comparison": MODE_COMPARISON,
}
MODE_KEY_TO_LABEL = {value: key for key, value in MODE_LABEL_TO_KEY.items()}

LOAD_INTERPRETATION_LABEL_TO_KEY = {
    "Auto (detect from Pd/Qd)": "auto",
    "Normalized (p/q as factors)": "normalized",
    "Absolute (p/q are demands)": "absolute",
}

LINE_COLOR_MODE_LABEL_TO_KEY = {
    "Flow on base MVA": "flow_pu",
    "Line rating (%)": "loading_pct",
}
LINE_COLOR_MODE_KEY_TO_LABEL = {value: key for key, value in LINE_COLOR_MODE_LABEL_TO_KEY.items()}

BFSA_MERIT_ORDER_LABEL_TO_KEY = {
    "Dummy merit-order clearing": "dummy",
    "Use market clearing prices (MCPA/MCPR)": "mcp",
}
BFSA_MERIT_ORDER_KEY_TO_LABEL = {value: key for key, value in BFSA_MERIT_ORDER_LABEL_TO_KEY.items()}

BFSA_CONVERGENCE_LABEL_TO_KEY = {
    "Voltage": "voltage",
    "Current-squared / losses": "current",
    "Both voltage and current": "both",
}
BFSA_CONVERGENCE_KEY_TO_LABEL = {value: key for key, value in BFSA_CONVERGENCE_LABEL_TO_KEY.items()}


def _filename_token(value: object, fallback: str = "none") -> str:
    token = str(value if value is not None else fallback).strip().lower()
    token = re.sub(r"\s+", "_", token)
    token = re.sub(r"[^a-z0-9_.-]+", "_", token)
    token = re.sub(r"_+", "_", token).strip("_.-")
    return token or fallback


def _timestep_database_filename(
    start_idx: int,
    end_idx: int,
    *,
    warm_start_mode: object,
    allow_load_shedding: bool,
    load_mode: object,
) -> str:
    timestamp = datetime.now().strftime("%Hh%M_%d%m")
    warm_start = _filename_token(warm_start_mode)
    shedding = "shedding" if allow_load_shedding else "no_shedding"
    load_mode_token = _filename_token(load_mode)
    return f"{timestamp}_{start_idx}_{end_idx}_{warm_start}_{shedding}_{load_mode_token}.db"


def _list_files(folder: Path, suffixes: tuple[str, ...]) -> list[str]:
    if not folder.exists() or not folder.is_dir():
        return []
    return sorted(
        f.name for f in folder.iterdir() if f.is_file() and f.suffix.lower() in suffixes
    )


def _list_base_mva_files(folder: Path) -> list[str]:
    if not folder.exists() or not folder.is_dir():
        return []
    files = []
    for f in folder.iterdir():
        if not f.is_file():
            continue
        if f.suffix.lower() in {"", ".csv"}:
            files.append(f.name)
    return sorted(files)


def _pick_default(options: list[str], exact: str, prefix: str) -> str:
    if exact in options:
        return exact
    prefixed = [name for name in options if name.startswith(prefix)]
    if prefixed:
        return prefixed[0]
    return options[0] if options else ""


def _pick_default_bus_mapping(options: list[str]) -> str:
    """Prefer standard bus mapping filenames, then fall back to close matches."""
    if not options:
        return ""

    lower_to_name = {name.lower(): name for name in options}
    if "bus_mapping.csv" in lower_to_name:
        return lower_to_name["bus_mapping.csv"]
    if "bus_maping.csv" in lower_to_name:
        return lower_to_name["bus_maping.csv"]

    for name in options:
        stem_lower = Path(name).stem.lower()
        if stem_lower.startswith("bus_mapping"):
            return name
    for name in options:
        stem_lower = Path(name).stem.lower()
        if stem_lower.startswith("bus_maping"):
            return name
    return ""


def _get_network_results_dir(network_name: str) -> Path:
    return os.path.join(data_path / network_name / "results")


def _get_categorized_output_dir(output_path: Path, category: str) -> Path:
    """Return the folder for a categorized artifact near the selected output."""
    base = Path(output_path)
    if base.name.lower() == "results":
        return base.parent / category
    return base / category


def _build_case_from_job(job: dict, load_index: int | None = None):
    input_folder = Path(job["input_folder"])
    case = load_case_from_folder(
        input_folder,
        load_index=load_index,
        mpc_base_mva_filename=job.get("mpc_base_mva_filename", "mpc_base_mva"),
        mpc_bus_filename=job["mpc_bus_filename"],
        mpc_branch_filename=job["mpc_branch_filename"],
        p_load_filename=job["p_load_filename"],
        q_load_filename=job["q_load_filename"],
        bus_mapping_filename=job.get("bus_mapping_filename") or None,
        generators_filename=job.get("generators_filename", "generators.xlsx"),
        load_interpretation_mode=job.get("load_interpretation_mode", "auto"),
    )
    return orient_radial_network(case)


def _run_estimator_pipeline_result(
    case,
    job: dict,
    *,
    show_net_bus_plot: bool = False,
    show_merit_order_plot: bool = False,
) -> OptimizationResult:
    """Run dispatch, physical, and economic estimators through the method registry."""
    methods = job.get("methods") or methods_config_from_legacy_job(job)
    dispatch_config = methods.get("dispatch", {})
    physical_config = methods.get("physical", {})
    economic_config = methods.get("economic", {})

    if not bool(dispatch_config.get("enabled", False)):
        raise ValueError("Estimator pipeline requires dispatch estimator to be enabled.")
    if not bool(physical_config.get("enabled", False)):
        raise ValueError("Estimator pipeline requires physical estimator to be enabled.")
    if not bool(economic_config.get("enabled", False)):
        raise ValueError("Estimator pipeline requires economic estimator to be enabled.")

    dispatch_estimator = build_method(
        "dispatch",
        str(dispatch_config.get("method", "dummy_merit_order")),
        dispatch_config.get("options") or {},
    )
    physical_estimator = build_method(
        "physical",
        str(physical_config.get("method", "bfsa")),
        physical_config.get("options") or {},
    )
    economic_estimator = build_method(
        "economic",
        str(economic_config.get("method", "bfsa_dlmp")),
        economic_config.get("options") or {},
    )

    return run_estimator_stages(
        case, dispatch_estimator, physical_estimator, economic_estimator,
        show_net_bus_plot=show_net_bus_plot, show_merit_order_plot=show_merit_order_plot,
    )


def _optimization_result_to_opf_result(result: OptimizationResult) -> OPFResult:
    """Compatibility wrapper for the shared result adapter."""
    return opf_result_from_optimization(result)


def _estimator_pipeline_result_as_opf_result(
    case,
    job: dict,
    *,
    show_net_bus_plot: bool = False,
    show_merit_order_plot: bool = False,
) -> OPFResult:
    """Run the estimator pipeline and adapt the result for legacy plotting helpers."""
    return _optimization_result_to_opf_result(
        _run_estimator_pipeline_result(
            case,
            job,
            show_net_bus_plot=show_net_bus_plot,
            show_merit_order_plot=show_merit_order_plot,
        )
    )


def _build_estimator_pipeline_solver_from_methods(methods: dict) -> EstimatorPipelineSolver:
    dispatch_config = methods.get("dispatch", {})
    physical_config = methods.get("physical", {})
    economic_config = methods.get("economic", {})
    return EstimatorPipelineSolver(
        dispatch_method=str(dispatch_config.get("method", "dummy_merit_order")),
        physical_method=str(physical_config.get("method", "bfsa")),
        economic_method=str(economic_config.get("method", "bfsa_dlmp")),
        dispatch_options=dispatch_config.get("options") or {},
        physical_options=physical_config.get("options") or {},
        economic_options=economic_config.get("options") or {},
    )


def _build_solvers_from_job(job: dict) -> dict:
    mode = str(job.get("mode", MODE_SOCP)).strip().lower()
    methods = job.get("methods") or methods_config_from_legacy_job(job)

    if mode == MODE_SOCP:
        opf_config = methods.get("opf", {})
        opf_method = str(opf_config.get("method", job.get("opf_model", "socp")))
        return {
            opf_method: build_method("opf", opf_method, opf_config.get("options") or {})
        }

    if mode == MODE_BFSA:
        physical_config = methods.get("physical", {})
        physical_method = str(physical_config.get("method", job.get("physical_state_method", "bfsa")))
        return {
            physical_method: _build_estimator_pipeline_solver_from_methods(methods)
        }

    if mode == MODE_COMPARISON:
        opf_config = methods.get("opf", {})
        physical_config = methods.get("physical", {})
        opf_method = str(opf_config.get("method", job.get("opf_model", "socp")))
        physical_method = str(physical_config.get("method", job.get("physical_state_method", "bfsa")))
        return {
            opf_method: build_method("opf", opf_method, opf_config.get("options") or {}),
            physical_method: _build_estimator_pipeline_solver_from_methods(methods),
        }

    raise ValueError(f"Unsupported mode '{mode}'")


def _job_allows_load_shedding(job: dict) -> bool:
    mode = str(job.get("mode", MODE_SOCP)).strip().lower()
    if mode == MODE_COMPARISON:
        return bool(job.get("comparison_allow_load_shedding", False))
    if mode == MODE_SOCP:
        return bool(job.get("socp_allow_load_shedding", False))
    return False


def _solve_socp_with_optional_warm_start(
    case,
    *,
    solver_name: str,
    tee: bool,
    allow_load_shedding: bool,
    warm_start_mode: str = "None",
):
    """Solve SOCP, optionally seeding it from a same-timestep BFSA run."""
    socp_solver = SOCPSolver(
        solver_name=solver_name,
        tee=tee,
        allow_load_shedding=allow_load_shedding,
    )

    warm_start = None
    selected_warm_start = str(warm_start_mode).strip()
    if selected_warm_start == "BFSA t":
        print("Computing BFSA physical warm start for the current timestep...")
        dispatch = build_method("dispatch", "dummy_merit_order").estimate(case)
        physical_estimator = build_method("physical", "bfsa")
        warm_start = physical_estimator.estimate(case, dispatch=dispatch)
        if warm_start.solve_time is not None:
            print(f"  BFSA warm start solved in {warm_start.solve_time:.3f} seconds")
    elif selected_warm_start == "SOCP t-1":
        print("Warm start 'SOCP t-1' is only applied in multi-timestep runs; using no seed here.")

    return socp_solver.solve(case, warm_start=warm_start)


def _run_single_case_job(job: dict, case) -> None:
    mode = str(job.get("mode", MODE_SOCP)).strip().lower()
    output_path = Path(job.get("output_path") or "").expanduser()
    show_plots = bool(job.get("show_plots", False))
    export_plots = bool(job.get("export_plots", False))
    line_color_mode = str(job.get("line_color_mode", "Flow on base MVA")).strip()
    solver_names = job.get("solver_names") or [mode]

    if mode == MODE_SOCP:
        print(f"Solving with SOCP OPF (solver: {job.get('socp_solver_name', 'mosek')})...")
        result = _solve_socp_with_optional_warm_start(
            case,
            solver_name=job.get("socp_solver_name", "mosek"),
            tee=bool(job.get("socp_verbose", False)),
            allow_load_shedding=bool(job.get("socp_allow_load_shedding", False)),
            warm_start_mode=str(job.get("socp_warm_start", "None")),
        )
        print(f"[OK] Solved in {result.solve_time:.3f} seconds")
        print(f"  Optimal Cost: ${result.cost:.2f}")
        print(f"  Convergence: {result.convergence_info}")
        print()
        _print_voltage_preview(result)
        _print_flow_preview(result)
        _print_lmp_preview(result, "Active LMPs")
        summary_path = output_path / "socp_run_summary.txt"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            "\n".join(
                [
                    f"mode=socp",
                    f"solve_time_s={result.solve_time:.6f}",
                    f"cost={result.cost:.6f}",
                    f"convergence={result.convergence_info}",
                ]
            ),
            encoding="utf-8",
        )
        return

    if mode == MODE_BFSA:
        show_net = bool(job.get("bfsa_show_plot", False))
        show_merit = bool(job.get("bfsa_show_merit", False))
        viz_theme = str(job.get("bfsa_theme", "dark"))

        print("Solving with estimator pipeline...")
        result = _estimator_pipeline_result_as_opf_result(
            case,
            job,
            show_net_bus_plot=show_net,
            show_merit_order_plot=show_merit,
        )

        print(f"[OK] Solved in {result.solve_time:.3f} seconds")
        print(f"  Convergence: {result.convergence_info}")
        print()
        _print_voltage_preview(result)
        _print_flow_preview(result)
        _print_lmp_preview(result, "Distribution LMPs")

        if export_plots or show_plots:
            set_visual_theme(viz_theme)
            set_line_color_mode(LINE_COLOR_MODE_LABEL_TO_KEY.get(line_color_mode, "flow_pu"))
            network_dir = output_path / "network"
            network_dir.mkdir(parents=True, exist_ok=True)

            net, bus_id_map, trans_df, dlmp, bus_voltages = build_plot_inputs(case, result)
            verify_load_balance(case, result)

            fig_white = create_white_network_plot(net, bus_id_map, trans_df, dlmp)
            white_path = network_dir / "bfsa_white.html"
            export_network_plot(fig_white, white_path)

            fig_p = create_combined_plot(
                net,
                bus_id_map,
                trans_df,
                dlmp,
                bus_voltages,
                title="BFSA network - lambda_p",
                lmp_type="lambda_p",
                line_color_mode=LINE_COLOR_MODE_LABEL_TO_KEY.get(line_color_mode, "flow_pu"),
            )
            export_network_plot(fig_p, network_dir / "bfsa_lambda_p.html")

            fig_q = create_combined_plot(
                net,
                bus_id_map,
                trans_df,
                dlmp,
                bus_voltages,
                title="BFSA network - lambda_q",
                lmp_type="lambda_q",
                line_color_mode=LINE_COLOR_MODE_LABEL_TO_KEY.get(line_color_mode, "flow_pu"),
            )
            export_network_plot(fig_q, network_dir / "bfsa_lambda_q.html")

            if show_plots:
                webbrowser.open_new_tab(white_path.resolve().as_uri())

        print(f"[OK] Results exported to {output_path}")
        summary_path = output_path / "bfsa_run_summary.txt"
        summary_path.write_text(
            "\n".join(
                [
                    f"mode=bfsa",
                    f"solve_time_s={result.solve_time:.6f}",
                    f"convergence={result.convergence_info}",
                    f"output_dir={output_path}",
                ]
            ),
            encoding="utf-8",
        )
        return

    if mode == MODE_COMPARISON:
        theme = str(job.get("comparison_theme", "dark"))
        export_plots = bool(job.get("comparison_export", False))
        show_plots = bool(job.get("comparison_show", False))

        print("Running SOCP OPF...")
        opf_solver = build_method(
            "opf",
            "socp",
            {
                "solver_name": "mosek",
                "allow_load_shedding": bool(job.get("comparison_allow_load_shedding", False)),
            },
        )
        opf_result = opf_solver.solve_optimization(case)

        print("Running estimator pipeline...")
        estimator_result = _run_estimator_pipeline_result(case, job)
        comparison = OptimizationComparison(candidate=estimator_result, reference=opf_result)
        stats = comparison.summary_stats()

        print("Voltage Deviations:")
        voltage_stats = stats["physical"]["voltage"]
        print(f"  Mean: {voltage_stats['abs_mean']}")
        print(f"  Max:  {voltage_stats['max_abs']}")
        print()
        print("LMP Deviations:")
        lmp_stats = stats.get("economic", {}).get("lambda_p", {})
        print(f"  Mean: {lmp_stats.get('abs_mean')}")
        print(f"  Max:  {lmp_stats.get('max_abs')}")
        print()
        print("Flow Deviations:")
        flow_stats = stats["physical"]["flow_p"]
        print(f"  Mean: {flow_stats['abs_mean']}")
        print(f"  Max:  {flow_stats['max_abs']}")
        print()
        print("Solve Times:")
        print(f"  SOCP: {opf_result.solve_time if opf_result.solve_time is not None else 'N/A'} seconds")
        print(f"  Estimator: {estimator_result.solve_time if estimator_result.solve_time is not None else 'N/A'} seconds")
        speedup = None
        if opf_result.solve_time and estimator_result.solve_time:
            speedup = float(opf_result.solve_time) / float(estimator_result.solve_time)
        print(f"  Speedup: {speedup:.2f}x" if speedup is not None else "  Speedup: N/A")
        print()

        output_path.mkdir(parents=True, exist_ok=True)
        opf_legacy = _optimization_result_to_opf_result(opf_result)
        estimator_legacy = _optimization_result_to_opf_result(estimator_result)

        if export_plots or show_plots:
            set_visual_theme(theme)
            network_dir = output_path / "network"
            network_dir.mkdir(parents=True, exist_ok=True)
            solver_views = [("SOCP", opf_legacy), ("Estimator", estimator_legacy)]

            for solver_name, result in solver_views:
                net, bus_id_map, trans_df, dlmp, bus_voltages = build_plot_inputs(case, result)
                verify_load_balance(case, result)
                export_network_plot(create_white_network_plot(net, bus_id_map, trans_df, dlmp), network_dir / f"{solver_name.lower()}_white.html")
                export_network_plot(
                    create_combined_plot(
                        net,
                        bus_id_map,
                        trans_df,
                        dlmp,
                        bus_voltages,
                        title=f"{solver_name} network - lambda_p",
                        lmp_type="lambda_p",
                        line_color_mode=LINE_COLOR_MODE_LABEL_TO_KEY.get(str(job.get("comparison_line_color_mode", "Flow on base MVA")), "flow_pu"),
                    ),
                    network_dir / f"{solver_name.lower()}_lambda_p.html",
                )
                export_network_plot(
                    create_combined_plot(
                        net,
                        bus_id_map,
                        trans_df,
                        dlmp,
                        bus_voltages,
                        title=f"{solver_name} network - lambda_q",
                        lmp_type="lambda_q",
                        line_color_mode=LINE_COLOR_MODE_LABEL_TO_KEY.get(str(job.get("comparison_line_color_mode", "Flow on base MVA")), "flow_pu"),
                    ),
                    network_dir / f"{solver_name.lower()}_lambda_q.html",
                )

            export_network_plot(
                create_deviation_plot(
                    *build_plot_inputs(case, opf_legacy)[:3],
                    case,
                    lmp_p_dev={} if comparison.economic is None else comparison.economic.deviations["lambda_p"],
                    flow_dev=comparison.physical.deviations["flow_p"],
                    title="Estimator vs SOCP Deviations",
                ),
                network_dir / "deviation_plot.html",
            )
            export_side_by_side_dashboard(network_dir / "comparison_dashboard.html", left_solver="estimator", right_solver="socp")
            if show_plots:
                webbrowser.open_new_tab((network_dir / "comparison_dashboard.html").resolve().as_uri())

        print(f"[OK] Results exported to {output_path}")
        summary_path = output_path / "comparison_run_summary.txt"
        summary_path.write_text(
            "\n".join(
                [
                    f"mode=comparison",
                    f"socp_solve_time_s={opf_result.solve_time}",
                    f"estimator_solve_time_s={estimator_result.solve_time}",
                    f"speedup={speedup if speedup is not None else 'N/A'}",
                    f"summary={stats}",
                    f"output_dir={output_path}",
                ]
            ),
            encoding="utf-8",
        )
        return

    raise ValueError(f"Unsupported mode '{mode}'")


def _execute_run_job(job: dict) -> None:
    load_mode = str(job.get("load_mode", "timestep")).strip().lower()
    kill_flag = [False]

    if load_mode == "range":
        if not bool(job.get("export_to_db", True)):
            raise ValueError("Range mode requires Export to SQLite to be enabled.")

        network = str(job.get("network", "")).strip()
        if not network:
            raise ValueError("Please select a network.")

        start_idx = int(job.get("start_idx", 0))
        end_idx = int(job.get("end_idx", 0))
        if start_idx < 0 or end_idx < start_idx:
            raise ValueError("Invalid timestep range (must be: 0 <= start <= end).")

        input_folder = Path(job["input_folder"])
        if not input_folder.exists():
            raise ValueError(f"{input_folder} does not exist.")

        required_files = {
            "Bus File": job.get("mpc_bus_filename", ""),
            "Branch File": job.get("mpc_branch_filename", ""),
            "P Load File": job.get("p_load_filename", ""),
            "Q Load File": job.get("q_load_filename", ""),
        }
        missing_required = [label for label, fname in required_files.items() if not fname]
        if missing_required:
            raise ValueError(f"Missing file selection for {', '.join(missing_required)}.")

        case_config = {
            "mpc_base_mva_filename": job.get("mpc_base_mva_filename", "mpc_base_mva"),
            "mpc_bus_filename": job["mpc_bus_filename"],
            "mpc_branch_filename": job["mpc_branch_filename"],
            "p_load_filename": job["p_load_filename"],
            "q_load_filename": job["q_load_filename"],
            "bus_mapping_filename": job.get("bus_mapping_filename") or None,
            "generators_filename": job.get("generators_filename", "generators.xlsx"),
            "load_interpretation_mode": job.get("load_interpretation_mode", "auto"),
        }

        output_dir = Path(job.get("output_path") or "")
        output_dir.mkdir(parents=True, exist_ok=True)
        db_dir = _get_categorized_output_dir(output_dir, "db")
        db_dir.mkdir(parents=True, exist_ok=True)
        db_path = db_dir / _timestep_database_filename(
            start_idx,
            end_idx,
            warm_start_mode=job.get("socp_warm_start", "None"),
            allow_load_shedding=_job_allows_load_shedding(job),
            load_mode=job.get("load_interpretation_mode", job.get("load_mode", "auto")),
        )
        solvers = _build_solvers_from_job(job)
        methods = job.get("methods") or methods_config_from_legacy_job(job)
        warm_start_estimator_name = str(
            methods.get("physical", {}).get("method", job.get("physical_state_method", ""))
        )

        print(f"Running multi-timestep solve [{start_idx}, {end_idx}]...")
        multi_solver = MultiTimestepSolver(
            input_folder=input_folder,
            output_db=db_path,
            case_config=case_config,
            solvers=solvers,
        )
        result = multi_solver.run_range(
            start_idx=start_idx,
            end_idx=end_idx,
            solver_names=list(solvers.keys()),
            scenario_label=f"Range {str(job.get('mode', MODE_SOCP)).upper()}",
            verbose=False,
            kill_flag=kill_flag,
            continue_on_socp_infeasible=bool(job.get("continue_on_socp_infeasible", True)),
            warm_start_prev=bool(str(job.get("socp_warm_start", "None")) == "SOCP t-1"),
            warm_start_bfsa=bool(str(job.get("socp_warm_start", "None")) == "BFSA t"),
            warm_start_solver_name=warm_start_estimator_name if warm_start_estimator_name in solvers else None,
        )
        multi_solver.print_summary(result)

        for solver_name in solvers.keys():
            print(f"\nQuerying {solver_name.upper()} statistics...")
            for metric in ["voltage", "p_d", "q_d", "lmp_p", "lmp_q"]:
                stats = multi_solver.query_metric_stats(start_idx, end_idx, solver_name, metric, verbose=True)
                if "error" not in stats:
                    print(f"  {metric}: mean={stats['mean']:.6f}, max={stats['max']:.6f}, min={stats['min']:.6f}")

        print(f"[OK] Multi-timestep execution complete! Results: {db_path}")
        return

    network = str(job.get("network", "")).strip()
    if not network:
        raise ValueError("Please select a network.")

    input_folder = Path(job["input_folder"])
    if not input_folder.exists():
        raise ValueError(f"{input_folder} does not exist.")

    load_mode_label = str(job.get("load_mode_label", "Timestep")).strip()
    if load_mode_label.lower() == "range":
        return

    load_index = _resolve_load_index(
        scenario_mode=load_mode_label,
        timestep_text=str(job.get("timestep_text", "0")),
        input_folder=input_folder,
        mpc_bus_filename=job["mpc_bus_filename"],
        p_load_filename=job["p_load_filename"],
        bus_mapping_filename=job.get("bus_mapping_filename") or "",
    )

    print(f"\nLoading case from {network}...")
    print(f"  Load scenario: {load_mode_label} (index {load_index})")
    case = _build_case_from_job(job, load_index=load_index)
    print(f"  Buses: {len(case.buses)}")
    print(f"  Branches: {len(case.branches)}")
    if hasattr(case, "generators") and hasattr(case, "loads"):
        print(f"  Generators: {len(case.generators)}")
        print(f"  Loads: {len(case.loads)}")
    print()

    _run_single_case_job(job, case)


def _print_voltage_preview(result: object) -> None:
    print("Bus Voltages (p.u.):")
    for bus_id, v_sq in sorted(result.voltages.items())[:5]:
        v = v_sq ** 0.5
        print(f"  Bus {bus_id}: {v:.4f}")
    if len(result.voltages) > 5:
        print(f"  ... and {len(result.voltages) - 5} more buses")
    print()


def _print_flow_preview(result: object) -> None:
    print("Branch Flows (p.u.):")
    for (i, j), (p, q) in sorted(result.flows.items())[:3]:
        s = (p ** 2 + q ** 2) ** 0.5
        print(f"  Line {i}-{j}: P={p:.4f}, Q={q:.4f}, S={s:.4f}")
    if len(result.flows) > 3:
        print(f"  ... and {len(result.flows) - 3} more branches")
    print()


def _print_lmp_preview(result: object, title: str) -> None:
    print(f"{title} ($/MWh):")
    for bus_id, lmp in sorted(result.duals_p.items())[:5]:
        print(f"  Bus {bus_id}: ${lmp:.2f}")
    if len(result.duals_p) > 5:
        print(f"  ... and {len(result.duals_p) - 5} more buses")


def _resolve_load_index(
    *,
    scenario_mode: str,
    timestep_text: str,
    input_folder: Path,
    mpc_bus_filename: str,
    p_load_filename: str,
    bus_mapping_filename: str,
) -> int:
    """Resolve GUI load scenario selection to a concrete load_index."""
    mode = scenario_mode.strip().lower()

    if mode == "timestep":
        if not timestep_text.strip():
            raise ValueError("Please enter a timestep index for 'Timestep' mode.")
        try:
            idx = int(timestep_text.strip())
        except ValueError as exc:
            raise ValueError(
                f"Invalid timestep '{timestep_text}'. Expected a non-negative integer."
            ) from exc
        if idx < 0:
            raise ValueError("Timestep index must be non-negative.")
        return idx

    mpc_bus_path = input_folder / mpc_bus_filename
    p_load_path = input_folder / p_load_filename
    bus_mapping_path = input_folder / bus_mapping_filename if bus_mapping_filename else None

    try:
        p_df = pd.read_csv(p_load_path, sep=None, engine="python", index_col=0)
    except Exception as exc:
        raise ValueError(f"Could not read p_load file '{p_load_path}': {exc}") from exc

    if p_df.empty:
        raise ValueError(f"p_load file has no rows: {p_load_path}")

    # Merge duplicate bus columns like "30" + "30.1" created by pandas.
    p_df = p_df.copy()
    p_df.columns = [str(col).split(".", 1)[0] for col in p_df.columns]
    if p_df.columns.duplicated().any():
        p_df = p_df.T.groupby(level=0, sort=False).sum().T

    p_numeric = p_df.apply(pd.to_numeric, errors="coerce")
    if p_numeric.empty:
        raise ValueError(f"p_load file has no usable load columns: {p_load_path}")

    bus_mapping: dict[str, int] = {}
    if bus_mapping_path is not None:
        try:
            mapping_df = pd.read_csv(bus_mapping_path, sep=None, engine="python")
        except Exception as exc:
            raise ValueError(f"Could not read bus mapping file '{bus_mapping_path}': {exc}") from exc

        if mapping_df.empty or len(mapping_df.columns) < 2:
            raise ValueError(
                f"Bus mapping file must contain at least two columns (file: {bus_mapping_path})."
            )

        normalized_cols = {
            re.sub(r"[^a-z0-9]", "", str(c).lower()): c for c in mapping_df.columns
        }
        source_col = normalized_cols.get("timeseriesid")
        target_col = normalized_cols.get("busi") or normalized_cols.get("bus")
        existing_col = normalized_cols.get("existingload")

        # Backward-compatible fallback: first two columns.
        if source_col is None or target_col is None:
            source_col = mapping_df.columns[0]
            target_col = mapping_df.columns[1]
            existing_col = None

        for _, row in mapping_df.iterrows():
            if existing_col is not None:
                existing_raw = str(row.get(existing_col, "")).strip().lower()
                if existing_raw in {"false", "0", "no", "n", "f"}:
                    continue

            source = str(row[source_col]).strip()
            if not source or source.lower() == "nan":
                continue
            try:
                bus_mapping[source] = int(row[target_col])
            except Exception as exc:
                raise ValueError(
                    f"Invalid bus mapping entry for '{source}' in {bus_mapping_path}: {exc}"
                ) from exc

    # Detect profile mode from mpc_bus (same trigger used by loader): non-zero Pd/Qd.
    try:
        mpc_bus = pd.read_csv(mpc_bus_path, sep=None, engine="python")
    except Exception as exc:
        raise ValueError(f"Could not read mpc_bus file '{mpc_bus_path}': {exc}") from exc

    normalized_cols = {
        re.sub(r"[^a-z0-9]", "", str(c).lower()): c for c in mpc_bus.columns
    }
    bus_col = normalized_cols.get("bus") or normalized_cols.get("busi")
    pd_col = normalized_cols.get("pd")
    qd_col = normalized_cols.get("qd")

    if bus_col is None or pd_col is None:
        raise ValueError(
            f"mpc_bus file must contain bus and Pd columns for load-scenario selection (file: {mpc_bus_path})."
        )

    pd_by_bus = {}
    qd_any_nonzero = False
    for _, row in mpc_bus.iterrows():
        try:
            bus_id = int(row[bus_col])
            pd_by_bus[bus_id] = float(row.get(pd_col, 0.0))
            if qd_col is not None and abs(float(row.get(qd_col, 0.0))) > 1e-12:
                qd_any_nonzero = True
        except Exception:
            continue

    pd_any_nonzero = any(abs(val) > 1e-12 for val in pd_by_bus.values())
    force_normalized_profiles = pd_any_nonzero or qd_any_nonzero

    if force_normalized_profiles:
        # Physical units: sum over buses (load_factor * Pd_max_at_bus).
        totals = []
        for _, row in p_numeric.fillna(0.0).iterrows():
            total = 0.0
            for bus_col_name, p_factor in row.items():
                bus_col_str = str(bus_col_name).strip()
                if bus_mapping and bus_col_str in bus_mapping:
                    bus_id = bus_mapping[bus_col_str]
                else:
                    try:
                        bus_id = int(bus_col_str)
                    except ValueError:
                        continue
                total += float(p_factor) * float(pd_by_bus.get(bus_id, 0.0))
            totals.append(total)
        combined = pd.Series(totals, index=p_numeric.index)
    else:
        # Already in physical units: sum active load per mapped/numeric bus column.
        totals = []
        for _, row in p_numeric.fillna(0.0).iterrows():
            total = 0.0
            for bus_col_name, p_val in row.items():
                bus_col_str = str(bus_col_name).strip()
                if bus_mapping and bus_col_str in bus_mapping:
                    total += float(p_val)
                else:
                    try:
                        int(bus_col_str)
                        total += float(p_val)
                    except ValueError:
                        continue
            totals.append(total)
        combined = pd.Series(totals, index=p_numeric.index)

    if combined.empty:
        raise ValueError("Could not compute load totals for scenario selection.")

    combined_vals = combined.to_numpy(dtype=float)

    if mode == "average":
        target = float(combined.mean())
        return int((combined - target).abs().to_numpy().argmin())
    if mode == "high":
        return int(combined_vals.argmax())
    if mode == "low":
        return int(combined_vals.argmin())

    raise ValueError(f"Unsupported load scenario mode '{scenario_mode}'.")


def build_solver_gui(initial_mode: str = MODE_SOCP) -> bool | None:
    """Build and run the unified solver GUI.

    Args:
        initial_mode: One of 'socp', 'bfsa', or 'comparison'.
    """
    try:
        import tkinter as tk
        from tkinter import ttk
    except ImportError:
        print("tkinter not available; falling back to CLI mode.")
        return None

    mode_key = initial_mode if initial_mode in MODE_KEY_TO_LABEL else MODE_SOCP
    data_folder = data_path
    available_networks = []
    if data_folder.exists():
        available_networks = sorted(d.name for d in data_folder.iterdir() if d.is_dir())

    window = tk.Tk()
    window.title("COBRAS Solver Runner")
    window.geometry("980x820")
    window.minsize(640, 420)
    window.resizable(True, True)
    window.grid_rowconfigure(0, weight=1)
    window.grid_columnconfigure(0, weight=1)

    scroll_canvas = tk.Canvas(window, highlightthickness=0)
    scroll_bar = ttk.Scrollbar(window, orient="vertical", command=scroll_canvas.yview)
    scroll_canvas.configure(yscrollcommand=scroll_bar.set)
    scroll_canvas.grid(row=0, column=0, sticky="nsew")
    scroll_bar.grid(row=0, column=1, sticky="ns")

    content = ttk.Frame(scroll_canvas)
    content_window = scroll_canvas.create_window((0, 0), window=content, anchor="nw")
    content.grid_columnconfigure(0, weight=1)

    def _update_scroll_region(_event: tk.Event | None = None) -> None:
        scroll_canvas.configure(scrollregion=scroll_canvas.bbox("all"))

    def _sync_content_width(event: tk.Event) -> None:
        scroll_canvas.itemconfigure(content_window, width=event.width)

    def _on_mousewheel(event: tk.Event) -> None:
        if event.delta:
            scroll_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    content.bind("<Configure>", _update_scroll_region)
    scroll_canvas.bind("<Configure>", _sync_content_width)
    window.bind_all("<MouseWheel>", _on_mousewheel)

    title = ttk.Label(content, text="Cobras Solver Runner", font=("Arial", 16, "bold"))
    title.grid(row=0, column=0, columnspan=3, pady=10)

    input_section = ttk.LabelFrame(content, text="1. Input files")
    input_section.grid(row=1, column=0, sticky="ew", padx=10, pady=(4, 8))
    input_section.grid_columnconfigure(1, weight=1)

    analysis_section = ttk.LabelFrame(content, text="2. Analysis parameters")
    analysis_section.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 8))
    analysis_section.grid_columnconfigure(1, weight=1)

    acopf_section = ttk.LabelFrame(content, text="3. ACOPF model")
    acopf_section.grid(row=3, column=0, sticky="ew", padx=10, pady=(0, 8))
    acopf_section.grid_columnconfigure(1, weight=1)

    estimator_section = ttk.LabelFrame(content, text="4. Estimators and parameters")
    estimator_section.grid(row=4, column=0, sticky="ew", padx=10, pady=(0, 8))
    estimator_section.grid_columnconfigure(1, weight=1)

    action_section = ttk.Frame(content)
    action_section.grid(row=5, column=0, sticky="ew", padx=10, pady=(0, 8))
    action_section.grid_columnconfigure(0, weight=0)
    action_section.grid_columnconfigure(1, weight=0)
    action_section.grid_columnconfigure(2, weight=0)

    opf_specs = list_method_specs("opf")
    opf_label_to_key = {spec.label: key for key, spec in opf_specs.items()}
    opf_key_to_label = {key: spec.label for key, spec in opf_specs.items()}
    default_opf_key = "socp" if "socp" in opf_specs else next(iter(opf_specs), "")
    dispatch_specs = list_method_specs("dispatch")
    physical_specs = list_method_specs("physical")
    economic_specs = list_method_specs("economic")
    dispatch_label_to_key = {spec.label: key for key, spec in dispatch_specs.items()}
    physical_label_to_key = {spec.label: key for key, spec in physical_specs.items()}
    economic_label_to_key = {spec.label: key for key, spec in economic_specs.items()}
    dispatch_key_to_label = {key: spec.label for key, spec in dispatch_specs.items()}
    physical_key_to_label = {key: spec.label for key, spec in physical_specs.items()}
    economic_key_to_label = {key: spec.label for key, spec in economic_specs.items()}
    default_dispatch_key = "dummy_merit_order" if "dummy_merit_order" in dispatch_specs else next(iter(dispatch_specs), "")
    default_physical_key = "bfsa" if "bfsa" in physical_specs else next(iter(physical_specs), "")
    default_economic_key = "bfsa_dlmp" if "bfsa_dlmp" in economic_specs else next(iter(economic_specs), "")

    ttk.Label(acopf_section, text="ACOPF model:").grid(row=0, column=0, sticky="w", padx=10, pady=5)
    opf_model_var = tk.StringVar(value=opf_key_to_label.get(default_opf_key, default_opf_key))
    opf_model_combo = ttk.Combobox(
        acopf_section,
        textvariable=opf_model_var,
        values=list(opf_label_to_key.keys()),
        state="readonly",
        width=50,
    )
    opf_model_combo.grid(row=0, column=1, padx=10, pady=5, sticky="ew")

    ttk.Label(input_section, text="Network:").grid(row=0, column=0, sticky="w", padx=10, pady=5)
    network_var = tk.StringVar(value=available_networks[0] if available_networks else "")
    network_combo = ttk.Combobox(
        input_section,
        textvariable=network_var,
        values=available_networks,
        state="readonly",
        width=50,
    )
    network_combo.grid(row=0, column=1, padx=10, pady=5, sticky="ew")

    ttk.Label(input_section, text="Bus:").grid(row=1, column=0, sticky="w", padx=10, pady=5)
    mpc_bus_var = tk.StringVar(value="")
    mpc_bus_combo = ttk.Combobox(input_section, textvariable=mpc_bus_var, values=[], state="readonly", width=50)
    mpc_bus_combo.grid(row=1, column=1, padx=10, pady=5, sticky="ew")
    mpc_bus_reset = ttk.Button(input_section, text="x", width=3)
    mpc_bus_reset.grid(row=1, column=2, padx=5, pady=5)

    ttk.Label(input_section, text="Branch:").grid(row=2, column=0, sticky="w", padx=10, pady=5)
    mpc_branch_var = tk.StringVar(value="")
    mpc_branch_combo = ttk.Combobox(input_section, textvariable=mpc_branch_var, values=[], state="readonly", width=50)
    mpc_branch_combo.grid(row=2, column=1, padx=10, pady=5, sticky="ew")
    mpc_branch_reset = ttk.Button(input_section, text="x", width=3)
    mpc_branch_reset.grid(row=2, column=2, padx=5, pady=5)

    ttk.Label(input_section, text="P load:").grid(row=3, column=0, sticky="w", padx=10, pady=5)
    p_load_var = tk.StringVar(value="")
    p_load_combo = ttk.Combobox(input_section, textvariable=p_load_var, values=[], state="readonly", width=50)
    p_load_combo.grid(row=3, column=1, padx=10, pady=5, sticky="ew")
    p_load_reset = ttk.Button(input_section, text="x", width=3)
    p_load_reset.grid(row=3, column=2, padx=5, pady=5)

    ttk.Label(input_section, text="Load interpretation:").grid(row=3, column=3, sticky="w", padx=5, pady=5)
    load_interp_var = tk.StringVar(value="Auto (detect from Pd/Qd)")
    load_interp_combo = ttk.Combobox(
        input_section,
        textvariable=load_interp_var,
        values=list(LOAD_INTERPRETATION_LABEL_TO_KEY.keys()),
        state="readonly",
        width=28,
    )
    load_interp_combo.grid(row=3, column=4, padx=5, pady=5, sticky="w")

    ttk.Label(input_section, text="Q load:").grid(row=4, column=0, sticky="w", padx=10, pady=5)
    q_load_var = tk.StringVar(value="")
    q_load_combo = ttk.Combobox(input_section, textvariable=q_load_var, values=[], state="readonly", width=50)
    q_load_combo.grid(row=4, column=1, padx=10, pady=5, sticky="ew")
    q_load_reset = ttk.Button(input_section, text="x", width=3)
    q_load_reset.grid(row=4, column=2, padx=5, pady=5)

    ttk.Label(input_section, text="Bus mapping:").grid(row=5, column=0, sticky="w", padx=10, pady=5)
    bus_mapping_var = tk.StringVar(value="")
    bus_mapping_combo = ttk.Combobox(
        input_section,
        textvariable=bus_mapping_var,
        values=[],
        state="readonly",
        width=50,
    )
    bus_mapping_combo.grid(row=5, column=1, padx=10, pady=5, sticky="ew")
    bus_mapping_reset = ttk.Button(input_section, text="x", width=3)
    bus_mapping_reset.grid(row=5, column=2, padx=5, pady=5)

    ttk.Label(input_section, text="Generators:").grid(row=6, column=0, sticky="w", padx=10, pady=5)
    gen_var = tk.StringVar(value="")
    gen_combo = ttk.Combobox(input_section, textvariable=gen_var, values=[], state="readonly", width=50)
    gen_combo.grid(row=6, column=1, padx=10, pady=5, sticky="ew")
    gen_reset = ttk.Button(input_section, text="x", width=3)
    gen_reset.grid(row=6, column=2, padx=5, pady=5)

    ttk.Label(input_section, text="Base MVA:").grid(row=7, column=0, sticky="w", padx=10, pady=5)
    base_mva_var = tk.StringVar(value="")
    base_mva_combo = ttk.Combobox(input_section, textvariable=base_mva_var, values=[], state="readonly", width=50)
    base_mva_combo.grid(row=7, column=1, padx=10, pady=5, sticky="ew")
    base_mva_reset = ttk.Button(input_section, text="x", width=3)
    base_mva_reset.grid(row=7, column=2, padx=5, pady=5)

    ttk.Label(input_section, text="Output directory:").grid(row=8, column=0, sticky="w", padx=10, pady=5)
    initial_network = network_var.get().strip()
    output_var = tk.StringVar(
        value=str(_get_network_results_dir(initial_network)) if initial_network else ""
    )
    output_entry = ttk.Entry(input_section, textvariable=output_var, width=52, state="readonly")
    output_entry.grid(row=8, column=1, padx=10, pady=5, sticky="ew")

    # The pipeline determines whether this analysis runs an OPF reference, the
    # estimator pipeline, or both for comparison.  It belongs with the temporal
    # analysis settings rather than the ACOPF-method-specific controls.
    ttk.Label(analysis_section, text="Pipeline:").grid(row=0, column=0, sticky="w", padx=10, pady=5)
    mode_var = tk.StringVar(value=MODE_KEY_TO_LABEL[mode_key])
    mode_combo = ttk.Combobox(
        analysis_section,
        textvariable=mode_var,
        values=list(MODE_LABEL_TO_KEY.keys()),
        state="readonly",
        width=50,
    )
    mode_combo.grid(row=0, column=1, padx=10, pady=5, sticky="ew")

    ttk.Label(analysis_section, text="Load scenario:").grid(row=1, column=0, sticky="w", padx=10, pady=5)
    load_scenario_var = tk.StringVar(value="Timestep")
    load_scenario_combo = ttk.Combobox(
        analysis_section,
        textvariable=load_scenario_var,
        values=["Timestep", "Average", "High", "Low", "Range"],
        state="readonly",
        width=50,
    )
    load_scenario_combo.grid(row=1, column=1, padx=10, pady=5, sticky="ew")

    ttk.Label(analysis_section, text="Timestep index:").grid(row=2, column=0, sticky="w", padx=10, pady=5)
    timestep_var = tk.StringVar(value="0")
    timestep_entry = ttk.Entry(analysis_section, textvariable=timestep_var, width=52)
    timestep_entry.grid(row=2, column=1, padx=10, pady=5, sticky="ew")

    # Range mode fields (hidden by default)
    start_idx_label = ttk.Label(analysis_section, text="Timestep start:")
    start_idx_label.grid(row=3, column=0, sticky="w", padx=10, pady=5)
    start_idx_var = tk.StringVar(value="0")
    start_idx_entry = ttk.Entry(analysis_section, textvariable=start_idx_var, width=52)
    start_idx_entry.grid(row=3, column=1, padx=10, pady=5, sticky="ew")
    start_idx_entry.grid_remove()  # Hide initially
    start_idx_label.grid_remove()  # Hide label initially

    end_idx_label = ttk.Label(analysis_section, text="Timestep end:")
    end_idx_label.grid(row=4, column=0, sticky="w", padx=10, pady=5)
    end_idx_var = tk.StringVar(value="10")
    end_idx_entry = ttk.Entry(analysis_section, textvariable=end_idx_var, width=52)
    end_idx_entry.grid(row=4, column=1, padx=10, pady=5, sticky="ew")
    end_idx_entry.grid_remove()  # Hide initially
    end_idx_label.grid_remove()  # Hide label initially

    export_to_db_var = tk.BooleanVar(value=True)
    export_to_db_check = ttk.Checkbutton(
        analysis_section,
        text="Export multi-timestep results to SQLite",
        variable=export_to_db_var,
    )
    export_to_db_check.grid(row=5, column=0, columnspan=2, sticky="w", padx=10, pady=5)
    export_to_db_check.grid_remove()  # Hide initially

    opf_continue_on_infeasible_var = tk.BooleanVar(value=True)
    opf_continue_on_infeasible_check = ttk.Checkbutton(
        acopf_section,
        text="Continue if current step is infeasible and store NULL",
        variable=opf_continue_on_infeasible_var,
    )
    opf_continue_on_infeasible_check.grid(row=2, column=0, columnspan=2, sticky="w", padx=10, pady=5)
    opf_continue_on_infeasible_check.grid_remove()  # Hide initially

    opf_warm_start_label = ttk.Label(acopf_section, text="Warm start:")
    opf_warm_start_label.grid(row=3, column=0, sticky="w", padx=10, pady=5)
    opf_warm_start_var = tk.StringVar(value="None")
    opf_warm_start_combo = ttk.Combobox(
        acopf_section,
        textvariable=opf_warm_start_var,
        values=["None", "ACOPF previous timestep", "Electrical state estimator"],
        state="readonly",
        width=50,
    )
    opf_warm_start_combo.grid(row=3, column=1, padx=10, pady=5, sticky="ew")
    opf_warm_start_combo.grid_remove()
    opf_warm_start_label.grid_remove()

    ttk.Label(acopf_section, text="Solver:").grid(row=4, column=0, sticky="w", padx=10, pady=5)
    opf_solver_backend_var = tk.StringVar(value="mosek")
    opf_solver_backend_combo = ttk.Combobox(
        acopf_section,
        textvariable=opf_solver_backend_var,
        values=["mosek", "gurobi", "cplex"],
        state="readonly",
        width=50,
    )
    opf_solver_backend_combo.grid(row=4, column=1, padx=10, pady=5, sticky="ew")

    opf_allow_load_shedding_var = tk.BooleanVar(value=False)
    opf_allow_load_shedding_check = ttk.Checkbutton(
        acopf_section,
        text="Allow load shedding",
        variable=opf_allow_load_shedding_var,
    )
    opf_allow_load_shedding_check.grid(row=5, column=0, columnspan=2, sticky="w", padx=10, pady=5)

    opf_verbose_var = tk.BooleanVar(value=False)
    opf_verbose_check = ttk.Checkbutton(
        acopf_section,
        text="Verbose solver output",
        variable=opf_verbose_var,
    )
    opf_verbose_check.grid(row=6, column=0, columnspan=2, sticky="w", padx=10, pady=5)

    ttk.Label(estimator_section, text="Dispatch:").grid(row=0, column=0, sticky="w", padx=10, pady=5)
    dispatch_method_var = tk.StringVar(value=dispatch_key_to_label.get(default_dispatch_key, default_dispatch_key))
    dispatch_method_combo = ttk.Combobox(
        estimator_section,
        textvariable=dispatch_method_var,
        values=list(dispatch_label_to_key.keys()),
        state="readonly",
        width=50,
    )
    dispatch_method_combo.grid(row=0, column=1, padx=10, pady=5, sticky="ew")

    ttk.Label(estimator_section, text="Electrical state:").grid(row=1, column=0, sticky="w", padx=10, pady=5)
    physical_method_var = tk.StringVar(value=physical_key_to_label.get(default_physical_key, default_physical_key))
    physical_method_combo = ttk.Combobox(
        estimator_section,
        textvariable=physical_method_var,
        values=list(physical_label_to_key.keys()),
        state="readonly",
        width=50,
    )
    physical_method_combo.grid(row=1, column=1, padx=10, pady=5, sticky="ew")

    ttk.Label(estimator_section, text="Economic state:").grid(row=2, column=0, sticky="w", padx=10, pady=5)
    economic_method_var = tk.StringVar(value=economic_key_to_label.get(default_economic_key, default_economic_key))
    economic_method_combo = ttk.Combobox(
        estimator_section,
        textvariable=economic_method_var,
        values=list(economic_label_to_key.keys()),
        state="readonly",
        width=50,
    )
    economic_method_combo.grid(row=2, column=1, padx=10, pady=5, sticky="ew")

    mode_options_expanded = tk.BooleanVar(value=True)
    mode_toggle_button = ttk.Button(estimator_section, text="v Method parameters")
    mode_toggle_button.grid(row=3, column=0, columnspan=3, sticky="w", padx=10, pady=(8, 2))

    mode_frame = ttk.LabelFrame(estimator_section, text="Estimator parameters")

    status_label = ttk.Label(content, text="", foreground="blue")
    status_label.grid(row=6, column=0, pady=(6, 8))

    mode_widgets: dict[str, object] = {}
    current_mode_key = mode_key

    file_fields = {
        "mpc_bus": {"var": mpc_bus_var, "combo": mpc_bus_combo},
        "mpc_branch": {"var": mpc_branch_var, "combo": mpc_branch_combo},
        "p_load": {"var": p_load_var, "combo": p_load_combo},
        "q_load": {"var": q_load_var, "combo": q_load_combo},
        "bus_mapping": {"var": bus_mapping_var, "combo": bus_mapping_combo},
        "generators": {"var": gen_var, "combo": gen_combo},
        "base_mva": {"var": base_mva_var, "combo": base_mva_combo},
    }
    folder_file_overrides: dict[str, dict[str, str]] = {}
    folder_defaults_cache: dict[str, dict[str, str]] = {}
    is_refreshing = False

    def _update_output_dir(*_args: object) -> None:
        selected_network = network_var.get().strip()
        output_var.set(str(_get_network_results_dir(selected_network)) if selected_network else "")

    def _current_network_key() -> str:
        return network_var.get().strip()

    def _store_current_selections(*_args: object) -> None:
        nonlocal is_refreshing
        if is_refreshing:
            return
        network_key_local = _current_network_key()
        if not network_key_local:
            return
        folder_file_overrides[network_key_local] = {
            key_name: cfg["var"].get().strip()
            for key_name, cfg in file_fields.items()
            if cfg["var"].get().strip()
        }

    def _refresh_file_dropdowns(*_args: object) -> None:
        nonlocal is_refreshing
        selected_network = network_var.get().strip()
        folder = data_folder / selected_network if selected_network else data_folder

        csv_files = _list_files(folder, (".csv",))
        gen_files = _list_files(folder, (".xlsx", ".xls", ".csv"))
        base_mva_files = _list_base_mva_files(folder)

        defaults = {
            "mpc_bus": _pick_default(csv_files, "mpc_bus.csv", "mpc_bus"),
            "mpc_branch": _pick_default(csv_files, "mpc_branch.csv", "mpc_branch"),
            "p_load": _pick_default(csv_files, "p_load.csv", "p_load"),
            "q_load": _pick_default(csv_files, "q_load.csv", "q_load"),
            "bus_mapping": _pick_default_bus_mapping(csv_files),
            "generators": _pick_default(gen_files, "generators.xlsx", "generators"),
            "base_mva": _pick_default(base_mva_files, "mpc_base_mva", "mpc_base_mva"),
        }
        if selected_network:
            folder_defaults_cache[selected_network] = defaults
        selected_overrides = folder_file_overrides.get(selected_network, {}) if selected_network else {}

        is_refreshing = True
        try:
            file_fields["mpc_bus"]["combo"]["values"] = csv_files
            file_fields["mpc_branch"]["combo"]["values"] = csv_files
            file_fields["p_load"]["combo"]["values"] = csv_files
            file_fields["q_load"]["combo"]["values"] = csv_files
            file_fields["bus_mapping"]["combo"]["values"] = [""] + csv_files
            file_fields["generators"]["combo"]["values"] = gen_files
            file_fields["base_mva"]["combo"]["values"] = base_mva_files

            for key_name, cfg in file_fields.items():
                options = list(cfg["combo"].cget("values"))
                remembered = selected_overrides.get(key_name, "")
                default_value = defaults.get(key_name, "")
                # Treat empty remembered values as "no override" so defaults can apply.
                chosen = remembered if remembered and remembered in options else default_value
                cfg["var"].set(chosen)
        finally:
            is_refreshing = False

    def _reset_field_to_default(field_key: str) -> None:
        network_key_local = _current_network_key()
        if not network_key_local:
            return
        default_value = folder_defaults_cache.get(network_key_local, {}).get(field_key, "")
        file_fields[field_key]["var"].set(default_value)
        _store_current_selections()

    def _set_status(text: str, color: str) -> None:
        # UI callbacks can arrive after the window is closed; ignore stale updates.
        try:
            if not window.winfo_exists() or not status_label.winfo_exists():
                return
            status_label.config(text=text, foreground=color)
        except tk.TclError:
            return

    def _selected_opf_model_key() -> str:
        label = opf_model_var.get().strip()
        return opf_label_to_key.get(label, default_opf_key)

    def _selected_dispatch_method_key() -> str:
        label = dispatch_method_var.get().strip()
        return dispatch_label_to_key.get(label, default_dispatch_key)

    def _selected_physical_method_key() -> str:
        label = physical_method_var.get().strip()
        return physical_label_to_key.get(label, default_physical_key)

    def _selected_economic_method_key() -> str:
        label = economic_method_var.get().strip()
        return economic_label_to_key.get(label, default_economic_key)

    def _legacy_opf_warm_start_key() -> str:
        label = opf_warm_start_var.get().strip()
        if label == "ACOPF previous timestep":
            return "SOCP t-1"
        if label == "Electrical state estimator":
            return "BFSA t"
        return "None"

    def _fit_window_to_content() -> None:
        """Keep the launcher large enough for the currently visible controls."""
        window.update_idletasks()
        screen_w = window.winfo_screenwidth()
        screen_h = window.winfo_screenheight()
        target_w = min(max(content.winfo_reqwidth() + 48, 980), max(screen_w - 80, 640))
        target_h = min(max(content.winfo_reqheight() + 24, 820), max(screen_h - 80, 420))
        window.geometry(f"{target_w}x{target_h}")
        _update_scroll_region()

    def _update_timestep_entry_state(*_args: object) -> None:
        mode = load_scenario_var.get().strip().lower()
        if mode == "timestep":
            timestep_entry.state(["!disabled"])
            start_idx_entry.grid_remove()
            start_idx_label.grid_remove()
            end_idx_entry.grid_remove()
            end_idx_label.grid_remove()
            export_to_db_check.grid_remove()
            opf_continue_on_infeasible_check.grid_remove()
            opf_warm_start_label.grid_remove()
            opf_warm_start_combo.grid_remove()
        elif mode == "range":
            timestep_entry.state(["disabled"])
            start_idx_entry.grid()
            start_idx_label.grid()
            end_idx_entry.grid()
            end_idx_label.grid()
            export_to_db_check.grid()
            opf_continue_on_infeasible_check.grid()
            opf_warm_start_label.grid()
            opf_warm_start_combo.grid()
        else:
            timestep_entry.state(["disabled"])
            start_idx_entry.grid_remove()
            start_idx_label.grid_remove()
            end_idx_entry.grid_remove()
            end_idx_label.grid_remove()
            export_to_db_check.grid_remove()
            opf_continue_on_infeasible_check.grid_remove()
            opf_warm_start_label.grid_remove()
            opf_warm_start_combo.grid_remove()
        _fit_window_to_content()


        


    def _clear_mode_frame() -> None:
        for child in mode_frame.winfo_children():
            child.destroy()
        mode_widgets.clear()

    def _set_mode_options_expanded(expanded: bool) -> None:
        mode_options_expanded.set(expanded)
        if expanded:
            mode_toggle_button.config(text="v Estimator parameters")
            mode_frame.grid(row=4, column=0, columnspan=4, sticky="ew", padx=10, pady=(0, 6))
        else:
            mode_toggle_button.config(text="> Estimator parameters")
            mode_frame.grid_remove()
        _fit_window_to_content()

    def _toggle_mode_options() -> None:
        _set_mode_options_expanded(not bool(mode_options_expanded.get()))

    mode_toggle_button.configure(command=_toggle_mode_options)

    def _render_mode_controls(*_args: object) -> None:
        nonlocal current_mode_key
        selected_mode_label = mode_var.get().strip() or MODE_KEY_TO_LABEL[MODE_SOCP]
        current_mode_key = MODE_LABEL_TO_KEY.get(selected_mode_label, MODE_SOCP)

        _clear_mode_frame()
        mode_frame.grid_columnconfigure(0, weight=1)
        mode_frame.grid_columnconfigure(1, weight=1)

        if current_mode_key == MODE_SOCP:
            ttk.Label(
                mode_frame,
                text="No estimator pipeline selected for this run.",
            ).grid(row=0, column=0, columnspan=2, sticky="w", padx=10, pady=5)
        elif current_mode_key == MODE_BFSA:
            ttk.Label(mode_frame, text="Max Iterations:").grid(row=0, column=0, sticky="w", padx=10, pady=5)
            max_iter_var = tk.StringVar(value="100")
            max_iter_entry = ttk.Entry(mode_frame, textvariable=max_iter_var, width=52)
            max_iter_entry.grid(row=0, column=1, padx=10, pady=5, sticky="ew")

            ttk.Label(mode_frame, text="Tolerance:").grid(row=1, column=0, sticky="w", padx=10, pady=5)
            tol_var = tk.StringVar(value="0.001")
            tol_entry = ttk.Entry(mode_frame, textvariable=tol_var, width=52)
            tol_entry.grid(row=1, column=1, padx=10, pady=5, sticky="ew")

            ttk.Label(mode_frame, text="Convergence Check:").grid(
                row=2, column=0, sticky="w", padx=10, pady=5
            )
            convergence_check_var = tk.StringVar(value=BFSA_CONVERGENCE_KEY_TO_LABEL["both"])
            convergence_check_combo = ttk.Combobox(
                mode_frame,
                textvariable=convergence_check_var,
                values=list(BFSA_CONVERGENCE_LABEL_TO_KEY.keys()),
                state="readonly",
                width=50,
            )
            convergence_check_combo.grid(row=2, column=1, padx=10, pady=5, sticky="ew")

            show_var = tk.BooleanVar(value=False)
            show_check = ttk.Checkbutton(
                mode_frame,
                text="Show net bus demand plot",
                variable=show_var,
            )
            show_check.grid(row=3, column=0, sticky="w", padx=10, pady=5)

            merit_var = tk.BooleanVar(value=False)
            merit_check = ttk.Checkbutton(
                mode_frame,
                text="Show merit-order clearing curves (P and Q)",
                variable=merit_var,
            )
            merit_check.grid(row=3, column=1, sticky="w", padx=10, pady=5)

            ttk.Label(mode_frame, text="Merit Clearing Mode:").grid(
                row=4, column=0, sticky="w", padx=10, pady=5
            )
            merit_mode_var = tk.StringVar(value=BFSA_MERIT_ORDER_KEY_TO_LABEL["dummy"])
            merit_mode_combo = ttk.Combobox(
                mode_frame,
                textvariable=merit_mode_var,
                values=list(BFSA_MERIT_ORDER_LABEL_TO_KEY.keys()),
                state="readonly",
                width=50,
            )
            merit_mode_combo.grid(row=4, column=1, padx=10, pady=5, sticky="ew")

            mcp_row = ttk.Frame(mode_frame)
            mcp_row.grid(row=5, column=0, columnspan=2, padx=10, pady=5, sticky="ew")
            for idx in range(4):
                mcp_row.columnconfigure(idx, weight=1, uniform="mcp_row")

            ttk.Label(mcp_row, text="MCPA:").grid(row=0, column=0, sticky="w")
            mcp_active_var = tk.StringVar(value="0.0")
            mcp_active_entry = ttk.Entry(mcp_row, textvariable=mcp_active_var, width=24)
            mcp_active_entry.grid(row=0, column=1, padx=(6, 14), sticky="ew")

            ttk.Label(mcp_row, text="MCPR:").grid(row=0, column=2, sticky="w")
            mcp_reactive_var = tk.StringVar(value="0.0")
            mcp_reactive_entry = ttk.Entry(mcp_row, textvariable=mcp_reactive_var, width=24)
            mcp_reactive_entry.grid(row=0, column=3, padx=(6, 0), sticky="ew")

            def _sync_bfsa_merit_inputs(*_args: object) -> None:
                enabled = BFSA_MERIT_ORDER_LABEL_TO_KEY.get(merit_mode_var.get(), "dummy") == "mcp"
                state = "normal" if enabled else "disabled"
                mcp_active_entry.configure(state=state)
                mcp_reactive_entry.configure(state=state)

            merit_mode_var.trace_add("write", _sync_bfsa_merit_inputs)
            _sync_bfsa_merit_inputs()

            mode_widgets["bfsa_max_iter_var"] = max_iter_var
            mode_widgets["bfsa_tol_var"] = tol_var
            mode_widgets["bfsa_convergence_check_var"] = convergence_check_var
            mode_widgets["bfsa_show_plot_var"] = show_var
            mode_widgets["bfsa_show_merit_var"] = merit_var
            mode_widgets["bfsa_merit_order_mode_var"] = merit_mode_var
            mode_widgets["bfsa_mcp_active_price_var"] = mcp_active_var
            mode_widgets["bfsa_mcp_reactive_price_var"] = mcp_reactive_var
        else:
            ttk.Label(mode_frame, text="BFSA Merit Clearing Mode:").grid(
                row=0, column=0, sticky="w", padx=10, pady=5
            )
            comparison_merit_mode_var = tk.StringVar(value=BFSA_MERIT_ORDER_KEY_TO_LABEL["dummy"])
            comparison_merit_mode_combo = ttk.Combobox(
                mode_frame,
                textvariable=comparison_merit_mode_var,
                values=list(BFSA_MERIT_ORDER_LABEL_TO_KEY.keys()),
                state="readonly",
                width=50,
            )
            comparison_merit_mode_combo.grid(row=0, column=1, padx=10, pady=5, sticky="ew")

            comparison_mcp_row = ttk.Frame(mode_frame)
            comparison_mcp_row.grid(row=1, column=0, columnspan=2, padx=10, pady=5, sticky="ew")
            for idx in range(4):
                comparison_mcp_row.columnconfigure(idx, weight=1, uniform="mcp_row")

            ttk.Label(comparison_mcp_row, text="BFSA MCPA:").grid(row=0, column=0, sticky="w")
            comparison_mcp_active_var = tk.StringVar(value=default_BFSA_MCPA)
            comparison_mcp_active_entry = ttk.Entry(comparison_mcp_row, textvariable=comparison_mcp_active_var, width=24)
            comparison_mcp_active_entry.grid(row=0, column=1, padx=(6, 14), sticky="ew")

            ttk.Label(comparison_mcp_row, text="BFSA MCPR:").grid(row=0, column=2, sticky="w")
            comparison_mcp_reactive_var = tk.StringVar(value=default_BFSA_MCPR)
            comparison_mcp_reactive_entry = ttk.Entry(comparison_mcp_row, textvariable=comparison_mcp_reactive_var, width=24)
            comparison_mcp_reactive_entry.grid(row=0, column=3, padx=(6, 0), sticky="ew")

            def _sync_comparison_bfsa_merit_inputs(*_args: object) -> None:
                enabled = BFSA_MERIT_ORDER_LABEL_TO_KEY.get(comparison_merit_mode_var.get(), "dummy") == "mcp"
                state = "normal" if enabled else "disabled"
                comparison_mcp_active_entry.configure(state=state)
                comparison_mcp_reactive_entry.configure(state=state)

            comparison_merit_mode_var.trace_add("write", _sync_comparison_bfsa_merit_inputs)
            _sync_comparison_bfsa_merit_inputs()

            mode_widgets["comparison_bfsa_merit_order_mode_var"] = comparison_merit_mode_var
            mode_widgets["comparison_bfsa_mcp_active_price_var"] = comparison_mcp_active_var
            mode_widgets["comparison_bfsa_mcp_reactive_price_var"] = comparison_mcp_reactive_var


    mpc_bus_reset.configure(command=lambda: _reset_field_to_default("mpc_bus"))
    mpc_branch_reset.configure(command=lambda: _reset_field_to_default("mpc_branch"))
    p_load_reset.configure(command=lambda: _reset_field_to_default("p_load"))
    q_load_reset.configure(command=lambda: _reset_field_to_default("q_load"))
    bus_mapping_reset.configure(command=lambda: _reset_field_to_default("bus_mapping"))
    gen_reset.configure(command=lambda: _reset_field_to_default("generators"))
    base_mva_reset.configure(command=lambda: _reset_field_to_default("base_mva"))

    for cfg in file_fields.values():
        cfg["var"].trace_add("write", _store_current_selections)

    def _on_network_change(*_args: object) -> None:
        _update_output_dir()
        _refresh_file_dropdowns()

    network_var.trace_add("write", _on_network_change)
    mode_var.trace_add("write", _render_mode_controls)

    _refresh_file_dropdowns()
    _render_mode_controls()
    _set_mode_options_expanded(True)
    _update_timestep_entry_state()

    load_scenario_var.trace_add("write", _update_timestep_entry_state)

    # Subprocess launch and kill support
    active_run_processes: list[dict[str, object]] = []

    def _update_run_controls() -> None:
        try:
            if not window.winfo_exists() or not kill_button.winfo_exists():
                return
            kill_button.config(state="normal" if active_run_processes else "disabled")
        except tk.TclError:
            return

    def _collect_run_job() -> dict:
        warm_start_mode = _legacy_opf_warm_start_key()
        opf_model_key = _selected_opf_model_key()
        dispatch_method_key = _selected_dispatch_method_key()
        physical_method_key = _selected_physical_method_key()
        economic_method_key = _selected_economic_method_key()
        job = {
            "mode": current_mode_key,
            "opf_model": opf_model_key,
            "dispatch_method": dispatch_method_key,
            "physical_state_method": physical_method_key,
            "economic_state_method": economic_method_key,
            "network": network_var.get().strip(),
            "input_folder": str(data_folder / network_var.get().strip()) if network_var.get().strip() else "",
            "output_path": output_var.get().strip() or str(ROOT / "run_outputs"),
            "load_mode": load_scenario_var.get().strip().lower(),
            "load_mode_label": load_scenario_var.get().strip(),
            "timestep_text": timestep_var.get().strip(),
            "start_idx": start_idx_var.get().strip(),
            "end_idx": end_idx_var.get().strip(),
            "export_to_db": bool(export_to_db_var.get()),
            "continue_on_socp_infeasible": bool(opf_continue_on_infeasible_var.get()),
            "opf_continue_on_infeasible": bool(opf_continue_on_infeasible_var.get()),
            "mpc_base_mva_filename": base_mva_var.get().strip() or "mpc_base_mva",
            "mpc_bus_filename": mpc_bus_var.get().strip(),
            "mpc_branch_filename": mpc_branch_var.get().strip(),
            "p_load_filename": p_load_var.get().strip(),
            "q_load_filename": q_load_var.get().strip(),
            "bus_mapping_filename": bus_mapping_var.get().strip(),
            "generators_filename": gen_var.get().strip() or "generators.xlsx",
            "load_interpretation_mode": LOAD_INTERPRETATION_LABEL_TO_KEY.get(load_interp_var.get(), "auto"),
            "opf_warm_start": opf_warm_start_var.get().strip(),
            "socp_warm_start": warm_start_mode,
            "show_plots": False,
            "export_plots": False,
            "line_color_mode": "Flow on base MVA",
        }

        if current_mode_key == MODE_SOCP:
            job.update(
                {
                    "opf_solver_backend": opf_solver_backend_var.get() or "mosek",
                    "opf_verbose": bool(opf_verbose_var.get()),
                    "opf_allow_load_shedding": bool(opf_allow_load_shedding_var.get()),
                    "socp_solver_name": opf_solver_backend_var.get() or "mosek",
                    "socp_verbose": bool(opf_verbose_var.get()),
                    "socp_allow_load_shedding": bool(opf_allow_load_shedding_var.get()),
                }
            )
        elif current_mode_key == MODE_BFSA:
            job.update(
                {
                    "bfsa_max_iter": int(mode_widgets.get("bfsa_max_iter_var", tk.StringVar(value="100")).get()),
                    "bfsa_tol": float(mode_widgets.get("bfsa_tol_var", tk.StringVar(value="1e-3")).get()),
                    "bfsa_convergence_check": BFSA_CONVERGENCE_LABEL_TO_KEY.get(
                        mode_widgets.get(
                            "bfsa_convergence_check_var",
                            tk.StringVar(value=BFSA_CONVERGENCE_KEY_TO_LABEL["both"]),
                        ).get(),
                        "both",
                    ),
                    "bfsa_show_plot": bool(mode_widgets.get("bfsa_show_plot_var", tk.BooleanVar(value=False)).get()),
                    "bfsa_show_merit": bool(mode_widgets.get("bfsa_show_merit_var", tk.BooleanVar(value=False)).get()),
                    "bfsa_merit_order_mode": BFSA_MERIT_ORDER_LABEL_TO_KEY.get(
                        mode_widgets.get(
                            "bfsa_merit_order_mode_var",
                            tk.StringVar(value=BFSA_MERIT_ORDER_KEY_TO_LABEL["dummy"]),
                        ).get(),
                        "dummy",
                    ),
                    "bfsa_mcp_active_price": float(
                        mode_widgets.get("bfsa_mcp_active_price_var", tk.StringVar(value="0.0")).get()
                    ),
                    "bfsa_mcp_reactive_price": float(
                        mode_widgets.get("bfsa_mcp_reactive_price_var", tk.StringVar(value="0.0")).get()
                    ),
                }
            )
            job["show_plots"] = False
            job["export_plots"] = False
        else:
            job.update(
                {
                    "comparison_allow_load_shedding": bool(opf_allow_load_shedding_var.get()),
                    "comparison_bfsa_merit_order_mode": BFSA_MERIT_ORDER_LABEL_TO_KEY.get(
                        mode_widgets.get(
                            "comparison_bfsa_merit_order_mode_var",
                            tk.StringVar(value=BFSA_MERIT_ORDER_KEY_TO_LABEL["dummy"]),
                        ).get(),
                        "dummy",
                    ),
                    "comparison_bfsa_mcp_active_price": float(
                        mode_widgets.get("comparison_bfsa_mcp_active_price_var", tk.StringVar(value="0.0")).get()
                    ),
                    "comparison_bfsa_mcp_reactive_price": float(
                        mode_widgets.get("comparison_bfsa_mcp_reactive_price_var", tk.StringVar(value="0.0")).get()
                    ),
                }
            )
            job["show_plots"] = False
            job["export_plots"] = True

        if job["load_mode_label"].strip().lower() == "range":
            job["start_idx"] = start_idx_var.get().strip()
            job["end_idx"] = end_idx_var.get().strip()

        job["methods"] = methods_config_from_legacy_job(job)
        job["methods"]["opf"]["method"] = opf_model_key
        job["methods"]["dispatch"]["method"] = dispatch_method_key
        job["methods"]["physical"]["method"] = physical_method_key
        job["methods"]["economic"]["method"] = economic_method_key

        return job

    def _poll_run_processes() -> None:
        active_run_processes[:] = [record for record in active_run_processes if record["process"].poll() is None]
        _update_run_controls()
        if window.winfo_exists():
            window.after(1000, _poll_run_processes)

    def _kill_computation() -> None:
        """Stop ongoing subprocesses."""
        for record in active_run_processes:
            process = record["process"]
            if process.poll() is None:
                try:
                    process.terminate()
                except Exception:
                    pass
        _set_status("Stopping computation subprocesses...", "orange")
        _update_run_controls()

    def _run_selected_mode_wrapper() -> None:
        """Launch the current solver job in a subprocess."""
        job = _collect_run_job()
        output_dir = Path(job["output_path"])
        output_dir.mkdir(parents=True, exist_ok=True)
        log_dir = _get_categorized_output_dir(output_dir, "log")
        log_dir.mkdir(parents=True, exist_ok=True)

        job_file = tempfile.NamedTemporaryFile("w", delete=False, suffix=".json", prefix="powersoc_job_", dir=log_dir)
        json.dump(job, job_file)
        job_file.close()

        log_file = tempfile.NamedTemporaryFile("w", delete=False, suffix=".log", prefix="powersoc_run_", dir=log_dir)
        log_path = Path(log_file.name)
        log_file.close()
        job["log_path"] = str(log_path)
        Path(job_file.name).write_text(json.dumps(job), encoding="utf-8")

        try:
            process = subprocess.Popen(
                [sys.executable, "-u", "-m", "libs.shared.gui_runner", "--run-job-json", job_file.name],
                cwd=str(ROOT),
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as ex:
            _set_status(f"Error: could not launch subprocess - {ex}", "red")
            return

        active_run_processes.append({"process": process, "job_file": job_file.name, "log_path": str(log_path)})
        _update_run_controls()
        _set_status(
            f"Launched subprocess PID {process.pid}. Log: {log_path.name}",
            "blue",
        )

    run_button = ttk.Button(action_section, text="Run", command=_run_selected_mode_wrapper, width=11)
    run_button.grid(row=0, column=0, pady=(2, 6), ipady=8)

    kill_button = ttk.Button(action_section, text="Kill", command=_kill_computation, width=11, state="disabled")
    kill_button.grid(row=0, column=1, pady=(2, 6), ipady=8, padx=(2, 0))

    plots_button = ttk.Button(action_section, text="Plots", command=_launch_plot_gui_async, width=11)
    plots_button.grid(row=0, column=2, pady=(2, 6), ipady=8, padx=(2, 10))

    _poll_run_processes()
    window.mainloop()
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="COBRAS solver GUI runner")
    parser.add_argument("--run-job-json", type=str, default=None, help="Execute a serialized solver job and exit")
    parser.add_argument("--mode", type=str, default=MODE_SOCP, choices=[MODE_SOCP, MODE_BFSA, MODE_COMPARISON])
    args = parser.parse_args(argv)

    if args.run_job_json:
        job_path = Path(args.run_job_json)
        job = json.loads(job_path.read_text(encoding="utf-8"))
        log_path = Path(job.get("log_path") or job_path.with_suffix(".log"))
        log_path.parent.mkdir(parents=True, exist_ok=True)
        exit_code = 0
        try:
            with open(log_path, "a", encoding="utf-8") as log_file:
                original_stdout = sys.stdout
                original_stderr = sys.stderr
                sys.stdout = _TeeStream(original_stdout, log_file)
                sys.stderr = _TeeStream(original_stderr, log_file)
                try:
                    print(f"Logging run output to {log_path}")
                    _execute_run_job(job)
                except Exception:
                    exit_code = 1
                    traceback.print_exc()
                finally:
                    sys.stdout = original_stdout
                    sys.stderr = original_stderr
        finally:
            job_path.unlink(missing_ok=True)
        return exit_code

    build_solver_gui(initial_mode=args.mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
