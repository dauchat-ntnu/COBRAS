import sys
import math
import sqlite3
import pytest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from libs.shared import BranchData, BusData, GeneratorData, LoadData, OPFResult, PowerFlowCase
from libs.shared.timestep_database import TimestepDatabase
from libs.visualization import (
    compute_duration_curve,
    create_branch_boxplot,
    create_branch_timeseries_plot,
    create_bus_boxplot,
    create_bus_timeseries_plot,
    create_database_dashboard,
    create_duration_curve_plot,
    create_generator_timeseries_plot,
    create_price_duration_curve_plot,
    create_solve_time_timeseries_plot,
)
from examples.generate_plots import (
    _comparison_network_solvers,
    _should_use_network_comparison,
    build_comparison_value_tables,
    comparison_table_filename_stem,
    format_min_max_id_range,
)


def _build_case() -> PowerFlowCase:
    bus1 = BusData(bus_id=1, bus_type=3)
    bus2 = BusData(bus_id=2, bus_type=1)
    branch = BranchData(branch_id=0, from_bus=1, to_bus=2, r=0.01, x=0.03)
    generator = GeneratorData(gen_id=0, bus_id=1, p_max=10.0, q_max=5.0)
    load = LoadData(bus_id=2, p_d=0.4, q_d=0.1)

    return PowerFlowCase(
        buses=[bus1, bus2],
        branches=[branch],
        generators=[generator],
        loads=[load],
        root_bus=1,
    )


def _build_result(voltage: float, lmp_p: float, flow_p: float, solve_time: float) -> OPFResult:
    return OPFResult(
        voltages={1: 1.0, 2: voltage},
        flows={(1, 2): (flow_p, 0.02)},
        duals_p={1: 20.0, 2: lmp_p},
        duals_q={1: 2.0, 2: 1.0},
        generator_output={0: (0.6, 0.1)},
        cost=100.0,
        convergence_info={"termination_message": "optimal", "iterations": 12},
        solver_name="socp",
        solve_time=solve_time,
    )


def _build_multi_generator_case() -> PowerFlowCase:
    bus1 = BusData(bus_id=1, bus_type=3)
    bus2 = BusData(bus_id=2, bus_type=1)
    branch = BranchData(branch_id=0, from_bus=1, to_bus=2, r=0.01, x=0.03)
    gen0 = GeneratorData(gen_id=0, bus_id=1, p_max=10.0, q_max=5.0)
    gen1 = GeneratorData(gen_id=1, bus_id=2, p_max=8.0, q_max=4.0)
    load = LoadData(bus_id=2, p_d=0.4, q_d=0.1)

    return PowerFlowCase(
        buses=[bus1, bus2],
        branches=[branch],
        generators=[gen0, gen1],
        loads=[load],
        root_bus=1,
        base_mva=50.0,
    )


def _populate_database(db_path: Path) -> TimestepDatabase:
    case = _build_case()
    database = TimestepDatabase(str(db_path))
    database.create_tables(case)

    for timestep in (0, 1):
        socp_result = _build_result(1.0 + 0.01 * timestep, 24.0 + timestep, 0.50 + 0.05 * timestep, 0.10)
        bfsa_result = _build_result(0.99 + 0.01 * timestep, 23.0 + timestep, 0.45 + 0.04 * timestep, 0.12)
        database.store_timestep_result(timestep, "scenario", "socp", socp_result, case=case)
        database.store_timestep_result(timestep, "scenario", "bfsa", bfsa_result, case=case)

    return database


def _set_database_value_to_null(db_path: Path, table: str, where_values: dict[str, object], metric: str) -> None:
    assignments = " AND ".join(f"{key} = ?" for key in where_values)
    with sqlite3.connect(db_path) as con:
        con.execute(
            f"UPDATE {table} SET {metric} = NULL WHERE {assignments}",
            tuple(where_values.values()),
        )


def test_bus_timeseries_plot_exports_html(tmp_path: Path) -> None:
    database = _populate_database(tmp_path / "results.db")

    fig, html_path = create_bus_timeseries_plot(
        database,
        bus_id=2,
        solver_names=("socp", "bfsa"),
        metric="lmp_p",
        output_dir=tmp_path,
        file_stem="bus_prices",
    )

    assert len(fig.data) == 2
    assert html_path is not None
    assert html_path.exists()
    assert html_path.name == "bus_prices.html"


def test_bus_demand_timeseries_plot_uses_demand_metric(tmp_path: Path) -> None:
    database = _populate_database(tmp_path / "results.db")

    fig, html_path = create_bus_timeseries_plot(
        database,
        bus_id=2,
        solver_names=("socp", "bfsa"),
        metric="p_d",
        output_dir=tmp_path,
        file_stem="bus_demand",
    )

    assert len(fig.data) == 2
    assert list(fig.data[0].y) == [0.4, 0.4]
    assert html_path is not None
    assert html_path.exists()
    assert html_path.name == "bus_demand.html"


def test_bus_timeseries_plot_accepts_multiple_buses(tmp_path: Path) -> None:
    database = _populate_database(tmp_path / "results.db")

    fig, _ = create_bus_timeseries_plot(
        database,
        bus_ids=(1, 2),
        solver_names=("socp", "bfsa"),
        metric="lmp_p",
    )

    assert len(fig.data) == 4
    assert [trace.name for trace in fig.data] == [
        "SOCP bus 1",
        "BFSA bus 1",
        "SOCP bus 2",
        "BFSA bus 2",
    ]
    assert list(fig.data[2].y) == [24.0, 25.0]


def test_compute_duration_curve_sorts_values_descending() -> None:
    x_values, y_values = compute_duration_curve([3.0, float("nan"), 1.0, 5.0])

    assert x_values.tolist() == [0.0, 0.5, 1.0]
    assert y_values.tolist() == [5.0, 3.0, 1.0]


def test_price_duration_curve_plots_multiple_buses_and_solvers(tmp_path: Path) -> None:
    database = _populate_database(tmp_path / "results.db")

    fig, html_path = create_price_duration_curve_plot(
        database,
        bus_ids=(1, 2),
        solver_names=("socp", "bfsa"),
        metric="lmp_p",
        output_dir=tmp_path,
        file_stem="price_duration",
    )

    assert len(fig.data) == 4
    assert fig.data[1].x[0] == 0.0
    assert fig.data[1].x[-1] == 1.0
    assert list(fig.data[1].y) == [25.0, 24.0]
    assert html_path is not None
    assert html_path.exists()
    assert html_path.name == "price_duration.html"


def test_duration_curve_can_plot_branch_flow(tmp_path: Path) -> None:
    database = _populate_database(tmp_path / "results.db")

    fig, _ = create_duration_curve_plot(
        database,
        entity_kind="branch",
        entity_ids=(0,),
        solver_names=("socp",),
        metric="flow_p",
    )

    assert len(fig.data) == 1
    assert fig.data[0].name == "SOCP branch 0"
    assert list(fig.data[0].y) == [0.55, 0.5]
    assert "Active flow" in fig.layout.title.text


def test_price_duration_curve_can_plot_socp_bfsa_error(tmp_path: Path) -> None:
    database = _populate_database(tmp_path / "results.db")

    fig, _ = create_price_duration_curve_plot(
        database,
        bus_ids=(2,),
        solver_names=(),
        metric="lmp_p",
        duration_mode="error",
    )

    assert len(fig.data) == 1
    assert fig.data[0].name == "BFSA-SOCP (%) bus 2"
    assert list(fig.data[0].y) == pytest.approx([-4.0, -100.0 / 24.0])
    assert "BFSA-SOCP" in fig.layout.title.text


def test_bus_timeseries_can_mask_bfsa_where_socp_is_nan(tmp_path: Path) -> None:
    db_path = tmp_path / "results.db"
    database = _populate_database(db_path)
    _set_database_value_to_null(
        db_path,
        "Res_Buses",
        {"timestep": 1, "solver_name": "socp", "bus_id": 2},
        "lmp_p",
    )

    fig, _ = create_bus_timeseries_plot(
        database,
        bus_id=2,
        solver_names=("bfsa",),
        metric="lmp_p",
        exclude_nan_timesteps=True,
    )

    assert list(fig.data[0].x) == [0, 1]
    assert fig.data[0].y[0] == 23.0
    assert math.isnan(fig.data[0].y[1])
    assert database.get_bus_timeseries(2, 0, 1, "bfsa", "lmp_p") == [(0, 23.0), (1, 24.0)]


def test_price_duration_excludes_bfsa_values_at_socp_nan_timesteps(tmp_path: Path) -> None:
    db_path = tmp_path / "results.db"
    database = _populate_database(db_path)
    _set_database_value_to_null(
        db_path,
        "Res_Buses",
        {"timestep": 1, "solver_name": "socp", "bus_id": 2},
        "lmp_p",
    )

    fig, _ = create_price_duration_curve_plot(
        database,
        bus_ids=(2,),
        solver_names=("bfsa",),
        metric="lmp_p",
        exclude_nan_timesteps=True,
    )

    assert list(fig.data[0].y) == [23.0]


def test_branch_timeseries_can_mask_bfsa_where_socp_is_nan(tmp_path: Path) -> None:
    db_path = tmp_path / "results.db"
    database = _populate_database(db_path)
    _set_database_value_to_null(
        db_path,
        "Res_Branches",
        {"timestep": 1, "solver_name": "socp", "branch_id": 0},
        "flow_p",
    )

    fig, _ = create_branch_timeseries_plot(
        database,
        branch_id=0,
        solver_names=("bfsa",),
        metric="flow_p",
        exclude_nan_timesteps=True,
    )

    assert fig.data[0].y[0] == 0.45
    assert math.isnan(fig.data[0].y[1])
    assert database.get_branch_timeseries(0, 0, 1, "bfsa", "flow_p") == [(0, 0.45), (1, 0.49)]


def test_dashboard_plot_combines_bus_and_branch_series(tmp_path: Path) -> None:
    database = _populate_database(tmp_path / "results.db")

    fig, html_path = create_database_dashboard(
        database,
        bus_id=2,
        branch_id=0,
        solver_names=("socp", "bfsa"),
        start_timestep=0,
        end_timestep=1,
        output_dir=tmp_path,
        file_stem="dashboard",
    )

    assert len(fig.data) == 4
    assert html_path is not None
    assert html_path.exists()
    assert html_path.name == "dashboard.html"


def test_dashboard_plot_accepts_bus_demand_metric(tmp_path: Path) -> None:
    database = _populate_database(tmp_path / "results.db")

    fig, _ = create_database_dashboard(
        database,
        bus_id=2,
        branch_id=0,
        solver_names=("socp", "bfsa"),
        bus_metric="p_d",
        branch_metric="flow_p",
        start_timestep=0,
        end_timestep=1,
        output_dir=tmp_path,
        file_stem="dashboard_demand",
    )

    assert len(fig.data) == 4
    assert fig.layout.annotations[0].text.startswith("Bus 2 - Active power demand")


def test_generator_timeseries_plot_scales_and_fills_missing_rows(tmp_path: Path) -> None:
    case = _build_multi_generator_case()
    database = TimestepDatabase(str(tmp_path / "results.db"))
    database.create_tables(case)

    result = OPFResult(
        voltages={1: 1.0, 2: 0.99},
        flows={(1, 2): (0.5, 0.02)},
        duals_p={1: 20.0, 2: 19.0},
        duals_q={1: 2.0, 2: 1.0},
        generator_output={0: (0.6, 0.1)},
        cost=100.0,
        convergence_info={"termination_message": "optimal"},
        solver_name="socp",
        solve_time=0.1,
    )

    database.store_timestep_result(0, "scenario", "socp", result, case=case)

    missing_rows = database.get_generator_timeseries(1, 0, 0, "socp")
    assert missing_rows == [(0, 0.0)]

    fig_pu, html_path_pu = create_generator_timeseries_plot(
        database,
        solver_name="socp",
        absolute=False,
        output_dir=tmp_path,
        file_stem="generator_pu",
    )
    assert len(fig_pu.data) == 2
    assert fig_pu.data[0].y[0] == 0.6
    assert fig_pu.data[1].y[0] == 0.0
    assert html_path_pu is not None
    assert html_path_pu.exists()

    fig_abs, html_path_abs = create_generator_timeseries_plot(
        database,
        solver_name="socp",
        absolute=True,
        base_mva=case.base_mva,
        output_dir=tmp_path,
        file_stem="generator_abs",
    )
    assert len(fig_abs.data) == 2
    assert fig_abs.data[0].y[0] == 30.0
    assert html_path_abs is not None
    assert html_path_abs.exists()


def test_solve_time_timeseries_plot_uses_timesteps_on_x_axis(tmp_path: Path) -> None:
    database = _populate_database(tmp_path / "results.db")

    fig, html_path = create_solve_time_timeseries_plot(
        database,
        solver_names=("socp",),
        output_dir=tmp_path,
        file_stem="solve_times",
    )

    assert len(fig.data) == 1
    assert list(fig.data[0].x) == [0, 1]
    assert list(fig.data[0].y) == [0.1, 0.1]
    assert fig.layout.yaxis.title.text == "Solve time (s)"
    assert html_path is not None
    assert html_path.exists()
    assert html_path.name == "solve_times.html"


def test_solve_time_timeseries_plot_shows_bfsa_iterations_on_left_axis(tmp_path: Path) -> None:
    database = _populate_database(tmp_path / "results.db")

    fig, html_path = create_solve_time_timeseries_plot(
        database,
        solver_names=("bfsa", "socp"),
        output_dir=tmp_path,
        file_stem="solve_times_bfsa",
    )

    assert len(fig.data) == 3
    assert list(fig.data[0].y) == [12, 12]
    assert fig.layout.yaxis.title.text == "BFSA iterations"
    assert fig.layout.yaxis2.title.text == "Solve time (s)"
    assert html_path is not None
    assert html_path.exists()


def test_branch_timeseries_plot_handles_single_metric(tmp_path: Path) -> None:
    database = _populate_database(tmp_path / "results.db")

    fig, html_path = create_branch_timeseries_plot(
        database,
        branch_id=0,
        solver_names=("socp",),
        metric="flow_p",
        output_dir=tmp_path,
        file_stem="branch_flow",
    )

    assert len(fig.data) == 1
    assert html_path is not None
    assert html_path.exists()


def test_bus_boxplot_groups_solvers_by_bus(tmp_path: Path) -> None:
    database = _populate_database(tmp_path / "results.db")

    fig, html_path = create_bus_boxplot(
        database,
        solver_names=("socp", "bfsa"),
        metric="lmp_p",
        bus_ids=(1, 2),
        output_dir=tmp_path,
        file_stem="bus_boxplot",
    )

    assert len(fig.data) == 2
    assert fig.layout.boxmode == "group"
    assert html_path is not None
    assert html_path.exists()
    assert html_path.name == "bus_boxplot.html"


def test_branch_boxplot_can_plot_all_branches(tmp_path: Path) -> None:
    database = _populate_database(tmp_path / "results.db")

    fig, html_path = create_branch_boxplot(
        database,
        solver_names=("socp", "bfsa"),
        metric="flow_p",
        branch_ids=None,
        output_dir=tmp_path,
        file_stem="branch_boxplot",
    )

    assert len(fig.data) == 2
    assert html_path is not None
    assert html_path.exists()


def test_network_plot_switches_to_comparison_only_for_bfsa_and_socp() -> None:
    assert _comparison_network_solvers(["socp"]) == ["socp"]
    assert _comparison_network_solvers(["bfsa"]) == ["bfsa"]
    assert _comparison_network_solvers(["bfsa", "socp"]) == ["bfsa", "socp"]
    assert _comparison_network_solvers(["socp", "bfsa", "other"]) == ["socp", "bfsa"]
    assert not _should_use_network_comparison(["socp"])
    assert not _should_use_network_comparison(["bfsa"])
    assert _should_use_network_comparison(["bfsa", "socp"])
    assert _should_use_network_comparison(["socp", "bfsa", "other"])


def test_format_min_max_id_range_for_gui_defaults() -> None:
    assert format_min_max_id_range([3, 1, 2]) == "1-3"
    assert format_min_max_id_range([4]) == "4"
    assert format_min_max_id_range([]) == ""


def test_comparison_value_table_accepts_single_solver(tmp_path: Path) -> None:
    database = _populate_database(tmp_path / "results.db")

    value_df, time_df = build_comparison_value_tables(
        database,
        ["bfsa"],
        0,
        1,
        table_type="aggregated",
    )

    lmp_row = value_df[
        (value_df["mode"] == "bfsa")
        & (value_df["entity_kind"] == "bus")
        & (value_df["metric"] == "lmp_p")
    ].iloc[0]
    assert lmp_row["count"] == 4
    assert lmp_row["mean"] == 21.75
    assert set(time_df["solver"]) == {"bfsa"}


def test_comparison_absolute_difference_uses_solver_value_delta(tmp_path: Path) -> None:
    database = _populate_database(tmp_path / "results.db")

    value_df, time_df = build_comparison_value_tables(
        database,
        ["bfsa", "socp"],
        0,
        1,
        table_type="aggregated",
        difference=True,
    )

    lmp_row = value_df[
        (value_df["mode"] == "bfsa-socp")
        & (value_df["entity_kind"] == "bus")
        & (value_df["metric"] == "lmp_p")
    ].iloc[0]
    assert lmp_row["count"] == 4
    assert lmp_row["mean"] == -0.5
    assert "mean_%" not in value_df.columns
    assert set(time_df["solver"]) == {"bfsa", "socp", "bfsa-socp"}


def test_comparison_table_filename_includes_timesteps_and_single_solver() -> None:
    assert (
        comparison_table_filename_stem(
            "database_plot",
            "comparison_aggregated_values",
            3,
            7,
            ["bfsa"],
            difference=False,
        )
        == "database_plot_comparison_aggregated_values_ts3-7_bfsa"
    )


def test_comparison_table_filename_omits_solver_for_difference() -> None:
    assert (
        comparison_table_filename_stem(
            "database_plot",
            "comparison_aggregated_absolute_difference",
            3,
            7,
            ["bfsa", "socp"],
            difference=True,
        )
        == "database_plot_comparison_aggregated_absolute_difference_ts3-7"
    )
