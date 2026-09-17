"""
Database-backed visualizations for multi-timestep COBRAS results.

This module turns rows stored in ``TimestepDatabase`` into interactive Plotly
figures for bus prices, branch flows, and compact comparison dashboards.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from libs.shared.timestep_database import TimestepDatabase


DatabaseLike = Union[TimestepDatabase, str, Path]

PLOTLY_HIGH_RES_EXPORT_CONFIG = {
    "toImageButtonOptions": {
        "format": "png",
        "scale": 4,
    },
}

_BUS_METRIC_LABELS = {
    "voltage": "Voltage squared (p.u. squared)",
    "p_d": "Active power demand (p.u.)",
    "q_d": "Reactive power demand (p.u.)",
    "lmp_p": "Active LMP",
    "lmp_q": "Reactive LMP",
}

_BRANCH_METRIC_LABELS = {
    "flow_p": "Active flow (p.u.)",
    "flow_q": "Reactive flow (p.u.)",
    "ell": "Current squared (p.u.)",
}

_GENERATOR_Y_LABELS = {
    False: "Active generation (p.u.)",
    True: "Active generation (MW)",
}

_SOLVER_COLORS = ["#2563eb", "#dc2626", "#16a34a", "#7c3aed", "#ea580c", "#0891b2"]
_REFERENCE_SOLVER_FOR_NAN_MASK = "socp"


def _coerce_database(database: DatabaseLike) -> TimestepDatabase:
    if isinstance(database, TimestepDatabase):
        return database

    database_path = Path(database)
    if not database_path.exists():
        raise FileNotFoundError(f"Database file not found: {database_path}")

    return TimestepDatabase(str(database_path))


def _resolve_timestep_bounds(
    database: TimestepDatabase,
    start_timestep: Optional[int],
    end_timestep: Optional[int],
) -> Tuple[int, int]:
    timesteps = database.get_timerange()
    if not timesteps:
        raise ValueError(f"Database does not contain any timestep results: {database.filename}")

    resolved_start = timesteps[0] if start_timestep is None else int(start_timestep)
    resolved_end = timesteps[-1] if end_timestep is None else int(end_timestep)

    if resolved_start > resolved_end:
        raise ValueError("start_timestep must be less than or equal to end_timestep")

    return resolved_start, resolved_end


def _metric_label(metric: str, entity_kind: str) -> str:
    if entity_kind == "bus":
        return _BUS_METRIC_LABELS.get(metric, metric.replace("_", " ").title())
    return _BRANCH_METRIC_LABELS.get(metric, metric.replace("_", " ").title())


def _series_for_solver(
    database: TimestepDatabase,
    entity_kind: str,
    entity_id: int,
    metric: str,
    solver_name: str,
    start_timestep: int,
    end_timestep: int,
) -> list[tuple[int, float]]:
    if entity_kind == "bus":
        return database.get_bus_timeseries(entity_id, start_timestep, end_timestep, solver_name, metric)
    if entity_kind == "branch":
        return database.get_branch_timeseries(entity_id, start_timestep, end_timestep, solver_name, metric)
    raise ValueError(f"Unsupported entity kind: {entity_kind}")


def _generator_series_for_solver(
    database: TimestepDatabase,
    generator_id: int,
    solver_name: str,
    start_timestep: int,
    end_timestep: int,
) -> list[tuple[int, float]]:
    return database.get_generator_timeseries(generator_id, start_timestep, end_timestep, solver_name)


def _solver_color(index: int) -> str:
    return _SOLVER_COLORS[index % len(_SOLVER_COLORS)]


def _to_float(value: object) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _is_nan_value(value: object) -> bool:
    return not np.isfinite(_to_float(value))


def _socp_nan_timesteps(
    database: TimestepDatabase,
    entity_kind: str,
    entity_id: int,
    metric: str,
    start_timestep: int,
    end_timestep: int,
) -> set[int]:
    if entity_kind == "generator":
        reference_rows = _generator_series_for_solver(
            database,
            entity_id,
            _REFERENCE_SOLVER_FOR_NAN_MASK,
            start_timestep,
            end_timestep,
        )
    else:
        reference_rows = _series_for_solver(
            database,
            entity_kind,
            entity_id,
            metric,
            _REFERENCE_SOLVER_FOR_NAN_MASK,
            start_timestep,
            end_timestep,
        )

    return {int(timestep) for timestep, value in reference_rows if _is_nan_value(value)}


def _mask_rows_at_timesteps(
    rows: Sequence[tuple[int, float]],
    masked_timesteps: set[int],
) -> list[tuple[int, float]]:
    if not masked_timesteps:
        return [(int(timestep), _to_float(value)) for timestep, value in rows]

    return [
        (int(timestep), float("nan") if int(timestep) in masked_timesteps else _to_float(value))
        for timestep, value in rows
    ]


def _series_for_plot(
    database: TimestepDatabase,
    entity_kind: str,
    entity_id: int,
    metric: str,
    solver_name: str,
    start_timestep: int,
    end_timestep: int,
    exclude_nan_timesteps: bool,
) -> list[tuple[int, float]]:
    rows = _series_for_solver(database, entity_kind, entity_id, metric, solver_name, start_timestep, end_timestep)
    if not exclude_nan_timesteps:
        return _mask_rows_at_timesteps(rows, set())

    masked_timesteps = _socp_nan_timesteps(database, entity_kind, entity_id, metric, start_timestep, end_timestep)
    return _mask_rows_at_timesteps(rows, masked_timesteps)


def _generator_series_for_plot(
    database: TimestepDatabase,
    generator_id: int,
    solver_name: str,
    start_timestep: int,
    end_timestep: int,
    exclude_nan_timesteps: bool,
) -> list[tuple[int, float]]:
    rows = _generator_series_for_solver(database, generator_id, solver_name, start_timestep, end_timestep)
    if not exclude_nan_timesteps:
        return _mask_rows_at_timesteps(rows, set())

    masked_timesteps = _socp_nan_timesteps(
        database,
        "generator",
        generator_id,
        "dispatch_p",
        start_timestep,
        end_timestep,
    )
    return _mask_rows_at_timesteps(rows, masked_timesteps)


def compute_duration_curve(values: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """Return exceedance fractions and values sorted from high to low.

    The returned x-axis runs from 0 to 1. ``y[0]`` is the maximum value and
    ``y[-1]`` is the minimum value, so ``y`` is directly usable as a duration
    curve for prices, loads, or any other sampled quantity.
    """

    finite_values = np.asarray(values, dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        raise ValueError("Cannot compute a duration curve from an empty array")

    sorted_values = np.sort(finite_values)[::-1]
    if sorted_values.size == 1:
        exceedance_fraction = np.array([0.0])
    else:
        exceedance_fraction = np.linspace(0.0, 1.0, sorted_values.size)

    return exceedance_fraction, sorted_values


def _finalize_figure(
    fig: go.Figure,
    title: str,
    yaxis_title: str,
    output_dir: Optional[Path],
    file_stem: Optional[str],
    show: bool,
) -> tuple[go.Figure, Optional[Path]]:
    fig.update_layout(
        title=title,
        template="plotly_white",
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0.0},
        margin={"l": 70, "r": 30, "t": 90, "b": 60},
    )
    if yaxis_title:
        fig.update_yaxes(title_text=yaxis_title)

    saved_path = None
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = file_stem or "database_timeseries"
        saved_path = output_dir / f"{stem}.html"
        fig.write_html(str(saved_path), include_plotlyjs="cdn", config=PLOTLY_HIGH_RES_EXPORT_CONFIG)

    if show:
        fig.show(config=PLOTLY_HIGH_RES_EXPORT_CONFIG)

    return fig, saved_path


def _resolve_entity_ids(
    database: TimestepDatabase,
    entity_kind: str,
    entity_ids: Optional[Sequence[int]],
) -> list[int]:
    if entity_ids is not None:
        return [int(entity_id) for entity_id in entity_ids]
    if entity_kind == "bus":
        return database.get_bus_ids()
    if entity_kind == "branch":
        return database.get_branch_ids()
    raise ValueError(f"Unsupported entity kind: {entity_kind}")


def _entity_boxplot(
    database: DatabaseLike,
    entity_kind: str,
    entity_ids: Optional[Sequence[int]],
    solver_names: Sequence[str],
    metric: str,
    start_timestep: Optional[int] = None,
    end_timestep: Optional[int] = None,
    output_dir: Optional[Path] = None,
    file_stem: Optional[str] = None,
    show: bool = False,
    exclude_nan_timesteps: bool = False,
    difference: bool = False,
    difference_mode: str = "percent",
) -> tuple[go.Figure, Optional[Path]]:
    db = _coerce_database(database)
    start_timestep, end_timestep = _resolve_timestep_bounds(db, start_timestep, end_timestep)
    resolved_ids = _resolve_entity_ids(db, entity_kind, entity_ids)

    fig = go.Figure()
    series_found = False
    x_title = "Bus ID" if entity_kind == "bus" else "Line ID"

    for index, solver_name in enumerate(solver_names):
        if difference and index > 0:
            continue

        x_values: list[str] = []
        y_values: list[float] = []
        trace_name = solver_name.upper()

        for entity_id in resolved_ids:
            if difference and len(solver_names) >= 2 and "bfsa" in [s.lower() for s in solver_names] and "socp" in [s.lower() for s in solver_names]:
                bfsa_rows = _series_for_plot(db, entity_kind, entity_id, metric, "bfsa", start_timestep, end_timestep, exclude_nan_timesteps)
                socp_rows = _series_for_plot(db, entity_kind, entity_id, metric, "socp", start_timestep, end_timestep, exclude_nan_timesteps)
                bfsa_map = {int(t): _to_float(v) for t, v in bfsa_rows}
                socp_map = {int(t): _to_float(v) for t, v in socp_rows}
                shared = sorted(set(bfsa_map.keys()) & set(socp_map.keys()))
                if not shared:
                    continue
                series_vals = []
                for t in shared:
                    a = bfsa_map[t]
                    b = socp_map[t]
                    if difference_mode == "raw":
                        series_vals.append(a - b)
                    else:
                        series_vals.append(100.0 * (a - b) / (abs(b) if abs(b) > 0 else float("nan")))
                if series_vals:
                    x_values.extend([str(entity_id)] * len(series_vals))
                    y_values.extend(series_vals)
                    series_found = True
                    trace_name = "BFSA-SOCP (%)" if difference_mode != "raw" else "BFSA-SOCP"
            else:
                rows = _series_for_plot(
                    db,
                    entity_kind,
                    entity_id,
                    metric,
                    solver_name,
                    start_timestep,
                    end_timestep,
                    exclude_nan_timesteps,
                )
                if not rows:
                    continue
                series_found = True
                x_values.extend([str(entity_id)] * len(rows))
                y_values.extend([_to_float(value) for _, value in rows])

        if x_values:
            fig.add_trace(
                go.Box(
                    x=x_values,
                    y=y_values,
                    name=trace_name,
                    marker_color=_solver_color(index),
                    boxmean=True,
                    offsetgroup=trace_name,
                    hovertemplate=f"{trace_name}<br>%{{x}}<br>Value %{{y:.6f}}<extra></extra>",
                )
            )

    if not series_found:
        entity_label = "bus" if entity_kind == "bus" else "branch"
        raise ValueError(
            f"No data found for selected {entity_label} IDs and metric '{metric}' in range [{start_timestep}, {end_timestep}]"
        )

    title = f"{entity_kind.title()} distribution for {_metric_label(metric, entity_kind)}"
    yaxis_title = _metric_label(metric, entity_kind)
    fig.update_layout(boxmode="group")
    fig.update_xaxes(title_text=x_title, categoryorder="array", categoryarray=[str(entity_id) for entity_id in resolved_ids])
    return _finalize_figure(fig, title, yaxis_title, output_dir, file_stem, show)


def create_bus_timeseries_plot(
    database: DatabaseLike,
    bus_id: Optional[Union[int, Sequence[int]]] = None,
    solver_names: Sequence[str] = (),
    metric: str = "lmp_p",
    start_timestep: Optional[int] = None,
    end_timestep: Optional[int] = None,
    output_dir: Optional[Path] = None,
    file_stem: Optional[str] = None,
    show: bool = False,
    exclude_nan_timesteps: bool = False,
    difference: bool = False,
    difference_mode: str = "percent",
    bus_ids: Optional[Sequence[int]] = None,
) -> tuple[go.Figure, Optional[Path]]:
    """Create a Plotly line chart for one or more buses across one or more solvers."""

    db = _coerce_database(database)
    start_timestep, end_timestep = _resolve_timestep_bounds(db, start_timestep, end_timestep)
    if bus_ids is not None:
        resolved_bus_ids = [int(value) for value in bus_ids]
    elif bus_id is None:
        resolved_bus_ids = []
    elif isinstance(bus_id, (str, bytes)):
        resolved_bus_ids = [int(bus_id)]
    else:
        try:
            resolved_bus_ids = [int(value) for value in bus_id]  # type: ignore[arg-type]
        except TypeError:
            resolved_bus_ids = [int(bus_id)]

    if not resolved_bus_ids:
        raise ValueError("At least one bus ID is required for a bus timeseries plot")

    fig = go.Figure()
    series_found = False
    trace_index = 0
    multiple_buses = len(resolved_bus_ids) > 1

    # Difference mode: compute BFSA - SOCP series (raw or percent)
    if difference and len(solver_names) >= 2 and "bfsa" in [s.lower() for s in solver_names] and "socp" in [s.lower() for s in solver_names]:
        for bus_index, resolved_bus_id in enumerate(resolved_bus_ids):
            bfsa_rows = _series_for_plot(db, "bus", resolved_bus_id, metric, "bfsa", start_timestep, end_timestep, exclude_nan_timesteps)
            socp_rows = _series_for_plot(db, "bus", resolved_bus_id, metric, "socp", start_timestep, end_timestep, exclude_nan_timesteps)
            bfsa_map = {int(t): _to_float(v) for t, v in bfsa_rows}
            socp_map = {int(t): _to_float(v) for t, v in socp_rows}
            shared = sorted(set(bfsa_map.keys()) & set(socp_map.keys()))
            if not shared:
                continue
            x_values = shared
            y_values = []
            for t in shared:
                a = bfsa_map[t]
                b = socp_map[t]
                if difference_mode == "raw":
                    y = a - b
                else:
                    y = 100.0 * (a - b) / (abs(b) if abs(b) > 0 else float("nan"))
                y_values.append(y)
            series_found = True
            base_name = "BFSA-SOCP (%)" if difference_mode != "raw" else "BFSA-SOCP"
            name = f"{base_name} bus {resolved_bus_id}" if multiple_buses else base_name
            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=y_values,
                    mode="lines+markers",
                    name=name,
                    line={"width": 2.2, "color": _solver_color(bus_index)},
                    marker={"size": 6},
                    hovertemplate=(
                        f"Difference {base_name}<br>"
                        f"Bus {resolved_bus_id}<br>"
                        "Timestep %{x}<br>Value %{y:.6f}<extra></extra>"
                    ),
                )
            )
    else:
        for resolved_bus_id in resolved_bus_ids:
            for solver_name in solver_names:
                rows = _series_for_plot(
                    db,
                    "bus",
                    resolved_bus_id,
                    metric,
                    solver_name,
                    start_timestep,
                    end_timestep,
                    exclude_nan_timesteps,
                )
                if not rows:
                    continue
                series_found = True
                x_values = [row[0] for row in rows]
                y_values = [_to_float(row[1]) for row in rows]
                name = f"{solver_name.upper()} bus {resolved_bus_id}" if multiple_buses else solver_name.upper()
                fig.add_trace(
                    go.Scatter(
                        x=x_values,
                        y=y_values,
                        mode="lines+markers",
                        name=name,
                        line={"width": 2.2, "color": _solver_color(trace_index)},
                        marker={"size": 6},
                        hovertemplate=(
                            f"Solver {solver_name.upper()}<br>"
                            f"Bus {resolved_bus_id}<br>"
                            "Timestep %{x}<br>Value %{y:.6f}<extra></extra>"
                        ),
                    )
                )
                trace_index += 1

    if not series_found:
        raise ValueError(
            f"No data found for buses {resolved_bus_ids} and metric '{metric}' in range [{start_timestep}, {end_timestep}]"
        )

    bus_label = f"Bus {resolved_bus_ids[0]}" if len(resolved_bus_ids) == 1 else f"Buses {', '.join(str(value) for value in resolved_bus_ids)}"
    title = f"{bus_label} {_metric_label(metric, 'bus')} over time"
    yaxis_title = _metric_label(metric, "bus")
    return _finalize_figure(fig, title, yaxis_title, output_dir, file_stem, show)


def create_duration_curve_plot(
    database: DatabaseLike,
    entity_kind: str,
    entity_ids: Sequence[int],
    solver_names: Sequence[str],
    metric: str,
    start_timestep: Optional[int] = None,
    end_timestep: Optional[int] = None,
    output_dir: Optional[Path] = None,
    file_stem: Optional[str] = None,
    show: bool = False,
    exclude_nan_timesteps: bool = False,
    duration_mode: str = "series",
    difference: bool = False,
    difference_mode: str = "percent",
) -> tuple[go.Figure, Optional[Path]]:
    """Create duration curves for one or more bus or branch metrics."""

    entity_kind = entity_kind.strip().lower()
    if entity_kind not in {"bus", "branch"}:
        raise ValueError("Duration curve entity_kind must be 'bus' or 'branch'.")

    valid_metrics = {"bus": {"voltage", "p_d", "q_d", "lmp_p", "lmp_q"}, "branch": {"flow_p", "flow_q"}}
    if metric not in valid_metrics[entity_kind]:
        raise ValueError(f"Metric '{metric}' is not valid for {entity_kind} duration curves.")

    db = _coerce_database(database)
    start_timestep, end_timestep = _resolve_timestep_bounds(db, start_timestep, end_timestep)
    resolved_entity_ids = [int(entity_id) for entity_id in entity_ids]
    if not resolved_entity_ids:
        raise ValueError(f"At least one {entity_kind} ID is required for a duration curve")

    fig = go.Figure()
    series_found = False
    trace_index = 0
    duration_mode_key = duration_mode.strip().lower().replace("_", "-")
    entity_label = "Bus" if entity_kind == "bus" else "Branch"
    metric_label = _metric_label(metric, entity_kind)

    if duration_mode_key in {"error", "bfsa-socp", "bfsa-socp error", "socp-bfsa", "socp-bfsa error"}:
        for entity_id in resolved_entity_ids:
            socp_rows = _series_for_plot(
                db,
                entity_kind,
                entity_id,
                metric,
                "socp",
                start_timestep,
                end_timestep,
                exclude_nan_timesteps,
            )
            bfsa_rows = _series_for_plot(
                db,
                entity_kind,
                entity_id,
                metric,
                "bfsa",
                start_timestep,
                end_timestep,
                exclude_nan_timesteps,
            )
            socp_by_timestep = {int(timestep): _to_float(value) for timestep, value in socp_rows}
            bfsa_by_timestep = {int(timestep): _to_float(value) for timestep, value in bfsa_rows}
            shared_timesteps = sorted(set(socp_by_timestep) & set(bfsa_by_timestep))

            # BFSA - SOCP, with percent error using SOCP as the reference.
            values = [
                (
                    bfsa_by_timestep[t] - socp_by_timestep[t]
                    if difference_mode == "raw"
                    else 100.0
                    * (bfsa_by_timestep[t] - socp_by_timestep[t])
                    / (abs(socp_by_timestep[t]) if abs(socp_by_timestep[t]) > 0 else float("nan"))
                )
                for t in shared_timesteps
            ]

            if not values:
                continue

            try:
                x_values, y_values = compute_duration_curve(values)
            except ValueError:
                continue

            series_found = True
            name_prefix = "BFSA-SOCP (%)" if difference_mode != "raw" else "BFSA-SOCP"
            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=y_values,
                    mode="lines",
                    name=f"{name_prefix} {entity_label.lower()} {entity_id}",
                    line={"width": 2.2, "color": _solver_color(trace_index)},
                    hovertemplate=(
                        f"Error {entity_label} {entity_id}<br>"
                        "Exceedance % %{x:.2%}<br>"
                        "Error %{y:.6f}<extra></extra>"
                    ),
                )
            )
            trace_index += 1
    elif duration_mode_key in {"series", "solvers", "socp/bfsa"}:
        for solver_name in solver_names:
            for entity_id in resolved_entity_ids:
                rows = _series_for_plot(
                    db,
                    entity_kind,
                    entity_id,
                    metric,
                    solver_name,
                    start_timestep,
                    end_timestep,
                    exclude_nan_timesteps,
                )
                if not rows:
                    continue

                try:
                    x_values, y_values = compute_duration_curve([value for _, value in rows])
                except ValueError:
                    continue

                series_found = True
                fig.add_trace(
                    go.Scatter(
                        x=x_values,
                        y=y_values,
                        mode="lines",
                        name=f"{solver_name.upper()} {entity_label.lower()} {entity_id}",
                        line={"width": 2.2, "color": _solver_color(trace_index)},
                        hovertemplate=(
                            f"Solver {solver_name.upper()}<br>"
                            f"{entity_label} {entity_id}<br>"
                            "Exceedance % %{x:.2%}<br>"
                            "Value %{y:.6f}<extra></extra>"
                        ),
                    )
                )
                trace_index += 1
    else:
        raise ValueError("Duration mode must be 'series' or 'error'.")

    if not series_found:
        raise ValueError(
            f"No data found for selected {entity_kind}s and metric '{metric}' in range [{start_timestep}, {end_timestep}]"
        )

    fig.update_xaxes(title_text="Fraction of timesteps with value at or above level", range=[0, 1])
    if duration_mode_key in {"error", "bfsa-socp", "bfsa-socp error", "socp-bfsa", "socp-bfsa error"}:
        title = f"Duration curve for {metric_label} error (BFSA-SOCP)"
        yaxis_title = f"{metric_label} error (BFSA-SOCP)"
    else:
        title = f"Duration curve for {metric_label}"
        yaxis_title = metric_label
    return _finalize_figure(fig, title, yaxis_title, output_dir, file_stem, show)


def create_price_duration_curve_plot(
    database: DatabaseLike,
    bus_ids: Sequence[int],
    solver_names: Sequence[str],
    metric: str = "lmp_p",
    start_timestep: Optional[int] = None,
    end_timestep: Optional[int] = None,
    output_dir: Optional[Path] = None,
    file_stem: Optional[str] = None,
    show: bool = False,
    exclude_nan_timesteps: bool = False,
    duration_mode: str = "series",
) -> tuple[go.Figure, Optional[Path]]:
    """Create price duration curves for one or more buses and solvers."""

    return create_duration_curve_plot(
        database=database,
        entity_kind="bus",
        entity_ids=bus_ids,
        solver_names=solver_names,
        metric=metric,
        start_timestep=start_timestep,
        end_timestep=end_timestep,
        output_dir=output_dir,
        file_stem=file_stem,
        show=show,
        exclude_nan_timesteps=exclude_nan_timesteps,
        duration_mode=duration_mode,
    )


def create_bus_boxplot(
    database: DatabaseLike,
    solver_names: Sequence[str],
    metric: str = "lmp_p",
    bus_ids: Optional[Sequence[int]] = None,
    start_timestep: Optional[int] = None,
    end_timestep: Optional[int] = None,
    output_dir: Optional[Path] = None,
    file_stem: Optional[str] = None,
    show: bool = False,
    exclude_nan_timesteps: bool = False,
    difference: bool = False,
    difference_mode: str = "percent",
 ) -> tuple[go.Figure, Optional[Path]]:
    """Create a grouped box plot for bus metrics across one or more solvers."""

    return _entity_boxplot(
        database=database,
        entity_kind="bus",
        entity_ids=bus_ids,
        solver_names=solver_names,
        metric=metric,
        start_timestep=start_timestep,
        end_timestep=end_timestep,
        output_dir=output_dir,
        file_stem=file_stem,
        show=show,
        exclude_nan_timesteps=exclude_nan_timesteps,
        difference=difference,
        difference_mode=difference_mode,
    )


def create_branch_timeseries_plot(
    database: DatabaseLike,
    branch_id: int,
    solver_names: Sequence[str],
    metric: str = "flow_p",
    start_timestep: Optional[int] = None,
    end_timestep: Optional[int] = None,
    output_dir: Optional[Path] = None,
    file_stem: Optional[str] = None,
    show: bool = False,
    exclude_nan_timesteps: bool = False,
    difference: bool = False,
    difference_mode: str = "percent",
) -> tuple[go.Figure, Optional[Path]]:
    """Create a Plotly line chart for one branch across one or more solvers."""

    db = _coerce_database(database)
    start_timestep, end_timestep = _resolve_timestep_bounds(db, start_timestep, end_timestep)

    fig = go.Figure()
    series_found = False

    # Difference mode: compute BFSA - SOCP series (raw or percent)
    if difference and len(solver_names) >= 2 and "bfsa" in [s.lower() for s in solver_names] and "socp" in [s.lower() for s in solver_names]:
        bfsa_rows = _series_for_plot(db, "branch", branch_id, metric, "bfsa", start_timestep, end_timestep, exclude_nan_timesteps)
        socp_rows = _series_for_plot(db, "branch", branch_id, metric, "socp", start_timestep, end_timestep, exclude_nan_timesteps)
        bfsa_map = {int(t): _to_float(v) for t, v in bfsa_rows}
        socp_map = {int(t): _to_float(v) for t, v in socp_rows}
        shared = sorted(set(bfsa_map.keys()) & set(socp_map.keys()))
        if not shared:
            raise ValueError(f"No overlapping timesteps for BFSA and SOCP for branch {branch_id}")
        x_values = shared
        y_values = []
        for t in shared:
            a = bfsa_map[t]
            b = socp_map[t]
            if difference_mode == "raw":
                y = a - b
            else:
                y = 100.0 * (a - b) / (abs(b) if abs(b) > 0 else float("nan"))
            y_values.append(y)
        series_found = True
        name = "BFSA-SOCP (%)" if difference_mode != "raw" else "BFSA-SOCP"
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode="lines+markers",
                name=name,
                line={"width": 2.2, "color": _solver_color(0)},
                marker={"size": 6},
                hovertemplate=f"Difference {name}<br>Timestep %{{x}}<br>Value %{{y:.6f}}<extra></extra>",
            )
        )
    else:
        for index, solver_name in enumerate(solver_names):
            rows = _series_for_plot(
                db,
                "branch",
                branch_id,
                metric,
                solver_name,
                start_timestep,
                end_timestep,
                exclude_nan_timesteps,
            )
            if not rows:
                continue
            series_found = True
            x_values = [row[0] for row in rows]
            y_values = [_to_float(row[1]) for row in rows]
            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=y_values,
                    mode="lines+markers",
                    name=solver_name.upper(),
                    line={"width": 2.2, "color": _solver_color(index)},
                    marker={"size": 6},
                    hovertemplate=f"Solver {solver_name.upper()}<br>Timestep %{{x}}<br>Value %{{y:.6f}}<extra></extra>",
                )
            )

    if not series_found:
        raise ValueError(
            f"No data found for branch {branch_id} and metric '{metric}' in range [{start_timestep}, {end_timestep}]"
        )

    title = f"Branch {branch_id} {_metric_label(metric, 'branch')} over time"
    yaxis_title = _metric_label(metric, "branch")
    return _finalize_figure(fig, title, yaxis_title, output_dir, file_stem, show)


def create_branch_boxplot(
    database: DatabaseLike,
    solver_names: Sequence[str],
    metric: str = "flow_p",
    branch_ids: Optional[Sequence[int]] = None,
    start_timestep: Optional[int] = None,
    end_timestep: Optional[int] = None,
    output_dir: Optional[Path] = None,
    file_stem: Optional[str] = None,
    show: bool = False,
    exclude_nan_timesteps: bool = False,
    difference: bool = False,
    difference_mode: str = "percent",
 ) -> tuple[go.Figure, Optional[Path]]:
    """Create a grouped box plot for branch metrics across one or more solvers."""

    return _entity_boxplot(
        database=database,
        entity_kind="branch",
        entity_ids=branch_ids,
        solver_names=solver_names,
        metric=metric,
        start_timestep=start_timestep,
        end_timestep=end_timestep,
        output_dir=output_dir,
        file_stem=file_stem,
        show=show,
        exclude_nan_timesteps=exclude_nan_timesteps,
        difference=difference,
        difference_mode=difference_mode,
    )


def create_generator_timeseries_plot(
    database: DatabaseLike,
    solver_name: str,
    absolute: bool = False,
    base_mva: Optional[float] = None,
    generator_ids: Optional[Sequence[int]] = None,
    start_timestep: Optional[int] = None,
    end_timestep: Optional[int] = None,
    output_dir: Optional[Path] = None,
    file_stem: Optional[str] = None,
    show: bool = False,
    exclude_nan_timesteps: bool = False,
    difference: bool = False,
    difference_mode: str = "percent",
) -> tuple[go.Figure, Optional[Path]]:
    """Create a Plotly line chart for each generator across time."""

    db = _coerce_database(database)
    start_timestep, end_timestep = _resolve_timestep_bounds(db, start_timestep, end_timestep)
    resolved_ids = [int(generator_id) for generator_id in generator_ids] if generator_ids is not None else db.get_generator_ids()

    if not resolved_ids:
        raise ValueError("Database does not contain any generators to plot")

    scale = float(base_mva if base_mva is not None else db.get_base_mva())
    if absolute and scale <= 0.0:
        raise ValueError("base_mva must be positive when plotting absolute generation")

    fig = go.Figure()
    series_found = False

    if difference:
        for index, generator_id in enumerate(resolved_ids):
            bfsa_rows = _generator_series_for_plot(
                db,
                generator_id,
                "bfsa",
                start_timestep,
                end_timestep,
                exclude_nan_timesteps,
            )
            socp_rows = _generator_series_for_plot(
                db,
                generator_id,
                "socp",
                start_timestep,
                end_timestep,
                exclude_nan_timesteps,
            )
            bfsa_map = {int(t): _to_float(v) for t, v in bfsa_rows}
            socp_map = {int(t): _to_float(v) for t, v in socp_rows}
            shared = sorted(set(bfsa_map.keys()) & set(socp_map.keys()))
            if not shared:
                continue
            x_values = shared
            y_values = []
            for t in shared:
                a = bfsa_map[t]
                b = socp_map[t]
                if difference_mode == "raw":
                    v = a - b
                else:
                    v = 100.0 * (a - b) / (abs(b) if abs(b) > 0 else float("nan"))
                y_values.append(v)
            if absolute:
                y_values = [val * scale for val in y_values]
            series_found = True
            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=y_values,
                    mode="lines+markers",
                    name=f"Gen {generator_id}",
                    line={"width": 2.2, "color": _solver_color(index)},
                    marker={"size": 6},
                    hovertemplate=f"Generator {generator_id}<br>Timestep %{{x}}<br>Value %{{y:.6f}}<extra></extra>",
                )
            )
    else:
        for index, generator_id in enumerate(resolved_ids):
            rows = _generator_series_for_plot(
                db,
                generator_id,
                solver_name,
                start_timestep,
                end_timestep,
                exclude_nan_timesteps,
            )
            if not rows:
                continue

            series_found = True
            x_values = [row[0] for row in rows]
            if absolute:
                y_values = [_to_float(row[1]) * scale for row in rows]
            else:
                y_values = [_to_float(row[1]) for row in rows]

            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=y_values,
                    mode="lines+markers",
                    name=f"Gen {generator_id}",
                    line={"width": 2.2, "color": _solver_color(index)},
                    marker={"size": 6},
                    hovertemplate=f"Generator {generator_id}<br>Timestep %{{x}}<br>Value %{{y:.6f}}<extra></extra>",
                )
            )

    if not series_found:
        raise ValueError(
            f"No data found for solver '{solver_name}' in range [{start_timestep}, {end_timestep}]"
        )

    title = f"Generator active power over time ({solver_name.upper()}, {'MW' if absolute else 'p.u.'})"
    yaxis_title = _GENERATOR_Y_LABELS[absolute]
    return _finalize_figure(fig, title, yaxis_title, output_dir, file_stem, show)


def create_solve_time_timeseries_plot(
    database: DatabaseLike,
    solver_names: Sequence[str],
    start_timestep: Optional[int] = None,
    end_timestep: Optional[int] = None,
    output_dir: Optional[Path] = None,
    file_stem: Optional[str] = None,
    show: bool = False,
) -> tuple[go.Figure, Optional[Path]]:
    """Create a Plotly line chart for solver solve time across time."""

    db = _coerce_database(database)
    start_timestep, end_timestep = _resolve_timestep_bounds(db, start_timestep, end_timestep)
    resolved_solver_names = [str(solver_name).strip().lower() for solver_name in solver_names if str(solver_name).strip()]
    if not resolved_solver_names:
        raise ValueError("At least one solver name is required for a solve-time plot")

    bfsa_rows = db.get_solve_time_and_iteration_timeseries(start_timestep, end_timestep, "bfsa")
    bfsa_iterations = [(int(row[0]), row[2]) for row in bfsa_rows if row[2] is not None]
    bfsa_solve_times = [(int(row[0]), row[1]) for row in bfsa_rows if row[1] is not None]
    show_iterations = bool(bfsa_iterations) and "bfsa" in resolved_solver_names

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    series_found = False

    if show_iterations:
        x_values = [timestep for timestep, _ in bfsa_iterations]
        y_values = [int(value) for _, value in bfsa_iterations]
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode="lines+markers",
                name="BFSA iterations",
                line={"width": 2.2, "color": _solver_color(0)},
                marker={"size": 6},
                hovertemplate="BFSA iterations<br>Timestep %{x}<br>Iterations %{y}<extra></extra>",
            ),
            secondary_y=False,
        )
        series_found = True

    for index, solver_name in enumerate(resolved_solver_names):
        rows = db.get_solve_time_timeseries(start_timestep, end_timestep, solver_name)
        if not rows:
            continue

        series_found = True
        x_values = [int(row[0]) for row in rows]
        y_values = [_to_float(row[1]) for row in rows]
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode="lines+markers",
                name=solver_name.upper(),
                line={"width": 2.2, "color": _solver_color(index)},
                marker={"size": 6},
                hovertemplate=f"Solver {solver_name.upper()}<br>Timestep %{{x}}<br>Solve time %{{y:.6f}} s<extra></extra>",
            ),
            secondary_y=show_iterations,
        )

    if not series_found:
        raise ValueError(
            f"No solve-time data found for selected solver(s) in range [{start_timestep}, {end_timestep}]"
        )

    title = "BFSA iterations and solver solve time over time" if show_iterations else "Solver solve time over time"
    yaxis_title = "BFSA iterations" if show_iterations else "Solve time (s)"
    fig.update_layout(
        title=title,
        template="plotly_white",
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0.0},
        margin={"l": 70, "r": 70 if show_iterations else 30, "t": 90, "b": 60},
    )
    fig.update_yaxes(title_text=yaxis_title, secondary_y=False)
    if show_iterations:
        fig.update_yaxes(title_text="Solve time (s)", secondary_y=True)

    saved_path = None
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = file_stem or "database_timeseries"
        saved_path = output_dir / f"{stem}.html"
        fig.write_html(str(saved_path), include_plotlyjs="cdn", config=PLOTLY_HIGH_RES_EXPORT_CONFIG)

    if show:
        fig.show(config=PLOTLY_HIGH_RES_EXPORT_CONFIG)

    return fig, saved_path


def create_database_dashboard(
    database: DatabaseLike,
    bus_id: int,
    branch_id: int,
    solver_names: Sequence[str] = ("socp", "bfsa"),
    bus_metric: str = "lmp_p",
    branch_metric: str = "flow_p",
    start_timestep: Optional[int] = None,
    end_timestep: Optional[int] = None,
    output_dir: Optional[Path] = None,
    file_stem: Optional[str] = None,
    show: bool = False,
    exclude_nan_timesteps: bool = False,
    difference: bool = False,
    difference_mode: str = "percent",
) -> tuple[go.Figure, Optional[Path]]:
    """Create a two-row dashboard for a bus metric and a branch metric."""

    db = _coerce_database(database)
    start_timestep, end_timestep = _resolve_timestep_bounds(db, start_timestep, end_timestep)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=(
            f"Bus {bus_id} - {_metric_label(bus_metric, 'bus')}",
            f"Branch {branch_id} - {_metric_label(branch_metric, 'branch')}",
        ),
    )

    bus_found = False
    branch_found = False

    if difference and len(solver_names) >= 2 and "bfsa" in [s.lower() for s in solver_names] and "socp" in [s.lower() for s in solver_names]:
        # Bus difference series
        bfsa_bus = _series_for_plot(db, "bus", bus_id, bus_metric, "bfsa", start_timestep, end_timestep, exclude_nan_timesteps)
        socp_bus = _series_for_plot(db, "bus", bus_id, bus_metric, "socp", start_timestep, end_timestep, exclude_nan_timesteps)
        bfsa_map = {int(t): _to_float(v) for t, v in bfsa_bus}
        socp_map = {int(t): _to_float(v) for t, v in socp_bus}
        shared_bus = sorted(set(bfsa_map.keys()) & set(socp_map.keys()))
        if shared_bus:
            bus_found = True
            x_bus = shared_bus
            y_bus = []
            for t in shared_bus:
                a = bfsa_map[t]
                b = socp_map[t]
                if difference_mode == "raw":
                    y_bus.append(a - b)
                else:
                    y_bus.append(100.0 * (a - b) / (abs(b) if abs(b) > 0 else float("nan")))
            fig.add_trace(
                go.Scatter(
                    x=x_bus,
                    y=y_bus,
                    mode="lines+markers",
                    name="BFSA-SOCP bus",
                    line={"width": 2.2, "color": _solver_color(0)},
                    marker={"size": 6},
                    hovertemplate=f"Difference BFSA-SOCP<br>Timestep %{{x}}<br>Value %{{y:.6f}}<extra></extra>",
                ),
                row=1,
                col=1,
            )

        # Branch difference series
        bfsa_branch = _series_for_plot(db, "branch", branch_id, branch_metric, "bfsa", start_timestep, end_timestep, exclude_nan_timesteps)
        socp_branch = _series_for_plot(db, "branch", branch_id, branch_metric, "socp", start_timestep, end_timestep, exclude_nan_timesteps)
        bfsa_bmap = {int(t): _to_float(v) for t, v in bfsa_branch}
        socp_bmap = {int(t): _to_float(v) for t, v in socp_branch}
        shared_branch = sorted(set(bfsa_bmap.keys()) & set(socp_bmap.keys()))
        if shared_branch:
            branch_found = True
            x_br = shared_branch
            y_br = []
            for t in shared_branch:
                a = bfsa_bmap[t]
                b = socp_bmap[t]
                if difference_mode == "raw":
                    y_br.append(a - b)
                else:
                    y_br.append(100.0 * (a - b) / (abs(b) if abs(b) > 0 else float("nan")))
            fig.add_trace(
                go.Scatter(
                    x=x_br,
                    y=y_br,
                    mode="lines+markers",
                    name="BFSA-SOCP branch",
                    line={"width": 2.2, "dash": "dash", "color": _solver_color(0)},
                    marker={"size": 6},
                    hovertemplate=f"Difference BFSA-SOCP<br>Timestep %{{x}}<br>Value %{{y:.6f}}<extra></extra>",
                ),
                row=2,
                col=1,
            )
    else:
        for index, solver_name in enumerate(solver_names):
            bus_rows = _series_for_plot(
                db,
                "bus",
                bus_id,
                bus_metric,
                solver_name,
                start_timestep,
                end_timestep,
                exclude_nan_timesteps,
            )
            if bus_rows:
                bus_found = True
                fig.add_trace(
                    go.Scatter(
                        x=[row[0] for row in bus_rows],
                        y=[_to_float(row[1]) for row in bus_rows],
                        mode="lines+markers",
                        name=f"{solver_name.upper()} bus",
                        line={"width": 2.2, "color": _solver_color(index)},
                        marker={"size": 6},
                        hovertemplate=f"Solver {solver_name.upper()}<br>Timestep %{{x}}<br>Value %{{y:.6f}}<extra></extra>",
                    ),
                    row=1,
                    col=1,
                )

            branch_rows = _series_for_plot(
                db,
                "branch",
                branch_id,
                branch_metric,
                solver_name,
                start_timestep,
                end_timestep,
                exclude_nan_timesteps,
            )
            if branch_rows:
                branch_found = True
                fig.add_trace(
                    go.Scatter(
                        x=[row[0] for row in branch_rows],
                        y=[_to_float(row[1]) for row in branch_rows],
                        mode="lines+markers",
                        name=f"{solver_name.upper()} branch",
                        line={"width": 2.2, "dash": "dash", "color": _solver_color(index)},
                        marker={"size": 6},
                        hovertemplate=f"Solver {solver_name.upper()}<br>Timestep %{{x}}<br>Value %{{y:.6f}}<extra></extra>",
                    ),
                    row=2,
                    col=1,
                )

    if not bus_found and not branch_found:
        raise ValueError(
            f"No data found for bus {bus_id} or branch {branch_id} in range [{start_timestep}, {end_timestep}]"
        )

    fig.update_xaxes(title_text="Timestep", row=2, col=1)
    fig.update_yaxes(title_text=_metric_label(bus_metric, "bus"), row=1, col=1)
    fig.update_yaxes(title_text=_metric_label(branch_metric, "branch"), row=2, col=1)

    title = f"Database results over time for bus {bus_id} and branch {branch_id}"
    return _finalize_figure(fig, title, "", output_dir, file_stem, show)
