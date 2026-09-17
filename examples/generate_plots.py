"""
Generate plots from a results database.

A small user-facing GUI for opening a SQLite results
database and creating bus, branch, or dashboard plots from the stored data.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from tkinter import BOTH, Canvas, LEFT, RIGHT, StringVar, Tk, filedialog, messagebox, Toplevel
from tkinter import scrolledtext
from tkinter import ttk
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from libs.visualization import (  # noqa: E402
    create_branch_boxplot,
    create_branch_timeseries_plot,
    create_bus_boxplot,
    create_bus_timeseries_plot,
    create_database_dashboard,
    create_duration_curve_plot,
    create_merit_order_plot,
    create_generator_timeseries_plot,
    create_solve_time_timeseries_plot,
    export_network_plot,
    export_side_by_side_dashboard,
    set_line_color_mode,
    set_visual_theme,
)
from libs.shared.timestep_database import TimestepDatabase  # noqa: E402
from libs.shared.io import load_case_from_folder  # noqa: E402
from comparison.benchmark import Benchmark, SolverComparison  # noqa: E402


BUS_METRICS = ("voltage", "p_d", "q_d", "lmp_p", "lmp_q")
BRANCH_METRICS = ("flow_p", "flow_q", "ell")
NETWORK_METRICS = ("lmp_p", "lmp_q", "voltage")
DURATION_METRICS = ("ALMP", "RLMP", "Flows over lines", "Voltage")
NETWORK_THEMES = ("dark", "light", "white")
LINE_COLOR_MODE_LABEL_TO_KEY = {
    "Flow on base MVA": "flow_pu",
    "Line rating (%)": "loading_pct",
}
PLOT_TYPES = ("bus", "branch", "dashboard", "box", "network", "generator", "merit", "duration", "solve_time", "comparison_table", "lmp_stats", "pq_case_counts")
BOX_ENTITY_KINDS = ("bus", "branch")
# Allow explicit solver selection or a 'Difference' mode mapping to BFSA-SOCP error
DURATION_METRICS = ("ALMP", "RLMP", "Flows over lines", "Voltage")
DURATION_METRIC_CONFIG = {
    "ALMP": ("bus", "lmp_p"),
    "RLMP": ("bus", "lmp_q"),
    "Flows over lines": ("branch", "flow_p"),
    "Voltage": ("bus", "voltage"),
}
NETWORK_COMPARISON_SOLVERS = ("bfsa", "socp")
PLOTLY_HIGH_RES_EXPORT_CONFIG = {
    "toImageButtonOptions": {
        "format": "png",
        "scale": 4,
    },
}


COMPARISON_BUS_METRICS = ("voltage", "p_d", "q_d", "lmp_p", "lmp_q")
COMPARISON_BRANCH_METRICS = ("flow_p", "flow_q", "ell")


def _normalize_difference_mode(mode: str) -> str:
    """Normalize UI/backend aliases for solver difference tables."""
    normalized = str(mode or "").strip().lower()
    return "absolute" if normalized in {"absolute", "abs", "raw"} else "percent"


def _comparison_table_filename_part(value: str) -> str:
    """Return a conservative filename component for comparison table exports."""
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value)).strip("_")


def comparison_table_filename_stem(
    base_stem: str | None,
    suffix: str,
    start_timestep: int,
    end_timestep: int,
    solver_names: list[str],
    *,
    difference: bool = False,
) -> str:
    parts = []
    if base_stem:
        parts.append(_comparison_table_filename_part(base_stem))
    parts.append(suffix)
    parts.append(f"ts{int(start_timestep)}-{int(end_timestep)}")
    if not difference and len(solver_names) == 1:
        parts.append(_comparison_table_filename_part(solver_names[0]))
    return "_".join(part for part in parts if part)


def _comparison_stats(values: list[float]) -> dict:
    clean = np.array([float(value) for value in values if value is not None and np.isfinite(float(value))])
    if clean.size == 0:
        return {
            "count": 0,
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
            "abs_mean": np.nan,
            "abs_median": np.nan,
            "abs_max": np.nan,
            "p25": np.nan,
            "p75": np.nan,
            "p95": np.nan,
        }

    abs_values = np.abs(clean)
    return {
        "count": int(clean.size),
        "mean": float(np.mean(clean)),
        "median": float(np.median(clean)),
        "std": float(np.std(clean)),
        "min": float(np.min(clean)),
        "max": float(np.max(clean)),
        "abs_mean": float(np.mean(abs_values)),
        "abs_median": float(np.median(abs_values)),
        "abs_max": float(np.max(abs_values)),
        "p25": float(np.percentile(clean, 25)),
        "p75": float(np.percentile(clean, 75)),
        "p95": float(np.percentile(clean, 95)),
    }


def _format_comparison_value(value: float, decimals: int = 3) -> float:
    if value is None:
        return value
    try:
        if np.isnan(value) or np.isinf(value):
            return value
    except TypeError:
        return value
    return round(float(value), decimals)


def _comparison_stats_columns(values: list[float]) -> dict:
    stats = _comparison_stats(values)
    return {
        "count": stats["count"],
        "mean": _format_comparison_value(stats["mean"]),
        "median": _format_comparison_value(stats["median"]),
        "std": _format_comparison_value(stats["std"]),
        "min": _format_comparison_value(stats["min"]),
        "max": _format_comparison_value(stats["max"]),
        "abs_mean": _format_comparison_value(stats["abs_mean"]),
        "abs_median": _format_comparison_value(stats["abs_median"]),
        "abs_max": _format_comparison_value(stats["abs_max"]),
        "p25": _format_comparison_value(stats["p25"]),
        "p75": _format_comparison_value(stats["p75"]),
        "p95": _format_comparison_value(stats["p95"]),
    }


def _collect_comparison_series(
    db: TimestepDatabase,
    solver_name: str,
    start_timestep: int,
    end_timestep: int,
) -> dict[tuple[str, str, int], dict[int, float]]:
    """Collect value series keyed by entity kind, metric, and entity id."""
    series: dict[tuple[str, str, int], dict[int, float]] = {}

    for bus_id in db.get_bus_ids():
        for metric in COMPARISON_BUS_METRICS:
            rows = db.get_bus_timeseries(bus_id, start_timestep, end_timestep, solver_name, metric)
            series[("bus", metric, int(bus_id))] = {
                int(timestep): float(value)
                for timestep, value in rows
                if value is not None
            }

    for branch_id in db.get_branch_ids():
        for metric in COMPARISON_BRANCH_METRICS:
            rows = db.get_branch_timeseries(branch_id, start_timestep, end_timestep, solver_name, metric)
            series[("branch", metric, int(branch_id))] = {
                int(timestep): float(value)
                for timestep, value in rows
                if value is not None
            }

    for gen_id in db.get_generator_ids():
        rows = db.get_generator_timeseries(gen_id, start_timestep, end_timestep, solver_name)
        p_values: dict[int, float] = {}
        q_values: dict[int, float] = {}
        # get_generator_timeseries currently exposes active dispatch. Query Q directly when available.
        for timestep, p_value in rows:
            if p_value is not None:
                p_values[int(timestep)] = float(p_value)
        series[("generator", "dispatch_p", int(gen_id))] = p_values

        import sqlite3

        con = sqlite3.connect(db.filename)
        try:
            with con:
                cur = con.cursor()
                cur.execute(
                    "SELECT timestep, dispatch_q FROM Res_Generators "
                    "WHERE gen_id = ? AND timestep >= ? AND timestep <= ? AND solver_name = ? "
                    "ORDER BY timestep",
                    (gen_id, start_timestep, end_timestep, solver_name),
                )
                for timestep, q_value in cur.fetchall():
                    if q_value is not None:
                        q_values[int(timestep)] = float(q_value)
        finally:
            con.close()
        series[("generator", "dispatch_q", int(gen_id))] = q_values

    return series


def build_comparison_value_tables(
    db: TimestepDatabase,
    solver_names: list[str],
    start_timestep: int,
    end_timestep: int,
    *,
    table_type: str = "detailed",
    difference: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build non-percent comparison-table data from raw solver values."""
    if not solver_names:
        raise ValueError("Comparison table requires at least one solver name.")

    all_series = {
        solver_name: _collect_comparison_series(db, solver_name, start_timestep, end_timestep)
        for solver_name in solver_names
    }

    rows: list[dict] = []
    if difference:
        if len(solver_names) < 2:
            raise ValueError("Absolute difference table requires two solver names.")
        solver1, solver2 = solver_names[0], solver_names[1]
        keys = sorted(set(all_series[solver1]) & set(all_series[solver2]))
        for entity_kind, metric, entity_id in keys:
            first = all_series[solver1][(entity_kind, metric, entity_id)]
            second = all_series[solver2][(entity_kind, metric, entity_id)]
            values = [first[timestep] - second[timestep] for timestep in sorted(set(first) & set(second))]
            if not values:
                continue
            rows.append(
                {
                    "mode": f"{solver1}-{solver2}",
                    "entity_kind": entity_kind,
                    "metric": metric,
                    "entity_id": entity_id,
                    **_comparison_stats_columns(values),
                }
            )
    else:
        for solver_name in solver_names:
            for (entity_kind, metric, entity_id), values_by_timestep in sorted(all_series[solver_name].items()):
                values = [values_by_timestep[timestep] for timestep in sorted(values_by_timestep)]
                if not values:
                    continue
                rows.append(
                    {
                        "mode": solver_name,
                        "entity_kind": entity_kind,
                        "metric": metric,
                        "entity_id": entity_id,
                        **_comparison_stats_columns(values),
                    }
                )

    value_df = pd.DataFrame(rows)
    if table_type == "aggregated" and not value_df.empty:
        aggregated_rows = []
        for group_keys, _group_df in value_df.groupby(["mode", "entity_kind", "metric"], sort=True):
            mode, entity_kind, metric = group_keys
            values: list[float] = []
            if difference:
                solver1, solver2 = solver_names[0], solver_names[1]
                keys = [
                    key
                    for key in set(all_series[solver1]) & set(all_series[solver2])
                    if key[0] == entity_kind and key[1] == metric
                ]
                for key in keys:
                    first = all_series[solver1][key]
                    second = all_series[solver2][key]
                    values.extend(first[timestep] - second[timestep] for timestep in sorted(set(first) & set(second)))
            else:
                for solver_name in solver_names:
                    if solver_name != mode:
                        continue
                    keys = [key for key in all_series[solver_name] if key[0] == entity_kind and key[1] == metric]
                    for key in keys:
                        values.extend(all_series[solver_name][key].values())
            if not values:
                continue
            aggregated_rows.append(
                {
                    "mode": mode,
                    "entity_kind": entity_kind,
                    "metric": metric,
                    **_comparison_stats_columns(values),
                }
            )
        value_df = pd.DataFrame(aggregated_rows)

    time_rows = []
    for solver_name in solver_names:
        values = [
            float(value)
            for _timestep, value in db.get_solve_time_timeseries(start_timestep, end_timestep, solver_name)
            if value is not None
        ]
        if values:
            time_rows.append({"solver": solver_name, **_comparison_stats_columns(values)})

    if difference and len(solver_names) >= 2:
        first_times = {
            int(timestep): float(value)
            for timestep, value in db.get_solve_time_timeseries(start_timestep, end_timestep, solver_names[0])
            if value is not None
        }
        second_times = {
            int(timestep): float(value)
            for timestep, value in db.get_solve_time_timeseries(start_timestep, end_timestep, solver_names[1])
            if value is not None
        }
        time_diffs = [first_times[timestep] - second_times[timestep] for timestep in sorted(set(first_times) & set(second_times))]
        if time_diffs:
            time_rows.append({"solver": f"{solver_names[0]}-{solver_names[1]}", **_comparison_stats_columns(time_diffs)})

    return value_df, pd.DataFrame(time_rows)


def parse_id_selection(raw_text: str) -> list[int]:
    """Parse comma-separated IDs with inclusive ranges like '3-7'."""

    ids: list[int] = []
    for token in raw_text.split(","):
        token = token.strip()
        if not token:
            continue

        if "-" in token:
            parts = [part.strip() for part in token.split("-", 1)]
            if len(parts) != 2 or not parts[0] or not parts[1]:
                raise ValueError(f"Invalid ID range '{token}'. Use x-y, for example 3-7.")

            start_id = int(parts[0])
            end_id = int(parts[1])
            if start_id > end_id:
                raise ValueError(f"Invalid ID range '{token}'. Start must be <= end.")
            ids.extend(range(start_id, end_id + 1))
        else:
            ids.append(int(token))

    return ids


def format_min_max_id_range(ids) -> str:
    """Format an ID collection as min-max text for GUI defaults."""

    id_values = [int(value) for value in ids]
    if not id_values:
        return ""

    min_id = min(id_values)
    max_id = max(id_values)
    if min_id == max_id:
        return str(min_id)
    return f"{min_id}-{max_id}"


def _comparison_network_solvers(solver_names: list[str]) -> list[str]:
    selected: list[str] = []
    for solver_name in solver_names:
        normalized = solver_name.strip().lower()
        if normalized in NETWORK_COMPARISON_SOLVERS and normalized not in selected:
            selected.append(normalized)
    return selected


def _should_use_network_comparison(solver_names: list[str]) -> bool:
    return len(_comparison_network_solvers(solver_names)) == 2


class PlotGeneratorGUI:
    """Simple Tkinter launcher for database-backed plot helpers."""

    def __init__(self) -> None:
        self.root = Tk()
        self.root.title("COBRAS Plot Interface")
        self.root.geometry("900x720")
        self.root.minsize(760, 520)
        self.root.resizable(True, True)
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        self.database_path = StringVar()
        self.output_dir = StringVar()
        self.plot_type = StringVar(value="bus")
        self.solver_option = StringVar(value="both")
        self.element_id = StringVar(value="2")
        self.branch_id = StringVar(value="0")
        self.start_timestep = StringVar(value="0")
        self.end_timestep = StringVar(value="10")
        self.metric = StringVar(value="lmp_p")
        self.difference_mode = StringVar(value="percent")
        self.branch_metric = StringVar(value="flow_p")
        self.network_theme = StringVar(value="dark")
        self.network_line_color_mode = StringVar(value="Flow on base MVA")
        # network_timestep removed; use start_timestep for network plots
        # merit_component removed; use `metric` field for merit plots (P/Q)
        self.generator_absolute = StringVar(value="0")
        self.exclude_nan_timesteps = StringVar(value="0")
        # duration mode removed; derive duration behavior from `solver_option`
        self.show_plot = StringVar(value="1")
        self.file_stem = StringVar(value="database_plot")
        self.box_entity_kind = StringVar(value="bus")
        # unified element ID(s) field — may contain comma-separated or ranges
        self.comparison_table_type = StringVar(value="detailed")
        self.copy_to_clipboard = StringVar(value="0")
        self._last_box_bus_range_default = ""

        self.status_var = StringVar(value="Select a database and configure the plot.")
        self._build_ui()
        self.solver_option.trace_add("write", lambda *_args: self._sync_enabled_state())
        self._sync_enabled_state()

    def _build_ui(self) -> None:
        shell = ttk.Frame(self.root)
        shell.grid(row=0, column=0, sticky="nsew")
        shell.grid_rowconfigure(0, weight=1)
        shell.grid_columnconfigure(0, weight=1)

        canvas = Canvas(shell, borderwidth=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(shell, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

        outer = ttk.Frame(canvas, padding=16)
        canvas_window = canvas.create_window((0, 0), window=outer, anchor="nw")

        def _sync_scroll_region(_event=None) -> None:
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _sync_content_width(event) -> None:
            canvas.itemconfigure(canvas_window, width=event.width)

        def _scroll_content(event) -> None:
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        def _bind_mousewheel(_event) -> None:
            canvas.bind_all("<MouseWheel>", _scroll_content)

        def _unbind_mousewheel(_event) -> None:
            canvas.unbind_all("<MouseWheel>")

        outer.bind("<Configure>", _sync_scroll_region)
        canvas.bind("<Configure>", _sync_content_width)
        canvas.bind("<Enter>", _bind_mousewheel)
        canvas.bind("<Leave>", _unbind_mousewheel)

        title = ttk.Label(outer, text="COBRAS Plot Interface", font=("Segoe UI", 16, "bold"))
        title.pack(anchor="w", pady=(0, 12))

        subtitle = ttk.Label(
            outer,
            text="Generate interactive plots directly from a stored multi-timestep SQLite database.",
        )
        subtitle.pack(anchor="w", pady=(0, 14))

        form = ttk.Frame(outer)
        form.pack(fill=BOTH, expand=True)

        form.columnconfigure(1, weight=1)
        form.columnconfigure(3, weight=1)

        self._add_path_row(form, 0, "Database file", self.database_path, self._browse_database)
        self._add_path_row(form, 1, "Output folder", self.output_dir, self._browse_output)

        ttk.Label(form, text="Plot type").grid(row=2, column=0, sticky="w", pady=6)
        plot_kind = ttk.Combobox(form, textvariable=self.plot_type, values=PLOT_TYPES, state="readonly")
        plot_kind.grid(row=2, column=1, sticky="ew", pady=6)
        plot_kind.bind("<<ComboboxSelected>>", lambda _event: self._sync_enabled_state())

        ttk.Label(form, text="Solver options").grid(row=2, column=2, sticky="w", padx=(16, 0), pady=6)
        self.solver_options_combo = ttk.Combobox(
            form,
            textvariable=self.solver_option,
            values=("bfsa", "socp", "both", "difference"),
            state="readonly",
            width=20,
        )
        self.solver_options_combo.grid(row=2, column=3, sticky="ew", pady=6)
        self.solver_options_combo.bind("<<ComboboxSelected>>", lambda _event: self._sync_enabled_state())

        # Difference mode (applies when solver options == 'difference')
        ttk.Label(form, text="Difference mode").grid(row=1, column=2, sticky="w", padx=(16, 0), pady=6)
        self.difference_mode_combo = ttk.Combobox(
            form,
            textvariable=self.difference_mode,
            values=("percent", "absolute"),
            state="readonly",
            width=12,
        )
        self.difference_mode_combo.grid(row=1, column=3, sticky="ew", pady=6)

        self.element_id_label, self.element_id_entry = self._add_text_row(form, 3, "Element IDs", self.element_id)
        self.branch_id_label, self.branch_id_entry = self._add_text_row(form, 4, "Dashboard branch ID", self.branch_id)

        ttk.Label(form, text="Element type").grid(row=7, column=0, sticky="w", pady=6)
        self.box_kind_combo = ttk.Combobox(form, textvariable=self.box_entity_kind, values=BOX_ENTITY_KINDS, state="readonly")
        self.box_kind_combo.grid(row=7, column=1, sticky="ew", pady=6)
        self.box_kind_combo.bind("<<ComboboxSelected>>", lambda _event: self._sync_enabled_state())

        # Unified element ID(s) field is the `Element IDs` entry at row 3

        ttk.Label(form, text="Network theme").grid(row=8, column=0, sticky="w", pady=6)
        self.network_theme_combo = ttk.Combobox(form, textvariable=self.network_theme, values=NETWORK_THEMES, state="readonly")
        self.network_theme_combo.grid(row=8, column=1, sticky="ew", pady=6)

        ttk.Label(form, text="Network line color").grid(row=9, column=0, sticky="w", pady=6)
        self.network_line_color_combo = ttk.Combobox(
            form,
            textvariable=self.network_line_color_mode,
            values=list(LINE_COLOR_MODE_LABEL_TO_KEY.keys()),
            state="readonly",
        )
        self.network_line_color_combo.grid(row=9, column=1, sticky="ew", pady=6)

        # Merit-order component merged into `metric` when merit plot selected.
        # Keep the UI minimal: `metric` combobox will include P/Q when plot_type == 'merit'.

        self.metric_label = ttk.Label(form, text="Metric")
        self.metric_label.grid(row=3, column=2, sticky="w", padx=(16, 0), pady=6)
        self.metric_combo = ttk.Combobox(form, textvariable=self.metric, values=BUS_METRICS, state="readonly")
        self.metric_combo.grid(row=3, column=3, sticky="ew", pady=6)

        self.branch_metric_label = ttk.Label(form, text="Branch metric")
        self.branch_metric_label.grid(row=4, column=2, sticky="w", padx=(16, 0), pady=6)
        self.branch_metric_combo = ttk.Combobox(form, textvariable=self.branch_metric, values=BRANCH_METRICS, state="readonly")
        self.branch_metric_combo.grid(row=4, column=3, sticky="ew", pady=6)

        # Duration mode UI removed; duration behavior is controlled by 'Solver options'

        self.comparison_table_label = ttk.Label(form, text="Table type")
        self.comparison_table_label.grid(row=9, column=2, sticky="w", padx=(16, 0), pady=6)
        self.comparison_table_combo = ttk.Combobox(
            form,
            textvariable=self.comparison_table_type,
            values=("detailed", "aggregated"),
            state="readonly",
        )
        self.comparison_table_combo.grid(row=9, column=3, sticky="ew", pady=6)

        ttk.Label(form, text="Start timestep").grid(row=5, column=0, sticky="w", pady=6)
        ttk.Entry(form, textvariable=self.start_timestep).grid(row=5, column=1, sticky="ew", pady=6)

        ttk.Label(form, text="End timestep").grid(row=5, column=2, sticky="w", padx=(16, 0), pady=6)
        self.end_timestep_entry = ttk.Entry(form, textvariable=self.end_timestep)
        self.end_timestep_entry.grid(row=5, column=3, sticky="ew", pady=6)

        ttk.Label(form, text="File stem").grid(row=6, column=0, sticky="w", pady=6)
        ttk.Entry(form, textvariable=self.file_stem).grid(row=6, column=1, sticky="ew", pady=6)

        ttk.Checkbutton(form, text="Open plot after generating", variable=self.show_plot).grid(
            row=6, column=2, columnspan=2, sticky="w", padx=(16, 0), pady=6
        )

        ttk.Checkbutton(form, text="Copy table to clipboard (when applicable)", variable=self.copy_to_clipboard).grid(
            row=6, column=2, columnspan=2, sticky="e", padx=(16, 0), pady=6
        )

        ttk.Checkbutton(form, text="Exclude NaN timesteps", variable=self.exclude_nan_timesteps).grid(
            row=8, column=2, columnspan=2, sticky="w", padx=(16, 0), pady=6
        )

        self.generator_absolute_check = ttk.Checkbutton(
            form,
            text="Plot generator output in absolute units",
            variable=self.generator_absolute,
        )
        self.generator_absolute_check.grid(row=7, column=2, columnspan=2, sticky="w", padx=(16, 0), pady=6)

        actions = ttk.Frame(outer)
        actions.pack(fill="x", pady=(10, 8))

        self.generate_button = ttk.Button(actions, text="Generate plot", command=self._generate_plot)
        self.generate_button.pack(side=LEFT)
        self.progress_bar = ttk.Progressbar(actions, mode="indeterminate", length=220)
        ttk.Button(actions, text="Quit", command=self.root.destroy).pack(side=RIGHT)

        status = ttk.Label(outer, textvariable=self.status_var, relief="sunken", anchor="w", padding=(8, 6))
        status.pack(fill="x", pady=(12, 0))

        self._fit_window_to_content()

    def _fit_window_to_content(self) -> None:
        """Size the launcher to visible content without exceeding the screen."""
        self.root.update_idletasks()
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        target_w = min(max(self.root.winfo_reqwidth() + 24, 900), max(screen_w - 80, 760))
        target_h = min(max(self.root.winfo_reqheight() + 24, 720), max(screen_h - 80, 520))
        self.root.geometry(f"{target_w}x{target_h}")

    def _add_path_row(self, parent, row: int, label: str, variable: StringVar, command) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=6)
        entry = ttk.Entry(parent, textvariable=variable)
        entry.grid(row=row, column=1, sticky="ew", pady=6)
        ttk.Button(parent, text="Browse", command=command).grid(row=row, column=2, columnspan=2, sticky="w", padx=(16, 0), pady=6)

    def _add_text_row(self, parent, row: int, label: str, variable: StringVar):
        label_widget = ttk.Label(parent, text=label)
        label_widget.grid(row=row, column=0, sticky="w", pady=6)
        entry = ttk.Entry(parent, textvariable=variable)
        entry.grid(row=row, column=1, sticky="ew", pady=6)
        return label_widget, entry

    def _browse_database(self) -> None:
        path = filedialog.askopenfilename(
            title="Select SQLite database",
            filetypes=(("SQLite database", "*.db *.sqlite *.sqlite3"), ("All files", "*.*")),
        )
        if path:
            self.database_path.set(path)
            self._sync_range_from_database(path)
            self._auto_set_output_folder(path)

    def _auto_set_output_folder(self, db_path: str) -> None:
        """Automatically set output folder near selected database.

        If DB is in <grid>/db or <grid>/results, use <grid>/plots.
        Otherwise use <db_parent>/plots.
        """
        db_file = Path(db_path)
        if db_file.parent.name.lower() in {"db", "results"}:
            plots_folder = db_file.parent.parent / "plots"
        else:
            plots_folder = db_file.parent / "plots"
        self.output_dir.set(str(plots_folder))

    def _sync_range_from_database(self, path: str) -> None:
        """Populate start/end fields from the selected database timerange."""
        try:
            db_handle = TimestepDatabase(path)
            timesteps = db_handle.get_timerange()
            if timesteps:
                self.start_timestep.set(str(timesteps[0]))
                self.end_timestep.set(str(timesteps[-1]))
                self.status_var.set(
                    f"Loaded timestep range [{timesteps[0]}, {timesteps[-1]}] from selected database."
                )
            else:
                self.status_var.set("Selected database has no timestep rows.")
            self._sync_box_bus_element_default()
        except Exception as exc:
            self.status_var.set(f"Could not read timestep range: {exc}")

    def _browse_output(self) -> None:
        path = filedialog.askdirectory(title="Select output folder")
        if path:
            self.output_dir.set(path)

    def _sync_enabled_state(self) -> None:
        plot_type = self.plot_type.get()
        box_enabled = plot_type == "box"
        box_bus_enabled = box_enabled and self.box_entity_kind.get().strip() == "bus"
        dashboard_enabled = plot_type == "dashboard"
        network_enabled = plot_type == "network"
        pq_case_counts_enabled = plot_type == "pq_case_counts"
        duration_enabled = plot_type == "duration"
        comparison_enabled = plot_type == "comparison_table"
        single_metric_enabled = plot_type in {"bus", "branch", "dashboard", "box", "network", "duration", "merit"}
        self.element_id_label.configure(text="Dashboard bus ID" if dashboard_enabled else "Element ID")
        self.element_id_entry.configure(
            state="normal" if plot_type in {"bus", "branch", "dashboard", "box", "duration"} else "disabled"
        )
        self.branch_id_entry.configure(state="normal" if dashboard_enabled else "disabled")
        self.metric_label.configure(text="Dashboard bus metric" if dashboard_enabled else "Metric")
        self.metric_combo.configure(state="readonly" if single_metric_enabled else "disabled")
        self.branch_metric_combo.configure(state="readonly" if (dashboard_enabled or network_enabled) else "disabled")
        # The value is ignored unless solver option is 'difference', but keeping the
        # selector editable avoids Tk state lag before the first generated plot.
        self.difference_mode_combo.configure(state="readonly")
        # duration mode controlled by solver_option; no UI element to enable/disable
        self.comparison_table_combo.configure(state="readonly" if comparison_enabled else "disabled")
        self.box_kind_combo.configure(state="readonly" if box_enabled else "disabled")
        self.network_theme_combo.configure(state="readonly" if network_enabled else "disabled")
        self.network_line_color_combo.configure(state="readonly" if network_enabled else "disabled")
        # Disable end timestep for network plots; network timestep will be taken from start_timestep
        self.end_timestep_entry.configure(state="disabled" if network_enabled else "normal")
        self.generator_absolute_check.configure(state="normal" if plot_type == "generator" else "disabled")

        if plot_type == "branch" or (box_enabled and not box_bus_enabled):
            metric_values = BRANCH_METRICS
        elif network_enabled:
            metric_values = NETWORK_METRICS
        elif duration_enabled:
            metric_values = DURATION_METRICS
        elif plot_type == "merit":
            metric_values = ("P", "Q")
        else:
            metric_values = BUS_METRICS
        self.metric_combo.configure(values=metric_values)
        if self.metric.get() not in metric_values:
            self.metric.set(metric_values[0])
        self._sync_box_bus_element_default()
        self._fit_window_to_content()

    def _sync_box_bus_element_default(self) -> None:
        """Default box/bus plots to the full bus ID range from the selected DB."""

        if self.plot_type.get() != "box" or self.box_entity_kind.get().strip() != "bus":
            return

        database_path = self.database_path.get().strip()
        if not database_path:
            return

        try:
            bus_range = format_min_max_id_range(TimestepDatabase(database_path).get_bus_ids())
        except Exception:
            return

        if not bus_range:
            return

        current_value = self.element_id.get().strip()
        defaultish_values = {"", "2", "all"}
        if self._last_box_bus_range_default:
            defaultish_values.add(self._last_box_bus_range_default)

        if current_value.lower() in {value.lower() for value in defaultish_values}:
            self.element_id.set(bus_range)
            self._last_box_bus_range_default = bus_range

    def _set_status(self, text: str) -> None:
        """Update the status label from the GUI thread or a worker thread."""
        if threading.current_thread() is threading.main_thread():
            self.status_var.set(text)
        else:
            self.root.after(0, self.status_var.set, text)

    def _set_busy(self, busy: bool) -> None:
        if busy:
            self.generate_button.configure(state="disabled")
            self.progress_bar.pack(side=LEFT, padx=(12, 0))
            self.progress_bar.start(12)
            self._set_status("Generating plot...")
        else:
            self.progress_bar.stop()
            self.progress_bar.pack_forget()
            self.generate_button.configure(state="normal")
            self._sync_enabled_state()

    def _load_matching_case_for_database(self, db: TimestepDatabase, database_file: Path, load_index: int | None = None):
        """Load the case whose topology best matches the selected database."""
        # Fast path: when DB is under <grid>/db or <grid>/results, load that grid directly.
        db_parent = database_file.parent
        if db_parent.name.lower() in {"db", "results"}:
            candidate_folder = db_parent.parent
            try:
                candidate_case = load_case_from_folder(candidate_folder, load_index=load_index)
                self._set_status(f"Using network data from {candidate_folder}")
                return candidate_case
            except Exception as exc:
                raise RuntimeError(
                    "Database is inside a 'db' or 'results' folder, but loading the parent grid failed: "
                    f"{candidate_folder}\n{exc}"
                ) from exc

        db_bus_ids = set(db.get_bus_ids())
        if not db_bus_ids:
            raise RuntimeError("Database has no Grid_Buses entries, cannot match network case.")
        db_branch_edges = {
            tuple(sorted((int(fbus), int(tbus))))
            for fbus, tbus in db.get_branch_endpoints()
        }

        data_root = ROOT / "data"
        best_case = None
        best_score = -1.0
        best_folder = None
        best_bus_overlap = 0
        best_branch_overlap = 0

        for network_folder in data_root.iterdir():
            if not network_folder.is_dir():
                continue
            try:
                candidate_case = load_case_from_folder(network_folder, load_index=load_index)
            except Exception:
                continue

            case_bus_ids = {int(bus.bus_id) for bus in candidate_case.buses}
            case_branch_edges = {
                tuple(sorted((int(br.from_bus), int(br.to_bus))))
                for br in candidate_case.branches
            }

            bus_intersection = len(case_bus_ids & db_bus_ids)
            branch_intersection = len(case_branch_edges & db_branch_edges)

            bus_union = len(case_bus_ids | db_bus_ids)
            branch_union = len(case_branch_edges | db_branch_edges)
            bus_jaccard = (bus_intersection / bus_union) if bus_union > 0 else 0.0
            branch_jaccard = (branch_intersection / branch_union) if branch_union > 0 else 0.0

            # Branch topology is more discriminative than bus IDs for similarly-numbered grids.
            score = (0.35 * bus_jaccard) + (0.65 * branch_jaccard)

            # Prefer exact topology match whenever possible.
            if case_bus_ids == db_bus_ids and case_branch_edges == db_branch_edges:
                return candidate_case

            if score > best_score:
                best_score = score
                best_case = candidate_case
                best_folder = network_folder
                best_bus_overlap = bus_intersection
                best_branch_overlap = branch_intersection

        if best_case is None:
            raise RuntimeError("Could not load any valid PowerFlowCase from data folder.")

        # If no exact match, require high topology similarity to avoid plotting the wrong grid.
        if best_score < 0.85:
            raise RuntimeError(
                "Could not find a matching case for this database. "
                f"Best candidate score={best_score:.3f}, "
                f"bus overlap={best_bus_overlap}/{len(db_bus_ids)}, "
                f"branch overlap={best_branch_overlap}/{max(len(db_branch_edges), 1)}"
                + (f" in {best_folder}." if best_folder is not None else ".")
            )

        self._set_status(
            "Warning: no exact topology match found; using closest case "
            f"(score={best_score:.3f}, buses={best_bus_overlap}/{len(db_bus_ids)}, "
            f"branches={best_branch_overlap}/{max(len(db_branch_edges), 1)})."
        )
        return best_case

    def _generate_plot(self) -> None:
        database_path = self.database_path.get().strip()
        if not database_path:
            messagebox.showerror("Missing database", "Please select a SQLite database file.")
            return

        database_file = Path(database_path)
        if not database_file.exists():
            messagebox.showerror("Missing database", f"Database file not found:\n{database_file}")
            return

        try:
            job = {
                "database_file": database_file,
                "start_timestep": int(self.start_timestep.get().strip()),
                "end_timestep": int(self.end_timestep.get().strip()),
                # Map the single-select GUI option to the backend solver names list.
                # 'bfsa' -> ['bfsa'], 'socp' -> ['socp'], 'both' -> ['socp','bfsa'],
                # 'difference' -> ['bfsa','socp'] and triggers a difference/deviation plot.
                "solver_option": self.solver_option.get().strip().lower(),
                "solver_names": [],
                "difference": False,
                "output_path": Path(self.output_dir.get().strip()) if self.output_dir.get().strip() else None,
                "file_stem": self.file_stem.get().strip() or None,
                "show_plot": self.show_plot.get() == "1",
                "difference_mode": self.difference_mode.get().strip().lower(),
                "exclude_nan_timesteps": self.exclude_nan_timesteps.get() == "1",
                "plot_type": self.plot_type.get(),
                "element_id": self.element_id.get().strip(),
                "branch_id": self.branch_id.get().strip(),
                "metric": self.metric.get().strip(),
                "branch_metric": self.branch_metric.get().strip(),
                "box_entity_kind": self.box_entity_kind.get().strip(),
                "network_theme": self.network_theme.get().strip(),
                "network_line_color_mode": self.network_line_color_mode.get().strip(),
                # network_timestep removed; use start_timestep for network plots
                "merit_component": None,
                "generator_absolute": self.generator_absolute.get() == "1",
                "comparison_table_type": self.comparison_table_type.get().strip(),
                "lmp_solver_option": self.solver_option.get().strip().lower(),
                "pq_case_counts_solver_option": self.solver_option.get().strip().lower(),
                "copy_to_clipboard": self.copy_to_clipboard.get() == "1",
            }
        except ValueError:
            messagebox.showerror("Invalid range", "Start and end timestep must be integers.")
            return

        # Translate solver option into solver_names used by plotting backends.
        opt = job.get("solver_option", "").lower()
        if opt == "bfsa":
            job["solver_names"] = ["bfsa"]
        elif opt == "socp":
            job["solver_names"] = ["socp"]
        elif opt == "both":
            job["solver_names"] = ["socp", "bfsa"]
        elif opt == "difference" or opt == "delta":
            # Difference mode compares BFSA vs SOCP (BFSA - SOCP)
            job["solver_names"] = ["bfsa", "socp"]
            job["difference"] = True
        else:
            # Fallback: keep existing comma-separated behavior (rare)
            job["solver_names"] = [name.strip() for name in str(self.solver_option.get()).split(",") if name.strip()]

        duration_error = job["plot_type"] == "duration" and job.get("solver_option", "") == "difference"
        comparison_table = job["plot_type"] == "comparison_table"
        if job["plot_type"] != "merit" and not job["solver_names"] and not duration_error and not comparison_table:
            messagebox.showerror("Missing solver", "Please provide at least one solver name.")
            return
        
        if comparison_table and not job["solver_names"]:
            messagebox.showerror("Missing solver", "Comparison table requires at least one solver name.")
            return
        if job["plot_type"] == "lmp_stats" and job["lmp_solver_option"] not in {"bfsa", "socp", "both"}:
            messagebox.showerror("Missing solver", "LMP stats requires solver option BFSA, SOCP, or BOTH.")
            return
        if job["plot_type"] == "pq_case_counts" and job["pq_case_counts_solver_option"] not in {"bfsa", "socp", "both"}:
            messagebox.showerror("Missing solver", "PQ case counts requires solver option BFSA, SOCP, or BOTH.")
            return

        self._set_busy(True)
        worker = threading.Thread(target=self._run_generate_plot_worker, args=(job,), daemon=True)
        worker.start()

    def _run_generate_plot_worker(self, job: dict) -> None:
        try:
            saved_path = self._run_generate_plot_job(job)
        except Exception as exc:
            self.root.after(0, self._finish_generate_plot, False, str(exc), None)
            return

        message = f"Plot saved to {saved_path}" if saved_path is not None else "Plot generated successfully."
        self.root.after(0, self._finish_generate_plot, True, message, saved_path)

    def _finish_generate_plot(self, success: bool, message: str, saved_path) -> None:
        self._set_busy(False)
        self.status_var.set(message if success else f"Failed: {message}")
        if success:
            messagebox.showinfo("Done", self.status_var.get())
        else:
            messagebox.showerror("Plot generation failed", message)

    def _run_generate_plot_job(self, job: dict):
        database_file = job["database_file"]
        start_timestep = job["start_timestep"]
        end_timestep = job["end_timestep"]
        solver_names = job["solver_names"]
        output_path = job["output_path"]
        file_stem = job["file_stem"]
        show_plot = job["show_plot"]
        exclude_nan_timesteps = job["exclude_nan_timesteps"]
        plot_type = job["plot_type"]

        available_timesteps = TimestepDatabase(str(database_file)).get_timerange()
        if not available_timesteps:
            raise RuntimeError(f"No timestep results found in:\n{database_file}")

        min_timestep = available_timesteps[0]
        max_timestep = available_timesteps[-1]
        if end_timestep < min_timestep or start_timestep > max_timestep:
            raise RuntimeError(
                "Requested range does not overlap database timesteps.\n\n"
                f"Requested: [{start_timestep}, {end_timestep}]\n"
                f"Available: [{min_timestep}, {max_timestep}]"
            )

        if plot_type == "bus":
            # Allow comma-separated/range element IDs; use first for single-element plots
            ids = parse_id_selection(job.get("element_id", ""))
            if not ids:
                raise RuntimeError("No element ID provided for bus plot.")
            fig, saved_path = create_bus_timeseries_plot(
                database_file,
                bus_ids=ids,
                solver_names=solver_names,
                metric=job["metric"],
                start_timestep=start_timestep,
                end_timestep=end_timestep,
                difference=job.get("difference", False),
                difference_mode=job.get("difference_mode", "percent"),
                output_dir=output_path,
                file_stem=file_stem,
                show=show_plot,
                exclude_nan_timesteps=exclude_nan_timesteps,
            )
        elif plot_type == "branch":
            ids = parse_id_selection(job.get("element_id", ""))
            if not ids:
                raise RuntimeError("No element ID provided for branch plot.")
            fig, saved_path = create_branch_timeseries_plot(
                database_file,
                branch_id=int(ids[0]),
                solver_names=solver_names,
                metric=job["metric"],
                start_timestep=start_timestep,
                end_timestep=end_timestep,
                difference=job.get("difference", False),
                difference_mode=job.get("difference_mode", "percent"),
                output_dir=output_path,
                file_stem=file_stem,
                show=show_plot,
                exclude_nan_timesteps=exclude_nan_timesteps,
            )
        elif plot_type == "box":
            raw_ids = job.get("element_id", "").lower()
            entity_ids = None if not raw_ids or raw_ids == "all" else parse_id_selection(raw_ids)
            if job["box_entity_kind"] == "bus":
                fig, saved_path = create_bus_boxplot(
                    database_file,
                    solver_names=solver_names,
                    metric=job["metric"],
                    bus_ids=entity_ids,
                    start_timestep=start_timestep,
                    end_timestep=end_timestep,
                    difference=job.get("difference", False),
                    difference_mode=job.get("difference_mode", "percent"),
                    output_dir=output_path,
                    file_stem=file_stem,
                    show=show_plot,
                    exclude_nan_timesteps=exclude_nan_timesteps,
                )
            else:
                fig, saved_path = create_branch_boxplot(
                    database_file,
                    solver_names=solver_names,
                    metric=job["metric"],
                    branch_ids=entity_ids,
                    start_timestep=start_timestep,
                    end_timestep=end_timestep,
                    difference=job.get("difference", False),
                    difference_mode=job.get("difference_mode", "percent"),
                    output_dir=output_path,
                    file_stem=file_stem,
                    show=show_plot,
                    exclude_nan_timesteps=exclude_nan_timesteps,
                )
        elif plot_type == "network":
                if job.get("difference", False):
                    fig, saved_path = self._generate_network_difference_plot(
                        database_file,
                        timestep=int(job["start_timestep"]),
                        metric=job["metric"],
                        branch_metric=job.get("branch_metric", "flow_p"),
                        theme=job["network_theme"],
                        output_dir=output_path,
                        show=show_plot,
                        difference_mode=job.get("difference_mode", "percent"),
                    )
                else:
                    comparison_solvers = _comparison_network_solvers(solver_names)
                    if len(comparison_solvers) == 2:
                        fig, saved_path = self._generate_network_comparison_plot(
                            database_file,
                            timestep=int(job["start_timestep"]),
                            solver_names=comparison_solvers,
                            theme=job["network_theme"],
                            output_dir=output_path,
                            show=show_plot,
                            line_color_mode=job.get("network_line_color_mode", "Flow on base MVA"),
                        )
                    else:
                        fig, saved_path = self._generate_network_plot(
                            database_file,
                            timestep=int(job["start_timestep"]),
                            solver_name=solver_names[0],
                            metric=job["metric"],
                            theme=job["network_theme"],
                            output_dir=output_path,
                            file_stem=file_stem,
                            show=show_plot,
                            line_color_mode=job.get("network_line_color_mode", "Flow on base MVA"),
                        )
        elif plot_type == "merit":
            fig, saved_path = self._generate_merit_order_plot(
                database_file,
                timestep=int(job["start_timestep"]),
                component=str(job.get("metric", "P")).upper(),
                output_dir=output_path,
                file_stem=file_stem,
                show=show_plot,
            )
        elif plot_type == "generator":
            if len(solver_names) > 1:
                self._set_status("Generator plots use the first solver name from the list.")
            fig, saved_path = create_generator_timeseries_plot(
                database_file,
                solver_name=solver_names[0],
                absolute=job["generator_absolute"],
                start_timestep=start_timestep,
                end_timestep=end_timestep,
                output_dir=output_path,
                file_stem=file_stem,
                show=show_plot,
                exclude_nan_timesteps=exclude_nan_timesteps,
                difference=job.get("difference", False),
                difference_mode=job.get("difference_mode", "percent"),
            )
        elif plot_type == "solve_time":
            fig, saved_path = create_solve_time_timeseries_plot(
                database_file,
                solver_names=solver_names,
                start_timestep=start_timestep,
                end_timestep=end_timestep,
                output_dir=output_path,
                file_stem=file_stem,
                show=show_plot,
            )
        elif plot_type == "duration":
            entity_kind, metric = DURATION_METRIC_CONFIG.get(
                job["metric"],
                DURATION_METRIC_CONFIG["ALMP"],
            )
            raw_ids = job["element_id"]
            if not raw_ids:
                raise RuntimeError("No element ID provided for duration plot.")

            # Derive duration behavior from solver_option: 'difference' -> error mode
            duration_mode_flag = "error" if job.get("solver_option", "") == "difference" else "series"
            duration_difference = job.get("difference", False) or job.get("solver_option", "") == "difference"

            fig, saved_path = create_duration_curve_plot(
                database_file,
                entity_kind=entity_kind,
                entity_ids=parse_id_selection(raw_ids),
                solver_names=solver_names,
                metric=metric,
                start_timestep=start_timestep,
                end_timestep=end_timestep,
                output_dir=output_path,
                file_stem=file_stem,
                show=show_plot,
                exclude_nan_timesteps=exclude_nan_timesteps,
                duration_mode=("error" if duration_mode_flag == "error" else "series"),
                difference=duration_difference,
                difference_mode=job.get("difference_mode", "percent"),
            )
        elif plot_type == "comparison_table":
            saved_path = self._generate_comparison_table(
                database_file,
                solver_names=solver_names,
                output_dir=output_path,
                table_type=job.get("comparison_table_type", "detailed"),
                start_timestep=start_timestep,
                end_timestep=end_timestep,
                file_stem=file_stem,
                difference=job.get("difference", False),
                difference_mode=job.get("difference_mode", "percent"),
            )
            fig = None
        elif plot_type == "pq_case_counts":
            saved_path, df = self._generate_pq_case_counts(
                database_file,
                solver_option=job.get("pq_case_counts_solver_option", "bfsa"),
                start_timestep=start_timestep,
                end_timestep=end_timestep,
                output_dir=output_path,
                file_stem=file_stem,
                copy_to_clipboard=job.get("copy_to_clipboard", False),
            )
            if job.get("copy_to_clipboard", False):
                self._show_dataframe_popup(df, f"PQ case counts - {job.get('pq_case_counts_solver_option', 'bfsa')}")
            fig = None
        elif plot_type == "lmp_stats":
            saved_path = self._generate_lmp_stats(
                database_file,
                solver_option=job.get("lmp_solver_option", "socp"),
                start_timestep=start_timestep,
                end_timestep=end_timestep,
                output_dir=output_path,
                file_stem=file_stem,
                copy_to_clipboard=job.get("copy_to_clipboard", False),
            )
            fig = None
        else:
            fig, saved_path = create_database_dashboard(
                database_file,
                bus_id=int(job["element_id"]),
                branch_id=int(job["branch_id"]),
                solver_names=solver_names,
                difference=job.get("difference", False),
                difference_mode=job.get("difference_mode", "percent"),
                bus_metric=job["metric"] if job["metric"] in BUS_METRICS else "lmp_p",
                branch_metric=job["branch_metric"],
                start_timestep=start_timestep,
                end_timestep=end_timestep,
                output_dir=output_path,
                file_stem=file_stem,
                show=show_plot,
                exclude_nan_timesteps=exclude_nan_timesteps,
            )

        if fig is not None:
            del fig
        return saved_path

    def _generate_network_plot(
        self,
        database_file,
        timestep: int,
        solver_name: str,
        metric: str,
        theme: str,
        output_dir=None,
        file_stem=None,
        show=True,
        line_color_mode: str = "Flow on base MVA",
    ):
        """Generate network visualization plot from database results."""
        from pathlib import Path as PathlibPath
        import libs.visualization.network as net_module
        
        # Set theme
        set_visual_theme(theme)
        resolved_line_color_mode = LINE_COLOR_MODE_LABEL_TO_KEY.get(line_color_mode, str(line_color_mode).strip())
        set_line_color_mode(resolved_line_color_mode)
        
        # Load the database
        db = TimestepDatabase(str(database_file))
        case = self._load_matching_case_for_database(db, Path(database_file), load_index=timestep)
        
        # Retrieve timestep result from database
        result = db.retrieve_timestep_result(timestep, solver_name, case)
        
        # Build plot inputs - this returns (net, bus_id_map, trans_df, dlmp, bus_voltages)
        net, bus_id_map, trans_df, dlmp, bus_voltages = net_module.build_plot_inputs(case, result)
        
        # Verify and print load balance
        net_module.verify_load_balance(case, result)
        
        # Map GUI metric name to the bus color metric used by the network plot.
        bus_color_metric = {
            "lmp_p": "lambda_p",
            "lmp_q": "lambda_q",
            "voltage": "voltage",
        }.get(str(metric).strip().lower(), "lambda_p")
        
        # Create the plot
        fig = net_module.create_combined_plot(
            net=net,
            bus_id_map=bus_id_map,
            trans_df=trans_df,
            dlmp=dlmp,
            bus_voltages=bus_voltages,
            title=f"Network Visualization - {solver_name} - Timestep {timestep} - {metric}",
            lmp_type=bus_color_metric,
            line_color_mode=resolved_line_color_mode,
        )
        
        # Export and save
        saved_path = None
        if output_dir:
            output_path = PathlibPath(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            if file_stem:
                filename = f"{file_stem}_{solver_name}_ts{timestep}_{metric}_{theme}.html"
            else:
                filename = f"network_{solver_name}_ts{timestep}_{metric}_{theme}.html"
            
            saved_path = output_path / filename
            net_module.export_network_plot(fig, saved_path)
        
        if show:
            fig.show(config=PLOTLY_HIGH_RES_EXPORT_CONFIG)
        
        return fig, saved_path

    def _generate_merit_order_plot(
        self,
        database_file,
        timestep: int,
        component: str,
        output_dir=None,
        file_stem=None,
        show=True,
    ):
        """Generate a BFSA-style merit-order clearing curve from a database timestep."""
        from pathlib import Path as PathlibPath

        component = component.upper().strip()
        if component not in {"P", "Q"}:
            raise ValueError("Merit-order component must be 'P' or 'Q'.")

        db = TimestepDatabase(str(database_file))
        case = self._load_matching_case_for_database(db, Path(database_file), load_index=timestep)

        quantity_attr = "p_max_available" if component == "P" else "q_max_available"
        cost_attr = "c_p" if component == "P" else "c_q"
        total_load = sum(float(load.p_d if component == "P" else load.q_d) for load in case.loads)

        offers_sorted = []
        for gen in case.generators:
            if int(gen.status) <= 0:
                continue

            capacity = float(getattr(gen, quantity_attr, 0.0) or 0.0)
            if capacity <= 0.0:
                continue

            offers_sorted.append(
                {
                    "gen_id": int(gen.gen_id),
                    "bus_id": int(gen.bus_id),
                    "name": f"G{gen.gen_id}@B{gen.bus_id}",
                    "capacity": capacity,
                    "cost": float(getattr(gen, cost_attr, 0.0)),
                    "is_load_excess": False,
                }
            )

        total_capacity = float(sum(float(offer["capacity"]) for offer in offers_sorted))
        offers_sorted.append(
            {
                "gen_id": -1,
                "bus_id": int(case.root_bus),
                "name": "LOAD_EXCESS",
                "capacity": max(total_load - total_capacity, 0.0),
                "cost": 0.0,
                "is_load_excess": True,
            }
        )

        offers_sorted = sorted(
            offers_sorted,
            key=lambda offer: (float(offer["cost"]), -float(offer["capacity"]), int(offer["gen_id"])),
        )

        marginal_price = 0.0
        if total_load > 0.0:
            remaining = total_load
            setting_offer = offers_sorted[-1]
            for offer in offers_sorted:
                take = min(float(offer["capacity"]), max(remaining, 0.0))
                if take > 0.0:
                    setting_offer = offer
                remaining -= take
                if remaining <= 1e-12:
                    break
            marginal_price = float(setting_offer["cost"])

        fig = create_merit_order_plot(
            offers_sorted=offers_sorted,
            total_load=total_load,
            marginal_price=marginal_price,
            component_label=component,
        )

        saved_path = None
        if output_dir:
            output_path = PathlibPath(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            if file_stem:
                filename = f"{file_stem}_ts{timestep}_merit_{component.lower()}.html"
            else:
                filename = f"merit_order_ts{timestep}_{component.lower()}.html"

            saved_path = output_path / filename
            export_network_plot(fig, saved_path)

        if show:
            fig.show(config=PLOTLY_HIGH_RES_EXPORT_CONFIG)

        return fig, saved_path

    def _generate_network_comparison_plot(
        self,
        database_file,
        timestep: int,
        solver_names: list[str],
        theme: str,
        output_dir=None,
        show=True,
        line_color_mode: str = "Flow on base MVA",
    ):
        """Generate the standard side-by-side network comparison dashboard."""
        from pathlib import Path as PathlibPath
        import webbrowser
        import libs.visualization.network as net_module

        if len(solver_names) < 2:
            raise ValueError("Network comparison requires at least two solver names.")

        set_visual_theme(theme)
        resolved_line_color_mode = LINE_COLOR_MODE_LABEL_TO_KEY.get(line_color_mode, str(line_color_mode).strip())
        set_line_color_mode(resolved_line_color_mode)

        db = TimestepDatabase(str(database_file))
        case = self._load_matching_case_for_database(db, Path(database_file), load_index=timestep)

        output_path = PathlibPath(output_dir) if output_dir else PathlibPath(database_file).parent / "network"
        output_path.mkdir(parents=True, exist_ok=True)

        for solver_name in solver_names[:2]:
            result = db.retrieve_timestep_result(timestep, solver_name, case)
            net, bus_id_map, trans_df, dlmp, bus_voltages = net_module.build_plot_inputs(case, result)
            net_module.verify_load_balance(case, result)

            white_fig = net_module.create_white_network_plot(net, bus_id_map, trans_df, dlmp)
            net_module.export_network_plot(white_fig, output_path / f"{solver_name}_white.html")

            fig_p = net_module.create_combined_plot(
                net=net,
                bus_id_map=bus_id_map,
                trans_df=trans_df,
                dlmp=dlmp,
                bus_voltages=bus_voltages,
                title=f"Network - {solver_name} - Timestep {timestep} - lambda_p",
                lmp_type="lambda_p",
                line_color_mode=resolved_line_color_mode,
            )
            net_module.export_network_plot(fig_p, output_path / f"{solver_name}_lambda_p.html")

            fig_q = net_module.create_combined_plot(
                net=net,
                bus_id_map=bus_id_map,
                trans_df=trans_df,
                dlmp=dlmp,
                bus_voltages=bus_voltages,
                title=f"Network - {solver_name} - Timestep {timestep} - lambda_q",
                lmp_type="lambda_q",
                line_color_mode=resolved_line_color_mode,
            )
            net_module.export_network_plot(fig_q, output_path / f"{solver_name}_lambda_q.html")

        dashboard_path = export_side_by_side_dashboard(
            output_path / "comparison_dashboard.html",
            left_solver=solver_names[0],
            right_solver=solver_names[1],
        )

        if show:
            webbrowser.open_new_tab(dashboard_path.resolve().as_uri())

        return None, dashboard_path

    def _generate_network_difference_plot(
        self,
        database_file,
        timestep: int,
        metric: str,
        branch_metric: str,
        theme: str,
        output_dir=None,
        show=True,
        difference_mode: str = "percent",
    ):
        """Generate a BFSA - SOCP difference plot for selected bus and branch metrics.

        difference_mode: 'percent' or 'raw' — controls whether deviations are
        percent-based (100*(bfsa-socp)/abs(bfsa)) or raw (bfsa - socp).
        """
        from pathlib import Path as PathlibPath
        import libs.visualization.network as net_module

        set_visual_theme(theme)

        db = TimestepDatabase(str(database_file))
        case = self._load_matching_case_for_database(db, Path(database_file), load_index=timestep)

        # Retrieve BFSA and SOCP results (BFSA - SOCP)
        try:
            result_bfsa = db.retrieve_timestep_result(timestep, "bfsa", case)
            result_socp = db.retrieve_timestep_result(timestep, "socp", case)
        except Exception as exc:
            raise RuntimeError(f"Could not retrieve BFSA/SOCP results for difference plot: {exc}") from exc

        bus_metric = str(metric).strip().lower()
        branch_metric = str(branch_metric).strip().lower()
        if bus_metric not in NETWORK_METRICS:
            raise ValueError(f"Unsupported network bus metric '{metric}'. Choose one of {NETWORK_METRICS}.")
        if branch_metric not in BRANCH_METRICS:
            raise ValueError(f"Unsupported network branch metric '{branch_metric}'. Choose one of {BRANCH_METRICS}.")

        def _bus_value(result, bus_id: int, selected_metric: str) -> float:
            if selected_metric == "voltage":
                return float(result.voltages.get(bus_id, 0.0))
            if selected_metric == "lmp_q":
                return float(result.duals_q.get(bus_id, 0.0))
            return float(result.duals_p.get(bus_id, 0.0))

        def _flow_component(result, edge: tuple[int, int], selected_metric: str) -> float:
            component_idx = 1 if selected_metric == "flow_q" else 0
            if edge in result.flows:
                return float(result.flows[edge][component_idx])
            reverse_edge = (edge[1], edge[0])
            if reverse_edge in result.flows:
                return float(result.flows[reverse_edge][component_idx])
            return 0.0

        # Compute deviations: percent via SolverComparison, or raw (BFSA - SOCP).
        difference_mode = str(difference_mode).lower().strip()
        if difference_mode == "raw":
            bus_dev = {}
            for b in case.buses:
                bus_id = int(b.bus_id)
                bus_dev[bus_id] = (
                    _bus_value(result_bfsa, bus_id, bus_metric)
                    - _bus_value(result_socp, bus_id, bus_metric)
                )

            branch_dev = {}
            for br in case.branches:
                edge = (int(br.from_bus), int(br.to_bus))
                branch_dev[edge] = (
                    _flow_component(result_bfsa, edge, branch_metric)
                    - _flow_component(result_socp, edge, branch_metric)
                )

        else:
            comparison = SolverComparison(case, result_bfsa, result_socp, names=("BFSA", "SOCP"))
            devs = comparison.deviations
            bus_dev = devs.get(f"{bus_metric}_dev", {})
            branch_dev = devs.get(f"{branch_metric}_dev", {})

        # Build plot inputs from BFSA (reference for layout)
        net, bus_id_map, trans_df, dlmp, bus_voltages = net_module.build_plot_inputs(case, result_bfsa)
        net_module.verify_load_balance(case, result_bfsa)

        # Create deviation figure (dev_is_percent controls labels/units)
        fig = net_module.create_deviation_plot(
            net,
            bus_id_map,
            trans_df,
            case,
            lmp_p_dev=bus_dev,
            flow_dev=branch_dev,
            title=f"Difference bus {bus_metric}, branch {branch_metric} - BFSA - SOCP",
            label_maps={"bus_metric": bus_metric, "branch_metric": branch_metric},
            dev_is_percent=(difference_mode != "raw"),
        )

        saved_path = None
        if output_dir:
            output_path = PathlibPath(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            filename = f"difference_ts{timestep}_bus_{bus_metric}_branch_{branch_metric}_{theme}.html"
            saved_path = output_path / filename
            net_module.export_network_plot(fig, saved_path)

        if show:
            fig.show(config=PLOTLY_HIGH_RES_EXPORT_CONFIG)

        return fig, saved_path

    def _generate_pq_case_counts(
        self,
        database_file,
        solver_option: str,
        start_timestep: int,
        end_timestep: int,
        output_dir=None,
        file_stem=None,
        copy_to_clipboard: bool = False,
    ) -> tuple[str, object]:
        """Generate a pq-case-count summary table from the selected database."""
        db = TimestepDatabase(str(database_file))
        available = db.get_timerange()
        if not available:
            raise RuntimeError("Database has no timesteps for pq case counts.")

        t0 = max(start_timestep, available[0])
        t1 = min(end_timestep, available[-1])
        if t0 > t1:
            raise RuntimeError("No overlapping timesteps in requested range for pq case counts.")

        if output_dir:
            out = Path(output_dir)
        else:
            out = Path(database_file).parent / "pq_case_counts"
        out.mkdir(parents=True, exist_ok=True)

        stem = file_stem or "pq_case_counts"
        option = str(solver_option).strip().lower()
        if option not in {"bfsa", "socp", "both"}:
            raise ValueError("PQ case counts requires solver option BFSA, SOCP, or BOTH.")

        solver_names = [option] if option in {"bfsa", "socp"} else ["bfsa", "socp"]
        first_saved_path = None
        first_df = None
        for solver_name in solver_names:
            excel_path = out / f"{stem}_{solver_name}_pq_case_counts.xlsx"
            df = db.get_bfsa_pq_case_counts(
                solver_name=solver_name,
                start_timestep=t0,
                end_timestep=t1,
                export_to_excel=True,
                excel_path=excel_path,
            )
            if first_saved_path is None:
                first_saved_path = str(excel_path)
                first_df = df

            if copy_to_clipboard:
                self.root.after(0, lambda frame=df, name=solver_name: self._show_dataframe_popup(frame, f"PQ case counts - {name}"))

        self._set_status(
            "PQ case counts saved: " + ", ".join(f"{stem}_{solver}_pq_case_counts.xlsx" for solver in solver_names)
        )
        return first_saved_path or "", first_df if first_df is not None else db.get_bfsa_pq_case_counts(solver_names[0], start_timestep=t0, end_timestep=t1)

    def _generate_comparison_table(
        self,
        database_file,
        solver_names: list,
        output_dir=None,
        table_type: str = "detailed",
        start_timestep: int = 0,
        end_timestep: int = 0,
        file_stem=None,
        difference: bool = False,
        difference_mode: str = "percent",
    ) -> str:
        """
        Generate comparison tables from database results (error metrics and times separated).
        
        Args:
            database_file: Path to SQLite database
            solver_names: List of solver names to compare
            output_dir: Output directory for CSVs
            table_type: "detailed" for per-case or "aggregated" for summary
            file_stem: Optional file stem for output
            
        Returns:
            Path to saved output directory
        """
        from pathlib import Path as PathlibPath
        
        if not solver_names:
            raise ValueError("Comparison table requires at least one solver name.")

        normalized_difference_mode = _normalize_difference_mode(difference_mode)
        
        db = TimestepDatabase(str(database_file))
        available_timesteps = db.get_timerange()
        if not available_timesteps:
            raise RuntimeError("Database has no timestep results for comparison tables.")

        t0 = max(start_timestep, available_timesteps[0])
        t1 = min(end_timestep, available_timesteps[-1])
        if t1 < t0:
            raise RuntimeError(
                "Requested comparison-table range does not overlap database timesteps.\n\n"
                f"Requested: [{start_timestep}, {end_timestep}]\n"
                f"Available: [{available_timesteps[0]}, {available_timesteps[-1]}]"
            )

        use_percent_difference = difference and normalized_difference_mode == "percent"
        if use_percent_difference:
            if len(solver_names) < 2:
                raise ValueError("Percent difference table requires two solver names.")

            solver1_name = solver_names[0]
            solver2_name = solver_names[1]
            case = self._load_matching_case_for_database(db, Path(database_file), load_index=0)

            # Create mock solver objects that retrieve results from database
            class DatabaseSolver:
                def __init__(self, db, solver_name, case, start_timestep, end_timestep):
                    self.db = db
                    self.solver_name = solver_name
                    self.case = case
                    self.start_timestep = start_timestep
                    self.end_timestep = end_timestep

                def solve(self, _case):
                    """Retrieve result from database."""
                    return self.db.retrieve_timestep_range_result(
                        self.start_timestep,
                        self.end_timestep,
                        self.solver_name,
                        self.case,
                    )

            solver1 = DatabaseSolver(db, solver1_name, case, t0, t1)
            solver2 = DatabaseSolver(db, solver2_name, case, t0, t1)

            # Create benchmark comparison
            benchmark = Benchmark(solver1, solver2, names=(solver1_name, solver2_name))
            benchmark.run_on_case(case)

            # Generate appropriate tables
            if table_type == "aggregated":
                data_df, time_df = benchmark.generate_aggregated_comparison_tables_split()
                data_suffix = "comparison_aggregated_errors"
                time_suffix = "comparison_aggregated_times"
            else:
                data_df, time_df = benchmark.generate_summary_tables_split()
                data_suffix = "comparison_detailed_errors"
                time_suffix = "comparison_detailed_times"
        else:
            data_df, time_df = build_comparison_value_tables(
                db,
                solver_names,
                t0,
                t1,
                table_type=table_type,
                difference=difference,
            )
            table_label = "absolute_difference" if difference else "values"
            if table_type == "aggregated":
                data_suffix = f"comparison_aggregated_{table_label}"
                time_suffix = "comparison_aggregated_times"
            else:
                data_suffix = f"comparison_detailed_{table_label}"
                time_suffix = "comparison_detailed_times"
        
        # Save to CSV
        output_path = None
        if output_dir:
            output_path = PathlibPath(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            error_filename = (
                comparison_table_filename_stem(
                    file_stem,
                    data_suffix,
                    t0,
                    t1,
                    solver_names,
                    difference=difference,
                )
                + ".csv"
            )
            time_filename = (
                comparison_table_filename_stem(
                    file_stem,
                    time_suffix,
                    t0,
                    t1,
                    solver_names,
                    difference=difference,
                )
                + ".csv"
            )
            
            error_path = output_path / error_filename
            time_path = output_path / time_filename
            
            data_df.to_csv(error_path, index=False)
            time_df.to_csv(time_path, index=False)

            self._set_status(f"Comparison tables saved:\n  Data: {error_filename}\n  Times: {time_filename}")
        
        return str(output_path) if output_path else None

    def _generate_lmp_stats(
        self,
        database_file,
        solver_option: str,
        start_timestep: int,
        end_timestep: int,
        output_dir=None,
        file_stem=None,
        copy_to_clipboard: bool = False,
    ) -> str:
        """
        Export LMP value statistics for a given solver across a timestep range.

        Produces two CSVs per solver: active LMP (`lmp_p`) and reactive LMP (`lmp_q`).
        Optionally copies the tables to clipboard when `copy_to_clipboard` is True.
        """
        import pandas as pd
        from pathlib import Path as PathlibPath

        db = TimestepDatabase(str(database_file))
        available = db.get_timerange()
        if not available:
            raise RuntimeError("Database has no timesteps for LMP statistics.")

        # Clip requested range to available timesteps
        t0 = max(start_timestep, available[0])
        t1 = min(end_timestep, available[-1])
        timesteps = [t for t in available if t0 <= t <= t1]
        if not timesteps:
            raise RuntimeError("No overlapping timesteps in requested range for LMP stats.")

        option = str(solver_option).strip().lower()
        if option not in {"bfsa", "socp", "both"}:
            raise ValueError("LMP stats requires solver option BFSA, SOCP, or BOTH.")

        try:
            case = self._load_matching_case_for_database(db, Path(database_file), load_index=None)
        except Exception:
            # Best-effort fallback: construct a minimal PowerFlowCase from DB metadata so we can
            # reconstruct OPFResult objects. This avoids failing when the local `data/` folder
            # doesn't contain matching grid files.
            from libs.shared.data import PowerFlowCase, BusData, BranchData

            bus_ids = db.get_bus_ids()
            buses = [BusData(bus_id=int(b)) for b in bus_ids]
            branches = []
            for fbus, tbus in db.get_branch_endpoints():
                branches.append(BranchData(branch_id=0, from_bus=int(fbus), to_bus=int(tbus), r=0.0, x=0.0))

            case = PowerFlowCase(buses=buses, branches=branches, generators=[], loads=[], root_bus=int(bus_ids[0]) if bus_ids else 0)

        solver_names = [option] if option in {"bfsa", "socp"} else ["bfsa", "socp"]

        def stats_from_lists(values: list):
            arr = np.array(values)
            return {
                "count": int(arr.size),
                "mean": float(np.mean(arr)),
                "median": float(np.median(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "p25": float(np.percentile(arr, 25)),
                "p75": float(np.percentile(arr, 75)),
                "p95": float(np.percentile(arr, 95)),
            }

        saved_dir = None
        # Provide a reasonable default output directory if none supplied
        if output_dir:
            out = Path(output_dir)
        else:
            out = Path(database_file).parent / "lmp_stats"

        out.mkdir(parents=True, exist_ok=True)
        stem = file_stem or "lmp_stats"
        saved_names: list[str] = []

        for solver_name in solver_names:
            lmp_p_vals: dict = {}
            lmp_q_vals: dict = {}

            for ts in timesteps:
                try:
                    res = db.retrieve_timestep_result(ts, solver_name, case)
                except Exception:
                    continue

                if hasattr(res, "duals_p") and res.duals_p:
                    for bus_id, val in res.duals_p.items():
                        lmp_p_vals.setdefault(bus_id, []).append(float(val))

                if hasattr(res, "duals_q") and res.duals_q:
                    for bus_id, val in res.duals_q.items():
                        lmp_q_vals.setdefault(bus_id, []).append(float(val))

            rows_p = []
            for bus, vals in sorted(lmp_p_vals.items(), key=lambda x: int(x[0])):
                s = stats_from_lists(vals)
                rows_p.append({"bus": int(bus), **{k: round(v, 3) for k, v in s.items()}})

            rows_q = []
            for bus, vals in sorted(lmp_q_vals.items(), key=lambda x: int(x[0])):
                s = stats_from_lists(vals)
                rows_q.append({"bus": int(bus), **{k: round(v, 3) for k, v in s.items()}})

            df_p = pd.DataFrame(rows_p)
            df_q = pd.DataFrame(rows_q)

            p_path = out / f"{stem}_{solver_name}_lmp_p_stats.csv"
            q_path = out / f"{stem}_{solver_name}_lmp_q_stats.csv"
            df_p.to_csv(p_path, index=False)
            df_q.to_csv(q_path, index=False)
            saved_names.extend([p_path.name, q_path.name])

            if copy_to_clipboard:
                if not df_p.empty:
                    self.root.after(0, lambda frame=df_p, name=solver_name: self._show_dataframe_popup(frame, f"Active LMPs - {name}"))
                if not df_q.empty:
                    self.root.after(0, lambda frame=df_q, name=solver_name: self._show_dataframe_popup(frame, f"Reactive LMPs - {name}"))

        saved_dir = out
        self._set_status("LMP stats saved: " + ", ".join(saved_names))

        return str(saved_dir) if saved_dir is not None else None

    def _show_dataframe_popup(self, df, title: str = "Table") -> None:
        """Display a DataFrame in a popup with a Copy button to copy its text to the clipboard."""

        def _show():
            # use Toplevel when main root exists
            if hasattr(self, "root") and self.root is not None:
                win = Toplevel(self.root)
            else:
                win = Toplevel()

            win.title(title)
            win.geometry("800x600")

            txt = scrolledtext.ScrolledText(win, wrap="none")
            txt.pack(fill=BOTH, expand=True, padx=8, pady=8)
            txt.insert("1.0", df.to_string(index=False))
            txt.configure(state="disabled")

            btn_frame = ttk.Frame(win)
            btn_frame.pack(fill="x", padx=8, pady=(0, 8))

            def _copy():
                try:
                    content = df.to_csv(index=False, sep="\t")
                    # use the main root clipboard if available
                    if hasattr(self, "root") and self.root is not None:
                        self.root.clipboard_clear()
                        self.root.clipboard_append(content)
                    else:
                        win.clipboard_clear()
                        win.clipboard_append(content)
                    self._set_status(f"{title} copied to clipboard.")
                except Exception as exc:
                    self._set_status(f"Copy failed: {exc}")

            copy_btn = ttk.Button(btn_frame, text="Copy to clipboard", command=_copy)
            copy_btn.pack(side=LEFT, padx=(0, 8))

            close_btn = ttk.Button(btn_frame, text="Close", command=win.destroy)
            close_btn.pack(side=RIGHT)

        # Ensure popup runs on main thread
        try:
            self.root.after(0, _show)
        except Exception:
            # fallback: call directly
            _show()

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    """Launch the plot generator GUI."""
    PlotGeneratorGUI().run()


if __name__ == "__main__":
    main()
