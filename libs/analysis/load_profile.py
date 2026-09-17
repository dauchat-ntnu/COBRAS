"""
Load profile analysis utilities for MATPOWER-style network folders.

This module focuses on the p_load.csv and q_load.csv files and provides:
- load profile loading with duplicate bus-column merging
- per-bus summary statistics
- box plot exports for active and reactive demand
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd


@dataclass(frozen=True)
class LoadProfileAnalysisResult:
    """Container for load profile analysis outputs."""

    p_load: pd.DataFrame
    q_load: pd.DataFrame
    summary: pd.DataFrame
    plot_path: Optional[Path] = None
    summary_path: Optional[Path] = None






def load_profile_timeseries(folder_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load active and reactive demand time series from a MATPOWER-style folder.

    Returns the merged p_load/q_load dataframes with the Date column used as the index.
    Duplicate bus columns are merged automatically.
    """

    from libs.shared.io import _merge_dot_suffix_columns, _sanitize_load_profile_dataframe

    folder_path = Path(folder_path)
    p_load_df = pd.read_csv(folder_path / "p_load.csv", sep=None, engine="python", index_col=0)
    q_load_df = pd.read_csv(folder_path / "q_load.csv", sep=None, engine="python", index_col=0)

    p_load_df = _merge_dot_suffix_columns(p_load_df)
    q_load_df = _merge_dot_suffix_columns(q_load_df)

    # Use the same validation as the solver-facing loader.
    p_load_df = _sanitize_load_profile_dataframe(p_load_df)
    q_load_df = _sanitize_load_profile_dataframe(q_load_df)

    p_cols = set(p_load_df.columns)
    q_cols = set(q_load_df.columns)
    if p_cols != q_cols:
        missing_in_q = sorted(p_cols - q_cols)
        missing_in_p = sorted(q_cols - p_cols)
        raise ValueError(
            "p_load.csv and q_load.csv have mismatched bus columns after normalization. "
            f"Missing in q_load: {missing_in_q}. Missing in p_load: {missing_in_p}."
        )

    ordered_columns = sorted(p_load_df.columns, key=lambda value: int(str(value)))
    p_load_df = p_load_df.loc[:, ordered_columns]
    q_load_df = q_load_df.loc[:, ordered_columns]

    return p_load_df, q_load_df


def summarize_load_profiles(p_load_df: pd.DataFrame, q_load_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-bus summary statistics for active and reactive demand."""

    bus_ids = [int(str(col)) for col in p_load_df.columns]
    p_summary = p_load_df.apply(pd.to_numeric, errors="coerce").agg(["mean", "median", "min", "max", "std"]).T
    q_summary = q_load_df.apply(pd.to_numeric, errors="coerce").agg(["mean", "median", "min", "max", "std"]).T

    summary = pd.DataFrame(
        {
            "bus": bus_ids,
            "p_mean": p_summary["mean"].to_numpy(),
            "p_median": p_summary["median"].to_numpy(),
            "p_min": p_summary["min"].to_numpy(),
            "p_max": p_summary["max"].to_numpy(),
            "p_std": p_summary["std"].to_numpy(),
            "q_mean": q_summary["mean"].to_numpy(),
            "q_median": q_summary["median"].to_numpy(),
            "q_min": q_summary["min"].to_numpy(),
            "q_max": q_summary["max"].to_numpy(),
            "q_std": q_summary["std"].to_numpy(),
            "sample_count": len(p_load_df.index),
        }
    ).sort_values("bus")

    return summary.reset_index(drop=True)


def plot_load_profile_timeseries(
    p_load_df: pd.DataFrame,
    output_dir: Optional[Path] = None,
    file_stem: str = "load_profile_timeseries",
    show: bool = False,
    normalize: bool = True,
    start_step: Optional[int] = None,
    end_step: Optional[int] = None,
) -> Optional[Path]:
    """
    Plot all normalized load profiles over time with one color per bus.

    Args:
        p_load_df: DataFrame with datetime index and bus columns
        output_dir: Directory to save PNG (optional)
        file_stem: Filename stem for saved plot
        show: Whether to display plot interactively
        normalize: If True, normalize each bus profile by its max absolute value
        start_step: Starting timestep index (inclusive, default 0)
        end_step: Ending timestep index (inclusive, default last step)

    Returns:
        Saved PNG path when output_dir is provided; otherwise returns None
    """

    try:
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
    except ImportError:
        return None

    p_load_numeric = p_load_df.apply(pd.to_numeric, errors="coerce")
    bus_ids = [int(str(col)) for col in p_load_df.columns]

    # Slice the dataframe to the specified range
    if start_step is None:
        start_step = 0
    if end_step is None:
        end_step = len(p_load_numeric) - 1

    # Clamp to valid range
    start_step = max(0, start_step)
    end_step = min(len(p_load_numeric) - 1, end_step)

    p_load_sliced = p_load_numeric.iloc[start_step : end_step + 1]

    # Normalize profiles if requested
    if normalize:
        scale_vals = p_load_sliced.abs().max(axis=0)
        scale_vals[scale_vals == 0] = 1.0  # Avoid division by zero
        p_plot = p_load_sliced / scale_vals
    else:
        p_plot = p_load_sliced

    # Create figure with enough width for all buses
    n_buses = len(bus_ids)
    fig_width = max(14.0, 0.15 * len(p_plot.index))
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    # Generate colors for each bus
    colormap = cm.get_cmap("tab20" if n_buses <= 20 else "hsv")
    colors = [colormap(i / max(n_buses - 1, 1)) for i in range(n_buses)]

    # Plot each bus's normalized profile
    time_index = range(len(p_plot))
    for (bus_col, bus_id), color in zip(zip(p_load_df.columns, bus_ids), colors):
        values = p_plot[bus_col].values
        ax.plot(time_index, values, label=f"Bus {bus_id}", color=color, linewidth=1.2, alpha=0.8)

    ax.set_xlabel("Time Step")
    ylabel_text = "Normalized Load (per-bus by max abs)" if normalize else "Load (p.u.)"
    ax.set_ylabel(ylabel_text)
    title = f"Load Profiles Over Time (Steps {start_step}–{end_step})"
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9, ncol=1)

    fig.tight_layout()

    saved_path = None
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        saved_path = output_dir / f"{file_stem}.png"
        fig.savefig(saved_path, dpi=180, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return saved_path


def plot_load_profile_timeseries_interactive(
    p_load_df: pd.DataFrame,
    output_dir: Optional[Path] = None,
    file_stem: str = "load_profile_timeseries_interactive",
    start_step: Optional[int] = None,
    end_step: Optional[int] = None,
    normalize: bool = True,
    show: bool = False,
) -> Optional[Path]:
    """
    Create an interactive HTML timeseries plot where bus labels can toggle traces.

    Clicking a bus label annotation at the top of the chart hides/shows that bus.
    """

    try:
        import plotly.graph_objects as go
    except ImportError:
        return None

    p_load_numeric = p_load_df.apply(pd.to_numeric, errors="coerce")
    bus_ids = [int(str(col)) for col in p_load_df.columns]

    if start_step is None:
        start_step = 0
    if end_step is None:
        end_step = len(p_load_numeric) - 1

    start_step = max(0, start_step)
    end_step = min(len(p_load_numeric) - 1, end_step)
    p_load_sliced = p_load_numeric.iloc[start_step : end_step + 1]

    if normalize:
        scale_vals = p_load_sliced.abs().max(axis=0)
        scale_vals[scale_vals == 0] = 1.0
        p_plot = p_load_sliced / scale_vals
    else:
        p_plot = p_load_sliced

    fig = go.Figure()
    for bus_col, bus_id in zip(p_load_df.columns, bus_ids):
        fig.add_trace(
            go.Scatter(
                x=list(range(len(p_plot))),
                y=p_plot[bus_col].tolist(),
                mode="lines",
                name=f"Bus {bus_id}",
                customdata=[bus_id] * len(p_plot),
                hovertemplate="Bus %{customdata}<br>Step %{x}<br>Value %{y:.4f}<extra></extra>",
                line={"width": 1.4},
            )
        )

    ylabel_text = "Normalized Load (per-bus by max abs)" if normalize else "Load (p.u.)"
    fig.update_layout(
        title=f"Interactive Load Profiles (Steps {start_step}-{end_step})",
        xaxis_title="Time Step",
        yaxis_title=ylabel_text,
        template="plotly_white",
        hovermode="x unified",
        legend={"orientation": "v", "x": 1.02, "y": 1.0},
        margin={"l": 70, "r": 220, "t": 130, "b": 70},
    )

    annotation_x_positions = [
        (i + 0.5) / max(len(bus_ids), 1)
        for i in range(len(bus_ids))
    ]
    annotations = []
    for x_pos, bus_id in zip(annotation_x_positions, bus_ids):
        annotations.append(
            {
                "xref": "paper",
                "yref": "paper",
                "x": x_pos,
                "y": 1.12,
                "text": f"Bus {bus_id}",
                "showarrow": False,
                "font": {"size": 10, "color": "#1f2937"},
                "align": "center",
                "captureevents": True,
                "bordercolor": "#cbd5e1",
                "borderwidth": 1,
                "borderpad": 2,
                "bgcolor": "#f8fafc",
            }
        )
    fig.update_layout(annotations=annotations)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        saved_path = output_dir / f"{file_stem}.html"
    else:
        saved_path = Path(file_stem).with_suffix(".html")

    bus_labels_json = json.dumps([f"Bus {bus_id}" for bus_id in bus_ids])
    post_script = f"""
const busLabels = {bus_labels_json};
const hiddenSet = new Set();
const gd = document.getElementById('{{plot_id}}');
function toggleBus(label) {{
  const traceIdx = busLabels.indexOf(label);
  if (traceIdx < 0) return;
  const hidden = hiddenSet.has(traceIdx);
  const newVisibility = hidden ? true : 'legendonly';
  Plotly.restyle(gd, {{visible: newVisibility}}, [traceIdx]);
  if (hidden) {{
    hiddenSet.delete(traceIdx);
  }} else {{
    hiddenSet.add(traceIdx);
  }}

  const nextAnnotations = gd.layout.annotations.map((ann) => {{
    if (ann.text !== label) return ann;
    return {{
      ...ann,
      font: {{...ann.font, color: hidden ? '#1f2937' : '#94a3b8'}},
      bgcolor: hidden ? '#f8fafc' : '#e2e8f0'
    }};
  }});
  Plotly.relayout(gd, {{annotations: nextAnnotations}});
}}

gd.on('plotly_clickannotation', (ev) => {{
  const label = ev && ev.annotation && ev.annotation.text;
  if (label) toggleBus(label);
}});
"""

    fig.write_html(str(saved_path), include_plotlyjs="cdn", post_script=post_script)

    if show:
        try:
            import webbrowser

            webbrowser.open_new_tab(saved_path.resolve().as_uri())
        except Exception:
            pass

    return saved_path


def plot_load_profile_boxplots(
    p_load_df: pd.DataFrame,
    q_load_df: pd.DataFrame,
    output_dir: Optional[Path] = None,
    file_stem: str = "bus_demand_boxplots",
    show: bool = False,
) -> Optional[Path]:
    """
    Create side-by-side box plots for active and reactive demand by bus.

    Returns the saved PNG path when output_dir is provided; otherwise returns None.
    """

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    bus_labels = [str(int(str(col))) for col in p_load_df.columns]
    p_values = [pd.to_numeric(p_load_df[col], errors="coerce").dropna().to_numpy() for col in p_load_df.columns]
    q_values = [pd.to_numeric(q_load_df[col], errors="coerce").dropna().to_numpy() for col in q_load_df.columns]

    fig_width = max(14.0, 0.35 * len(bus_labels))
    fig, axes = plt.subplots(2, 1, figsize=(fig_width, 10), sharex=True)

    axes[0].boxplot(p_values, labels=bus_labels, showfliers=False)
    axes[0].set_title("Active Demand by Bus")
    axes[0].set_ylabel("P load")
    axes[0].grid(True, axis="y", alpha=0.25)

    axes[1].boxplot(q_values, labels=bus_labels, showfliers=False)
    axes[1].set_title("Reactive Demand by Bus")
    axes[1].set_ylabel("Q load")
    axes[1].set_xlabel("Bus")
    axes[1].grid(True, axis="y", alpha=0.25)

    for axis in axes:
        axis.tick_params(axis="x", labelrotation=90)

    fig.suptitle("Bus Demand Distribution from p_load.csv / q_load.csv")
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    saved_path = None
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        saved_path = output_dir / f"{file_stem}.png"
        fig.savefig(saved_path, dpi=180, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return saved_path


def analyze_load_profiles(
    folder_path: Path,
    output_dir: Optional[Path] = None,
    show: bool = False,
    scale_factor: float = 1.0,
) -> LoadProfileAnalysisResult:
    """
    Run the complete load profile analysis for a MATPOWER-style folder.

    Exports a box plot PNG and a summary CSV when output_dir is provided.
    """

    folder_path = Path(folder_path)
    p_load_df, q_load_df = load_profile_timeseries(folder_path)

    if scale_factor != 1.0:
        p_load_df = p_load_df * scale_factor
        q_load_df = q_load_df * scale_factor

    summary = summarize_load_profiles(p_load_df, q_load_df)

    plot_path = None
    summary_path = None
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = output_dir / "bus_demand_summary.csv"
        summary.to_csv(summary_path, index=False)
        plot_path = plot_load_profile_boxplots(p_load_df, q_load_df, output_dir=output_dir, show=show)
    elif show:
        plot_load_profile_boxplots(p_load_df, q_load_df, output_dir=None, show=True)

    return LoadProfileAnalysisResult(
        p_load=p_load_df,
        q_load=q_load_df,
        summary=summary,
        plot_path=plot_path,
        summary_path=summary_path,
    )
