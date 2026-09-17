"""Plot sensitivity results for LMP deviations.

Reads the most recent sensitivity run under data/<network>/sensitivity (or use
`--run-dir` to point at a specific run folder). Produces two boxplots:
- Active LMP relative error: (BFSA - SOCP) / SOCP (zero where SOCP==0)
- Reactive LMP relative error: same as above for reactive LMPs

Box grouping: x-axis = ratio values; for each ratio, there are boxes for
the requested snapshot labels (e.g., average/high/low) colored distinctly.

Usage (from workspace root):
    python examples/plot_sensitivity_lmp.py
    python examples/plot_sensitivity_lmp.py --run-dir "data/MV_solar/sensitivity/..."
"""

from pathlib import Path
import sys
import argparse
import re
import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

# Ensure local imports work when running directly
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def find_latest_run(network_folder: Path) -> Path:
    sens_dir = network_folder / "sensitivity"
    if not sens_dir.exists():
        raise FileNotFoundError(f"No sensitivity directory at {sens_dir}")
    candidates = [d for d in sens_dir.iterdir() if d.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No runs found in {sens_dir}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def parse_scenario_label(label: str) -> dict:
    # expected format: sensitivity:<name>;ratio:<val>;label:<label>;gen:<id>
    out = {}
    parts = label.split(";")
    for p in parts:
        if ":" in p:
            k, v = p.split(":", 1)
            out[k.strip()] = v.strip()
    return out


def load_results(db_path: Path):
    con = sqlite3.connect(str(db_path))
    try:
        df_timesteps = pd.read_sql_query("SELECT timestep, scenario_label, solver_name FROM Res_Timesteps", con)
        df_buses = pd.read_sql_query("SELECT timestep, solver_name, bus_id, lmp_p, lmp_q FROM Res_Buses", con)
    finally:
        con.close()

    return df_timesteps, df_buses


def compute_errors(df_timesteps: pd.DataFrame, df_buses: pd.DataFrame):
    # Build mapping timestep -> parsed metadata from SOCP rows
    df_socp_meta = df_timesteps[df_timesteps['solver_name'].str.lower() == 'socp'].copy()
    df_socp_meta['meta'] = df_socp_meta['scenario_label'].apply(lambda s: parse_scenario_label(s))
    df_socp_meta['ratio'] = df_socp_meta['meta'].apply(lambda m: float(m.get('ratio', 0.0)))
    df_socp_meta['label'] = df_socp_meta['meta'].apply(lambda m: m.get('label', ''))

    # Prepare bus-level pivot: for each (timestep, bus_id) get socp and bfsa lmp values
    df_buses_pivot = df_buses.pivot_table(index=['timestep', 'bus_id'], columns='solver_name', values=['lmp_p', 'lmp_q'])
    # Flatten columns
    df_buses_pivot.columns = ['_'.join(col).strip() for col in df_buses_pivot.columns.values]
    df_buses_pivot = df_buses_pivot.reset_index()

    # Join with socp meta (on timestep)
    df = df_buses_pivot.merge(df_socp_meta[['timestep', 'ratio', 'label']], on='timestep', how='left')

    # Compute relative errors: (bfsa - socp) / socp, but zero where socp == 0 or NaN
    def rel_err(bfsa, socp):
        try:
            if socp is None or socp == 0 or np.isnan(socp):
                return 0.0
            return float((bfsa - socp) / socp)*100  # convert to percentage
        except Exception:
            return 0.0

    df['err_p'] = df.apply(lambda r: rel_err(r.get('lmp_p_bfsa', 0.0), r.get('lmp_p_socp', 0.0)), axis=1)
    df['err_q'] = df.apply(lambda r: rel_err(r.get('lmp_q_bfsa', 0.0), r.get('lmp_q_socp', 0.0)), axis=1)

    return df


def grouped_boxplot(df, value_col: str, ratios: list, labels_order: list, out_path: Path, title: str, outliers=True):
    # Prepare data lists in order: for each ratio, for each label
    data = []
    positions = []
    tick_positions = []
    tick_labels = []
    width = 0.12
    group_spacing = 0.75
    if len(labels_order) == 1:
        label_offsets = [0.0]
    else:
        label_offsets = np.linspace(-0.18, 0.18, len(labels_order))
    for i, ratio in enumerate(ratios):
        base = i * group_spacing
        tick_positions.append(base)
        tick_labels.append(f"{ratio:.3f}")
        for j, lab in enumerate(labels_order):
            sel = df[(df['ratio'] == ratio) & (df['label'] == lab)][value_col].dropna().values  # convert to percentage
            data.append(sel)
            positions.append(base + float(label_offsets[j]))

    fig, ax = plt.subplots(figsize=(max(8, len(ratios)*1.2), 6))
    bp = ax.boxplot(data, positions=positions, widths=width, patch_artist=True, manage_ticks=False, showfliers=outliers)

    # Color boxes by label
    colors = plt.cm.Set2.colors
    for idx, patch in enumerate(bp['boxes']):
        lab_idx = idx % len(labels_order)
        patch.set_facecolor(colors[lab_idx % len(colors)])

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_title(title)
    # Create legend for labels
    handles = [plt.Rectangle((0,0),1,1, color=colors[i % len(colors)]) for i in range(len(labels_order))]
    ax.legend(handles, labels_order, frameon=False)
    ax.margins(x=0.02)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    sns.despine()
    fig.savefig(str(out_path), dpi=200)
    plt.close(fig)


def main(network_name=None, run_dir=None, labels_order=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', default='80_bus_base', help='Network folder under data/')
    parser.add_argument('--run-dir', default=None, help='Optional explicit sensitivity run folder')
    parser.add_argument('--labels', default='average,high,low', help='Comma-separated snapshot labels in preferred order')
    args = parser.parse_args([] if any(value is not None for value in (network_name, run_dir, labels_order)) else None)

    network = network_name or args.network
    network_folder = Path('data') / network

    if run_dir is not None or args.run_dir:
        run_dir = Path(run_dir or args.run_dir)
    else:
        run_dir = find_latest_run(network_folder)

    print(f"Using run dir: {run_dir}")
    db_path = run_dir / 'db' / 'results.db'
    if not db_path.exists():
        db_path = run_dir / 'results.db'
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found at {db_path}")

    df_timesteps, df_buses = load_results(db_path)
    df = compute_errors(df_timesteps, df_buses)

    # Determine ratios and labels order
    ratios = sorted(df['ratio'].dropna().unique())
    labels_order = labels_order or [s.strip() for s in args.labels.split(',')]

    plots_dir = run_dir / 'plots'
    grouped_boxplot(df, 'err_p', ratios, labels_order, plots_dir / 'lmp_p_deviation_boxplot.png', 'Active LMP Relative Errors')
    grouped_boxplot(df, 'err_q', ratios, labels_order, plots_dir / 'lmp_q_deviation_boxplot.png', 'Reactive LMP Relative Errors', outliers=False)

    print(f"Plots written to {plots_dir}")


if __name__ == '__main__':
    main()
