"""
Notebook-derived network visualizations for PowerSOC results.

This module keeps the aggregate-notebook plotting APIs and behavior:
  - create_combined_plot
  - create_white_network_plot
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import matplotlib as mpl
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pandapower.plotting.plotly import simple_plotly

from libs.shared import OPFResult, PowerFlowCase


CONFIG = {
    "lmp_type_p": "lambda_p",
    "lmp_type_q": "lambda_q",
    "line_colorscale": "OrRd",
    "bus_colorscale": "viridis",
    "line_width": 2,
    "bus_size": 10,
    "show_arrows": True,
    "arrow_min_abs_flow_pu": 1e-3,
    # Line coloring mode: "flow_pu" (current flow in p.u.) or "loading_pct" (flow vs line rating).
    "line_color_mode": "flow_pu",
    "lmp_uniform_tol": 0.01,
    # Hide zero-valued fields from hover tooltips when True.
    "hide_zero_hover_fields": False,
    # Absolute tolerance used when evaluating whether a value is effectively zero.
    "hover_zero_tol": 1e-12,
    # General visualization theme option: "white" | "light" | "dark"
    "theme": "dark",
}

LINE_COLORBAR_LABELS = {
    "flow_pu": r"$\text{Line loading p.u.}$",
    "loading_pct": r"$\text{Line loading \%}$",
}

LMP_COLORBAR_LABELS = {
    "lambda_p": r"$\text{Active LMP }[\mathrm{€/MWh}]$",
    "lambda_q": r"$\text{Reactive LMP }[\mathrm{€/MVArh}]$",
    "voltage": r"$\text{Voltage }[\mathrm{p.u.}]$",
}

PLOTLY_HIGH_RES_EXPORT_CONFIG = {
    "toImageButtonOptions": {
        "format": "png",
        "scale": 4,
    },
}

THEME_CONFIG = {
    "dark": {
        "paper_bgcolor": "black",
        "plot_bgcolor": "black",
        "font_color": "white",
        "line_colorscale": "OrRd",
        "bus_colorscale": "viridis",
        "line_fallback_color": "white",
        "external_grid_color": None,
        "monochrome": False,
    },
    "light": {
        "paper_bgcolor": "white",
        "plot_bgcolor": "white",
        "font_color": "black",
        # High-contrast light mode: dark low-flow lines to red at high loading.
        "line_colorscale": "black_red",
        # Use teal scale for bus coloring in light theme as requested.
        "bus_colorscale": "teal_scale",
        "line_fallback_color": "black",
        "external_grid_color": "#d1d5db",
        "monochrome": False,
    },
    # White mode per user convention: black background with white lines/nodes.
    "white": {
        "paper_bgcolor": "black",
        "plot_bgcolor": "black",
        "font_color": "white",
        "line_colorscale": "Greys",
        "bus_colorscale": "teal_scale",
        "line_fallback_color": "white",
        "external_grid_color": None,
        "monochrome": True,
    },
}


_CUSTOM_MPL_COLORMAPS = {
    "black_red": mcolors.LinearSegmentedColormap.from_list(
        "black_red", ["#000000", "#8B0000", "#FF0000"]
    ),
    "green_orange": mcolors.LinearSegmentedColormap.from_list(
        "green_orange", ["#1B5E20", "#66BB6A", "#F9A825", "#EF6C00"]
    ),
    "teal_scale": mcolors.LinearSegmentedColormap.from_list(
        "teal_scale", ["#c7f9cc", "#52b788", "#1b4332"]
    ),
}

_CUSTOM_PLOTLY_COLORSCALES = {
    "black_red": [
        [0.0, "#000000"],
        [0.6, "#8B0000"],
        [1.0, "#FF0000"],
    ],
    "green_orange": [
        [0.0, "#1B5E20"],
        [0.35, "#66BB6A"],
        [0.7, "#F9A825"],
        [1.0, "#EF6C00"],
    ],
    "teal_scale": [
        [0.0, "#c7f9cc"],
        [0.5, "#52b788"],
        [1.0, "#1b4332"],
    ],
}


def _get_mpl_cmap(colorscale_name: str):
    if colorscale_name in _CUSTOM_MPL_COLORMAPS:
        return _CUSTOM_MPL_COLORMAPS[colorscale_name]
    return mpl.colormaps.get_cmap(colorscale_name)


def _get_plotly_colorscale(colorscale_name: str):
    if colorscale_name in _CUSTOM_PLOTLY_COLORSCALES:
        return _CUSTOM_PLOTLY_COLORSCALES[colorscale_name]
    return colorscale_name


def set_visual_theme(theme: str) -> None:
    """Set global visualization theme for network plots."""
    theme = str(theme).lower().strip()
    if theme not in THEME_CONFIG:
        raise ValueError(f"Unknown theme '{theme}'. Use one of: {sorted(THEME_CONFIG.keys())}")
    CONFIG["theme"] = theme


def get_visual_theme() -> str:
    """Get current global visualization theme."""
    return str(CONFIG.get("theme", "dark"))



def get_hide_zero_hover_fields() -> bool:
    """Get current hover field filtering option."""
    return bool(CONFIG.get("hide_zero_hover_fields", False))


def set_line_color_mode(mode: str) -> None:
    """Set the line coloring mode for network plots."""
    mode = str(mode).strip().lower()
    if mode not in {"flow_pu", "loading_pct"}:
        raise ValueError("Unknown line color mode. Use 'flow_pu' or 'loading_pct'.")
    CONFIG["line_color_mode"] = mode


def get_line_color_mode() -> str:
    """Get the current line coloring mode."""
    return str(CONFIG.get("line_color_mode", "flow_pu"))


def _theme_settings() -> Dict[str, str]:
    return THEME_CONFIG.get(get_visual_theme(), THEME_CONFIG["dark"])


def _is_effectively_zero(value: float, tol: float | None = None) -> bool:
    tol = float(CONFIG.get("hover_zero_tol", 1e-12)) if tol is None else float(tol)
    return bool(np.isfinite(value) and abs(float(value)) <= tol)


def _append_hover_value(
    lines: list[str],
    label: str,
    value: float,
    fmt: str,
    unit: str = "",
    hide_zero_fields: bool = False,
) -> None:
    """Append a formatted hover line unless filtered by the zero-hide option."""
    if hide_zero_fields and (not np.isfinite(value) or _is_effectively_zero(float(value))):
        return
    suffix = f" {unit}" if unit else ""
    lines.append(f"{label}: {format(float(value), fmt)}{suffix}")


def _aggregate_case_loads(case: PowerFlowCase) -> Tuple[pd.Series, pd.Series]:
    p_by_bus = {}
    q_by_bus = {}
    for load in case.loads:
        p_by_bus[load.bus_id] = p_by_bus.get(load.bus_id, 0.0) + float(load.p_d)
        q_by_bus[load.bus_id] = q_by_bus.get(load.bus_id, 0.0) + float(load.q_d)

    p_bus = pd.Series(p_by_bus, dtype=float).sort_index()
    q_bus = pd.Series(q_by_bus, dtype=float).sort_index()
    return p_bus, q_bus


def _aggregate_case_generation(case: PowerFlowCase, result: OPFResult) -> Tuple[pd.Series, pd.Series]:
    """Aggregate solved generator injections by bus in p.u. units."""
    p_by_bus = {}
    q_by_bus = {}

    # Map generator id to connected bus for robust lookup.
    gen_to_bus = {int(gen.gen_id): int(gen.bus_id) for gen in case.generators}

    for gen_id, pq in (result.generator_output or {}).items():
        if pq is None:
            continue

        bus_id = gen_to_bus.get(int(gen_id))
        if bus_id is None:
            continue

        p_val = float(pq[0]) if len(pq) > 0 else 0.0
        q_val = float(pq[1]) if len(pq) > 1 else 0.0
        p_by_bus[bus_id] = p_by_bus.get(bus_id, 0.0) + p_val
        q_by_bus[bus_id] = q_by_bus.get(bus_id, 0.0) + q_val

    p_bus = pd.Series(p_by_bus, dtype=float).sort_index()
    q_bus = pd.Series(q_by_bus, dtype=float).sort_index()
    return p_bus, q_bus


def build_pandapower_net_from_case(case: PowerFlowCase):
    """Create a pandapower net from a PowerFlowCase for notebook-style plotting."""
    import pandapower as pp

    net = pp.create_empty_network(sn_mva=case.base_mva)
    bus_id_map = {}

    for bus in case.buses:
        pp_bus = pp.create_bus(
            net,
            vn_kv=float(bus.base_kv),
            name=f"Bus_{bus.bus_id}",
            max_vm_pu=float(bus.v_max),
            min_vm_pu=float(bus.v_min),
        )
        bus_id_map[int(bus.bus_id)] = int(pp_bus)

    root_bus = case.root_bus
    if root_bus not in bus_id_map:
        raise ValueError(f"Root bus {root_bus} is not present in the bus map")

    pp.create_ext_grid(net, bus=bus_id_map[root_bus], vm_pu=1.0, name="EGT")

    for branch in case.branches:
        from_bus_idx = bus_id_map[int(branch.from_bus)]
        to_bus_idx = bus_id_map[int(branch.to_bus)]
        vn_kv = float(net.bus.at[from_bus_idx, "vn_kv"])
        max_i_ka = (
            float(branch.s_max) / (np.sqrt(3.0) * vn_kv)
            if np.isfinite(branch.s_max) and vn_kv > 0.0
            else 1.0
        )
        pp.create_line_from_parameters(
            net,
            from_bus=from_bus_idx,
            to_bus=to_bus_idx,
            length_km=1.0,
            r_ohm_per_km=float(branch.r),
            x_ohm_per_km=float(branch.x),
            c_nf_per_km=0.0,
            max_i_ka=max_i_ka,
            name=f"Line_{branch.from_bus}_{branch.to_bus}",
        )

    for load in case.loads:
        if load.bus_id in bus_id_map:
            pp.create_load(
                net,
                bus=bus_id_map[int(load.bus_id)],
                p_mw=float(load.p_d) * float(case.base_mva),
                q_mvar=float(load.q_d) * float(case.base_mva),
                name=f"Load_Bus{load.bus_id}",
                controllable=False,
            )

    return net, bus_id_map


def build_plot_inputs(case: PowerFlowCase, result: OPFResult):
    """Build the exact plotting inputs used by the notebook-derived functions."""
    global p_bus, q_bus, p_gen_bus, q_gen_bus, base_mva, load_shedding_enabled

    base_mva = float(case.base_mva)
    load_shedding_enabled = bool((result.convergence_info or {}).get("allow_load_shedding", False))
    p_bus, q_bus = _aggregate_case_loads(case)
    p_gen_bus, q_gen_bus = _aggregate_case_generation(case, result)

    net, bus_id_map = build_pandapower_net_from_case(case)

    dlmp = {
        int(bus): {
            "lambda_p": float(result.duals_p.get(int(bus), 0.0)),
            "lambda_q": float(result.duals_q.get(int(bus), 0.0)),
        }
        for bus in [b.bus_id for b in case.buses]
    }

    bus_voltages = {}
    for bus in [b.bus_id for b in case.buses]:
        v_sq = float(result.voltages.get(int(bus), 0.0))
        v_mag = float(np.sqrt(v_sq)) if np.isfinite(v_sq) and v_sq >= 0.0 else float("nan")
        bus_voltages[int(bus)] = {
            "v_sq": v_sq,
            "v": v_mag,
        }

    trans_rows = []
    for branch in case.branches:
        edge = (int(branch.from_bus), int(branch.to_bus))
        if edge in result.flows:
            p_flow, q_flow = result.flows[edge]
        elif (edge[1], edge[0]) in result.flows:
            p_flow, q_flow = result.flows[(edge[1], edge[0])]
        else:
            p_flow, q_flow = 0.0, 0.0

        trans_rows.append(
            {
                "fbus": int(branch.from_bus),
                "tbus": int(branch.to_bus),
                "r": float(branch.r),
                "x": float(branch.x),
                "lmax": float(branch.lmax),
                "ptrans": float(p_flow),
                "qtrans": float(q_flow),
                "strans": float((p_flow**2 + q_flow**2) ** 0.5),
            }
        )

    trans_df = pd.DataFrame(trans_rows)
    return net, bus_id_map, trans_df, dlmp, bus_voltages


def verify_load_balance(case: PowerFlowCase, result: OPFResult) -> None:
    """
    Verify and print load balance information.
    
    Displays:
    - Total load (P and Q)
    - Total generation by bus and generator
    - Load shedding/curtailment if enabled
    """
    solver = getattr(result, "solver_name", None) or "Unknown"
    print("\n" + "="*70)
    print(f"{solver} - LOAD BALANCE VERIFICATION")
    print("="*70)
    base_mva = float(case.base_mva)
    
    # Total loads
    total_p_load = sum(ld.p_d for ld in case.loads) * base_mva
    total_q_load = sum(ld.q_d for ld in case.loads) * base_mva
    print("\nTotal Load:")
    print(f"  Active Power (P):   {total_p_load:12.6f} MW")
    print(f"  Reactive Power (Q): {total_q_load:12.6f} MVAr")
    
    # Total generation by generator
    print("\nGeneration by Bus/Generator:")
    gen_by_bus = {}
    for gen_id, (pg, qg) in (result.generator_output or {}).items():
        gen = next((g for g in case.generators if g.gen_id == gen_id), None)
        if gen:
            bus_id = gen.bus_id
            if bus_id not in gen_by_bus:
                gen_by_bus[bus_id] = []
            gen_by_bus[bus_id].append((gen_id, pg, qg))
    
    total_p_gen = 0.0
    total_q_gen = 0.0
    for bus_id in sorted(gen_by_bus.keys()):
        print(f"  Bus {bus_id}:")
        for gen_id, pg, qg in gen_by_bus[bus_id]:
            print(f"    Gen {gen_id}: P={pg * base_mva:10.6f} MW, Q={qg * base_mva:10.6f} MVAr")
            total_p_gen += pg * base_mva
            total_q_gen += qg * base_mva
    
    print("\nTotal Generation:")
    print(f"  Active Power (P):   {total_p_gen:12.6f} MW")
    print(f"  Reactive Power (Q): {total_q_gen:12.6f} MVAr")
    
    # Load shedding/curtailment
    allow_shedding = bool((result.convergence_info or {}).get("allow_load_shedding", False))
    total_p_curt = float((result.convergence_info or {}).get("total_p_curtailment", 0.0)) * base_mva
    total_q_curt = float((result.convergence_info or {}).get("total_q_curtailment", 0.0)) * base_mva
    
    if allow_shedding:
        print("\nLoad Shedding:")
        print("  Enabled: True")
        print(f"  Active Power Shed:   {total_p_curt:12.6f} MW")
        print(f"  Reactive Power Shed: {total_q_curt:12.6f} MVAr")
    else:
        print("\nLoad Shedding: Disabled")
    
    # Balance check
    balance_p = total_p_gen - total_p_load + total_p_curt
    balance_q = total_q_gen - total_q_load + total_q_curt
    print("\nBalance (Gen - Load + Shed):")
    print(f"  Active Power:   {balance_p:12.6f} MW (should be ~0)")
    print(f"  Reactive Power: {balance_q:12.6f} MVAr (should be ~0)")
    print("="*70 + "\n")


def normalize_and_color(values, colorscale_name):
    """Normalize values and convert to hex colors."""
    vmin = min(values)
    vmax = max(values)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = _get_mpl_cmap(colorscale_name)
    colors = [mcolors.to_hex(cmap(norm(v))) for v in values]
    return colors, vmin, vmax


def find_bus_trace(fig, line_count):
    """
    Robustly find the bus trace index by looking for the scatter trace
    with the most points (buses), skipping line traces.
    """
    for i, trace in enumerate(fig.data):
        if i < line_count:
            continue
        if hasattr(trace, "mode") and trace.mode is not None and "markers" in trace.mode:
            if trace.x is not None and len(trace.x) > 1:
                return i
    return line_count


def _apply_external_grid_coloring(fig, theme):
    """Override pandapower's external-grid marker color when the active theme requests it."""

    color = theme.get("external_grid_color")
    if not color:
        return

    for trace in fig.data:
        name = str(getattr(trace, "name", "") or "").lower().replace("-", "_")
        if not any(token in name for token in ("external_grid", "ext_grid", "external grid", "ext grid")):
            continue
        if hasattr(trace, "marker") and trace.marker is not None:
            trace.marker.color = color
            if getattr(trace.marker, "line", None) is not None:
                trace.marker.line.color = color


def _capitalize_network_legend_names(fig):
    """Normalize pandapower's default network legend labels."""

    legend_names = {
        "lines": "Lines",
        "external grid": "External grid",
        "external_grid": "External grid",
        "ext grid": "External grid",
        "ext_grid": "External grid",
        "buses": "Buses",
    }
    for trace in fig.data:
        name = str(getattr(trace, "name", "") or "")
        normalized = name.strip().lower().replace("-", " ")
        trace.name = legend_names.get(normalized, name)




def _line_current_ka(
    s_pu: float,
    line,
    net,
    reverse_bus_map: dict[int, int],
    bus_voltages,
    system_base_mva: float,
) -> float:
    """Estimate the highest endpoint current from apparent power and bus voltage."""

    if not np.isfinite(s_pu):
        return float("nan")

    currents_ka: list[float] = []
    s_mva = abs(float(s_pu)) * float(system_base_mva)
    for pp_bus in (int(line.from_bus), int(line.to_bus)):
        try:
            vn_kv = float(net.bus.at[pp_bus, "vn_kv"])
        except (KeyError, TypeError, ValueError):
            vn_kv = float("nan")

        orig_bus = reverse_bus_map.get(pp_bus)
        v_pu = 1.0
        if isinstance(bus_voltages, dict) and orig_bus in bus_voltages:
            try:
                candidate_v_pu = float(bus_voltages[orig_bus].get("v", float("nan")))
            except (AttributeError, TypeError, ValueError):
                candidate_v_pu = float("nan")
            if np.isfinite(candidate_v_pu) and candidate_v_pu > 0.0:
                v_pu = candidate_v_pu

        voltage_kv = vn_kv * v_pu
        if np.isfinite(voltage_kv) and voltage_kv > 0.0:
            currents_ka.append(s_mva / (np.sqrt(3.0) * voltage_kv))

    return max(currents_ka) if currents_ka else float("nan")


def _line_loading_hover_value(
    s_pu: float,
    lmax: float,
    line,
    net,
    reverse_bus_map: dict[int, int],
    bus_voltages,
    system_base_mva: float,
    line_color_mode: str,
) -> tuple[str, float]:
    if not np.isfinite(s_pu):
        label = "loading_current" if line_color_mode == "loading_pct" else "loading_base_current"
        return label, float("nan")

    if line_color_mode == "loading_pct":
        from_bus = reverse_bus_map.get(int(line.from_bus))
        current_loading = _current_loading_from_lmax(s_pu, lmax, int(from_bus), bus_voltages) if from_bus is not None else float("nan")
        if np.isfinite(current_loading):
            return "loading_current", 100.0 * current_loading
        return "loading_current", float("nan")

    current_ka = _line_current_ka(s_pu, line, net, reverse_bus_map, bus_voltages, system_base_mva)
    current_base_ka = _line_current_ka(1.0, line, net, reverse_bus_map, None, system_base_mva)
    if np.isfinite(current_ka) and np.isfinite(current_base_ka) and current_base_ka > 0.0:
        return "loading_base_current", 100.0 * current_ka / current_base_ka
    return "loading_base_current", float("nan")


def _current_loading_from_lmax(
    s_pu: float,
    lmax: float,
    from_bus: int,
    bus_voltages,
) -> float:
    """Return I/Imax using ell/lmax from the SOCP current-limit constraint."""

    try:
        lmax_value = float(lmax)
    except (TypeError, ValueError):
        return float("nan")

    if not np.isfinite(s_pu) or not np.isfinite(lmax_value) or lmax_value <= 0.0:
        return float("nan")

    v_sq = 1.0
    if isinstance(bus_voltages, dict) and int(from_bus) in bus_voltages:
        try:
            candidate_v_sq = float(bus_voltages[int(from_bus)].get("v_sq", float("nan")))
        except (AttributeError, TypeError, ValueError):
            candidate_v_sq = float("nan")
        if np.isfinite(candidate_v_sq) and candidate_v_sq > 0.0:
            v_sq = candidate_v_sq

    ell = float(s_pu) ** 2 / v_sq
    return np.sqrt(ell / lmax_value)


def _line_current_pu(s_pu: float, from_bus: int, bus_voltages) -> float:
    """Return per-unit current magnitude sqrt(ell) for a branch sending end."""

    if not np.isfinite(s_pu):
        return float("nan")

    v_sq = 1.0
    if isinstance(bus_voltages, dict) and int(from_bus) in bus_voltages:
        try:
            candidate_v_sq = float(bus_voltages[int(from_bus)].get("v_sq", float("nan")))
        except (AttributeError, TypeError, ValueError):
            candidate_v_sq = float("nan")
        if np.isfinite(candidate_v_sq) and candidate_v_sq > 0.0:
            v_sq = candidate_v_sq

    return abs(float(s_pu)) / np.sqrt(v_sq)


def _apply_line_coloring(
    fig,
    net,
    trans_df,
    values_by_edge,
    reverse_bus_map,
    colorscale_name,
    theme,
    line_color_mode: str = "flow_pu",
    system_base_mva: float = 1.0,
    bus_voltages=None,
):
    """
    Apply line colors and hover text based on values_by_edge.
    
    Args:
        fig: plotly figure
        net: pandapower network
        trans_df: branch data
        values_by_edge: dict mapping (from_bus, to_bus) -> value
        reverse_bus_map: mapping from pandapower bus idx to original bus id
        colorscale_name: matplotlib colorscale name
        theme: theme settings dict
    
    Returns:
        (line_colors, vmin, vmax) - for colorbar creation
    """
    line_count = len(net.line)
    line_values = []
    lmax_by_edge = {}
    if "lmax" in trans_df.columns:
        for _, row in trans_df.iterrows():
            edge = (int(row["fbus"]), int(row["tbus"]))
            lmax_by_edge[edge] = float(row.get("lmax", float("nan")))
    
    for _, line in net.line.iterrows():
        pp_fbus = int(line.from_bus)
        pp_tbus = int(line.to_bus)
        bus_a = reverse_bus_map.get(pp_fbus)
        bus_b = reverse_bus_map.get(pp_tbus)
        
        if bus_a is None or bus_b is None:
            line_values.append(0)
            continue
        
        parent = int(min(bus_a, bus_b))
        child = int(max(bus_a, bus_b))
        edge_pc = (parent, child)
        edge_cp = (child, parent)
        
        val = 0
        lmax = float("nan")
        from_bus_for_limit = None
        if edge_pc in values_by_edge:
            val = abs(float(values_by_edge[edge_pc]))
            lmax = lmax_by_edge.get(edge_pc, float("nan"))
            from_bus_for_limit = edge_pc[0]
        elif edge_cp in values_by_edge:
            val = abs(float(values_by_edge[edge_cp]))
            lmax = lmax_by_edge.get(edge_cp, float("nan"))
            from_bus_for_limit = edge_cp[0]
        
        if line_color_mode == "loading_pct":
            current_loading = _current_loading_from_lmax(
                val,
                lmax,
                int(from_bus_for_limit) if from_bus_for_limit is not None else int(parent),
                bus_voltages,
            )
            if np.isfinite(current_loading):
                val = 100.0 * current_loading
            else:
                val = 0.0
        else:
            current_ka = _line_current_ka(
                val,
                line,
                net,
                reverse_bus_map,
                bus_voltages,
                system_base_mva,
            )
            current_base_ka = _line_current_ka(
                1.0,
                line,
                net,
                reverse_bus_map,
                None,
                system_base_mva,
            )
            if np.isfinite(current_ka) and np.isfinite(current_base_ka) and current_base_ka > 0.0:
                val = 100.0 * current_ka / current_base_ka
            else:
                val = 0.0

        line_values.append(val)
    
    # Normalize and color
    if line_values and max(line_values) > 0:
        line_colors, line_vmin, line_vmax = normalize_and_color(line_values, colorscale_name)
    else:
        line_colors = [theme["line_fallback_color"] for _ in line_values]
        line_vmin, line_vmax = 0, 1
    
    if theme.get("monochrome", False):
        line_colors = ["white" for _ in line_colors]
    
    # Apply colors to traces
    for i, line in net.line.iterrows():
        if i >= line_count or i >= len(fig.data):
            continue
        trace = fig.data[i]
        trace.line.color = line_colors[i]
    
    return line_colors, line_vmin, line_vmax, line_values


def _apply_bus_coloring(fig, net, bus_id_map, pp_to_orig, values_by_bus, colorscale_name, theme):
    """
    Apply bus colors and hover text based on values_by_bus.
    
    Args:
        fig: plotly figure
        net: pandapower network
        bus_id_map: mapping original bus id -> pandapower index
        pp_to_orig: mapping pandapower index -> original bus id
        values_by_bus: dict mapping original_bus_id -> value
        colorscale_name: matplotlib colorscale name
        theme: theme settings dict
    
    Returns:
        (color_values, vmin, vmax, uniform) - for colorbar and marker setup
    """
    line_count = len(net.line)
    color_values = []
    
    for pp_bus in net.bus.index:
        orig_bus = pp_to_orig.get(pp_bus)
        if orig_bus is not None and orig_bus in values_by_bus:
            color_values.append(abs(float(values_by_bus[orig_bus])))
        else:
            color_values.append(0)
    
    color_values = np.array(color_values, dtype=float)
    vmin = float(color_values.min())
    vmax = float(color_values.max())
    uniform = (vmax - vmin) < float(CONFIG.get("lmp_uniform_tol", 0.01))
    
    # Find and update bus trace
    bus_trace_idx = find_bus_trace(fig, line_count)
    
    if bus_trace_idx < len(fig.data):
        bt = fig.data[bus_trace_idx]
        
        if theme.get("monochrome", False):
            bt.marker.color = ["white" for _ in net.bus.index]
            bt.marker.showscale = False
        elif uniform:
            cmap = _get_mpl_cmap(colorscale_name)
            uniform_color = mcolors.to_hex(cmap(0.5))
            bt.marker.color = [uniform_color for _ in net.bus.index]
            bt.marker.showscale = False
        else:
            bt.marker.color = color_values
            bt.marker.colorscale = _get_plotly_colorscale(colorscale_name)
            bt.marker.cmin = vmin
            bt.marker.cmax = vmax
            bt.marker.showscale = False
        
        bt.marker.size = CONFIG["bus_size"]
    
    return color_values, vmin, vmax, uniform


def _add_dual_colorbars(fig, line_vmin, line_vmax, bus_vmin, bus_vmax, 
                        line_colorscale, bus_colorscale, line_title, bus_title, theme):
    """
    Add two colorbars to figure (one for lines, one for buses).
    Positioned horizontally below the plot.
    
    Args:
        fig: plotly figure
        line_vmin, line_vmax: line value range
        bus_vmin, bus_vmax: bus value range
        line_colorscale: colorscale for lines
        bus_colorscale: colorscale for buses
        line_title: colorbar title for lines
        bus_title: colorbar title for buses
        theme: theme settings dict
    """
    if theme.get("monochrome", False):
        return
    
    # Keep both horizontal bars on one row, side by side below the plot.
    colorbar_y = -0.20
    colorbar_len = 0.42
    colorbar_thickness = 22

    def _display_range(vmin, vmax):
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            return 0.0, 1.0
        if abs(float(vmax) - float(vmin)) >= 1e-12:
            return float(vmin), float(vmax)
        pad = max(abs(float(vmin)), 1.0) * 0.005
        return float(vmin) - pad, float(vmax) + pad

    def _add_horizontal_colorbar_label_above(text, colorbar_x):
        fig.add_annotation(
            text=text,
            x=colorbar_x,
            y=colorbar_y + 0.05,
            xref="paper",
            yref="paper",
            xanchor="center",
            yanchor="top",
            showarrow=False,
            font=dict(color=theme["font_color"], size=16),
        )

    line_cmin, line_cmax = _display_range(line_vmin, line_vmax)
    bus_cmin, bus_cmax = _display_range(bus_vmin, bus_vmax)

    # Line flow colorbar (left side)
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode="markers",
        marker=dict(
            colorscale=_get_plotly_colorscale(line_colorscale),
            cmin=line_cmin, cmax=line_cmax,
            showscale=True,
            colorbar=dict(
                title="",
                orientation="h",
                thickness=colorbar_thickness,
                len=colorbar_len,
                x=0.24,
                y=colorbar_y,
                xanchor="center",
                yanchor="top",
            ),
        ),
        hoverinfo="none",
        name="Line Values",
        showlegend=False,
    ))
    _add_horizontal_colorbar_label_above(line_title, 0.24)
    
    # Bus colorbar (right side)
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode="markers",
        marker=dict(
            colorscale=_get_plotly_colorscale(bus_colorscale),
            cmin=bus_cmin, cmax=bus_cmax,
            showscale=True,
            colorbar=dict(
                title="",
                orientation="h",
                thickness=colorbar_thickness,
                len=colorbar_len,
                x=0.76,
                y=colorbar_y,
                xanchor="center",
                yanchor="top",
            ),
        ),
        hoverinfo="none",
        name="Bus Values",
        showlegend=False,
    ))
    _add_horizontal_colorbar_label_above(bus_title, 0.76)


def create_combined_plot(
    net,
    bus_id_map,
    trans_df,
    dlmp,
    bus_voltages=None,
    title=None,
    lmp_type="lambda_p",
    label_maps=None,
    line_color_mode=None,
):
    """
    Interactive plot using plotly to display network with flow and LMP information.
    
    - Lines colored by power flow
    - Buses colored by LMP (lambda_p or lambda_q)
    - Two colorbars (one per dimension)
    - Mid-line arrows showing signed flow direction
    """
    if label_maps is None:
        label_maps = {}
    if line_color_mode is None:
        line_color_mode = get_line_color_mode()
    system_base_mva = float(globals().get("base_mva", 1.0))
    
    fig = simple_plotly(
        net,
        bus_size=CONFIG["bus_size"],
        line_width=CONFIG["line_width"],
        auto_open=False,
    )
    _capitalize_network_legend_names(fig)

    pp_to_orig = {pp_idx: orig for orig, pp_idx in bus_id_map.items()}
    reverse_bus_map = {v: k for k, v in bus_id_map.items()}
    theme = _theme_settings()
    _apply_external_grid_coloring(fig, theme)
    line_count = len(net.line)
    hide_zero_fields = get_hide_zero_hover_fields()
    flow_component = "qtrans" if lmp_type == "lambda_q" else "ptrans"
    
    # Build flow dict from trans_df for coloring
    flows_by_edge = {}
    for _, row in trans_df.iterrows():
        edge = (int(row["fbus"]), int(row["tbus"]))
        s_val = float(row.get("strans", 0.0))
        flows_by_edge[edge] = s_val
    
    # Apply line coloring using helper
    _, line_vmin, line_vmax, line_values = _apply_line_coloring(
        fig,
        net,
        trans_df,
        flows_by_edge,
        reverse_bus_map,
        theme["line_colorscale"],
        theme,
        line_color_mode=line_color_mode,
        system_base_mva=system_base_mva,
        bus_voltages=bus_voltages,
    )
    
    # Add flow hover text to lines
    for i, line in net.line.iterrows():
        if i >= line_count or i >= len(fig.data):
            continue
        
        trace = fig.data[i]
        pp_fbus, pp_tbus = int(line.from_bus), int(line.to_bus)
        bus_a, bus_b = reverse_bus_map.get(pp_fbus), reverse_bus_map.get(pp_tbus)
        
        p_int = q_int = s_int = i_pu = r_val = x_val = lmax = np.nan
        from_bus_for_current = None
        
        if bus_a is not None and bus_b is not None:
            parent, child = int(min(bus_a, bus_b)), int(max(bus_a, bus_b))
            match_pc = trans_df[(trans_df["fbus"] == parent) & (trans_df["tbus"] == child)]
            match_cp = trans_df[(trans_df["fbus"] == child) & (trans_df["tbus"] == parent)]
            
            if not match_pc.empty:
                row = match_pc.iloc[0]
                p_int, q_int = float(row.get("ptrans", np.nan)), float(row.get("qtrans", np.nan))
                r_val, x_val = float(row.get("r", np.nan)), float(row.get("x", np.nan))
                lmax = float(row.get("lmax", np.nan))
                from_bus_for_current = int(row.get("fbus", parent))
            elif not match_cp.empty:
                row = match_cp.iloc[0]
                p_int, q_int = -float(row.get("ptrans", np.nan)), -float(row.get("qtrans", np.nan))
                r_val, x_val = float(row.get("r", np.nan)), float(row.get("x", np.nan))
                lmax = float(row.get("lmax", np.nan))
                from_bus_for_current = int(row.get("fbus", child))
        
        if np.isfinite(p_int) and np.isfinite(q_int):
            s_int = float(np.sqrt(p_int**2 + q_int**2))
        if from_bus_for_current is not None:
            i_pu = _line_current_pu(s_int, from_bus_for_current, bus_voltages)
        loading_label, loading_pct = _line_loading_hover_value(
            s_int,
            lmax,
            line,
            net,
            reverse_bus_map,
            bus_voltages,
            system_base_mva,
            line_color_mode,
        )

        line_hover_lines = [f"Line {bus_a} -> {bus_b}"]
        _append_hover_value(line_hover_lines, "p_int", p_int, ".3f", "p.u.", hide_zero_fields)
        _append_hover_value(line_hover_lines, "q_int", q_int, ".3f", "p.u.", hide_zero_fields)
        _append_hover_value(line_hover_lines, "s_int", s_int, ".3f", "p.u.", hide_zero_fields)
        _append_hover_value(line_hover_lines, "i_int", i_pu, ".3f", "p.u.", hide_zero_fields)
        _append_hover_value(line_hover_lines, loading_label, loading_pct, ".2f", "%", hide_zero_fields)
        _append_hover_value(line_hover_lines, "r", r_val, ".3f", "", hide_zero_fields)
        _append_hover_value(line_hover_lines, "x", x_val, ".3f", "", hide_zero_fields)

        trace.text = "<br>".join(line_hover_lines)
        trace.hovertext = trace.text
        trace.hoverinfo = "text"
    
    # Add flow arrows
    if CONFIG["show_arrows"]:
        arrow_annotations = []
        arrow_half_fraction = 0.18
        arrow_min_abs_flow = float(CONFIG.get("arrow_min_abs_flow_pu", 1e-3))
        
        for i, line in net.line.iterrows():
            if i >= line_count or i >= len(fig.data):
                continue
            
            pp_fbus, pp_tbus = int(line.from_bus), int(line.to_bus)
            bus_a, bus_b = reverse_bus_map.get(pp_fbus), reverse_bus_map.get(pp_tbus)
            if bus_a is None or bus_b is None:
                continue
            
            parent, child = int(min(bus_a, bus_b)), int(max(bus_a, bus_b))
            match_pc = trans_df[(trans_df["fbus"] == parent) & (trans_df["tbus"] == child)]
            match_cp = trans_df[(trans_df["fbus"] == child) & (trans_df["tbus"] == parent)]
            
            signed_flow = 0
            if not match_pc.empty:
                signed_flow = float(match_pc.iloc[0][flow_component])
            elif not match_cp.empty:
                signed_flow = -float(match_cp.iloc[0][flow_component])
            else:
                continue

            # Do not draw arrows when flow is below the configured visibility threshold.
            if not np.isfinite(signed_flow) or abs(float(signed_flow)) < arrow_min_abs_flow:
                continue
            
            trace = fig.data[i]
            xs = [x for x in (trace.x or []) if x is not None]
            ys = [y for y in (trace.y or []) if y is not None]
            if len(xs) < 2 or len(ys) < 2:
                continue
            
            x0, y0, x1, y1 = float(xs[0]), float(ys[0]), float(xs[-1]), float(ys[-1])
            start_bus, end_bus = (parent, child) if signed_flow >= 0 else (child, parent)
            
            if int(bus_a) == start_bus and int(bus_b) == end_bus:
                sx, sy, ex, ey = x0, y0, x1, y1
            elif int(bus_a) == end_bus and int(bus_b) == start_bus:
                sx, sy, ex, ey = x1, y1, x0, y0
            else:
                continue
            
            mx, my = (sx + ex) / 2.0, (sy + ey) / 2.0
            dx, dy = (ex - sx), (ey - sy)
            ax, ay = mx - arrow_half_fraction * dx, my - arrow_half_fraction * dy
            xh, yh = mx + arrow_half_fraction * dx, my + arrow_half_fraction * dy
            
            line_color = theme["line_fallback_color"]
            if hasattr(trace, "line") and trace.line is not None and getattr(trace.line, "color", None):
                line_color = trace.line.color
            
            arrow_annotations.append(dict(
                x=xh, y=yh, ax=ax, ay=ay, xref="x", yref="y", axref="x", ayref="y",
                showarrow=True, arrowhead=3, arrowsize=1.0, arrowwidth=1.8,
                arrowcolor=line_color, opacity=0.95,
            ))
    else:
        arrow_annotations = []
    
    # Prepare selected bus values for coloring.
    if lmp_type == "voltage":
        lmp_values_by_bus = {
            int(orig): float(values.get("v", float("nan")))
            for orig, values in (bus_voltages or {}).items()
        }
    else:
        lmp_values_by_bus = {
            orig: float(dlmp[orig][lmp_type])
            for orig in dlmp
        }
    
    # Apply bus coloring using helper
    color_values, bus_vmin, bus_vmax, uniform_bus = _apply_bus_coloring(
        fig, net, bus_id_map, pp_to_orig, lmp_values_by_bus, 
        theme["bus_colorscale"], theme
    )
    
    # Prepare loads and generation for hover text
    if "p_bus" in globals() and "q_bus" in globals() and "base_mva" in globals():
        p_load_mw_by_pp = {
            pp_idx: float((pd.to_numeric(p_bus, errors="coerce").fillna(0.0) * float(base_mva)).get(orig, 0.0))
            for orig, pp_idx in bus_id_map.items()
        }
        q_load_mvar_by_pp = {
            pp_idx: float((pd.to_numeric(q_bus, errors="coerce").fillna(0.0) * float(base_mva)).get(orig, 0.0))
            for orig, pp_idx in bus_id_map.items()
        }
    elif not net.load.empty:
        load_by_bus = net.load.groupby("bus")[["p_mw", "q_mvar"]].sum()
        p_load_mw_by_pp = load_by_bus["p_mw"].to_dict()
        q_load_mvar_by_pp = load_by_bus["q_mvar"].to_dict()
    else:
        p_load_mw_by_pp = {}
        q_load_mvar_by_pp = {}
    
    # Prepare generation
    p_gen_mw_by_pp, q_gen_mvar_by_pp = {}, {}
    if "p_gen_bus" in globals() and "q_gen_bus" in globals() and "base_mva" in globals():
        p_gen_mw_by_pp = {
            pp_idx: float((pd.to_numeric(p_gen_bus, errors="coerce").fillna(0.0) * float(base_mva)).get(orig, 0.0))
            for orig, pp_idx in bus_id_map.items()
        }
        q_gen_mvar_by_pp = {
            pp_idx: float((pd.to_numeric(q_gen_bus, errors="coerce").fillna(0.0) * float(base_mva)).get(orig, 0.0))
            for orig, pp_idx in bus_id_map.items()
        }
    else:
        if not net.gen.empty and "bus" in net.gen.columns and "p_mw" in net.gen.columns:
            for bus_idx, value in net.gen.groupby("bus")["p_mw"].sum().items():
                p_gen_mw_by_pp[int(bus_idx)] = float(value)
        if not net.sgen.empty and "bus" in net.sgen.columns and "p_mw" in net.sgen.columns:
            for bus_idx, value in net.sgen.groupby("bus")["p_mw"].sum().items():
                p_gen_mw_by_pp[int(bus_idx)] = p_gen_mw_by_pp.get(int(bus_idx), 0.0) + float(value)
        if not net.sgen.empty and "bus" in net.sgen.columns and "q_mvar" in net.sgen.columns:
            for bus_idx, value in net.sgen.groupby("bus")["q_mvar"].sum().items():
                q_gen_mvar_by_pp[int(bus_idx)] = q_gen_mvar_by_pp.get(int(bus_idx), 0.0) + float(value)
        if hasattr(net, "res_gen") and not net.res_gen.empty and "q_mvar" in net.res_gen.columns:
            for bus_idx, value in net.gen["bus"].to_frame().join(net.res_gen[["q_mvar"]]).groupby("bus")["q_mvar"].sum().items():
                q_gen_mvar_by_pp[int(bus_idx)] = q_gen_mvar_by_pp.get(int(bus_idx), 0.0) + float(value)
        if hasattr(net, "res_ext_grid") and not net.res_ext_grid.empty:
            if "p_mw" in net.res_ext_grid.columns:
                for bus_idx, value in net.ext_grid["bus"].to_frame().join(net.res_ext_grid[["p_mw"]]).groupby("bus")["p_mw"].sum().items():
                    p_gen_mw_by_pp[int(bus_idx)] = p_gen_mw_by_pp.get(int(bus_idx), 0.0) + float(value)
            if "q_mvar" in net.res_ext_grid.columns:
                for bus_idx, value in net.ext_grid["bus"].to_frame().join(net.res_ext_grid[["q_mvar"]]).groupby("bus")["q_mvar"].sum().items():
                    q_gen_mvar_by_pp[int(bus_idx)] = q_gen_mvar_by_pp.get(int(bus_idx), 0.0) + float(value)
    
    # Create and apply bus hover text
    pp_to_lmp = {
        pp_idx: dlmp[orig]
        for orig, pp_idx in bus_id_map.items()
        if orig in dlmp
    }

    if bool(globals().get("load_shedding_enabled", False)):
        pp_load_shedding = {
            pp_idx: max(
                float(p_load_mw_by_pp.get(pp_idx, 0.0)) - float(p_gen_mw_by_pp.get(pp_idx, 0.0)),
                0.0,
            )
            for pp_idx in net.bus.index
        }
        active_load_shedding = any(float(val) > 0.0 for val in pp_load_shedding.values())
    else:
        pp_load_shedding = {}
        active_load_shedding = False
    
    hover_texts = []
    for pp_bus in net.bus.index:
        orig_bus = pp_to_orig.get(pp_bus, pp_bus)
        lmp_p = float(pp_to_lmp.get(pp_bus, {}).get("lambda_p", 0.0))
        lmp_q = float(pp_to_lmp.get(pp_bus, {}).get("lambda_q", 0.0))
        if isinstance(bus_voltages, dict) and orig_bus in bus_voltages:
            v_sq = float(bus_voltages[orig_bus].get("v_sq", float("nan")))
        else:
            v_sq = float("nan")
        p_load = float(p_load_mw_by_pp.get(pp_bus, 0.0))
        q_load = float(q_load_mvar_by_pp.get(pp_bus, 0.0))
        p_gen = float(p_gen_mw_by_pp.get(pp_bus, 0.0))
        q_gen = float(q_gen_mvar_by_pp.get(pp_bus, 0.0))

        bus_hover_lines = [f"Bus {orig_bus}"]
        # _append_hover_value(bus_hover_lines, "V", v_mag, ".4f", "p.u.", hide_zero_fields)
        _append_hover_value(bus_hover_lines, "V^2", v_sq, ".4f", "p.u.^2", hide_zero_fields)
        _append_hover_value(bus_hover_lines, "LMP_p", lmp_p, ".3f", "", hide_zero_fields)
        _append_hover_value(bus_hover_lines, "LMP_q", lmp_q, ".3f", "", hide_zero_fields)
        _append_hover_value(bus_hover_lines, "p_load", p_load, ".3f", "MW", hide_zero_fields)
        _append_hover_value(bus_hover_lines, "q_load", q_load, ".3f", "MVAr", hide_zero_fields)
        _append_hover_value(bus_hover_lines, "p_gen", p_gen, ".3f", "MW", hide_zero_fields)
        _append_hover_value(bus_hover_lines, "q_gen", q_gen, ".3f", "MVAr", hide_zero_fields)
        if active_load_shedding:
            _append_hover_value(bus_hover_lines, "load_shedding", pp_load_shedding.get(pp_bus, 0.0), ".3f", "MW", hide_zero_fields)
        hover_texts.append("<br>".join(bus_hover_lines))
    
    bus_trace_idx = find_bus_trace(fig, line_count)
    if bus_trace_idx < len(fig.data):
        bt = fig.data[bus_trace_idx]
        bt.text = hover_texts
        bt.hovertext = hover_texts
        bt.hoverinfo = "text"
    
    # Add colorbars
    if line_color_mode == "loading_pct":
        titleFlow = label_maps.get("flow", LINE_COLORBAR_LABELS["loading_pct"])
    else:
        titleFlow = label_maps.get("flow", LINE_COLORBAR_LABELS["flow_pu"])
    titleLMP = label_maps.get("lmp", LMP_COLORBAR_LABELS)
    
    _add_dual_colorbars(fig, line_vmin, line_vmax, bus_vmin, bus_vmax,
                       theme["line_colorscale"], theme["bus_colorscale"],
                       titleFlow, titleLMP.get(lmp_type, lmp_type), theme)
    
    colorbar_label_annotations = list(fig.layout.annotations or ())

    # Final layout
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center") if title else None,
        width=1200,
        height=1200 * 8 / 16,
        margin=dict(l=20, r=80, t=60, b=200),
        hovermode="closest",
        annotations=colorbar_label_annotations + arrow_annotations,
        legend=dict(x=0, y=0.0, xanchor="left", yanchor="bottom", orientation="v",
               bgcolor="rgba(0,0,0,0.35)", bordercolor="rgba(255,255,255,0.25)",
               borderwidth=1, font=dict(size=11)),
        showlegend=True,
        paper_bgcolor=theme["paper_bgcolor"],
        plot_bgcolor=theme["plot_bgcolor"],
        font=dict(color=theme["font_color"]),
    )
    
    return fig


def create_white_network_plot(net, bus_id_map, trans_df, dlmp):
    """Render the white mode network plot (black background, white lines/nodes)."""
    previous_theme = get_visual_theme()
    set_visual_theme("white")
    fig = create_combined_plot(
        net, bus_id_map, trans_df, dlmp, None,
        title="Network - White View",
        lmp_type="lambda_p",
    )
    set_visual_theme(previous_theme)
    return fig


def create_deviation_plot(net, bus_id_map, trans_df, case, lmp_p_dev, flow_dev, 
                          title="Deviation Plot (BFSA vs SOCP)", label_maps=None, dev_is_percent: bool = True):
    """
    Interactive plot showing BFSA-SOCP deviations on network topology.
    
    - Buses colored by the selected absolute bus deviation
    - Lines colored by the selected absolute branch-flow deviation
    - Hover text shows deviation magnitudes
    
    Args:
        net: pandapower network
        bus_id_map: mapping from original bus IDs to pandapower indices
        trans_df: branch data with flow information
        case: PowerFlowCase object
        lmp_p_dev: dict mapping bus_id -> selected bus deviation
        flow_dev: dict mapping edge tuple -> selected branch deviation
        title: plot title
        label_maps: optional custom colorbar labels
    
    Returns:
        plotly Figure object
    """
    if label_maps is None:
        label_maps = {}
    bus_metric = str(label_maps.get("bus_metric", "lmp_p")).strip().lower()
    branch_metric = str(label_maps.get("branch_metric", "flow_p")).strip().lower()
    bus_metric_labels = {
        "voltage": "Voltage",
        "lmp_p": "Active LMP",
        "lmp_q": "Reactive LMP",
    }
    branch_metric_labels = {
        "flow_p": "Active Flow",
        "flow_q": "Reactive Flow",
    }
    bus_label = bus_metric_labels.get(bus_metric, bus_metric)
    branch_label = branch_metric_labels.get(branch_metric, branch_metric)
    bus_colorbar_labels = {
        "voltage": "Bus |Delta V|",
        "lmp_p": "Bus |Delta lambda_p|",
        "lmp_q": "Bus |Delta lambda_q|",
    }
    branch_colorbar_labels = {
        "flow_p": "Branch |Delta P|",
        "flow_q": "Branch |Delta Q|",
    }
    
    fig = simple_plotly(
        net,
        bus_size=CONFIG["bus_size"],
        line_width=CONFIG["line_width"],
        auto_open=False,
    )
    _capitalize_network_legend_names(fig)
    
    pp_to_orig = {pp_idx: orig for orig, pp_idx in bus_id_map.items()}
    reverse_bus_map = {v: k for k, v in bus_id_map.items()}
    theme = _theme_settings()
    line_count = len(net.line)
    hide_zero_fields = get_hide_zero_hover_fields()
    
    # Apply line coloring directly by the selected absolute deviation.
    line_abs_values = []
    for _, line in net.line.iterrows():
        pp_fbus, pp_tbus = int(line.from_bus), int(line.to_bus)
        bus_a, bus_b = reverse_bus_map.get(pp_fbus), reverse_bus_map.get(pp_tbus)
        value = 0.0
        if bus_a is not None and bus_b is not None:
            parent, child = int(min(bus_a, bus_b)), int(max(bus_a, bus_b))
            edge_pc, edge_cp = (parent, child), (child, parent)
            if edge_pc in flow_dev:
                value = abs(float(flow_dev[edge_pc]))
            elif edge_cp in flow_dev:
                value = abs(float(flow_dev[edge_cp]))
        line_abs_values.append(value)

    if line_abs_values and max(line_abs_values) > 0:
        line_colors, line_vmin, line_vmax = normalize_and_color(line_abs_values, theme["line_colorscale"])
    else:
        line_colors = [theme["line_fallback_color"] for _ in line_abs_values]
        line_vmin, line_vmax = 0, 1
    if theme.get("monochrome", False):
        line_colors = ["white" for _ in line_colors]
    for color_idx, (_idx, _line) in enumerate(net.line.iterrows()):
        if color_idx < line_count and color_idx < len(fig.data) and color_idx < len(line_colors):
            fig.data[color_idx].line.color = line_colors[color_idx]
    
    # Add flow deviation hover text to lines
    for i, line in net.line.iterrows():
        if i >= line_count or i >= len(fig.data):
            continue
        
        trace = fig.data[i]
        pp_fbus, pp_tbus = int(line.from_bus), int(line.to_bus)
        bus_a, bus_b = reverse_bus_map.get(pp_fbus), reverse_bus_map.get(pp_tbus)
        
        if bus_a is not None and bus_b is not None:
            parent, child = int(min(bus_a, bus_b)), int(max(bus_a, bus_b))
            edge_pc, edge_cp = (parent, child), (child, parent)
            
            flow_dev_val = 0
            if edge_pc in flow_dev:
                flow_dev_val = float(flow_dev[edge_pc])
            elif edge_cp in flow_dev:
                flow_dev_val = float(flow_dev[edge_cp])
            
            line_hover_lines = [f"Line {bus_a} -> {bus_b}"]
            unit = "%" if dev_is_percent else ""
            _append_hover_value(line_hover_lines, f"|{branch_label} Dev|", abs(flow_dev_val), ".6f", unit, hide_zero_fields)
            _append_hover_value(line_hover_lines, f"{branch_label} Dev", flow_dev_val, "+.6f", unit, hide_zero_fields)
            trace.text = "<br>".join(line_hover_lines)
        else:
            trace.text = f"Line {pp_fbus} → {pp_tbus}"
        
        trace.hovertext = trace.text
        trace.hoverinfo = "text"
    
    # Apply bus coloring by the selected bus deviation
    color_values, bus_vmin, bus_vmax, uniform_bus = _apply_bus_coloring(
        fig, net, bus_id_map, pp_to_orig, lmp_p_dev,
        theme["bus_colorscale"], theme
    )
    
    # Create and apply bus hover text with deviations
    hover_texts = []
    for pp_bus in net.bus.index:
        orig_bus = pp_to_orig.get(pp_bus, pp_bus)
        lmp_dev = float(lmp_p_dev.get(pp_to_orig.get(pp_bus), 0.0))
        bus_hover_lines = [f"Bus {orig_bus}"]
        unit = "%" if dev_is_percent else ""
        _append_hover_value(bus_hover_lines, f"|{bus_label} Dev|", abs(lmp_dev), ".6f", unit, hide_zero_fields)
        _append_hover_value(bus_hover_lines, f"{bus_label} Dev", lmp_dev, "+.6f", unit, hide_zero_fields)
        hover_texts.append("<br>".join(bus_hover_lines))
    
    bus_trace_idx = find_bus_trace(fig, line_count)
    if bus_trace_idx < len(fig.data):
        bt = fig.data[bus_trace_idx]
        bt.text = hover_texts
        bt.hovertext = hover_texts
        bt.hoverinfo = "text"
    
    # Add colorbars
    bus_colorbar_base = bus_colorbar_labels.get(bus_metric, f"Bus |Delta {bus_label}|")
    branch_colorbar_base = branch_colorbar_labels.get(branch_metric, f"Branch |Delta {branch_label}|")
    if dev_is_percent:
        titleFlow = label_maps.get("flow", f"{branch_colorbar_base} (%)")
        titleLMP = label_maps.get("lmp", f"{bus_colorbar_base} (%)")
    else:
        branch_unit = " (p.u.)" if branch_metric in {"flow_p", "flow_q"} else ""
        titleFlow = label_maps.get("flow", f"{branch_colorbar_base}{branch_unit}")
        titleLMP = label_maps.get("lmp", bus_colorbar_base)
    
    _add_dual_colorbars(fig, line_vmin, line_vmax, bus_vmin, bus_vmax,
                       theme["line_colorscale"], theme["bus_colorscale"],
                       titleFlow, titleLMP, theme)
    
    # Final layout with MathJax support for LaTeX in colorbars
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center") if title else None,
        width=1200,
        height=1200 * 8 / 16,
        margin=dict(l=20, r=80, t=60, b=200),
        hovermode="closest",
        legend=dict(x=0, y=0.0, xanchor="left", yanchor="bottom", orientation="v",
               bgcolor="rgba(0,0,0,0.35)", bordercolor="rgba(255,255,255,0.25)",
               borderwidth=1, font=dict(size=11)),
        showlegend=True,
        paper_bgcolor=theme["paper_bgcolor"],
        plot_bgcolor=theme["plot_bgcolor"],
        font=dict(color=theme["font_color"]),
    )
    
    # Add MathJax script for LaTeX rendering in colorbar titles
    fig.add_annotation(
        text="",
        showarrow=False,
        visible=False,
        xref="paper", yref="paper",
        x=0, y=0,
    )
    
    return fig


def create_merit_order_plot(
    offers_sorted,
    total_load: float,
    marginal_price: float,
    component_label: str,
    title: str | None = None,
):
    """Create an interactive merit-order clearing curve plot."""
    fig = go.Figure()

    cumulative = 0.0
    for offer in offers_sorted or []:
        capacity = float(offer.get("capacity", 0.0))
        if capacity <= 0.0:
            continue

        cost = float(offer.get("cost", 0.0))
        name = str(offer.get("name", ""))
        is_excess = bool(offer.get("is_load_excess", False))
        color = "#f4a261" if is_excess else "#457b9d"
        hatch = "/" if is_excess else None

        marker_kwargs = {"color": color, "line": {"color": "black", "width": 1.0}}
        if hatch is not None:
            marker_kwargs["pattern"] = {"shape": hatch}

        fig.add_trace(
            go.Bar(
                x=[cumulative],
                y=[cost],
                width=[capacity],
                offset=0,
                base=0,
                marker=marker_kwargs,
                name=name,
                customdata=[[capacity, cost, is_excess]],
                hovertemplate=(
                    "%{fullData.name}<br>Capacity %{customdata[0]:.4f} p.u.<br>"
                    "Cost %{customdata[1]:.4f}<extra></extra>"
                ),
                showlegend=False,
            )
        )
        cumulative += capacity

    fig.add_vline(x=float(total_load), line_width=2, line_dash="dash", line_color="#d62828")
    fig.add_annotation(
        x=float(total_load),
        y=float(marginal_price),
        text="Clearing point",
        showarrow=True,
        arrowhead=2,
        arrowcolor="#2a9d8f",
        ax=30,
        ay=-30,
    )

    fig.add_scatter(
        x=[0.0, float(total_load)],
        y=[float(marginal_price), float(marginal_price)],
        mode="Lines",
        line={"color": "#2a9d8f", "dash": "dot", "width": 2},
        name="Clearing price",
        hoverinfo="skip",
        showlegend=True,
    )

    fig.add_scatter(
        x=[0.0, float(total_load)],
        y=[float(marginal_price), float(marginal_price)],
        mode="lines",
        line={"color": "#123292", "dash": "dot", "width": 2},
        name="Load position",
        hoverinfo="skip",
        showlegend=True,
    )

    fig.update_layout(
        title=title or f"BFSA Merit Order Clearing Curve ({component_label})",
        template="plotly_white",
        width=1100,
        height=480,
        xaxis_title=f"Cumulative offered {component_label} (p.u.)",
        yaxis_title="Marginal cost",
        hovermode="x unified",
        margin={"l": 70, "r": 30, "t": 80, "b": 60},
    )
    fig.update_xaxes(range=[0.0, max(cumulative, float(total_load)) * 1.05 if max(cumulative, float(total_load)) > 0 else 1.0])

    return fig


def export_network_plot(fig, output_path: Path) -> bool:
    if fig is None:
        return False

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write HTML with MathJax support for LaTeX rendering in colorbar titles
    export_config = {
        **PLOTLY_HIGH_RES_EXPORT_CONFIG,
        "responsive": True,
        "displaylogo": False,
    }

    fig.write_html(
        str(output_path),
        include_plotlyjs="cdn",
        include_mathjax="cdn",
        config=export_config,
    )
    return True


def export_side_by_side_dashboard(
    output_path: Path,
    left_solver: str = "bfsa",
    right_solver: str = "socp",
) -> Path:
    """
    Export a lightweight HTML dashboard for side-by-side solver comparison.

    The dashboard expects these files to exist in the same directory:
      - <solver>_white.html
      - <solver>_lambda_p.html
      - <solver>_lambda_q.html
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    left = str(left_solver).lower().strip()
    right = str(right_solver).lower().strip()

    html = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Network Comparison Dashboard</title>
  <style>
    :root {
      --bg: #0f172a;
      --panel: #111827;
      --panel-2: #1f2937;
      --text: #e5e7eb;
      --muted: #9ca3af;
      --accent: #22c55e;
      --border: #374151;
    }

    * { box-sizing: border-box; }

    html, body {
      width: 100%;
      margin: 0;
      overflow-x: hidden;
            overflow-y: auto;
    }

    body {
      font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
      background:
        radial-gradient(circle at 20% 10%, rgba(34, 197, 94, 0.16), transparent 38%),
        radial-gradient(circle at 85% 85%, rgba(56, 189, 248, 0.18), transparent 36%),
        var(--bg);
      color: var(--text);
      min-height: 100vh;
    }

    .container {
      width: 100%;
      max-width: none;
      margin: 0;
      padding: 16px;
    }

    .toolbar {
      display: flex;
      gap: 12px;
      align-items: center;
      flex-wrap: wrap;
      width: 100%;
      background: color-mix(in srgb, var(--panel) 92%, black 8%);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 12px 14px;
      margin-bottom: 14px;
      backdrop-filter: blur(2px);
    }

    .toolbar h1 {
      font-size: 18px;
      margin: 0;
      font-weight: 700;
      letter-spacing: 0.2px;
    }

    .toolbar .hint {
      color: var(--muted);
      font-size: 13px;
      margin-left: 4px;
      margin-right: auto;
    }

    .toolbar label {
      font-size: 14px;
      color: var(--muted);
    }

    select {
      border: 1px solid var(--border);
      background: var(--panel-2);
      color: var(--text);
      border-radius: 8px;
      padding: 8px 10px;
      font-size: 14px;
      outline: none;
    }

    .grid {
      display: flex;
      justify-content: space-between;
            align-items: flex-start;
      width: 100%;
      gap: 6%;
    }

    .pane {
      flex: 0 0 47%;
      max-width: 47%;
      min-width: 0;
      border: 1px solid var(--border);
      border-radius: 12px;
      background: color-mix(in srgb, var(--panel) 92%, black 8%);
            overflow: auto;
      min-height: 640px;
    }

    .pane-header {
      padding: 10px 12px;
      border-bottom: 1px solid var(--border);
      font-size: 13px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }

    iframe {
      width: 100%;
            height: 80vh;
            min-height: 680px;
      border: 0;
      background: #000;
            display: block;
    }

    @media (max-width: 1100px) {
      .grid {
        display: grid;
        grid-template-columns: 1fr;
        gap: 12px;
      }

      .pane {
        max-width: none;
        flex-basis: auto;
        min-height: 520px;
      }

      iframe {
                height: 72vh;
                min-height: 560px;
      }
    }
  </style>
</head>
<body>
  <div class="container">
    <div class="toolbar">
      <h1>Network Comparison</h1>
      <span class="hint">__LEFT__ vs __RIGHT__</span>
      <label for="mode">View</label>
      <select id="mode">
        <option value="white">White</option>
        <option value="lambda_p" selected>Active (lambda_p)</option>
        <option value="lambda_q">Reactive (lambda_q)</option>
      </select>
    </div>

    <div class="grid">
      <section class="pane">
        <div class="pane-header">__LEFT__</div>
        <iframe id="left-frame" loading="lazy" title="__LEFT__ network view"></iframe>
      </section>
      <section class="pane">
        <div class="pane-header">__RIGHT__</div>
        <iframe id="right-frame" loading="lazy" title="__RIGHT__ network view"></iframe>
      </section>
    </div>
  </div>

  <script>
    const modeSelect = document.getElementById("mode");
    const leftFrame = document.getElementById("left-frame");
    const rightFrame = document.getElementById("right-frame");

    function fileFor(solver, mode) {
      return `${solver}_${mode}.html`;
    }

    function updateFrames() {
      const mode = modeSelect.value;
      leftFrame.src = fileFor("__LEFT__", mode);
      rightFrame.src = fileFor("__RIGHT__", mode);
    }

    modeSelect.addEventListener("change", updateFrames);
    updateFrames();
  </script>
</body>
</html>
"""

    html = html.replace("__LEFT__", left).replace("__RIGHT__", right)
    output_path.write_text(html, encoding="utf-8")
    return output_path
