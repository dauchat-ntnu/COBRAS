"""Full radial KKT DLMP economic estimator utilities."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from libs.methods.physical.bfsa import BFSAElectricalSolver
from libs.shared import BranchData, PowerFlowCase, orient_radial_network


@dataclass(frozen=True)
class KKTSystem:
    """Dense matrix representation of the radial DLMP KKT system."""

    matrix: np.ndarray
    rhs: np.ndarray
    rows: list[str]
    columns: list[str]


@dataclass(frozen=True)
class KKTResult:
    """Solved KKT system plus extracted nodal prices."""

    system: KKTSystem
    solution: dict[str, float]
    dlmp_p: dict[int, float]
    dlmp_q: dict[int, float]
    rank: int
    residual_norm: float
    solve_mode: str


@dataclass(frozen=True)
class RadialDLMPKKTComputation:
    """End-to-end radial KKT computation result and timing metadata."""

    case: PowerFlowCase
    electrical_result: Any
    kkt_result: KKTResult
    lambda_p_seed: float
    lambda_q_seed: float
    seed_bus: int
    solve_electrical_duration: float
    build_duration: float
    solve_duration: float

    @property
    def system(self) -> KKTSystem:
        return self.kkt_result.system


def _branch_by_oriented_edge(case: PowerFlowCase) -> dict[tuple[int, int], BranchData]:
    branch_by_pair = {tuple(sorted((br.from_bus, br.to_bus))): br for br in case.branches if br.status == 1}
    return {
        (int(parent), int(child)): branch_by_pair[tuple(sorted((int(parent), int(child))))]
        for parent, child in case.tree_edges
    }


def _binding_sets(
    case: PowerFlowCase,
    v_sq: dict[int, float],
    line_state: dict[tuple[int, int], dict[str, float]],
    *,
    tolerance: float,
) -> tuple[set[int], set[int], set[tuple[int, int]]]:
    """Detect voltage and current constraints that are binding at the BFSA state."""

    bus_by_id = {bus.bus_id: bus for bus in case.buses}
    branch_by_edge = _branch_by_oriented_edge(case)
    v_lower: set[int] = set()
    v_upper: set[int] = set()
    ell_upper: set[tuple[int, int]] = set()

    for bus_id, bus in bus_by_id.items():
        if bus_id == case.root_bus:
            continue
        v = float(v_sq[bus_id])
        if abs(v - float(bus.v_min)) <= tolerance:
            v_lower.add(bus_id)
        if abs(v - float(bus.v_max)) <= tolerance:
            v_upper.add(bus_id)

    for edge, state in line_state.items():
        br = branch_by_edge[edge]
        lmax = float(getattr(br, "lmax", float("inf")))
        if math.isfinite(lmax) and lmax >= 0.0 and abs(float(state["ell"]) - lmax) <= tolerance:
            ell_upper.add(edge)

    return v_lower, v_upper, ell_upper


def _add_row(
    rows: list[str],
    coeffs: list[dict[str, float]],
    rhs: list[float],
    name: str,
    terms: dict[str, float],
    value: float = 0.0,
) -> None:
    rows.append(name)
    coeffs.append({k: v for k, v in terms.items() if abs(v) > 0.0})
    rhs.append(float(value))


def build_kkt_system(
    case: PowerFlowCase,
    electrical_result: Any,
    *,
    lambda_p_seed: float,
    lambda_q_seed: float,
    seed_bus: int | None = None,
    binding_tolerance: float = 1e-7,
    force_inactive_constraints: bool = False,
) -> KKTSystem:
    """
    Build A x = b for radial DLMP stationarity.

    Unknowns always include lambda_p, lambda_q at buses and lambda_v, mu_soc at
    branches. Thermal and bus-voltage duals are only included when the BFSA state
    is within binding_tolerance of the corresponding constraint, unless
    force_inactive_constraints=True, in which case all such duals are omitted.
    """

    if not case.tree_edges:
        case = orient_radial_network(case)

    v_sq = dict(electrical_result.voltages)
    line_state = electrical_result.convergence_info.get("line_state_bfsa")
    if line_state is None:
        raise KeyError("BFSA electrical_result is missing convergence_info['line_state_bfsa']")

    edges = [(int(i), int(j)) for i, j in case.tree_edges]
    buses = sorted(int(bus.bus_id) for bus in case.buses)
    seed_bus = int(seed_bus if seed_bus is not None else case.root_bus)
    branch_by_edge = _branch_by_oriented_edge(case)

    v_lower, v_upper, ell_upper = _binding_sets(
        case,
        v_sq,
        line_state,
        tolerance=binding_tolerance,
    )
    if force_inactive_constraints:
        v_lower, v_upper, ell_upper = set(), set(), set()

    columns: list[str] = []
    columns.extend(f"lambda_p[{bus}]" for bus in buses)
    columns.extend(f"lambda_q[{bus}]" for bus in buses)
    columns.extend(f"lambda_v[{i}->{j}]" for i, j in edges)
    columns.extend(f"mu_soc[{i}->{j}]" for i, j in edges)
    columns.extend(f"mu_ell_upper[{i}->{j}]" for i, j in sorted(ell_upper))
    columns.extend(f"mu_v_lower[{bus}]" for bus in sorted(v_lower))
    columns.extend(f"mu_v_upper[{bus}]" for bus in sorted(v_upper))

    rows: list[str] = []
    coeffs: list[dict[str, float]] = []
    rhs: list[float] = []

    children_by_bus: dict[int, list[int]] = {bus: [] for bus in buses}
    parent_by_bus: dict[int, int] = {}
    for parent, child in edges:
        children_by_bus[parent].append(child)
        parent_by_bus[child] = parent

    for i, j in edges:
        br = branch_by_edge[(i, j)]
        state = line_state[(i, j)]
        r = float(br.r)
        x = float(br.x)
        p_int = float(state["ptrans"])
        q_int = float(state["qtrans"])
        v_parent = float(v_sq[i])

        edge_label = f"{i}->{j}"
        _add_row(
            rows,
            coeffs,
            rhs,
            f"sta_p_int[{edge_label}]",
            {
                f"lambda_p[{i}]": 1.0,
                f"lambda_p[{j}]": -1.0,
                f"lambda_v[{edge_label}]": -2.0 * r,
                f"mu_soc[{edge_label}]": -2.0 * p_int,
            },
        )
        _add_row(
            rows,
            coeffs,
            rhs,
            f"sta_q_int[{edge_label}]",
            {
                f"lambda_q[{i}]": 1.0,
                f"lambda_q[{j}]": -1.0,
                f"lambda_v[{edge_label}]": -2.0 * x,
                f"mu_soc[{edge_label}]": -2.0 * q_int,
            },
        )
        ell_terms = {
            f"lambda_p[{j}]": r,
            f"lambda_q[{j}]": x,
            f"lambda_v[{edge_label}]": r**2 + x**2,
            f"mu_soc[{edge_label}]": v_parent,
        }
        if (i, j) in ell_upper:
            ell_terms[f"mu_ell_upper[{edge_label}]"] = -1.0
        _add_row(rows, coeffs, rhs, f"sta_ell[{edge_label}]", ell_terms)

    for bus in buses:
        if bus == case.root_bus:
            continue
        terms: dict[str, float] = {}
        parent = parent_by_bus.get(bus)
        if parent is not None:
            terms[f"lambda_v[{parent}->{bus}]"] = -1.0
        for child in children_by_bus.get(bus, []):
            state = line_state[(bus, child)]
            terms[f"lambda_v[{bus}->{child}]"] = terms.get(f"lambda_v[{bus}->{child}]", 0.0) + 1.0
            terms[f"mu_soc[{bus}->{child}]"] = terms.get(f"mu_soc[{bus}->{child}]", 0.0) + float(state["ell"])
        if bus in v_lower:
            terms[f"mu_v_lower[{bus}]"] = 1.0
        if bus in v_upper:
            terms[f"mu_v_upper[{bus}]"] = -1.0
        _add_row(rows, coeffs, rhs, f"sta_voltage_radial[{bus}]", terms)

    if children_by_bus.get(case.root_bus):
        root_child = children_by_bus[case.root_bus][0]
        _add_row(
            rows,
            coeffs,
            rhs,
            f"sta_voltage_radial[{case.root_bus}]",
            {f"lambda_v[{case.root_bus}->{root_child}]": 1.0},
        )

    _add_row(rows, coeffs, rhs, f"seed_lambda_p[{seed_bus}]", {f"lambda_p[{seed_bus}]": 1.0}, lambda_p_seed)
    _add_row(rows, coeffs, rhs, f"seed_lambda_q[{seed_bus}]", {f"lambda_q[{seed_bus}]": 1.0}, lambda_q_seed)

    col_index = {name: idx for idx, name in enumerate(columns)}
    matrix = np.zeros((len(rows), len(columns)), dtype=float)
    for row_idx, row_terms in enumerate(coeffs):
        for col_name, value in row_terms.items():
            matrix[row_idx, col_index[col_name]] = value

    return KKTSystem(matrix=matrix, rhs=np.array(rhs, dtype=float), rows=rows, columns=columns)


def solve_kkt_system(system: KKTSystem, case: PowerFlowCase) -> KKTResult:
    """Solve the explicit KKT system and extract bus DLMPs."""

    a = system.matrix
    b = system.rhs
    if a.shape[0] == a.shape[1]:
        try:
            x = np.linalg.solve(a, b)
            solve_mode = "solve"
            rank = a.shape[1]
        except np.linalg.LinAlgError:
            x, _, rank, _ = np.linalg.lstsq(a, b, rcond=None)
            solve_mode = "lstsq_singular_square"
    else:
        x, _, rank, _ = np.linalg.lstsq(a, b, rcond=None)
        solve_mode = "lstsq_rectangular"

    solution = {name: float(value) for name, value in zip(system.columns, x)}
    residual_norm = float(np.linalg.norm(a @ x - b, ord=2))

    dlmp_p = {int(bus.bus_id): solution[f"lambda_p[{int(bus.bus_id)}]"] for bus in case.buses}
    dlmp_q = {int(bus.bus_id): solution[f"lambda_q[{int(bus.bus_id)}]"] for bus in case.buses}

    return KKTResult(
        system=system,
        solution=solution,
        dlmp_p=dlmp_p,
        dlmp_q=dlmp_q,
        rank=int(rank),
        residual_norm=residual_norm,
        solve_mode=solve_mode,
    )


def seed_prices_from_bfsa(
    electrical_result: Any,
    case: PowerFlowCase,
    *,
    lambda_p_seed: float | None = None,
    lambda_q_seed: float | None = None,
    seed_bus: int | None = None,
) -> tuple[float, float, int]:
    """Resolve KKT seed prices from explicit overrides or BFSA merit-order metadata."""

    merit = electrical_result.convergence_info.get("merit_order", {})
    lambda_p = lambda_p_seed
    lambda_q = lambda_q_seed
    if lambda_p is None:
        lambda_p = float(merit.get("lambda_p_clear", 0.0))
    if lambda_q is None:
        lambda_q = float(merit.get("lambda_q_clear", 0.0))
    resolved_seed_bus = int(seed_bus if seed_bus is not None else merit.get("p_setting_bus", case.root_bus))
    return float(lambda_p), float(lambda_q), resolved_seed_bus


def compute_radial_dlmp_kkt(
    case: PowerFlowCase,
    *,
    electrical_result: Any | None = None,
    lambda_p_seed: float | None = None,
    lambda_q_seed: float | None = None,
    seed_bus: int | None = None,
    v_root_sq: float = 1.0,
    bfsa_tol: float = 1e-6,
    bfsa_max_iterations: int = 100,
    binding_tolerance: float = 1e-7,
    force_inactive_constraints: bool = False,
) -> RadialDLMPKKTComputation:
    """Run the full radial DLMP KKT workflow for an already loaded case."""

    case = orient_radial_network(case)
    solve_electrical_duration = 0.0
    if electrical_result is None:
        solver = BFSAElectricalSolver(
            config={
                "compute_losses": True,
                "tol": bfsa_tol,
                "max_iterations": bfsa_max_iterations,
                "v_root_sq": v_root_sq,
                "convergence_check": "both",
            }
        )
        solve_electrical_time = time.time()
        electrical_result = solver.solve_electrical(case, orient=False)
        solve_electrical_duration = time.time() - solve_electrical_time

    resolved_lambda_p, resolved_lambda_q, resolved_seed_bus = seed_prices_from_bfsa(
        electrical_result,
        case,
        lambda_p_seed=lambda_p_seed,
        lambda_q_seed=lambda_q_seed,
        seed_bus=seed_bus,
    )

    build_time = time.time()
    system = build_kkt_system(
        case,
        electrical_result,
        lambda_p_seed=resolved_lambda_p,
        lambda_q_seed=resolved_lambda_q,
        seed_bus=resolved_seed_bus,
        binding_tolerance=binding_tolerance,
        force_inactive_constraints=force_inactive_constraints,
    )
    build_duration = time.time() - build_time

    solve_time = time.time()
    kkt_result = solve_kkt_system(system, case)
    solve_duration = time.time() - solve_time

    return RadialDLMPKKTComputation(
        case=case,
        electrical_result=electrical_result,
        kkt_result=kkt_result,
        lambda_p_seed=resolved_lambda_p,
        lambda_q_seed=resolved_lambda_q,
        seed_bus=resolved_seed_bus,
        solve_electrical_duration=solve_electrical_duration,
        build_duration=build_duration,
        solve_duration=solve_duration,
    )
