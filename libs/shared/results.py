"""
Standardized framework result containers and compatibility adapters.

These classes separate physical state, economic state, and optimization output.
The legacy OPFResult remains supported through adapter helpers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class BusDuals:
    """Standard bus/node dual fields."""

    lambda_p: Optional[float] = None
    lambda_q: Optional[float] = None
    mu_v_up: Optional[float] = None
    mu_v_low: Optional[float] = None
    lambda_vref: Optional[float] = None


@dataclass
class BranchDuals:
    """Standard branch/line dual fields."""

    mu_int: Optional[float] = None
    mu_ell: Optional[float] = None


@dataclass
class GeneratorDuals:
    """Standard generator dual fields."""

    mu_pmax: Optional[float] = None
    mu_pmin: Optional[float] = None
    mu_qmin: Optional[float] = None
    mu_qmax: Optional[float] = None


@dataclass
class PhysicalStateResult:
    """Standard physical state estimate or physical part of an optimization result."""

    voltages: Dict[int, float]
    flows: Dict[Tuple[int, int], Tuple[float, float]]
    branch_currents: Dict[Tuple[int, int], float] = field(default_factory=dict)
    losses: Dict[Tuple[int, int], Tuple[float, float]] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    method_name: Optional[str] = None
    solve_time: Optional[float] = None


@dataclass
class DispatchResult:
    """Standard dispatch estimate or dispatch part of an optimization result."""

    generator_output: Dict[int, Tuple[float, float]] = field(default_factory=dict)
    objective: Optional[float] = None
    market_clearing: Dict[str, Any] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    method_name: Optional[str] = None
    solve_time: Optional[float] = None


@dataclass
class EconomicStateResult:
    """Standard economic state estimate."""

    lambda_p: Dict[int, float] = field(default_factory=dict)
    lambda_q: Dict[int, float] = field(default_factory=dict)
    bus_duals: Dict[int, BusDuals] = field(default_factory=dict)
    branch_duals: Dict[int, BranchDuals] = field(default_factory=dict)
    generator_duals: Dict[int, GeneratorDuals] = field(default_factory=dict)
    price_components: Dict[str, Any] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    method_name: Optional[str] = None
    solve_time: Optional[float] = None


@dataclass
class OptimizationResult:
    """Standard ACOPF/optimization result."""

    physical_state: PhysicalStateResult
    dispatch: DispatchResult = field(default_factory=DispatchResult)
    economic_state: Optional[EconomicStateResult] = None
    duals: List[Dict[str, Any]] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    method_name: Optional[str] = None
    solve_time: Optional[float] = None

    @property
    def generator_output(self) -> Dict[int, Tuple[float, float]]:
        """Compatibility accessor for dispatch generator output."""
        return self.dispatch.generator_output

    @generator_output.setter
    def generator_output(self, value: Dict[int, Tuple[float, float]]) -> None:
        self.dispatch.generator_output = value

    @property
    def objective(self) -> Optional[float]:
        """Compatibility accessor for dispatch objective."""
        return self.dispatch.objective

    @objective.setter
    def objective(self, value: Optional[float]) -> None:
        self.dispatch.objective = value


_DUAL_NAME_MAP = {
    "lambda_p_balance": ("bus", "lambda_p"),
    "lambda_q_balance": ("bus", "lambda_q"),
    "mu_v_lv_plus": ("bus", "mu_v_up"),
    "mu_v_lv_minus": ("bus", "mu_v_low"),
    "lambda_v_ref": ("bus", "lambda_vref"),
    "mu_int": ("branch", "mu_int"),
    "mu_l": ("branch", "mu_ell"),
    "mu_g_p_plus": ("generator", "mu_pmax"),
    "mu_g_p_minus": ("generator", "mu_pmin"),
    "mu_g_q_minus": ("generator", "mu_qmin"),
    "mu_g_q_plus": ("generator", "mu_qmax"),
}


def opf_result_from_optimization(result: OptimizationResult):
    """Adapt framework results for the database and legacy plotting API."""
    from .data import OPFResult
    economic = result.economic_state
    return OPFResult(
        voltages=dict(result.physical_state.voltages),
        flows=dict(result.physical_state.flows),
        duals_p={} if economic is None else dict(economic.lambda_p),
        duals_q={} if economic is None else dict(economic.lambda_q),
        generator_output=dict(result.dispatch.generator_output),
        cost=float(result.dispatch.objective or 0.0),
        convergence_info=dict(result.diagnostics),
        branch_currents=dict(result.physical_state.branch_currents),
        solver_name=result.method_name, solve_time=result.solve_time,
        socp_duals=list(result.duals),
    )


def physical_state_from_opf_result(result: Any) -> PhysicalStateResult:
    """Build a PhysicalStateResult from a legacy OPFResult-like object."""
    return PhysicalStateResult(
        voltages=dict(getattr(result, "voltages", {}) or {}),
        flows=dict(getattr(result, "flows", {}) or {}),
        branch_currents=dict(getattr(result, "branch_currents", {}) or {}),
        losses=_losses_from_convergence_info(getattr(result, "convergence_info", {}) or {}),
        diagnostics=dict(getattr(result, "convergence_info", {}) or {}),
        method_name=getattr(result, "solver_name", None),
        solve_time=getattr(result, "solve_time", None),
    )


def economic_state_from_opf_result(result: Any) -> EconomicStateResult:
    """Build an EconomicStateResult from a legacy OPFResult-like object."""
    lambda_p = dict(getattr(result, "duals_p", {}) or {})
    lambda_q = dict(getattr(result, "duals_q", {}) or {})
    economic = EconomicStateResult(
        lambda_p=lambda_p,
        lambda_q=lambda_q,
        bus_duals={
            int(bus_id): BusDuals(lambda_p=float(value), lambda_q=lambda_q.get(bus_id))
            for bus_id, value in lambda_p.items()
        },
        diagnostics=dict(getattr(result, "convergence_info", {}) or {}),
        method_name=getattr(result, "solver_name", None),
        solve_time=getattr(result, "solve_time", None),
    )

    for row in getattr(result, "socp_duals", []) or []:
        _apply_standard_dual_row(economic, row)

    return economic


def dispatch_from_opf_result(result: Any) -> DispatchResult:
    """Build a DispatchResult from a legacy OPFResult-like object."""
    convergence_info = dict(getattr(result, "convergence_info", {}) or {})
    merit_order = convergence_info.get("merit_order", {})
    market_clearing = merit_order if isinstance(merit_order, dict) else {}
    return DispatchResult(
        generator_output=dict(getattr(result, "generator_output", {}) or {}),
        objective=getattr(result, "cost", None),
        market_clearing=dict(market_clearing),
        diagnostics=convergence_info,
        method_name=getattr(result, "solver_name", None),
        solve_time=getattr(result, "solve_time", None),
    )


def optimization_result_from_opf_result(result: Any) -> OptimizationResult:
    """Build an OptimizationResult from a legacy OPFResult-like object."""
    return OptimizationResult(
        physical_state=physical_state_from_opf_result(result),
        dispatch=dispatch_from_opf_result(result),
        economic_state=economic_state_from_opf_result(result),
        duals=list(getattr(result, "socp_duals", []) or []),
        diagnostics=dict(getattr(result, "convergence_info", {}) or {}),
        method_name=getattr(result, "solver_name", None),
        solve_time=getattr(result, "solve_time", None),
    )


def _losses_from_convergence_info(convergence_info: Dict[str, Any]) -> Dict[Tuple[int, int], Tuple[float, float]]:
    line_state = convergence_info.get("line_state_bfsa")
    if line_state is None:
        merit_order = convergence_info.get("merit_order", {})
        if isinstance(merit_order, dict):
            line_state = merit_order.get("line_state_bfsa")
    if not isinstance(line_state, dict):
        return {}

    losses: Dict[Tuple[int, int], Tuple[float, float]] = {}
    for edge, state in line_state.items():
        if not isinstance(edge, tuple) or len(edge) != 2 or not isinstance(state, dict):
            continue
        losses[(int(edge[0]), int(edge[1]))] = (
            float(state.get("ploss", 0.0)),
            float(state.get("qloss", 0.0)),
        )
    return losses


def _apply_standard_dual_row(economic: EconomicStateResult, row: Dict[str, Any]) -> None:
    dual_name = str(row.get("dual_name", ""))
    mapping = _DUAL_NAME_MAP.get(dual_name)
    if mapping is None:
        return

    entity_type, field_name = mapping
    value = row.get("value")
    if value is None:
        return
    value = float(value)

    if entity_type == "bus":
        entity_id = row.get("entity_id")
        if entity_id is None:
            entity_id = _infer_root_bus_from_diagnostics(economic.diagnostics)
        if entity_id is None:
            return
        bus_duals = economic.bus_duals.setdefault(int(entity_id), BusDuals())
        setattr(bus_duals, field_name, value)
        if field_name == "lambda_p":
            economic.lambda_p[int(entity_id)] = value
        elif field_name == "lambda_q":
            economic.lambda_q[int(entity_id)] = value
        return

    entity_id = row.get("entity_id")
    if entity_id is None:
        return

    if entity_type == "branch":
        branch_duals = economic.branch_duals.setdefault(int(entity_id), BranchDuals())
        setattr(branch_duals, field_name, value)
    elif entity_type == "generator":
        generator_duals = economic.generator_duals.setdefault(int(entity_id), GeneratorDuals())
        setattr(generator_duals, field_name, value)


def _infer_root_bus_from_diagnostics(diagnostics: Dict[str, Any]) -> Optional[int]:
    radial_roots = diagnostics.get("radial_roots")
    if isinstance(radial_roots, list) and radial_roots:
        return int(radial_roots[0])
    return None
