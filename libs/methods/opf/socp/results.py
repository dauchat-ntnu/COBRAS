"""
Result extraction from Pyomo SOCP model to OPFResult.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import math
import pyomo.environ as pyo

from libs.shared import PowerFlowCase, OPFResult


def _val(x):
    """Extract float value from Pyomo variable/expression."""
    try:
        return float(x.value)
    except Exception:
        pass
    try:
        return float(x())
    except Exception:
        return None


def _is_valid_dual(val: float, max_allowed: float = 1e10) -> bool:
    """Check if a dual value is finite and within reasonable bounds."""
    if val is None:
        return False
    try:
        fval = float(val)
        if not math.isfinite(fval):
            return False
        if abs(fval) > max_allowed:
            return False
        return True
    except Exception:
        return False


def _dual(model, c, max_allowed: float = 1e10):
    """Extract dual value from constraint, sanitizing infinite/NaN values."""
    try:
        val = model.dual[c]
    except Exception:
        return None

    try:
        fval = float(val)
        if _is_valid_dual(fval, max_allowed=max_allowed):
            return fval
        else:
            return None
    except Exception:
        pass

    try:
        vals = [float(v) for v in val]
        return vals if vals else None
    except Exception:
        return None


def _dual_float(model, c, *, max_allowed: float = 1e10, negate: bool = False) -> Optional[float]:
    """Extract a single finite dual value, optionally flipping sign for convention."""
    val = _dual(model, c, max_allowed=max_allowed)
    if isinstance(val, list):
        if len(val) != 1:
            return None
        val = val[0]
    if val is None:
        return None
    try:
        fval = float(val)
    except Exception:
        return None
    if not _is_valid_dual(fval, max_allowed=max_allowed):
        return None
    return -fval if negate else fval


def _append_dual(
    rows: List[Dict[str, Any]],
    *,
    dual_name: str,
    entity_type: str,
    entity_id: Optional[int],
    value: Optional[float],
) -> None:
    if value is None:
        return
    rows.append(
        {
            "dual_name": dual_name,
            "entity_type": entity_type,
            "entity_id": entity_id,
            "value": float(value),
        }
    )


def _iter_component_children(component):
    try:
        yield from component.children()
    except Exception:
        return


def _soc_scalar_dual(model, soc_block, *, max_allowed: float = 1e10) -> Optional[float]:
    """
    Return a scalar SOC dual only when Pyomo/solver exposes exactly one scalar value.

    Conic duals are often vector-valued or represented on generated domain
    constraints, so ambiguous cases are intentionally omitted from persistence.
    """
    direct = _dual_float(model, soc_block, max_allowed=max_allowed)
    if direct is not None:
        return direct

    values: List[float] = []
    stack = list(_iter_component_children(soc_block))
    while stack:
        child = stack.pop()
        child_val = _dual(model, child, max_allowed=max_allowed)
        if isinstance(child_val, list):
            scalar_vals = []
            for val in child_val:
                try:
                    fval = float(val)
                except Exception:
                    continue
                if _is_valid_dual(fval, max_allowed=max_allowed):
                    scalar_vals.append(fval)
            values.extend(scalar_vals)
        elif child_val is not None:
            try:
                fval = float(child_val)
            except Exception:
                fval = None
            if fval is not None and _is_valid_dual(fval, max_allowed=max_allowed):
                values.append(fval)
        stack.extend(_iter_component_children(child))

    return values[0] if len(values) == 1 else None


def extract_opf_result(
    model: pyo.ConcreteModel,
    case: PowerFlowCase,
    solver_results: Optional[pyo.SolverResults] = None,
    solve_time: Optional[float] = None,
) -> OPFResult:
    """
    Extract OPFResult from solved Pyomo model.
    
    Args:
        model: Solved Pyomo model (from RadialSOCPModel.build())
        case: Original PowerFlowCase
        solver_results: Pyomo solver results object
        solve_time: Total solve time in seconds
    
    Returns:
        OPFResult with standardized format
    """
    
    # Extract bus voltages (squared)
    voltages: Dict[int, float] = {}
    for i in model.BUS:
        voltages[i] = _val(model.v[i])
    
    # Extract branch flows and current-squared values.
    flows: Dict[Tuple[int, int], Tuple[float, float]] = {}
    branch_currents: Dict[Tuple[int, int], float] = {}
    branches_by_id = {branch.branch_id: branch for branch in case.branches}
    for e in model.BRANCH:
        br = branches_by_id.get(e)
        if br:
            p_flow = _val(model.p[e])
            q_flow = _val(model.q[e])
            flows[(br.from_bus, br.to_bus)] = (p_flow, q_flow)
            if hasattr(model, "ell") and e in model.ell:
                ell_value = _val(model.ell[e])
                if ell_value is not None:
                    branch_currents[(br.from_bus, br.to_bus)] = float(ell_value)
    
    # Extract generator output
    generator_output: Dict[int, Tuple[float, float]] = {}
    for g in model.GEN:
        pg = _val(model.pg[g])
        qg = _val(model.qg[g])
        generator_output[g] = (pg, qg)
    
    # Extract LMPs (duals of power balance constraints)
    # Cap duals at reasonable bounds to catch numerical issues and unboundedness.
    duals_p: Dict[int, float] = {}
    duals_q: Dict[int, float] = {}
    max_dual = 1e8  # Flag suspicious solver behavior
    socp_duals: List[Dict[str, Any]] = []

    _append_dual(
        socp_duals,
        dual_name="lambda_v_ref",
        entity_type="system",
        entity_id=None,
        value=_dual_float(model, model.v_ref, max_allowed=max_dual),
    )
    
    for i in model.BUS:
        if i in model.p_balance:
            lmp_p = _dual(model, model.p_balance[i], max_allowed=max_dual)
            if lmp_p is not None:
                duals_p[i] = float(lmp_p)
            else:
                duals_p[i] = float("nan")
            _append_dual(
                socp_duals,
                dual_name="lambda_p_balance",
                entity_type="bus",
                entity_id=int(i),
                value=duals_p[i],
            )
        
        if i in model.q_balance:
            lmp_q = _dual(model, model.q_balance[i], max_allowed=max_dual)
            if lmp_q is not None:
                duals_q[i] = float(lmp_q)
            else:
                duals_q[i] = float("nan")
            _append_dual(
                socp_duals,
                dual_name="lambda_q_balance",
                entity_type="bus",
                entity_id=int(i),
                value=duals_q[i],
            )

        if hasattr(model, "v_upper") and i in model.v_upper:
            _append_dual(
                socp_duals,
                dual_name="mu_v_lv_plus",
                entity_type="bus",
                entity_id=int(i),
                value=_dual_float(model, model.v_upper[i], max_allowed=max_dual),
            )
        if hasattr(model, "v_lower") and i in model.v_lower:
            _append_dual(
                socp_duals,
                dual_name="mu_v_lv_minus",
                entity_type="bus",
                entity_id=int(i),
                value=_dual_float(model, model.v_lower[i], max_allowed=max_dual, negate=True),
            )

    for e in model.BRANCH:
        if hasattr(model, "voltage_drop") and e in model.voltage_drop:
            _append_dual(
                socp_duals,
                dual_name="lambda_v",
                entity_type="branch",
                entity_id=int(e),
                value=_dual_float(model, model.voltage_drop[e], max_allowed=max_dual),
            )
        if hasattr(model, "ell_upper") and e in model.ell_upper:
            _append_dual(
                socp_duals,
                dual_name="mu_l",
                entity_type="branch",
                entity_id=int(e),
                value=_dual_float(model, model.ell_upper[e], max_allowed=max_dual),
            )
        if hasattr(model, "ell_lower") and e in model.ell_lower:
            _append_dual(
                socp_duals,
                dual_name="mu_l_minus",
                entity_type="branch",
                entity_id=int(e),
                value=_dual_float(model, model.ell_lower[e], max_allowed=max_dual, negate=True),
            )
        if hasattr(model, "soc") and e in model.soc:
            _append_dual(
                socp_duals,
                dual_name="mu_int",
                entity_type="branch",
                entity_id=int(e),
                value=_soc_scalar_dual(model, model.soc[e], max_allowed=max_dual),
            )

    for g in model.GEN:
        if hasattr(model, "pg_lower") and g in model.pg_lower:
            _append_dual(
                socp_duals,
                dual_name="mu_g_p_minus",
                entity_type="generator",
                entity_id=int(g),
                value=_dual_float(model, model.pg_lower[g], max_allowed=max_dual, negate=True),
            )
        if hasattr(model, "pg_upper") and g in model.pg_upper:
            _append_dual(
                socp_duals,
                dual_name="mu_g_p_plus",
                entity_type="generator",
                entity_id=int(g),
                value=_dual_float(model, model.pg_upper[g], max_allowed=max_dual),
            )
        if hasattr(model, "qg_lower") and g in model.qg_lower:
            _append_dual(
                socp_duals,
                dual_name="mu_g_q_minus",
                entity_type="generator",
                entity_id=int(g),
                value=_dual_float(model, model.qg_lower[g], max_allowed=max_dual, negate=True),
            )
        if hasattr(model, "qg_upper") and g in model.qg_upper:
            _append_dual(
                socp_duals,
                dual_name="mu_g_q_plus",
                entity_type="generator",
                entity_id=int(g),
                value=_dual_float(model, model.qg_upper[g], max_allowed=max_dual),
            )
    
    # The model objective is formulated in per-unit, so rescale the reported
    # objective value to the case base power without changing the optimization.
    cost = _val(model.obj)
    if cost is not None:
        cost = float(cost) * float(case.base_mva)

    # Extract load curtailment totals (if present on the model)
    total_p_curtailment = 0.0
    total_q_curtailment = 0.0
    if hasattr(model, "p_curt"):
        for i in model.BUS:
            val = _val(model.p_curt[i])
            if val is not None:
                total_p_curtailment += float(val)
    if hasattr(model, "q_curt"):
        for i in model.BUS:
            val = _val(model.q_curt[i])
            if val is not None:
                total_q_curtailment += float(val)

    # Numerical guard: clip tiny negatives introduced by solver tolerances.
    if abs(total_p_curtailment) < 1e-9:
        total_p_curtailment = 0.0
    if abs(total_q_curtailment) < 1e-9:
        total_q_curtailment = 0.0
    
    # Convergence info
    convergence_info = {
        "converged": solver_results is not None and str(solver_results.solver.termination_condition) == "optimal",
        "termination_message": str(solver_results.solver.termination_condition) if solver_results else "unknown",
        "allow_load_shedding": bool(getattr(model, "allow_load_shedding", False)),
        "total_p_curtailment": total_p_curtailment,
        "total_q_curtailment": total_q_curtailment,
        "p_curtailment_by_bus": {i: _val(model.p_curt[i]) for i in model.BUS} if hasattr(model, "p_curt") else {},
        "q_curtailment_by_bus": {i: _val(model.q_curt[i]) for i in model.BUS} if hasattr(model, "q_curt") else {},
    }
    
    return OPFResult(
        voltages=voltages,
        flows=flows,
        duals_p=duals_p,
        duals_q=duals_q,
        generator_output=generator_output,
        cost=cost,
        convergence_info=convergence_info,
        branch_currents=branch_currents,
        solve_time=solve_time,
        socp_duals=socp_duals,
    )
