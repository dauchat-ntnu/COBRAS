"""
Solver wrapper for SOCP OPF.
"""

from __future__ import annotations

import time
from typing import Any, Optional
import pyomo.environ as pyo

from libs.shared import (
    MethodCapabilities,
    OPFResult,
    OptimizationResult,
    PhysicalStateResult,
    PowerFlowCase,
    optimization_result_from_opf_result,
    orient_radial_network,
)
from .model import RadialSOCPModel
from .results import extract_opf_result


class SOCPSolver:
    """
    SOCP OPF Solver wrapper.
    
    Builds and solves SOCP relaxation using Pyomo + MOSEK/IPOPT.
    
    Example:
        solver = SOCPSolver(solver_name="mosek", tee=False)
        result = solver.solve(case)
    """

    capabilities = MethodCapabilities(
        produces_physical_state=True,
        produces_economic_state=True,
        produces_dispatch=True,
        supports_warm_start=True,
        requires_physical_state=False,
    )
    
    def __init__(
        self,
        solver_name: str = "mosek",
        tee: bool = False,
        solver_options: Optional[dict[str, Any]] = None,
        allow_load_shedding: bool = False,
        load_curtailment_penalty_p: float = 1000,
        load_curtailment_penalty_q: float = 1000,
    ):
        """
        Initialize SOCP solver.
        
        Args:
            solver_name: Pyomo solver name ("mosek", "ipopt", etc.)
            tee: If True, stream solver output to console
            solver_options: Dict of solver-specific options
            allow_load_shedding: If True, enables nodal load-curtailment variables
            load_curtailment_penalty_p: Objective penalty coefficient for active curtailment
            load_curtailment_penalty_q: Objective penalty coefficient for reactive curtailment
        """
        self.solver_name = solver_name
        self.tee = tee
        self.solver_options = solver_options or {}
        self.allow_load_shedding = bool(allow_load_shedding)
        self.load_curtailment_penalty_p = float(load_curtailment_penalty_p)
        self.load_curtailment_penalty_q = float(load_curtailment_penalty_q)
        self._model_builder: Optional[RadialSOCPModel] = None
        self._model_signature: Optional[tuple[Any, ...]] = None

    @property
    def name(self) -> str:
        """Framework method name."""
        return f"SOCP-{self.solver_name}"

    @staticmethod
    def _case_signature(case: PowerFlowCase, allow_load_shedding: bool) -> tuple[Any, ...]:
        return RadialSOCPModel._case_signature(case, allow_load_shedding)

    @staticmethod
    def _legacy_warm_start(
        warm_start: "OPFResult | PhysicalStateResult | OptimizationResult | None",
    ) -> OPFResult | None:
        """Convert framework warm-start objects to the legacy OPFResult shape."""
        if warm_start is None:
            return None
        if isinstance(warm_start, OPFResult):
            return warm_start

        if isinstance(warm_start, OptimizationResult):
            physical_state = warm_start.physical_state
            generator_output = dict(warm_start.generator_output)
            cost = warm_start.objective
            diagnostics = dict(warm_start.diagnostics)
            method_name = warm_start.method_name
            solve_time = warm_start.solve_time
        elif isinstance(warm_start, PhysicalStateResult):
            physical_state = warm_start
            generator_output = dict(physical_state.diagnostics.get("generator_output", {}))
            cost = physical_state.diagnostics.get("cost", 0.0)
            diagnostics = dict(physical_state.diagnostics)
            method_name = physical_state.method_name
            solve_time = physical_state.solve_time
        else:
            raise TypeError(
                "warm_start must be OPFResult, PhysicalStateResult, OptimizationResult, or None"
            )

        return OPFResult(
            voltages=dict(physical_state.voltages),
            flows=dict(physical_state.flows),
            duals_p={},
            duals_q={},
            generator_output=generator_output,
            cost=cost,
            convergence_info=diagnostics,
            branch_currents=dict(physical_state.branch_currents),
            solver_name=method_name,
            solve_time=solve_time,
        )
    
    def solve(
        self,
        case: PowerFlowCase,
        orient: bool = True,
        warm_start: "OPFResult | PhysicalStateResult | OptimizationResult | None" = None,
    ) -> OPFResult:
        """
        Solve SOCP OPF for given case.
        
        Args:
            case: PowerFlowCase object
            orient: If True, auto-orient network if not already done
        
        Returns:
            OPFResult with voltages, flows, duals, generator output, and cost
        """
        total_start = time.perf_counter()
        warm_start = self._legacy_warm_start(warm_start)
        
        # Orient network if needed
        if not case.incoming_branches:
            if orient:
                case = orient_radial_network(case)
            else:
                raise ValueError("Case not oriented; set orient=True or call orient_radial_network(case) first")
        
        # Build or refresh model
        build_start = time.perf_counter()
        case_signature = self._case_signature(case, self.allow_load_shedding)
        if self._model_builder is None or self._model_signature != case_signature:
            self._model_builder = RadialSOCPModel(
                case,
                allow_load_shedding=self.allow_load_shedding,
                load_curtailment_penalty_p=self.load_curtailment_penalty_p,
                load_curtailment_penalty_q=self.load_curtailment_penalty_q,
            )
            m = self._model_builder.build()
        else:
            m = self._model_builder.update_case(case)
        self._model_signature = case_signature
        build_time = time.perf_counter() - build_start
        
        # Solve
        opt = pyo.SolverFactory(self.solver_name)
        if opt is None:
            raise ValueError(f"Solver '{self.solver_name}' not available. Install it or use a different solver.")
        
        # Set solver options
        for key, value in self.solver_options.items():
            opt.options[key] = value
        
        # Inject warm-start variable values when available (previous timestep result)
        if warm_start is not None:
            try:
                # Voltages (squared)
                for i, v_sq in (warm_start.voltages or {}).items():
                    if i in getattr(m, "v", {}):
                        try:
                            m.v[i].value = float(v_sq)
                        except Exception:
                            pass

                # Generator outputs
                for g, pq in (warm_start.generator_output or {}).items():
                    if g in getattr(m, "pg", {}):
                        try:
                            m.pg[g].value = float(pq[0])
                        except Exception:
                            pass
                    if g in getattr(m, "qg", {}):
                        try:
                            m.qg[g].value = float(pq[1])
                        except Exception:
                            pass

                # Branch flows: warm_start.flows keys are (from_bus, to_bus)
                branch_lookup = { (br.from_bus, br.to_bus): br.branch_id for br in case.branches }
                for (i, j), pq in (warm_start.flows or {}).items():
                    bid = branch_lookup.get((i, j))
                    if bid is None:
                        continue
                    if bid in getattr(m, "p", {}):
                        try:
                            m.p[bid].value = float(pq[0])
                        except Exception:
                            pass
                    if bid in getattr(m, "q", {}):
                        try:
                            m.q[bid].value = float(pq[1])
                        except Exception:
                            pass

                # Estimate ell from flows and sending-end voltage when possible: ell = (p^2 + q^2) / v_send
                for br in case.branches:
                    e = br.branch_id
                    key = (br.from_bus, br.to_bus)
                    if key in (warm_start.flows or {}) and br.from_bus in (warm_start.voltages or {}):
                        p_val, q_val = warm_start.flows[key]
                        v_send = warm_start.voltages.get(br.from_bus, None)
                        try:
                            if v_send is not None and float(v_send) > 1e-12:
                                ell_est = (float(p_val) ** 2 + float(q_val) ** 2) / float(v_send)
                                if e in getattr(m, "ell", {}):
                                    m.ell[e].value = float(ell_est)
                        except Exception:
                            pass
            except Exception:
                # Best-effort warm start; never fatal
                pass

        optimize_start = time.perf_counter()
        solver_results = opt.solve(m, tee=self.tee)
        solve_time = time.perf_counter() - optimize_start

        termination = None
        if solver_results is not None and getattr(solver_results, "solver", None) is not None:
            termination = solver_results.solver.termination_condition

        if termination in (
            pyo.TerminationCondition.infeasible,
            pyo.TerminationCondition.infeasibleOrUnbounded,
        ):
            raise ValueError(
                f"SOCP solve failed: termination_condition={termination}. "
                "The case is infeasible for the current snapshot and constraints."
            )

        if termination not in (pyo.TerminationCondition.optimal, pyo.TerminationCondition.locallyOptimal):
            raise ValueError(f"SOCP solve failed: termination_condition={termination}; no accepted optimal solution")
        
        # Extract results
        extract_start = time.perf_counter()
        result = extract_opf_result(m, case, solver_results, solve_time)
        extract_time = time.perf_counter() - extract_start
        total_time = time.perf_counter() - total_start

        if result.convergence_info is None:
            result.convergence_info = {}
        timing_info = result.convergence_info.setdefault("timing", {})
        timing_info["build"] = float(build_time)
        timing_info["optimize"] = float(solve_time)
        timing_info["extract"] = float(extract_time)
        timing_info["total"] = float(total_time)

        result.solver_name = self.solver_name
        
        return result

    def solve_optimization(
        self,
        case: PowerFlowCase,
        orient: bool = True,
        warm_start: "OPFResult | PhysicalStateResult | OptimizationResult | None" = None,
    ) -> OptimizationResult:
        """Solve SOCP and return the framework optimization result."""
        result = self.solve(case=case, orient=orient, warm_start=warm_start)
        optimization = optimization_result_from_opf_result(result)
        optimization.method_name = self.name
        optimization.physical_state.method_name = self.name
        if optimization.economic_state is not None:
            optimization.economic_state.method_name = self.name
        return optimization
