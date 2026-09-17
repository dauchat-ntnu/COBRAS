"""BFSA DLMP economic state estimator."""

from __future__ import annotations

from typing import Dict, Optional
from time import perf_counter

from libs.methods.physical.bfsa.solver import BFSASolver
from libs.shared import (
    DispatchResult,
    EconomicStateResult,
    MethodCapabilities,
    OPFResult,
    PhysicalStateResult,
    PowerFlowCase,
    economic_state_from_opf_result,
)


class BFSADLMPSolver(BFSASolver):
    """Economic estimator that propagates DLMPs from a BFSA physical state."""

    name = "BFSA-DLMP"
    capabilities = MethodCapabilities(
        produces_physical_state=False,
        produces_economic_state=True,
        produces_dispatch=False,
        supports_warm_start=False,
        requires_physical_state=True,
    )

    def estimate(
        self,
        case: PowerFlowCase,
        physical_state: PhysicalStateResult,
        *,
        dispatch: Optional[DispatchResult] = None,
        previous_economic_state: Optional[EconomicStateResult] = None,
        dlmp_seed_bus: Optional[int] = None,
    ) -> EconomicStateResult:
        """Estimate DLMPs from a previously estimated physical state."""
        start = perf_counter()
        electrical_result = self._opf_result_from_physical_state(physical_state)
        if str(self.config.get("merit_order_mode", "dummy")).lower() == "mcp":
            p_seed, q_seed, source = self._resolve_price_propagation_seeds({}, {})
            merit = dict(electrical_result.convergence_info.get("merit_order", {}))
            merit.update(lambda_p_clear=p_seed, lambda_q_clear=q_seed, price_source=source)
            electrical_result.convergence_info["merit_order"] = merit
        dlmp = self.propagate_dlmp(
            case=case,
            electrical_result=electrical_result,
            dlmp_seed_bus=dlmp_seed_bus,
        )
        opf_result = OPFResult(
            voltages=dict(physical_state.voltages),
            flows=dict(physical_state.flows),
            duals_p={int(bus_id): float(data["lambda_p"]) for bus_id, data in dlmp.items()},
            duals_q={int(bus_id): float(data["lambda_q"]) for bus_id, data in dlmp.items()},
            generator_output=dict(physical_state.diagnostics.get("generator_output", {})),
            cost=float(physical_state.diagnostics.get("cost", 0.0)),
            convergence_info={
                **dict(physical_state.diagnostics),
                "solver_stage": "dlmp",
            },
            branch_currents=dict(physical_state.branch_currents),
            solver_name=self.name,
            solve_time=perf_counter() - start,
        )
        return economic_state_from_opf_result(opf_result)

    def solve(
        self,
        case: PowerFlowCase,
        electrical_result: OPFResult,
        dlmp_seed_bus: Optional[int] = None,
    ) -> Dict[int, Dict[str, float]]:
        return self.propagate_dlmp(
            case=case,
            electrical_result=electrical_result,
            dlmp_seed_bus=dlmp_seed_bus,
        )
