"""Registry-backed method pipelines."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from libs.methods.defaults import get_method_defaults
from libs.methods.registry import build_method
from libs.methods.execution import run_estimator_stages
from libs.shared.results import opf_result_from_optimization
from libs.shared import OPFResult, PowerFlowCase


class EstimatorPipelineSolver:
    """Solver-like adapter for dispatch -> physical -> economic estimators."""

    name = "EstimatorPipeline"

    def __init__(
        self,
        *,
        dispatch_method: str = "dummy_merit_order",
        physical_method: str = "bfsa",
        economic_method: str = "bfsa_dlmp",
        dispatch_options: Optional[Mapping[str, Any]] = None,
        physical_options: Optional[Mapping[str, Any]] = None,
        economic_options: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.dispatch_method = dispatch_method
        self.physical_method = physical_method
        self.economic_method = economic_method
        self.dispatch_options = get_method_defaults("dispatch", dispatch_method)
        self.dispatch_options.update(dispatch_options or {})
        self.physical_options = get_method_defaults("physical", physical_method)
        self.physical_options.update(physical_options or {})
        self.economic_options = get_method_defaults("economic", economic_method)
        self.economic_options.update(economic_options or {})

    def solve(
        self,
        case: PowerFlowCase,
        *,
        show_net_bus_plot: bool = False,
        show_merit_order_plot: bool = False,
    ) -> OPFResult:
        """Run the estimator stages and adapt their output for persistence."""
        result = run_estimator_stages(
            case,
            build_method("dispatch", self.dispatch_method, self.dispatch_options),
            build_method("physical", self.physical_method, self.physical_options),
            build_method("economic", self.economic_method, self.economic_options),
            show_net_bus_plot=show_net_bus_plot,
            show_merit_order_plot=show_merit_order_plot,
        )
        return opf_result_from_optimization(result)
