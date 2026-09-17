"""State-object comparison utilities for the framework result model."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from statistics import mean, median
from typing import Dict, Iterable, Optional, Tuple

from libs.shared import (
    DispatchResult,
    EconomicStateResult,
    OptimizationResult,
    PhysicalStateResult,
)


ScalarDeviations = Dict[object, float]


def _percent_deviation(candidate: float, reference: float, *, zero_tol: float = 1e-12) -> float:
    """Return signed percent deviation relative to reference."""
    candidate = float(candidate)
    reference = float(reference)
    if abs(reference) <= zero_tol:
        return 0.0 if abs(candidate) <= zero_tol else float("inf")
    return 100.0 * (candidate - reference) / abs(reference)


def _shared_keys(left: Dict, right: Dict) -> Iterable:
    return sorted(set(left).intersection(right), key=str)


def _stats(values: Iterable[float]) -> Dict[str, float | int | None]:
    finite_values = [float(value) for value in values if isfinite(float(value))]
    if not finite_values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "max_abs": None,
            "abs_mean": None,
        }
    abs_values = [abs(value) for value in finite_values]
    return {
        "count": len(finite_values),
        "mean": mean(finite_values),
        "median": median(finite_values),
        "max_abs": max(abs_values),
        "abs_mean": mean(abs_values),
    }


def _compare_scalar_maps(candidate: Dict, reference: Dict) -> ScalarDeviations:
    return {
        key: _percent_deviation(candidate[key], reference[key])
        for key in _shared_keys(candidate, reference)
    }


def _compare_tuple_component_maps(
    candidate: Dict[object, Tuple[float, float]],
    reference: Dict[object, Tuple[float, float]],
    component_index: int,
) -> ScalarDeviations:
    return {
        key: _percent_deviation(candidate[key][component_index], reference[key][component_index])
        for key in _shared_keys(candidate, reference)
    }


@dataclass
class DispatchComparison:
    """Compare two standardized dispatch results."""

    candidate: DispatchResult
    reference: DispatchResult
    deviations: Dict[str, ScalarDeviations] = field(init=False)

    def __post_init__(self) -> None:
        self.deviations = {
            "p_g": _compare_tuple_component_maps(
                self.candidate.generator_output,
                self.reference.generator_output,
                0,
            ),
            "q_g": _compare_tuple_component_maps(
                self.candidate.generator_output,
                self.reference.generator_output,
                1,
            ),
            "objective": self._objective_deviation(),
        }

    def _objective_deviation(self) -> ScalarDeviations:
        if self.candidate.objective is None or self.reference.objective is None:
            return {}
        return {"objective": _percent_deviation(self.candidate.objective, self.reference.objective)}

    def summary_stats(self) -> Dict[str, Dict[str, float | int | None]]:
        return {metric: _stats(values.values()) for metric, values in self.deviations.items()}


@dataclass
class PhysicalStateComparison:
    """Compare two standardized physical state results."""

    candidate: PhysicalStateResult
    reference: PhysicalStateResult
    deviations: Dict[str, ScalarDeviations] = field(init=False)

    def __post_init__(self) -> None:
        self.deviations = {
            "voltage": _compare_scalar_maps(self.candidate.voltages, self.reference.voltages),
            "flow_p": _compare_tuple_component_maps(self.candidate.flows, self.reference.flows, 0),
            "flow_q": _compare_tuple_component_maps(self.candidate.flows, self.reference.flows, 1),
            "branch_current": _compare_scalar_maps(
                self.candidate.branch_currents,
                self.reference.branch_currents,
            ),
            "loss_p": _compare_tuple_component_maps(self.candidate.losses, self.reference.losses, 0),
            "loss_q": _compare_tuple_component_maps(self.candidate.losses, self.reference.losses, 1),
        }

    def summary_stats(self) -> Dict[str, Dict[str, float | int | None]]:
        return {metric: _stats(values.values()) for metric, values in self.deviations.items()}


@dataclass
class EconomicStateComparison:
    """Compare two standardized economic state results."""

    candidate: EconomicStateResult
    reference: EconomicStateResult
    deviations: Dict[str, ScalarDeviations] = field(init=False)

    def __post_init__(self) -> None:
        self.deviations = {
            "lambda_p": _compare_scalar_maps(self.candidate.lambda_p, self.reference.lambda_p),
            "lambda_q": _compare_scalar_maps(self.candidate.lambda_q, self.reference.lambda_q),
        }

    def summary_stats(self) -> Dict[str, Dict[str, float | int | None]]:
        return {metric: _stats(values.values()) for metric, values in self.deviations.items()}


@dataclass
class OptimizationComparison:
    """Compare two standardized optimization results by their component states."""

    candidate: OptimizationResult
    reference: OptimizationResult
    dispatch: DispatchComparison = field(init=False)
    physical: PhysicalStateComparison = field(init=False)
    economic: Optional[EconomicStateComparison] = field(init=False)

    def __post_init__(self) -> None:
        self.dispatch = DispatchComparison(self.candidate.dispatch, self.reference.dispatch)
        self.physical = PhysicalStateComparison(
            self.candidate.physical_state,
            self.reference.physical_state,
        )
        if self.candidate.economic_state is None or self.reference.economic_state is None:
            self.economic = None
        else:
            self.economic = EconomicStateComparison(
                self.candidate.economic_state,
                self.reference.economic_state,
            )

    def summary_stats(self) -> Dict[str, Dict[str, Dict[str, float | int | None]]]:
        summary = {
            "dispatch": self.dispatch.summary_stats(),
            "physical": self.physical.summary_stats(),
        }
        if self.economic is not None:
            summary["economic"] = self.economic.summary_stats()
        return summary
