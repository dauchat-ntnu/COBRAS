"""Public method interfaces for OPF solvers and state estimators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable

from .data import PowerFlowCase
from .results import DispatchResult, EconomicStateResult, OptimizationResult, PhysicalStateResult


@dataclass(frozen=True)
class MethodCapabilities:
    """Declare what a method consumes and produces."""

    produces_physical_state: bool
    produces_economic_state: bool
    produces_dispatch: bool
    supports_warm_start: bool
    requires_physical_state: bool
    requires_dispatch: bool = False
    requires_radial_network: bool = True


@runtime_checkable
class DispatchEstimator(Protocol):
    """Interface for dispatch or market-clearing estimators."""

    name: str
    capabilities: MethodCapabilities

    def estimate(
        self,
        case: PowerFlowCase,
        *,
        physical_state: Optional[PhysicalStateResult] = None,
        economic_state: Optional[EconomicStateResult] = None,
        previous_dispatch: Optional[DispatchResult] = None,
    ) -> DispatchResult:
        ...


@runtime_checkable
class PhysicalStateEstimator(Protocol):
    """Interface for physical state estimators."""

    name: str
    capabilities: MethodCapabilities

    def estimate(
        self,
        case: PowerFlowCase,
        *,
        dispatch: Optional[DispatchResult] = None,
        economic_state: Optional[EconomicStateResult] = None,
        previous_physical_state: Optional[PhysicalStateResult] = None,
    ) -> PhysicalStateResult:
        ...


@runtime_checkable
class EconomicStateEstimator(Protocol):
    """Interface for economic state estimators."""

    name: str
    capabilities: MethodCapabilities

    def estimate(
        self,
        case: PowerFlowCase,
        physical_state: PhysicalStateResult,
        *,
        dispatch: Optional[DispatchResult] = None,
        previous_economic_state: Optional[EconomicStateResult] = None,
    ) -> EconomicStateResult:
        ...


@runtime_checkable
class ACOPFSolver(Protocol):
    """Interface for ACOPF and relaxation-based optimization solvers."""

    name: str
    capabilities: MethodCapabilities

    def solve_optimization(
        self,
        case: PowerFlowCase,
        warm_start: Optional[PhysicalStateResult | OptimizationResult] = None,
    ) -> OptimizationResult:
        ...
