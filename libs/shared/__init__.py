"""
PowerSOC Shared Layer
====================
Common data structures and utilities for SOCP, BFSA, and other modules.

Public API:
-----------
Data Structures:
  - BusData
  - BranchData
  - GeneratorData
  - LoadData
  - PowerFlowCase
  - OPFResult

Functions:
  - load_case_from_folder()
  - orient_radial_network()
  - validate_case()
"""

from .data import (
    BusData,
    BranchData,
    GeneratorData,
    LoadData,
    PowerFlowCase,
    OPFResult,
    validate_case,
)

from .io import (
    load_case_from_folder,
)

from .network import (
    orient_radial_network,
)

from .interfaces import (
    MethodCapabilities,
    DispatchEstimator,
    PhysicalStateEstimator,
    EconomicStateEstimator,
    ACOPFSolver,
)

from .results import (
    BusDuals,
    BranchDuals,
    GeneratorDuals,
    DispatchResult,
    PhysicalStateResult,
    EconomicStateResult,
    OptimizationResult,
    dispatch_from_opf_result,
    physical_state_from_opf_result,
    economic_state_from_opf_result,
    optimization_result_from_opf_result,
)

from libs.analysis import (
  analyze_load_profiles,
  load_profile_timeseries,
  plot_load_profile_boxplots,
  plot_load_profile_timeseries,
  summarize_load_profiles,
)

__all__ = [
    "BusData",
    "BranchData",
    "GeneratorData",
    "LoadData",
    "PowerFlowCase",
    "OPFResult",
    "load_case_from_folder",
    "orient_radial_network",
    "validate_case",
    "MethodCapabilities",
    "DispatchEstimator",
    "PhysicalStateEstimator",
    "EconomicStateEstimator",
    "ACOPFSolver",
    "BusDuals",
    "BranchDuals",
    "GeneratorDuals",
    "DispatchResult",
    "PhysicalStateResult",
    "EconomicStateResult",
    "OptimizationResult",
    "dispatch_from_opf_result",
    "physical_state_from_opf_result",
    "economic_state_from_opf_result",
    "optimization_result_from_opf_result",
    "analyze_load_profiles",
    "load_profile_timeseries",
    "plot_load_profile_boxplots",
    "plot_load_profile_timeseries",
    "summarize_load_profiles",
]
