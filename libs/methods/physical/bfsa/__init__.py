"""BFSA physical state estimator method."""

from __future__ import annotations

from libs.methods.base import MethodSpec
from libs.methods.defaults import get_method_defaults, get_method_option_schema

from .algorithm import bfsa_pq_case1, run_electrical_bfsa_forest
from .solver import BFSAElectricalSolver


def build_bfsa_physical_estimator(**options):
    """Build the BFSA physical state estimator."""
    return BFSAElectricalSolver(config=dict(options))


BFSA_PHYSICAL_METHOD = MethodSpec(
    family="physical",
    key="bfsa",
    label="BFSA physical state",
    description="Backward-forward sweep physical state estimator for radial networks.",
    builder=build_bfsa_physical_estimator,
    capabilities=BFSAElectricalSolver.capabilities,
    default_options=get_method_defaults("physical", "bfsa"),
    option_schema=get_method_option_schema("physical", "bfsa"),
)


__all__ = [
    "BFSAElectricalSolver",
    "BFSA_PHYSICAL_METHOD",
    "bfsa_pq_case1",
    "build_bfsa_physical_estimator",
    "run_electrical_bfsa_forest",
]
