"""BFSA DLMP economic state estimator method."""

from __future__ import annotations

from libs.methods.base import MethodSpec
from libs.methods.defaults import get_method_defaults, get_method_option_schema

from .solver import BFSADLMPSolver


def build_bfsa_dlmp_estimator(**options):
    """Build the BFSA DLMP economic estimator."""
    return BFSADLMPSolver(config=dict(options))


BFSA_DLMP_METHOD = MethodSpec(
    family="economic",
    key="bfsa_dlmp",
    label="BFSA DLMP propagation",
    description="Economic state estimator that propagates DLMPs from a BFSA physical state.",
    builder=build_bfsa_dlmp_estimator,
    capabilities=BFSADLMPSolver.capabilities,
    default_options=get_method_defaults("economic", "bfsa_dlmp"),
    option_schema=get_method_option_schema("economic", "bfsa_dlmp"),
)


__all__ = ["BFSADLMPSolver", "BFSA_DLMP_METHOD", "build_bfsa_dlmp_estimator"]
