"""SOCP ACOPF method."""

from __future__ import annotations

from libs.methods.base import MethodSpec
from libs.methods.defaults import get_method_defaults, get_method_option_schema
from libs.shared import MethodCapabilities


def build_socp_solver(**options):
    """Build the SOCP solver as a framework OPF method."""
    from .solver import SOCPSolver

    return SOCPSolver(
        solver_name=options.get("solver_name", "mosek"),
        tee=bool(options.get("verbose", False)),
        allow_load_shedding=bool(options.get("allow_load_shedding", False)),
    )


SOCP_METHOD = MethodSpec(
    family="opf",
    key="socp",
    label="SOCP ACOPF",
    description="Second-order cone relaxation ACOPF solver.",
    builder=build_socp_solver,
    capabilities=MethodCapabilities(
        produces_physical_state=True,
        produces_economic_state=True,
        produces_dispatch=True,
        supports_warm_start=True,
        requires_physical_state=False,
    ),
    default_options={
        **get_method_defaults("opf", "socp"),
    },
    option_schema=get_method_option_schema("opf", "socp"),
)


def __getattr__(name: str):
    if name == "SOCPSolver":
        from .solver import SOCPSolver

        return SOCPSolver
    if name == "RadialSOCPModel":
        from .model import RadialSOCPModel

        return RadialSOCPModel
    if name == "extract_opf_result":
        from .results import extract_opf_result

        return extract_opf_result
    raise AttributeError(name)


__all__ = [
    "RadialSOCPModel",
    "SOCPSolver",
    "SOCP_METHOD",
    "build_socp_solver",
    "extract_opf_result",
]
