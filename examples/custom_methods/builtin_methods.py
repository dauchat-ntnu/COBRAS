"""Built-in method factories exposed as framework custom methods.

The imports are intentionally lazy so users can inspect and import the example
registry without installing every optional backend dependency.
"""

from __future__ import annotations

from libs.methods import build_method


DEFAULT_METHODS = {
    "opf": "socp",
    "dispatch": "dummy_merit_order",
    "physical": "bfsa",
    "economic": "bfsa_dlmp",
}


def build_opf_solver(method: str, **options):
    """Build an ACOPF solver by framework method key."""
    return build_method("opf", method, options)


def build_dispatch_estimator(method: str, **options):
    """Build a dispatch estimator by framework method key."""
    return build_method("dispatch", method, options)


def build_physical_estimator(method: str, **options):
    """Build a physical state estimator by framework method key."""
    return build_method("physical", method, options)


def build_economic_estimator(method: str, **options):
    """Build an economic state estimator by framework method key."""
    return build_method("economic", method, options)
