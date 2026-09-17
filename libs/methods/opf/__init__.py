"""Registered OPF methods."""

from .socp import SOCP_METHOD, build_socp_solver

__all__ = ["SOCP_METHOD", "build_socp_solver"]
