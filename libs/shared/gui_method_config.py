"""GUI method-family configuration helpers.

This module has no solver imports so GUI configuration can be tested without
requiring optional solver backends.
"""

from __future__ import annotations



METHOD_FAMILY_OPF = "opf"
METHOD_FAMILY_DISPATCH = "dispatch"
METHOD_FAMILY_PHYSICAL = "physical"
METHOD_FAMILY_ECONOMIC = "economic"
METHOD_FAMILY_COMPARISON = "comparison"

MODE_SOCP = "socp"
MODE_BFSA = "bfsa"
MODE_COMPARISON = "comparison"


def _method_job(enabled: bool, method: str, options: dict | None = None) -> dict:
    return {
        "enabled": bool(enabled),
        "method": method,
        "options": dict(options or {}),
    }


def empty_methods_job() -> dict:
    return {
        METHOD_FAMILY_OPF: _method_job(False, "socp"),
        METHOD_FAMILY_DISPATCH: _method_job(False, "dummy_merit_order"),
        METHOD_FAMILY_PHYSICAL: _method_job(False, "bfsa"),
        METHOD_FAMILY_ECONOMIC: _method_job(False, "bfsa_dlmp"),
        METHOD_FAMILY_COMPARISON: {"enabled": False},
    }


def methods_config_from_legacy_job(job: dict) -> dict:
    """Translate the current mode-based GUI job into method-family settings."""
    methods = empty_methods_job()
    mode = str(job.get("mode", MODE_SOCP)).strip().lower()

    if mode == MODE_SOCP:
        methods[METHOD_FAMILY_OPF] = _method_job(
            True,
            "socp",
            {
                "solver_name": job.get("socp_solver_name", "mosek"),
                "verbose": bool(job.get("socp_verbose", False)),
                "allow_load_shedding": bool(job.get("socp_allow_load_shedding", False)),
                "warm_start": job.get("socp_warm_start", "None"),
            },
        )
        return methods

    if mode == MODE_BFSA:
        dispatch_options = {
            "merit_order_mode": str(job.get("bfsa_merit_order_mode", "dummy")),
            "mcp_active_price": job.get("bfsa_mcp_active_price"),
            "mcp_reactive_price": job.get("bfsa_mcp_reactive_price"),
        }
        methods[METHOD_FAMILY_DISPATCH] = _method_job(True, "dummy_merit_order", dispatch_options)
        methods[METHOD_FAMILY_PHYSICAL] = _method_job(
            True,
            "bfsa",
            {
                "tol": float(job.get("bfsa_tol", 1e-3)),
                "max_iterations": int(job.get("bfsa_max_iter", 100)),
                "convergence_check": str(job.get("bfsa_convergence_check", "both")),
                "show_network_plot": bool(job.get("bfsa_show_plot", False)),
            },
        )
        methods[METHOD_FAMILY_ECONOMIC] = _method_job(True, "bfsa_dlmp", dispatch_options)
        return methods

    if mode == MODE_COMPARISON:
        dispatch_options = {
            "merit_order_mode": str(job.get("comparison_bfsa_merit_order_mode", "dummy")),
            "mcp_active_price": job.get("comparison_bfsa_mcp_active_price"),
            "mcp_reactive_price": job.get("comparison_bfsa_mcp_reactive_price"),
        }
        methods[METHOD_FAMILY_OPF] = _method_job(
            True,
            "socp",
            {
                "solver_name": "mosek",
                "allow_load_shedding": bool(job.get("comparison_allow_load_shedding", False)),
                "warm_start": job.get("socp_warm_start", "None"),
            },
        )
        methods[METHOD_FAMILY_DISPATCH] = _method_job(True, "dummy_merit_order", dispatch_options)
        methods[METHOD_FAMILY_PHYSICAL] = _method_job(True, "bfsa", {"compute_losses": True})
        methods[METHOD_FAMILY_ECONOMIC] = _method_job(True, "bfsa_dlmp", dispatch_options)
        methods[METHOD_FAMILY_COMPARISON] = {"enabled": True, "reference": "opf", "candidate": "estimator"}
        return methods

    raise ValueError(f"Unsupported mode '{mode}'")
