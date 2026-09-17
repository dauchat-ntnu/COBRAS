from pathlib import Path
import sys

sys.path.append(str(Path.cwd()))

from libs.shared.gui_method_config import (
    METHOD_FAMILY_COMPARISON,
    METHOD_FAMILY_DISPATCH,
    METHOD_FAMILY_ECONOMIC,
    METHOD_FAMILY_OPF,
    METHOD_FAMILY_PHYSICAL,
    MODE_BFSA,
    MODE_COMPARISON,
    MODE_SOCP,
    methods_config_from_legacy_job,
)


def test_socp_mode_enables_only_opf_method():
    methods = methods_config_from_legacy_job(
        {
            "mode": MODE_SOCP,
            "socp_solver_name": "mosek",
            "socp_verbose": True,
            "socp_allow_load_shedding": True,
            "socp_warm_start": "BFSA t",
        }
    )

    assert methods[METHOD_FAMILY_OPF]["enabled"] is True
    assert methods[METHOD_FAMILY_OPF]["method"] == "socp"
    assert methods[METHOD_FAMILY_OPF]["options"]["solver_name"] == "mosek"
    assert methods[METHOD_FAMILY_OPF]["options"]["warm_start"] == "BFSA t"
    assert methods[METHOD_FAMILY_DISPATCH]["enabled"] is False
    assert methods[METHOD_FAMILY_PHYSICAL]["enabled"] is False
    assert methods[METHOD_FAMILY_ECONOMIC]["enabled"] is False
    assert methods[METHOD_FAMILY_COMPARISON]["enabled"] is False


def test_bfsa_mode_enables_dispatch_physical_and_economic_estimators():
    methods = methods_config_from_legacy_job(
        {
            "mode": MODE_BFSA,
            "bfsa_max_iter": 25,
            "bfsa_tol": 1e-5,
            "bfsa_convergence_check": "voltage",
            "bfsa_merit_order_mode": "mcp",
            "bfsa_mcp_active_price": 42.0,
            "bfsa_mcp_reactive_price": 0.4,
        }
    )

    assert methods[METHOD_FAMILY_OPF]["enabled"] is False
    assert methods[METHOD_FAMILY_DISPATCH]["enabled"] is True
    assert methods[METHOD_FAMILY_DISPATCH]["method"] == "dummy_merit_order"
    assert methods[METHOD_FAMILY_PHYSICAL]["enabled"] is True
    assert methods[METHOD_FAMILY_PHYSICAL]["method"] == "bfsa"
    assert methods[METHOD_FAMILY_PHYSICAL]["options"]["max_iterations"] == 25
    assert methods[METHOD_FAMILY_PHYSICAL]["options"]["tol"] == 1e-5
    assert methods[METHOD_FAMILY_ECONOMIC]["enabled"] is True
    assert methods[METHOD_FAMILY_ECONOMIC]["method"] == "bfsa_dlmp"
    assert methods[METHOD_FAMILY_ECONOMIC]["options"]["mcp_active_price"] == 42.0


def test_comparison_mode_enables_opf_estimators_and_comparison():
    methods = methods_config_from_legacy_job(
        {
            "mode": MODE_COMPARISON,
            "comparison_allow_load_shedding": True,
            "comparison_bfsa_merit_order_mode": "dummy",
        }
    )

    assert methods[METHOD_FAMILY_OPF]["enabled"] is True
    assert methods[METHOD_FAMILY_OPF]["options"]["allow_load_shedding"] is True
    assert methods[METHOD_FAMILY_DISPATCH]["enabled"] is True
    assert methods[METHOD_FAMILY_PHYSICAL]["enabled"] is True
    assert methods[METHOD_FAMILY_ECONOMIC]["enabled"] is True
    assert methods[METHOD_FAMILY_COMPARISON]["enabled"] is True
