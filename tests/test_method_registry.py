from pathlib import Path
import sys

import pytest

sys.path.append(str(Path.cwd()))

from libs.methods import METHOD_REGISTRY, MethodSpec, build_method, get_method_spec, list_method_specs
from libs.shared import (
    BranchData,
    BusData,
    DispatchResult,
    EconomicStateResult,
    GeneratorData,
    LoadData,
    PhysicalStateResult,
    PowerFlowCase,
)


def test_builtin_method_registry_contains_expected_families_and_methods():
    assert set(METHOD_REGISTRY) == {"opf", "dispatch", "physical", "economic"}
    assert set(METHOD_REGISTRY["opf"]) == {"socp"}
    assert set(METHOD_REGISTRY["dispatch"]) == {"dummy_merit_order"}
    assert set(METHOD_REGISTRY["physical"]) == {"bfsa"}
    assert set(METHOD_REGISTRY["economic"]) == {"bfsa_dlmp"}


def test_every_registered_method_has_required_metadata():
    for family, methods in METHOD_REGISTRY.items():
        for key, spec in methods.items():
            assert isinstance(spec, MethodSpec)
            assert spec.family == family
            assert spec.key == key
            assert spec.label
            assert callable(spec.builder)
            assert spec.capabilities is not None


def test_registry_lookup_and_listing_are_copy_safe():
    spec = get_method_spec("opf", "socp")
    listed = list_method_specs("opf")

    assert listed == {"socp": spec}
    listed["new"] = spec
    assert "new" not in METHOD_REGISTRY["opf"]


def test_build_method_returns_none_for_disabled_method():
    assert build_method("opf", None) is None
    assert build_method("opf", "none") is None


def test_get_method_spec_raises_clear_error_for_unknown_method():
    try:
        get_method_spec("economic", "missing")
    except ValueError as exc:
        assert "Unknown economic method 'missing'" in str(exc)
        assert "bfsa_dlmp" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_registry_builds_dummy_dispatch_estimator():
    case = PowerFlowCase(
        buses=[BusData(bus_id=1, bus_type=3), BusData(bus_id=2)],
        branches=[BranchData(branch_id=0, from_bus=1, to_bus=2, r=0.01, x=0.03)],
        generators=[
            GeneratorData(gen_id=0, bus_id=1, p_max=0.3, q_max=0.2, c_p=5.0, c_q=1.0),
            GeneratorData(gen_id=1, bus_id=2, p_max=0.5, q_max=0.2, c_p=8.0, c_q=2.0),
        ],
        loads=[LoadData(bus_id=2, p_d=0.6, q_d=0.3)],
        root_bus=1,
        base_mva=100.0,
    )
    estimator = build_method("dispatch", "dummy_merit_order")

    dispatch = estimator.estimate(case)

    assert isinstance(dispatch, DispatchResult)
    assert dispatch.generator_output[0] == (0.3, 0.2)
    assert dispatch.generator_output[1] == pytest.approx((0.3, 0.1))
    assert dispatch.objective == 540.0


def test_registry_builds_bfsa_physical_estimator():
    case = PowerFlowCase(
        buses=[BusData(bus_id=1, bus_type=3), BusData(bus_id=2)],
        branches=[BranchData(branch_id=0, from_bus=2, to_bus=1, r=0.01, x=0.03)],
        generators=[GeneratorData(gen_id=0, bus_id=1, p_max=1.0, q_max=1.0, c_p=10.0, c_q=1.0)],
        loads=[LoadData(bus_id=2, p_d=0.2, q_d=0.05)],
        root_bus=1,
    )
    dispatch = build_method("dispatch", "dummy_merit_order").estimate(case)
    estimator = build_method("physical", "bfsa", {"max_iterations": 5, "tol": 1e-6})

    physical = estimator.estimate(case, dispatch=dispatch)

    assert isinstance(physical, PhysicalStateResult)
    assert set(physical.voltages) == {1, 2}
    assert (1, 2) in physical.flows


def test_registry_builds_bfsa_dlmp_economic_estimator():
    case = PowerFlowCase(
        buses=[BusData(bus_id=1, bus_type=3), BusData(bus_id=2)],
        branches=[BranchData(branch_id=0, from_bus=2, to_bus=1, r=0.01, x=0.03)],
        generators=[GeneratorData(gen_id=0, bus_id=1, p_max=1.0, q_max=1.0, c_p=10.0, c_q=1.0)],
        loads=[LoadData(bus_id=2, p_d=0.2, q_d=0.05)],
        root_bus=1,
    )
    dispatch = build_method("dispatch", "dummy_merit_order").estimate(case)
    physical = build_method("physical", "bfsa", {"max_iterations": 5, "tol": 1e-6}).estimate(
        case,
        dispatch=dispatch,
    )
    estimator = build_method("economic", "bfsa_dlmp")

    economic = estimator.estimate(case, physical, dispatch=dispatch)

    assert isinstance(economic, EconomicStateResult)
    assert set(economic.lambda_p) == {1, 2}
    assert set(economic.lambda_q) == {1, 2}
