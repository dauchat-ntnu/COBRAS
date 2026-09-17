"""Tests for state-object comparison utilities."""

import pytest

from comparison import (
    DispatchComparison,
    EconomicStateComparison,
    OptimizationComparison,
    PhysicalStateComparison,
)
from libs.shared import (
    DispatchResult,
    EconomicStateResult,
    OptimizationResult,
    PhysicalStateResult,
)


def test_dispatch_comparison_compares_generator_output_and_objective():
    candidate = DispatchResult(
        generator_output={1: (1.1, 0.45), 2: (0.4, 0.2)},
        objective=110.0,
    )
    reference = DispatchResult(
        generator_output={1: (1.0, 0.5), 2: (0.5, 0.2)},
        objective=100.0,
    )

    comparison = DispatchComparison(candidate, reference)

    assert comparison.deviations["p_g"][1] == pytest.approx(10.0)
    assert comparison.deviations["p_g"][2] == pytest.approx(-20.0)
    assert comparison.deviations["q_g"][1] == pytest.approx(-10.0)
    assert comparison.deviations["objective"]["objective"] == pytest.approx(10.0)
    assert comparison.summary_stats()["p_g"]["count"] == 2


def test_physical_state_comparison_compares_voltages_flows_currents_and_losses():
    candidate = PhysicalStateResult(
        voltages={1: 1.0, 2: 0.99},
        flows={(1, 2): (1.1, 0.45)},
        branch_currents={(1, 2): 0.21},
        losses={(1, 2): (0.011, 0.018)},
    )
    reference = PhysicalStateResult(
        voltages={1: 1.0, 2: 1.0},
        flows={(1, 2): (1.0, 0.5)},
        branch_currents={(1, 2): 0.2},
        losses={(1, 2): (0.01, 0.02)},
    )

    comparison = PhysicalStateComparison(candidate, reference)

    assert comparison.deviations["voltage"][2] == pytest.approx(-1.0)
    assert comparison.deviations["flow_p"][(1, 2)] == pytest.approx(10.0)
    assert comparison.deviations["flow_q"][(1, 2)] == pytest.approx(-10.0)
    assert comparison.deviations["branch_current"][(1, 2)] == pytest.approx(5.0)
    assert comparison.deviations["loss_p"][(1, 2)] == pytest.approx(10.0)
    assert comparison.deviations["loss_q"][(1, 2)] == pytest.approx(-10.0)


def test_economic_state_comparison_compares_lambda_maps():
    candidate = EconomicStateResult(
        lambda_p={1: 21.0, 2: 22.0},
        lambda_q={1: 4.5, 2: 5.5},
    )
    reference = EconomicStateResult(
        lambda_p={1: 20.0, 2: 20.0},
        lambda_q={1: 5.0, 2: 5.0},
    )

    comparison = EconomicStateComparison(candidate, reference)

    assert comparison.deviations["lambda_p"][1] == pytest.approx(5.0)
    assert comparison.deviations["lambda_p"][2] == pytest.approx(10.0)
    assert comparison.deviations["lambda_q"][1] == pytest.approx(-10.0)
    assert comparison.deviations["lambda_q"][2] == pytest.approx(10.0)


def test_optimization_comparison_delegates_to_component_comparisons():
    candidate = OptimizationResult(
        dispatch=DispatchResult(generator_output={1: (1.1, 0.45)}, objective=110.0),
        physical_state=PhysicalStateResult(
            voltages={1: 1.0},
            flows={(1, 2): (1.1, 0.45)},
        ),
        economic_state=EconomicStateResult(lambda_p={1: 21.0}, lambda_q={1: 4.5}),
    )
    reference = OptimizationResult(
        dispatch=DispatchResult(generator_output={1: (1.0, 0.5)}, objective=100.0),
        physical_state=PhysicalStateResult(
            voltages={1: 1.0},
            flows={(1, 2): (1.0, 0.5)},
        ),
        economic_state=EconomicStateResult(lambda_p={1: 20.0}, lambda_q={1: 5.0}),
    )

    comparison = OptimizationComparison(candidate, reference)
    summary = comparison.summary_stats()

    assert comparison.dispatch.deviations["objective"]["objective"] == pytest.approx(10.0)
    assert comparison.physical.deviations["flow_p"][(1, 2)] == pytest.approx(10.0)
    assert comparison.economic.deviations["lambda_p"][1] == pytest.approx(5.0)
    assert set(summary) == {"dispatch", "physical", "economic"}


def test_optimization_comparison_allows_missing_economic_state():
    candidate = OptimizationResult(
        dispatch=DispatchResult(generator_output={1: (1.0, 0.5)}, objective=100.0),
        physical_state=PhysicalStateResult(voltages={1: 1.0}, flows={}),
        economic_state=None,
    )
    reference = OptimizationResult(
        dispatch=DispatchResult(generator_output={1: (1.0, 0.5)}, objective=100.0),
        physical_state=PhysicalStateResult(voltages={1: 1.0}, flows={}),
        economic_state=None,
    )

    comparison = OptimizationComparison(candidate, reference)

    assert comparison.economic is None
    assert set(comparison.summary_stats()) == {"dispatch", "physical"}
