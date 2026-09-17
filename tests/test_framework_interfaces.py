"""Tests for the framework-level result and interface layer."""

from pathlib import Path
import sys

import pytest

sys.path.append(str(Path.cwd()))


from libs.shared import (
    BranchDuals,
    BranchData,
    BusData,
    BusDuals,
    DispatchResult,
    EconomicStateResult,
    GeneratorDuals,
    GeneratorData,
    LoadData,
    MethodCapabilities,
    OPFResult,
    OptimizationResult,
    PowerFlowCase,
    PhysicalStateResult,
    economic_state_from_opf_result,
    optimization_result_from_opf_result,
    physical_state_from_opf_result,
)


def test_new_result_types_are_publicly_importable():
    physical = PhysicalStateResult(
        voltages={1: 1.0},
        flows={(1, 2): (0.4, 0.1)},
        branch_currents={(1, 2): 0.17},
    )
    dispatch = DispatchResult(
        generator_output={7: (0.4, 0.1)},
        objective=12.0,
    )
    economic = EconomicStateResult(
        lambda_p={1: 10.0},
        lambda_q={1: 2.0},
        bus_duals={1: BusDuals(lambda_p=10.0, lambda_q=2.0)},
        branch_duals={3: BranchDuals(mu_int=0.5)},
        generator_duals={7: GeneratorDuals(mu_pmax=1.2)},
    )
    optimization = OptimizationResult(
        physical_state=physical,
        dispatch=dispatch,
        economic_state=economic,
    )

    assert optimization.physical_state.voltages[1] == 1.0
    assert optimization.economic_state.lambda_p[1] == 10.0
    assert optimization.generator_output[7] == (0.4, 0.1)


def test_method_capabilities_are_name_independent():
    capabilities = MethodCapabilities(
        produces_physical_state=True,
        produces_economic_state=False,
        produces_dispatch=False,
        supports_warm_start=False,
        requires_physical_state=False,
    )

    assert capabilities.produces_physical_state is True
    assert capabilities.requires_radial_network is True


def test_opf_result_adapters_preserve_standard_states():
    result = OPFResult(
        voltages={1: 1.0, 2: 0.98},
        flows={(1, 2): (0.4, 0.1)},
        duals_p={1: 10.0, 2: 11.0},
        duals_q={1: 2.0, 2: 2.2},
        generator_output={7: (0.4, 0.1)},
        cost=12.0,
        convergence_info={"line_state_bfsa": {(1, 2): {"ploss": 0.01, "qloss": 0.02}}},
        branch_currents={(1, 2): 0.17},
        solver_name="example",
        solve_time=0.2,
        socp_duals=[
            {"dual_name": "mu_v_lv_plus", "entity_type": "bus", "entity_id": 2, "value": 0.3},
            {"dual_name": "mu_l", "entity_type": "branch", "entity_id": 5, "value": 0.4},
            {"dual_name": "mu_g_p_plus", "entity_type": "generator", "entity_id": 7, "value": 0.5},
        ],
    )

    physical = physical_state_from_opf_result(result)
    economic = economic_state_from_opf_result(result)
    optimization = optimization_result_from_opf_result(result)

    assert physical.voltages == {1: 1.0, 2: 0.98}
    assert physical.losses[(1, 2)] == (0.01, 0.02)
    assert economic.lambda_p[2] == 11.0
    assert economic.bus_duals[2].mu_v_up == 0.3
    assert economic.branch_duals[5].mu_ell == 0.4
    assert economic.generator_duals[7].mu_pmax == 0.5
    assert optimization.objective == 12.0
    assert optimization.dispatch.objective == 12.0
    assert optimization.dispatch.generator_output == {7: (0.4, 0.1)}
    assert optimization.method_name == "example"


def test_bfsa_electrical_solver_estimate_returns_physical_state():
    from libs.methods.physical.bfsa import BFSAElectricalSolver
    from libs.methods.dispatch import DummyMeritOrderDispatchEstimator

    case = PowerFlowCase(
        buses=[BusData(bus_id=1, bus_type=3), BusData(bus_id=2)],
        branches=[BranchData(branch_id=0, from_bus=2, to_bus=1, r=0.01, x=0.03)],
        generators=[GeneratorData(gen_id=0, bus_id=1, p_max=1.0, q_max=1.0)],
        loads=[LoadData(bus_id=2, p_d=0.2, q_d=0.05)],
        root_bus=1,
    )

    estimator = BFSAElectricalSolver(config={"max_iterations": 5, "tol": 1e-6})
    dispatch = DummyMeritOrderDispatchEstimator().estimate(case)
    physical = estimator.estimate(case, dispatch=dispatch)

    assert estimator.capabilities.produces_physical_state is True
    assert estimator.capabilities.produces_economic_state is False
    assert isinstance(physical, PhysicalStateResult)
    assert set(physical.voltages) == {1, 2}
    assert (1, 2) in physical.flows
    assert physical.diagnostics["solver_stage"] == "electrical"


def test_dummy_merit_order_dispatch_estimator_returns_dispatch_result():
    from libs.methods.dispatch import DummyMeritOrderDispatchEstimator

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

    dispatch = DummyMeritOrderDispatchEstimator().estimate(case)

    assert isinstance(dispatch, DispatchResult)
    assert dispatch.generator_output[0] == (0.3, 0.2)
    assert dispatch.generator_output[1] == pytest.approx((0.3, 0.1))
    assert dispatch.market_clearing["lambda_p_clear"] == 8.0
    assert dispatch.market_clearing["lambda_q_clear"] == 2.0
    assert dispatch.market_clearing["p_gen_by_bus"] == {1: 0.3, 2: 0.3}
    assert dispatch.market_clearing["q_gen_by_bus"] == pytest.approx({1: 0.2, 2: 0.1})
    assert dispatch.objective == 540.0


def test_estimators_accept_optional_feedback_arguments():
    from libs.methods.economic.bfsa_dlmp import BFSADLMPSolver
    from libs.methods.dispatch import DummyMeritOrderDispatchEstimator
    from libs.methods.physical.bfsa import BFSAElectricalSolver

    case = PowerFlowCase(
        buses=[BusData(bus_id=1, bus_type=3), BusData(bus_id=2)],
        branches=[BranchData(branch_id=0, from_bus=2, to_bus=1, r=0.01, x=0.03)],
        generators=[GeneratorData(gen_id=0, bus_id=1, p_max=1.0, q_max=1.0, c_p=10.0, c_q=1.0)],
        loads=[LoadData(bus_id=2, p_d=0.2, q_d=0.05)],
        root_bus=1,
    )

    dispatch_estimator = DummyMeritOrderDispatchEstimator()
    initial_dispatch = dispatch_estimator.estimate(case)
    feedback_dispatch = dispatch_estimator.estimate(
        case,
        previous_dispatch=initial_dispatch,
    )
    physical = BFSAElectricalSolver(config={"max_iterations": 5, "tol": 1e-6}).estimate(
        case,
        dispatch=feedback_dispatch,
        previous_physical_state=None,
    )
    economic = BFSADLMPSolver().estimate(
        case,
        physical,
        dispatch=feedback_dispatch,
        previous_economic_state=None,
    )

    assert feedback_dispatch.generator_output == initial_dispatch.generator_output
    assert isinstance(physical, PhysicalStateResult)
    assert isinstance(economic, EconomicStateResult)


def test_bfsa_dlmp_solver_estimate_consumes_physical_state():
    from libs.methods.economic.bfsa_dlmp import BFSADLMPSolver
    from libs.methods.physical.bfsa import BFSAElectricalSolver

    case = PowerFlowCase(
        buses=[BusData(bus_id=1, bus_type=3), BusData(bus_id=2)],
        branches=[BranchData(branch_id=0, from_bus=2, to_bus=1, r=0.01, x=0.03)],
        generators=[GeneratorData(gen_id=0, bus_id=1, p_max=1.0, q_max=1.0, c_p=10.0, c_q=1.0)],
        loads=[LoadData(bus_id=2, p_d=0.2, q_d=0.05)],
        root_bus=1,
    )

    physical = BFSAElectricalSolver(config={"max_iterations": 5, "tol": 1e-6}).estimate(case)
    economic = BFSADLMPSolver().estimate(case, physical)

    assert isinstance(economic, EconomicStateResult)
    assert set(economic.lambda_p) == {1, 2}
    assert set(economic.lambda_q) == {1, 2}
    assert economic.method_name == "BFSA-DLMP"


def test_socp_solver_exposes_framework_capabilities():
    pytest.importorskip("pyomo")
    from libs.methods.opf.socp import SOCPSolver

    solver = SOCPSolver(solver_name="test-solver")

    assert solver.name == "SOCP-test-solver"
    assert solver.capabilities.produces_physical_state is True
    assert solver.capabilities.produces_economic_state is True
    assert solver.capabilities.produces_dispatch is True
    assert solver.capabilities.supports_warm_start is True


def test_socp_legacy_warm_start_accepts_physical_state():
    pytest.importorskip("pyomo")
    from libs.methods.opf.socp import SOCPSolver

    physical = PhysicalStateResult(
        voltages={1: 1.0, 2: 0.98},
        flows={(1, 2): (0.4, 0.1)},
        branch_currents={(1, 2): 0.17},
        diagnostics={"generator_output": {7: (0.4, 0.1)}, "cost": 12.0},
        method_name="BFSA-ELECTRICAL",
        solve_time=0.1,
    )

    warm_start = SOCPSolver._legacy_warm_start(physical)

    assert isinstance(warm_start, OPFResult)
    assert warm_start.voltages == physical.voltages
    assert warm_start.flows == physical.flows
    assert warm_start.generator_output == {7: (0.4, 0.1)}
    assert warm_start.branch_currents == {(1, 2): 0.17}
    assert warm_start.cost == 12.0


def test_socp_solve_optimization_adapts_legacy_result(monkeypatch):
    pytest.importorskip("pyomo")
    from libs.methods.opf.socp import SOCPSolver

    solver = SOCPSolver(solver_name="mock")
    legacy_result = OPFResult(
        voltages={1: 1.0, 2: 0.98},
        flows={(1, 2): (0.4, 0.1)},
        duals_p={1: 10.0, 2: 11.0},
        duals_q={1: 2.0, 2: 2.2},
        generator_output={7: (0.4, 0.1)},
        cost=12.0,
        convergence_info={"termination_message": "optimal"},
        branch_currents={(1, 2): 0.17},
        solver_name="mock",
        solve_time=0.3,
    )

    def fake_solve(case, orient=True, warm_start=None):
        return legacy_result

    monkeypatch.setattr(solver, "solve", fake_solve)

    case = PowerFlowCase(
        buses=[BusData(bus_id=1, bus_type=3), BusData(bus_id=2)],
        branches=[BranchData(branch_id=0, from_bus=1, to_bus=2, r=0.01, x=0.03)],
        generators=[GeneratorData(gen_id=7, bus_id=1)],
        loads=[],
        root_bus=1,
    )

    result = solver.solve_optimization(case)

    assert isinstance(result, OptimizationResult)
    assert result.objective == 12.0
    assert result.method_name == "SOCP-mock"
    assert result.physical_state.voltages[2] == 0.98
    assert result.economic_state.lambda_p[2] == 11.0
    assert result.generator_output[7] == (0.4, 0.1)
