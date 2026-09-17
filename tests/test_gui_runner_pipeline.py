from pathlib import Path
import sys

sys.path.append(str(Path.cwd()))

from libs.shared import (
    BranchData,
    BusData,
    DispatchResult,
    EconomicStateResult,
    GeneratorData,
    LoadData,
    OPFResult,
    OptimizationResult,
    PhysicalStateResult,
    PowerFlowCase,
)
from libs.shared.gui_method_config import MODE_BFSA, methods_config_from_legacy_job
from libs.shared.gui_runner import _estimator_pipeline_result_as_opf_result, _optimization_result_to_opf_result
from libs.methods import EstimatorPipelineSolver


def test_estimator_pipeline_result_as_opf_result_uses_method_registry():
    case = PowerFlowCase(
        buses=[BusData(bus_id=1, bus_type=3), BusData(bus_id=2)],
        branches=[BranchData(branch_id=0, from_bus=2, to_bus=1, r=0.01, x=0.03)],
        generators=[GeneratorData(gen_id=0, bus_id=1, p_max=1.0, q_max=1.0, c_p=10.0, c_q=1.0)],
        loads=[LoadData(bus_id=2, p_d=0.2, q_d=0.05)],
        root_bus=1,
    )
    job = {
        "mode": MODE_BFSA,
        "bfsa_max_iter": 5,
        "bfsa_tol": 1e-6,
        "bfsa_convergence_check": "both",
        "bfsa_merit_order_mode": "dummy",
        "bfsa_mcp_active_price": None,
        "bfsa_mcp_reactive_price": None,
    }
    job["methods"] = methods_config_from_legacy_job(job)

    result = _estimator_pipeline_result_as_opf_result(case, job)

    assert isinstance(result, OPFResult)
    assert result.solver_name == "EstimatorPipeline"
    assert set(result.voltages) == {1, 2}
    assert set(result.duals_p) == {1, 2}
    assert result.convergence_info["dispatch_method"] == "DUMMY-MERIT-ORDER"
    assert result.convergence_info["physical_method"] == "BFSA-ELECTRICAL"
    assert result.convergence_info["economic_method"] == "BFSA-DLMP"


def test_estimator_pipeline_solver_adapter_returns_opf_result():
    case = PowerFlowCase(
        buses=[BusData(bus_id=1, bus_type=3), BusData(bus_id=2)],
        branches=[BranchData(branch_id=0, from_bus=2, to_bus=1, r=0.01, x=0.03)],
        generators=[GeneratorData(gen_id=0, bus_id=1, p_max=1.0, q_max=1.0, c_p=10.0, c_q=1.0)],
        loads=[LoadData(bus_id=2, p_d=0.2, q_d=0.05)],
        root_bus=1,
    )
    solver = EstimatorPipelineSolver(physical_options={"max_iterations": 5, "tol": 1e-6})

    result = solver.solve(case)

    assert isinstance(result, OPFResult)
    assert result.solver_name == "EstimatorPipeline"
    assert set(result.voltages) == {1, 2}
    assert set(result.duals_p) == {1, 2}


def test_optimization_result_to_opf_result_preserves_plot_fields():
    standard = OptimizationResult(
        physical_state=PhysicalStateResult(
            voltages={1: 1.0, 2: 0.99},
            flows={(1, 2): (0.2, 0.05)},
            branch_currents={(1, 2): 0.04},
        ),
        dispatch=DispatchResult(generator_output={0: (0.2, 0.05)}, objective=12.0),
        economic_state=EconomicStateResult(lambda_p={1: 10.0, 2: 10.5}, lambda_q={1: 1.0, 2: 1.2}),
        diagnostics={"solver_stage": "test"},
        method_name="standard",
        solve_time=0.25,
    )

    legacy = _optimization_result_to_opf_result(standard)

    assert isinstance(legacy, OPFResult)
    assert legacy.voltages == {1: 1.0, 2: 0.99}
    assert legacy.flows == {(1, 2): (0.2, 0.05)}
    assert legacy.duals_p == {1: 10.0, 2: 10.5}
    assert legacy.duals_q == {1: 1.0, 2: 1.2}
    assert legacy.generator_output == {0: (0.2, 0.05)}
    assert legacy.cost == 12.0
    assert legacy.solver_name == "standard"
