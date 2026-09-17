from pathlib import Path
import sys

import pytest

sys.path.append(str(Path.cwd()))

from examples.run_speed_analysis import (
    _comparison_frame,
    _result_cost,
    _result_diagnostics,
    _solve_case,
)
from libs.shared import DispatchResult, OptimizationResult, PhysicalStateResult


class _LegacyResult:
    convergence_info = {"converged": True, "timing": {"total": 0.5}}
    cost = 12.0
    solve_time = 0.2


class _LegacySolver:
    def solve(self, case, orient=True):
        self.orient = orient
        return _LegacyResult()


class _EstimatorSolver:
    def solve(self, case):
        self.case = case
        return _LegacyResult()


def test_solve_case_passes_orient_false_only_when_supported():
    legacy_solver = _LegacySolver()
    estimator_solver = _EstimatorSolver()

    assert isinstance(_solve_case(legacy_solver, object()), _LegacyResult)
    assert legacy_solver.orient is False

    assert isinstance(_solve_case(estimator_solver, object()), _LegacyResult)
    assert estimator_solver.case is not None


def test_result_helpers_support_refactored_optimization_result():
    result = OptimizationResult(
        physical_state=PhysicalStateResult(voltages={}, flows={}),
        dispatch=DispatchResult(objective=42.0),
        diagnostics={"converged": True, "iterations": 3},
        solve_time=0.1,
    )

    assert _result_diagnostics(result) == {"converged": True, "iterations": 3}
    assert _result_cost(result) == 42.0


def test_comparison_frame_uses_wall_time_for_bfsa_ratio():
    frame = _comparison_frame(
        [
            {
                "solver": "BFSA",
                "horizon_start": 1,
                "horizon_end": 1,
                "horizon_steps": 1,
                "solve_time_s": 1.0,
                "solve_wall_time_s": 2.0,
            },
            {
                "solver": "SOCP",
                "horizon_start": 1,
                "horizon_end": 1,
                "horizon_steps": 1,
                "solve_time_s": 3.0,
                "solve_wall_time_s": 8.0,
            },
        ]
    )

    socp = frame.loc[frame["solver"] == "SOCP"].iloc[0]
    assert socp["runtime_ratio_vs_bfsa"] == pytest.approx(4.0)
