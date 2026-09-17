"""Regression checks for the publication review's numerical and API fixes."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from libs.shared import (
    BranchData, BusData, GeneratorData, LoadData, PowerFlowCase,
    load_case_from_folder, orient_radial_network,
)
from libs.shared.io import build_case_loader_from_folder
from libs.methods import EstimatorPipelineSolver


@pytest.fixture
def radial_case():
    return PowerFlowCase(
        buses=[BusData(1, bus_type=3), BusData(2)],
        branches=[BranchData(0, 1, 2, .01, .02)],
        generators=[GeneratorData(0, 1, p_max=2, q_max=2, c_p=20, c_q=2)],
        loads=[LoadData(2, .2, .05)], root_bus=1, base_mva=10,
    )


@pytest.fixture
def case_folder(tmp_path):
    (tmp_path / 'mpc_base_mva').write_text('10\n')
    pd.DataFrame([
        dict(bus=1, type=3, Pd=0, Qd=0, Vmin=.9, Vmax=1.1),
        dict(bus=2, type=1, Pd=0, Qd=0, Vmin=.9, Vmax=1.1),
    ]).to_csv(tmp_path / 'mpc_bus.csv', index=False)
    pd.DataFrame([dict(fbus=2, tbus=1, r=.01, x=.02)]).to_csv(tmp_path / 'mpc_branch.csv', index=False)
    pd.DataFrame({'2': [2., 3.]}, index=['t0', 't1']).to_csv(tmp_path / 'p_load.csv')
    pd.DataFrame({'2': [.5, .75]}, index=['t0', 't1']).to_csv(tmp_path / 'q_load.csv')
    pd.DataFrame([dict(bus=1, pmax=2, qmax=2, qmin=-2, cp=20, cq=2, inflow_profile='pv')]).to_excel(tmp_path / 'generators.xlsx', index=False)
    pd.DataFrame({'pv': [1., 1.2]}, index=['t0', 't1']).to_csv(tmp_path / 'profiles.csv')
    return tmp_path


def test_loaders_agree_and_snapshots_are_independent(case_folder):
    at = build_case_loader_from_folder(case_folder, load_interpretation_mode='absolute', override_voltage_bounds=True)
    first = at(0)
    second = at(1)
    direct = load_case_from_folder(case_folder, load_index=1, load_interpretation_mode='absolute', override_voltage_bounds=True)
    assert second == direct
    assert second.loads[0].p_d == pytest.approx(.3)
    assert second.buses[1].v_min == pytest.approx(.81)
    assert second.generators[0].p_max_available == pytest.approx(2.4)
    orient_radial_network(first)
    first.buses[0].v_max = 99
    assert second.branches[0].from_bus == 2
    assert second.buses[0].v_max == pytest.approx(1.21)
    assert first.branches[0].original_from_bus == 2


@pytest.mark.parametrize('invalid', [float('inf'), float('nan'), 'bad'])
def test_invalid_loads_rejected_by_both_loaders(case_folder, invalid):
    pd.DataFrame({'2': [2., invalid]}, index=['t0', 't1']).to_csv(case_folder / 'p_load.csv')
    for loader in (load_case_from_folder, build_case_loader_from_folder):
        with pytest.raises(ValueError, match='finite numeric'):
            loader(case_folder)


def test_load_timestamps_must_align(case_folder):
    pd.DataFrame({'2': [.5, .75]}, index=['t1', 't0']).to_csv(case_folder / 'q_load.csv')
    with pytest.raises(ValueError, match='ordered timestep labels'):
        load_case_from_folder(case_folder)


def test_generator_profiles_support_ordinal_selection(case_folder):
    pd.DataFrame({'pv': [.5, .75]}, index=['old0', 'old1']).to_csv(case_folder / 'profiles.csv')
    assert load_case_from_folder(case_folder, load_index=1).generators[0].p_max_available == 1.5


def test_negative_index_rejected(case_folder):
    with pytest.raises(ValueError, match='non-negative'):
        load_case_from_folder(case_folder, load_index=-1)


def test_parallel_branches_and_unknown_endpoints_rejected(radial_case):
    radial_case.branches.append(replace(radial_case.branches[0], branch_id=1))
    with pytest.raises(ValueError, match='Parallel'):
        orient_radial_network(radial_case)
    radial_case.branches[-1].to_bus = 99
    with pytest.raises(ValueError, match='endpoints'):
        orient_radial_network(radial_case)


def test_economic_mcp_options_apply_and_timing_is_complete(radial_case):
    solver = EstimatorPipelineSolver(economic_options={
        'merit_order_mode': 'mcp', 'mcp_active_price': 45, 'mcp_reactive_price': 4,
    })
    result = solver.solve(radial_case)
    assert result.duals_p[1] == pytest.approx(45)
    assert result.duals_q[1] == pytest.approx(4)
    assert result.solve_time > 0


def test_bfsa_plot_flags_invoke_diagnostic_plots(radial_case, monkeypatch):
    from libs.methods.physical.bfsa import BFSAElectricalSolver
    calls = []
    monkeypatch.setattr(BFSAElectricalSolver, '_plot_bus_net_demand', staticmethod(lambda *args: calls.append('net')))
    monkeypatch.setattr(BFSAElectricalSolver, '_plot_merit_order_component', staticmethod(lambda *args: calls.append('merit')))
    BFSAElectricalSolver().estimate(radial_case, show_net_bus_plot=True, show_merit_order_plot=True)
    assert calls == ['net', 'merit', 'merit']


def test_socp_omits_inactive_lines_and_honors_profile_qmin(radial_case):
    pytest.importorskip('pyomo')
    from libs.methods.opf.socp.model import RadialSOCPModel
    radial_case.branches.append(BranchData(1, 1, 2, .1, .2, status=0))
    radial_case.generators[0].q_min_available = -.1
    model = RadialSOCPModel(orient_radial_network(radial_case))
    built = model.build()
    assert list(built.BRANCH) == [0]
    assert built.q_g_min[0].value == -.1
    radial_case.generators[0].q_min_available = -.2
    assert model.update_case(radial_case).q_g_min[0].value == -.2


def test_missing_duals_remain_missing(radial_case):
    pytest.importorskip('pyomo')
    from libs.methods.opf.socp.model import RadialSOCPModel
    from libs.methods.opf.socp.results import extract_opf_result
    import math
    model = RadialSOCPModel(orient_radial_network(radial_case)).build()
    result = extract_opf_result(model, radial_case)
    assert all(math.isnan(v) for v in result.duals_p.values())


def test_failed_solver_does_not_extract_stale_solution(radial_case, monkeypatch):
    pyo = pytest.importorskip('pyomo.environ')
    from libs.methods.opf.socp.solver import SOCPSolver
    backend = SimpleNamespace(options={}, solve=lambda *a, **k: SimpleNamespace(
        solver=SimpleNamespace(termination_condition=pyo.TerminationCondition.maxTimeLimit)))
    monkeypatch.setattr(pyo, 'SolverFactory', lambda name: backend)
    with pytest.raises(ValueError, match='no accepted optimal solution'):
        SOCPSolver().solve(radial_case)


def test_full_kkt_has_small_stationarity_residual(radial_case):
    from libs.methods.economic.full_KKT import compute_radial_dlmp_kkt
    result = compute_radial_dlmp_kkt(radial_case, force_inactive_constraints=True)
    assert result.kkt_result.residual_norm < 1e-7
    assert result.kkt_result.dlmp_p[1] == pytest.approx(result.lambda_p_seed)


def test_range_reuses_same_timestep_warm_start(case_folder):
    from libs.shared.multi_timestep_solver import MultiTimestepSolver
    class Estimator:
        calls = 0
        def solve(self, case):
            self.calls += 1
            return EstimatorPipelineSolver().solve(case)
    class Reference:
        allow_load_shedding = False
        def solve(self, case, warm_start=None):
            assert warm_start is not None
            return warm_start
    estimator = Estimator()
    runner = MultiTimestepSolver(case_folder, case_folder / 'range.db',
        {'load_interpretation_mode': 'absolute'}, {'reference': Reference(), 'estimator': estimator})
    result = runner.run_range(0, 1, verbose=False, warm_start_mode='Estimator t')
    assert estimator.calls == 2
    assert len(result['solver_outputs']['reference']) == 2


def test_bfsa_ignores_inactive_branch_parameters(radial_case):
    from copy import deepcopy
    reference = EstimatorPipelineSolver().solve(deepcopy(radial_case))
    radial_case.branches.append(BranchData(1, 1, 2, 10, 20, status=0))
    actual = EstimatorPipelineSolver().solve(radial_case)
    assert actual.voltages == reference.voltages
    assert actual.duals_p == reference.duals_p


def test_database_stores_losses_curtailment_and_convergence(radial_case, tmp_path):
    import sqlite3
    from libs.shared.timestep_database import TimestepDatabase
    result = EstimatorPipelineSolver().solve(radial_case)
    result.convergence_info['p_curtailment_by_bus'] = {2: .03}
    database = TimestepDatabase(str(tmp_path / 'physical.db'))
    database.create_tables(radial_case)
    database.store_timestep_result(0, 'test', 'bfsa', result, radial_case)
    with sqlite3.connect(database.filename) as con:
        assert con.execute('SELECT load_shed FROM Res_Buses WHERE bus_id=2').fetchone()[0] == .03
        assert con.execute('SELECT flow_loss FROM Res_Branches').fetchone()[0] > 0
        assert con.execute('SELECT convergence_status FROM Res_Timesteps').fetchone()[0] == 'converged'


def test_userguide_python_examples(tmp_path):
    """Run every guide Python block except the licensed backend invocation."""
    import re
    from libs.methods.registry import METHOD_REGISTRY
    source = (Path(__file__).resolve().parents[1] / 'docs/documentation.tex').read_text(encoding='utf-8')
    blocks = re.findall(r'\\begin\{lstlisting\}\[language=Python\](.*?)\\end\{lstlisting\}', source, re.S)
    assert len(blocks) == 6
    namespace = {}
    original = dict(METHOD_REGISTRY['economic'])
    try:
        for block in blocks:
            block = block.split('# Requires an installed and licensed solver:')[0]
            block = block.replace('results/tutorial', tmp_path.as_posix())
            exec(compile(block, 'documentation.tex', 'exec'), namespace)
    finally:
        METHOD_REGISTRY['economic'].clear()
        METHOD_REGISTRY['economic'].update(original)
    assert (tmp_path / 'study.db').is_file()
    assert (tmp_path / 'prices.html').is_file()
