"""
Unit and Integration Tests for PowerSOC Libraries
"""

import pytest
from pathlib import Path
import pandas as pd
import networkx as nx
from libs.shared import (
    BusData, BranchData, LoadData, GeneratorData,
    PowerFlowCase, OPFResult,
    load_case_from_folder,
    orient_radial_network,
    validate_case,
)
from libs.shared.multi_timestep_solver import MultiTimestepSolver


class TestSharedLayer:
    """Tests for shared data structures."""
    
    def test_bus_data_creation(self):
        """Test BusData instantiation."""
        bus = BusData(bus_id=1, v_min=0.9, v_max=1.1)
        assert bus.bus_id == 1
        assert bus.v_min == 0.9
        assert bus.v_max == 1.1
    
    def test_branch_data_creation(self):
        """Test BranchData instantiation."""
        br = BranchData(branch_id=0, from_bus=1, to_bus=2, r=0.01, x=0.05, rateA=100.0)
        assert br.branch_id == 0
        assert br.from_bus == 1
        assert br.to_bus == 2
        assert br.s_max == 100.0
    
    def test_generator_data_creation(self):
        """Test GeneratorData instantiation."""
        gen = GeneratorData(gen_id=0, bus_id=1, p_max=100.0, q_max=50.0, c_p=30.0)
        assert gen.gen_id == 0
        assert gen.bus_id == 1
        assert gen.p_max == 100.0
        assert gen.c_p == 30.0

    def test_generator_available_pmax_defaults_to_static(self):
        """Generator should expose static p_max when no snapshot availability is set."""
        gen = GeneratorData(gen_id=0, bus_id=1, p_max=12.0)
        assert gen.get_p_max_available() == 12.0

    def test_generator_available_pmax_uses_snapshot_value(self):
        """Generator should expose snapshot-available p_max when provided."""
        gen = GeneratorData(gen_id=0, bus_id=1, p_max=12.0, p_max_available=4.5)
        assert gen.get_p_max_available() == 4.5

    def test_generator_available_qmax_defaults_to_static(self):
        """Generator should expose static q_max when no snapshot availability is set."""
        gen = GeneratorData(gen_id=0, bus_id=1, q_max=6.0)
        assert gen.get_q_max_available() == 6.0

    def test_generator_available_qmax_uses_snapshot_value(self):
        """Generator should expose snapshot-available q_max when provided."""
        gen = GeneratorData(gen_id=0, bus_id=1, q_max=6.0, q_max_available=2.5)
        assert gen.get_q_max_available() == 2.5
    
    def test_load_data_creation(self):
        """Test LoadData instantiation."""
        load = LoadData(bus_id=1, p_d=0.5, q_d=0.1)
        assert load.bus_id == 1
        assert load.p_d == 0.5
    
    def test_power_flow_case_validation(self):
        """Test case validation in constructor."""
        bus1 = BusData(bus_id=1)
        bus2 = BusData(bus_id=2)
        branch = BranchData(branch_id=0, from_bus=1, to_bus=2, r=0.01, x=0.05)
        
        case = PowerFlowCase(
            buses=[bus1, bus2],
            branches=[branch],
            generators=[],
            loads=[],
            root_bus=1,
        )
        
        assert case.root_bus == 1
        assert len(case.buses) == 2
        assert not case.incoming_branches  # Not yet oriented
    
    def test_ofd_result_creation(self):
        """Test OPFResult structure."""
        result = OPFResult(
            voltages={1: 1.0, 2: 0.95},
            flows={(1, 2): (0.5, 0.1)},
            duals_p={1: 25.0, 2: 30.0},
            duals_q={1: 5.0, 2: 6.0},
            generator_output={0: (1.0, 0.2)},
            cost=50.0,
        )
        
        assert result.cost == 50.0
        assert len(result.voltages) == 2
        assert result.voltages[1] == 1.0
        assert result.duals_p[1] == 25.0
    
    def test_ofd_result_to_dict(self):
        """Test OPFResult serialization."""
        result = OPFResult(
            voltages={1: 1.0},
            flows={},
            duals_p={1: 25.0},
            duals_q={1: 5.0},
            generator_output={},
            cost=50.0,
        )
        
        d = result.to_dict()
        assert isinstance(d, dict)
        assert d["cost"] == 50.0
        assert d["voltages"] == {1: 1.0}
    
    def test_orient_radial_network(self):
        """Test tree orientation."""
        bus1 = BusData(bus_id=1)
        bus2 = BusData(bus_id=2)
        bus3 = BusData(bus_id=3)
        
        branch1 = BranchData(branch_id=0, from_bus=1, to_bus=2, r=0.01, x=0.05)
        branch2 = BranchData(branch_id=1, from_bus=2, to_bus=3, r=0.02, x=0.06)
        
        case = PowerFlowCase(
            buses=[bus1, bus2, bus3],
            branches=[branch1, branch2],
            generators=[],
            loads=[],
            root_bus=1,
        )
        
        case = orient_radial_network(case)
        
        assert 0 in case.incoming_branches
        assert 1 in case.incoming_branches
        assert case.incoming_branches[0] == 2
        assert case.incoming_branches[1] == 3
        assert len(case.tree_edges) == 2

    def test_orient_radial_network_rewrites_reversed_branch_endpoints(self):
        """Orientation should canonicalize branch endpoints from root to leaves."""
        bus1 = BusData(bus_id=1)
        bus2 = BusData(bus_id=2)
        bus3 = BusData(bus_id=3)

        branch1 = BranchData(branch_id=0, from_bus=2, to_bus=1, r=0.01, x=0.05)
        branch2 = BranchData(branch_id=1, from_bus=3, to_bus=2, r=0.02, x=0.06)

        case = PowerFlowCase(
            buses=[bus1, bus2, bus3],
            branches=[branch1, branch2],
            generators=[],
            loads=[],
            root_bus=1,
        )

        case = orient_radial_network(case)

        assert [(br.branch_id, br.from_bus, br.to_bus) for br in case.branches] == [
            (0, 1, 2),
            (1, 2, 3),
        ]
        assert branch1.original_from_bus == 2
        assert branch1.original_to_bus == 1
        assert branch2.original_from_bus == 3
        assert branch2.original_to_bus == 2
        assert case.tree_edges == [(1, 2), (2, 3)]

    def test_orient_radial_network_preserves_original_branch_endpoints_on_reorientation(self):
        """Repeated orientation should not overwrite original input endpoint metadata."""
        bus1 = BusData(bus_id=1)
        bus2 = BusData(bus_id=2)
        branch = BranchData(branch_id=0, from_bus=2, to_bus=1, r=0.01, x=0.05)

        case = PowerFlowCase(
            buses=[bus1, bus2],
            branches=[branch],
            generators=[],
            loads=[],
            root_bus=1,
        )

        orient_radial_network(case)
        orient_radial_network(case)

        assert branch.from_bus == 1
        assert branch.to_bus == 2
        assert branch.original_from_bus == 2
        assert branch.original_to_bus == 1
    
    def test_orient_radial_network_not_connected(self):
        """Test that orient fails on disconnected network."""
        bus1 = BusData(bus_id=1)
        bus2 = BusData(bus_id=2)
        bus3 = BusData(bus_id=3)  # Isolated
        
        branch = BranchData(branch_id=0, from_bus=1, to_bus=2, r=0.01, x=0.05)
        
        case = PowerFlowCase(
            buses=[bus1, bus2, bus3],
            branches=[branch],
            generators=[],
            loads=[],
            root_bus=1,
        )
        
        with pytest.raises(ValueError, match="not connected"):
            orient_radial_network(case)
    
    def test_validate_case_success(self):
        """Test case validation succeeds for valid case."""
        bus1 = BusData(bus_id=1)
        bus2 = BusData(bus_id=2)
        gen = GeneratorData(gen_id=0, bus_id=1)
        load = LoadData(bus_id=2, p_d=0.5)
        branch = BranchData(branch_id=0, from_bus=1, to_bus=2, r=0.01, x=0.05)
        
        case = PowerFlowCase(
            buses=[bus1, bus2],
            branches=[branch],
            generators=[gen],
            loads=[load],
            root_bus=1,
        )
        
        # Should not raise
        validate_case(case)
    
    def test_validate_case_gen_bus_missing(self):
        """Test case validation fails when gen bus missing."""
        bus1 = BusData(bus_id=1)
        gen = GeneratorData(gen_id=0, bus_id=99)  # Non-existent bus
        
        case = PowerFlowCase(
            buses=[bus1],
            branches=[],
            generators=[gen],
            loads=[],
            root_bus=1,
        )
        
        with pytest.raises(ValueError, match="Generator buses"):
            validate_case(case)


class TestSOCPLibrary:
    """Tests for SOCP library."""

    def test_extract_opf_result_collects_named_socp_duals(self):
        """SOCP extraction should persist named duals using the documented convention."""
        pytest.importorskip("pyomo.environ")
        from libs.methods.opf.socp.results import extract_opf_result

        class Value:
            def __init__(self, value):
                self.value = value

        model = type("FakeModel", (), {})()
        model.BUS = [1, 2]
        model.BRANCH = [10]
        model.GEN = [5]
        model.v = {1: Value(1.0), 2: Value(0.98)}
        model.p = {10: Value(0.4)}
        model.q = {10: Value(0.1)}
        model.pg = {5: Value(0.4)}
        model.qg = {5: Value(0.1)}
        model.obj = Value(12.0)

        model.v_ref = object()
        model.v_upper = {2: object()}
        model.v_lower = {2: object()}
        model.p_balance = {1: object(), 2: object()}
        model.q_balance = {1: object(), 2: object()}
        model.voltage_drop = {10: object()}
        model.ell_upper = {10: object()}
        model.ell_lower = {10: object()}
        model.soc = {10: object()}
        model.pg_lower = {5: object()}
        model.pg_upper = {5: object()}
        model.qg_lower = {5: object()}
        model.qg_upper = {5: object()}
        model.dual = {
            model.v_ref: 1.0,
            model.v_upper[2]: 2.0,
            model.v_lower[2]: -3.0,
            model.p_balance[1]: 10.0,
            model.p_balance[2]: 11.0,
            model.q_balance[1]: 20.0,
            model.q_balance[2]: 21.0,
            model.voltage_drop[10]: 4.0,
            model.ell_upper[10]: 5.0,
            model.ell_lower[10]: -6.0,
            model.soc[10]: 7.0,
            model.pg_lower[5]: -8.0,
            model.pg_upper[5]: 9.0,
            model.qg_lower[5]: -10.0,
            model.qg_upper[5]: 12.0,
        }

        case = PowerFlowCase(
            buses=[BusData(bus_id=1), BusData(bus_id=2)],
            branches=[BranchData(branch_id=10, from_bus=1, to_bus=2, r=0.01, x=0.05)],
            generators=[GeneratorData(gen_id=5, bus_id=1)],
            loads=[],
            root_bus=1,
        )

        result = extract_opf_result(model, case)
        duals = {
            (row["dual_name"], row["entity_type"], row["entity_id"]): row["value"]
            for row in result.socp_duals
        }

        assert duals[("lambda_v_ref", "system", None)] == 1.0
        assert duals[("mu_v_lv_plus", "bus", 2)] == 2.0
        assert duals[("mu_v_lv_minus", "bus", 2)] == 3.0
        assert duals[("lambda_p_balance", "bus", 1)] == 10.0
        assert duals[("lambda_q_balance", "bus", 2)] == 21.0
        assert duals[("lambda_v", "branch", 10)] == 4.0
        assert duals[("mu_l", "branch", 10)] == 5.0
        assert duals[("mu_l_minus", "branch", 10)] == 6.0
        assert duals[("mu_int", "branch", 10)] == 7.0
        assert duals[("mu_g_p_minus", "generator", 5)] == 8.0
        assert duals[("mu_g_p_plus", "generator", 5)] == 9.0
        assert duals[("mu_g_q_minus", "generator", 5)] == 10.0
        assert duals[("mu_g_q_plus", "generator", 5)] == 12.0
    
    @pytest.mark.skipif(True, reason="Requires test data and MOSEK solver")
    def test_socp_solver_basic(self):
        """Test SOCP solver creates OPFResult."""
        from libs.methods.opf.socp import SOCPSolver
        
        # Create minimal test case
        bus1 = BusData(bus_id=1, v_min=0.95, v_max=1.05)
        bus2 = BusData(bus_id=2, v_min=0.95, v_max=1.05)
        
        br = BranchData(branch_id=0, from_bus=1, to_bus=2, r=0.01, x=0.05, rateA=100.0)
        gen = GeneratorData(gen_id=0, bus_id=1, p_max=100.0, q_max=50.0)
        load = LoadData(bus_id=2, p_d=0.5, q_d=0.1)
        
        case = PowerFlowCase(
            buses=[bus1, bus2],
            branches=[br],
            generators=[gen],
            loads=[load],
            root_bus=1,
        )
        case = orient_radial_network(case)
        
        solver = SOCPSolver(solver_name="mosek")
        result = solver.solve(case, orient=False)
        
        assert isinstance(result, OPFResult)
        assert result.cost is not None
        assert len(result.voltages) == 2


class TestBFSALibrary:
    """Tests for BFSA library."""
    
    def test_bfsa_solver_initialization(self):
        """Test estimator pipeline solver can be initialized."""
        from libs.methods import EstimatorPipelineSolver
        
        config = {
            "compute_losses": True,
            "tol": 1e-3,
            "max_iterations": 100,
        }
        solver = EstimatorPipelineSolver(physical_options=config)
        
        assert solver.physical_options["compute_losses"] is True
        assert solver.physical_options["max_iterations"] == 100
        assert solver.physical_options["convergence_check"] == "both"

    def test_bfsa_electrical_iterates_losses_until_tolerance(self):
        """Loss-enabled BFSA should allow repeated sweeps when tolerance requires it."""
        from libs.methods.physical.bfsa.algorithm import bfsa_electrical

        graph = nx.DiGraph()
        graph.add_edges_from([(1, 2), (2, 3)])
        branches = {
            (1, 2): BranchData(branch_id=1, from_bus=1, to_bus=2, r=0.02, x=0.04),
            (2, 3): BranchData(branch_id=2, from_bus=2, to_bus=3, r=0.03, x=0.05),
        }

        result = bfsa_electrical(
            radial_root=1,
            G=graph,
            p_bus={1: 0.0, 2: 0.4, 3: 0.3},
            q_bus={1: 0.0, 2: 0.1, 3: 0.08},
            line_lookup_undirected=branches,
            v_root_sq=1.0,
            config={
                "compute_losses": True,
                "tol": 1e-12,
                "max_iterations": 5,
                "convergence_check": "both",
            },
        )

        assert result["iterations"] > 1
        assert result["iterations"] <= 5
        assert result["convergence_check"] == "both"
        assert result["final_mismatch"] == max(
            result["voltage_mismatch"],
            result["current_mismatch"],
        )

    def test_bfsa_electrical_can_converge_on_current_only(self):
        """The convergence criterion can be switched to current-squared/loss feedback."""
        from libs.methods.physical.bfsa.algorithm import bfsa_electrical

        graph = nx.DiGraph()
        graph.add_edge(1, 2)
        branches = {
            (1, 2): BranchData(branch_id=1, from_bus=1, to_bus=2, r=0.01, x=0.03),
        }

        result = bfsa_electrical(
            radial_root=1,
            G=graph,
            p_bus={1: 0.0, 2: 0.5},
            q_bus={1: 0.0, 2: 0.2},
            line_lookup_undirected=branches,
            v_root_sq=1.0,
            config={
                "compute_losses": True,
                "tol": 1e-6,
                "max_iterations": 20,
                "convergence_check": "current",
            },
        )

        assert result["converged"] is True
        assert result["convergence_check"] == "current"
        assert result["final_mismatch"] == result["current_mismatch"]

    def test_bfsa_solver_uses_configured_mcp_seeds(self):
        """BFSA should use configured MCP prices when MCP mode is selected."""
        from libs.methods.physical.bfsa.solver import BFSASolver

        solver = BFSASolver(
            config={
                "merit_order_mode": "mcp",
                "mcp_active_price": 17.5,
                "mcp_reactive_price": 4.25,
            }
        )

        lambda_p, lambda_q, source = solver._resolve_price_propagation_seeds(
            {"marginal_price": 1.0},
            {"marginal_price": 2.0},
        )

        assert source == "mcp"
        assert lambda_p == 17.5
        assert lambda_q == 4.25
    
    @pytest.mark.skipif(True, reason="Requires test data")
    def test_bfsa_solver_basic(self):
        """Test BFSA solver returns OPFResult."""
        from libs.methods import EstimatorPipelineSolver
        
        # Create minimal radial case
        bus1 = BusData(bus_id=1)
        bus2 = BusData(bus_id=2)
        
        br = BranchData(branch_id=0, from_bus=1, to_bus=2, r=0.01, x=0.05)
        load = LoadData(bus_id=2, p_d=0.5, q_d=0.1)
        
        case = PowerFlowCase(
            buses=[bus1, bus2],
            branches=[br],
            generators=[],
            loads=[load],
            root_bus=1,
        )
        case = orient_radial_network(case)
        
        solver = EstimatorPipelineSolver()
        result = solver.solve(case)
        
        assert isinstance(result, OPFResult)
        assert len(result.voltages) == 2
        assert len(result.duals_p) == 2


class TestComparisonFramework:
    """Tests for comparison framework."""
    
    def test_solver_comparison_creation(self):
        """Test SolverComparison can be created."""
        from comparison import SolverComparison
        
        bus1 = BusData(bus_id=1)
        case = PowerFlowCase(
            buses=[bus1],
            branches=[],
            generators=[],
            loads=[],
            root_bus=1,
        )
        
        result1 = OPFResult(
            voltages={1: 1.0},
            flows={(1, 2): (1.0, 0.5)},
            duals_p={1: 25.0},
            duals_q={1: 5.0},
            generator_output={},
            cost=50.0,
        )
        
        result2 = OPFResult(
            voltages={1: 1.01},
            flows={(1, 2): (0.9, 0.45)},
            duals_p={1: 24.0},
            duals_q={1: 5.1},
            generator_output={},
            cost=51.0,
        )
        
        comparison = SolverComparison(case, result1, result2, names=("Solver1", "Solver2"))
        
        assert comparison.names == ("Solver1", "Solver2")
        assert "voltage_dev" in comparison.deviations
        assert comparison.deviations["flow_p_dev"][(1, 2)] == pytest.approx(11.111111111111109)
        assert comparison.deviations["flow_q_dev"][(1, 2)] == pytest.approx(11.111111111111109)
        stats = comparison.summary_stats()
        assert stats["flow_p"]["count"] == 1
        assert stats["flow_q"]["count"] == 1
        assert stats["objectives"]["Solver1"] == pytest.approx(50.0)
        assert stats["objectives"]["Solver2"] == pytest.approx(51.0)
        assert stats["objectives"]["gap"] == pytest.approx(1.0)

    def test_benchmark_summary_tables_include_flow_components(self):
        """Benchmark summary paths should use the declared metric list."""
        from comparison import Benchmark, SolverComparison

        bus1 = BusData(bus_id=1)
        case = PowerFlowCase(
            buses=[bus1],
            branches=[],
            generators=[],
            loads=[],
            root_bus=1,
        )

        result1 = OPFResult(
            voltages={1: 1.0},
            flows={(1, 2): (1.0, 0.5)},
            duals_p={1: 25.0},
            duals_q={1: 5.0},
            generator_output={},
            cost=50.0,
            solve_time=2.0,
        )
        result2 = OPFResult(
            voltages={1: 1.01},
            flows={(1, 2): (0.9, 0.45)},
            duals_p={1: 24.0},
            duals_q={1: 5.1},
            generator_output={},
            cost=51.0,
            solve_time=1.0,
        )

        benchmark = Benchmark(object(), object())
        benchmark.comparisons.append(SolverComparison(case, result1, result2))

        assert benchmark.metrics == ["voltage", "lmp_p", "lmp_q", "flow", "flow_p", "flow_q"]
        assert set(benchmark.generate_summary_table()["metric"]) == set(benchmark.metrics)
        error_df, _ = benchmark.generate_summary_tables_split()
        assert set(error_df["metric"]) == set(benchmark.metrics)
        assert set(benchmark.generate_aggregated_comparison_table()["metric"]) >= set(benchmark.metrics)
        aggregated_error_df, _ = benchmark.generate_aggregated_comparison_tables_split()
        assert set(aggregated_error_df["metric"]) == set(benchmark.metrics)


class TestGeneratorProfiles:
    """Tests for generator profile scaling from profiles.csv."""

    @staticmethod
    def _write_minimal_case_folder(tmp_path: Path) -> None:
        (tmp_path / "mpc_base_mva").write_text("100\n", encoding="utf-8")

        pd.DataFrame(
            [
                {"bus": 1, "type": 3, "Pd": 0.0, "Qd": 0.0, "Vmax": 1.1, "Vmin": 0.9},
                {"bus": 2, "type": 1, "Pd": 0.0, "Qd": 0.0, "Vmax": 1.1, "Vmin": 0.9},
            ]
        ).to_csv(tmp_path / "mpc_bus.csv", index=False)

        pd.DataFrame(
            [{"fbus": 1, "tbus": 2, "r": 0.01, "x": 0.02}]
        ).to_csv(tmp_path / "mpc_branch.csv", index=False)

        pd.DataFrame(
            [{"2": 0.2}],
            index=["t0"],
        ).to_csv(tmp_path / "p_load.csv")

        pd.DataFrame(
            [{"2": 0.05}],
            index=["t0"],
        ).to_csv(tmp_path / "q_load.csv")

    def test_profile_scales_available_generator_pmax(self, tmp_path: Path):
        """inflow_profile should scale available pmax while keeping static pmax unchanged."""
        self._write_minimal_case_folder(tmp_path)

        pd.DataFrame(
            [{"bus": 1, "pmax": 10.0, "qmax": 5.0, "status": 1, "inflow_profile": "pv"}]
        ).to_csv(tmp_path / "generators.csv", index=False)

        pd.DataFrame(
            [{"pv": 0.5}],
            index=["t0"],
        ).to_csv(tmp_path / "profiles.csv")

        case = load_case_from_folder(
            tmp_path,
            load_at="t0",
            generators_filename="generators.csv",
        )

        assert len(case.generators) == 1
        gen = case.generators[0]
        assert gen.p_max == 10.0
        assert gen.p_max_available == 5.0
        assert gen.q_max == 5.0
        assert gen.q_max_available == 2.5
        assert gen.q_min == -1.0
        assert gen.q_min_available == -0.5

    def test_missing_profile_column_falls_back_to_static_pmax(self, tmp_path: Path):
        """Missing profile column should silently fallback to static pmax."""
        self._write_minimal_case_folder(tmp_path)

        pd.DataFrame(
            [{"bus": 1, "pmax": 10.0, "qmax": 5.0, "status": 1, "inflow_profile": "missing"}]
        ).to_csv(tmp_path / "generators.csv", index=False)

        pd.DataFrame(
            [{"pv": 0.5}],
            index=["t0"],
        ).to_csv(tmp_path / "profiles.csv")

        case = load_case_from_folder(
            tmp_path,
            load_at="t0",
            generators_filename="generators.csv",
        )

        gen = case.generators[0]
        assert gen.p_max == 10.0
        assert gen.p_max_available == 10.0
        assert gen.q_min_available == -1.0

    def test_profile_negative_factor_raises(self, tmp_path: Path):
        """Capacity factors cannot reverse generator bounds."""
        self._write_minimal_case_folder(tmp_path)

        pd.DataFrame(
            [{"bus": 1, "pmax": 10.0, "qmax": 5.0, "status": 1, "inflow_profile": "pv"}]
        ).to_csv(tmp_path / "generators.csv", index=False)

        pd.DataFrame(
            [{"pv": -0.2}],
            index=["t0"],
        ).to_csv(tmp_path / "profiles.csv")

        with pytest.raises(ValueError, match="finite non-negative"):
            load_case_from_folder(
                tmp_path,
                load_at="t0",
                generators_filename="generators.csv",
            )

    def test_profile_uses_load_index_selector(self, tmp_path: Path):
        """Available pmax should use the same row selection policy as load profiles."""
        self._write_minimal_case_folder(tmp_path)

        # Overwrite load files with two timesteps so load_index selection is meaningful.
        pd.DataFrame(
            [{"2": 0.2}, {"2": 0.4}],
            index=["t0", "t1"],
        ).to_csv(tmp_path / "p_load.csv")
        pd.DataFrame(
            [{"2": 0.05}, {"2": 0.07}],
            index=["t0", "t1"],
        ).to_csv(tmp_path / "q_load.csv")

        pd.DataFrame(
            [{"bus": 1, "pmax": 10.0, "qmax": 5.0, "status": 1, "inflow_profile": "pv"}]
        ).to_csv(tmp_path / "generators.csv", index=False)
        pd.DataFrame(
            [{"pv": 0.2}, {"pv": 0.8}],
            index=["t0", "t1"],
        ).to_csv(tmp_path / "profiles.csv")

        case = load_case_from_folder(
            tmp_path,
            load_index=1,
            generators_filename="generators.csv",
        )

        assert case.generators[0].p_max_available == 8.0

    def test_multi_timestep_range_rejects_end_equal_snapshot_count(self, tmp_path: Path):
        """Range end_idx is inclusive, so a two-row profile only accepts end_idx <= 1."""
        self._write_minimal_case_folder(tmp_path)

        pd.DataFrame(
            [{"2": 0.2}, {"2": 0.4}],
            index=["t0", "t1"],
        ).to_csv(tmp_path / "p_load.csv")
        pd.DataFrame(
            [{"2": 0.05}, {"2": 0.07}],
            index=["t0", "t1"],
        ).to_csv(tmp_path / "q_load.csv")

        solver = MultiTimestepSolver(
            input_folder=tmp_path,
            output_db=tmp_path / "results.db",
            case_config={},
            solvers={"dummy": object()},
        )

        with pytest.raises(
            ValueError,
            match=r"end_idx=2 is out of bounds.*Valid timestep indices are 0-1.*inclusive",
        ):
            solver.run_range(start_idx=0, end_idx=2, solver_names=["dummy"], verbose=False)

    def test_multi_timestep_warm_start_uses_configured_estimator_name(self, tmp_path: Path):
        """ACOPF warm starts should not depend on literal socp/bfsa solver keys."""
        self._write_minimal_case_folder(tmp_path)

        class FakeEstimator:
            name = "Configured Estimator"

            def __init__(self) -> None:
                self.calls = 0

            def solve(self, case):
                self.calls += 1
                return OPFResult(
                    voltages={bus.bus_id: 1.0 for bus in case.buses},
                    flows={(branch.from_bus, branch.to_bus): (0.0, 0.0) for branch in case.branches},
                    duals_p={bus.bus_id: 0.0 for bus in case.buses},
                    duals_q={bus.bus_id: 0.0 for bus in case.buses},
                    generator_output={},
                    cost=0.0,
                    convergence_info={"solver_stage": "fake_estimator"},
                    solve_time=0.01,
                )

        class FakeACOPF:
            name = "Configured ACOPF"
            allow_load_shedding = False

            def __init__(self) -> None:
                self.warm_starts = []

            def solve(self, case, *, warm_start=None):
                self.warm_starts.append(warm_start)
                return OPFResult(
                    voltages={bus.bus_id: 1.0 for bus in case.buses},
                    flows={(branch.from_bus, branch.to_bus): (0.0, 0.0) for branch in case.branches},
                    duals_p={bus.bus_id: 1.0 for bus in case.buses},
                    duals_q={bus.bus_id: 1.0 for bus in case.buses},
                    generator_output={},
                    cost=1.0,
                    convergence_info={},
                    solve_time=0.02,
                )

        estimator = FakeEstimator()
        acopf = FakeACOPF()
        solver = MultiTimestepSolver(
            input_folder=tmp_path,
            output_db=tmp_path / "results.db",
            case_config={},
            solvers={"configured_acopf": acopf, "configured_estimator": estimator},
        )

        result = solver.run_range(
            start_idx=0,
            end_idx=0,
            solver_names=["configured_acopf"],
            verbose=False,
            warm_start_bfsa=True,
            warm_start_solver_name="configured_estimator",
        )

        assert estimator.calls == 1
        assert acopf.warm_starts[0] is not None
        acopf_result = result["solver_outputs"]["configured_acopf"][0]
        assert acopf_result.convergence_info["warm_start_source"] == "configured_estimator"
        assert result["selected_warm_start_mode"] == "Estimator t"

    def test_solver_comparison_stats(self):
        """Test SolverComparison statistics."""
        from comparison import SolverComparison
        
        bus1 = BusData(bus_id=1)
        case = PowerFlowCase(
            buses=[bus1],
            branches=[],
            generators=[],
            loads=[],
            root_bus=1,
        )
        
        result1 = OPFResult(
            voltages={1: 1.0},
            flows={},
            duals_p={1: 25.0},
            duals_q={1: 5.0},
            generator_output={},
            cost=50.0,
        )
        
        result2 = OPFResult(
            voltages={1: 1.0},
            flows={},
            duals_p={1: 25.0},
            duals_q={1: 5.0},
            generator_output={},
            cost=50.0,
        )
        
        comparison = SolverComparison(case, result1, result2)
        stats = comparison.summary_stats()
        
        assert "voltage" in stats
        assert "solve_times" in stats


class TestLoadNormalizationDetection:
    """Tests for automatic normalized-load mode in CSV loader."""

    def _write_case_files(
        self,
        case_dir: Path,
        base_mva: float,
        pd_bus2: float,
        qd_bus2: float,
        p_factor_bus2: float,
        q_factor_bus2: float,
        extra_load_bus: int = None,
    ) -> None:
        case_dir.mkdir(parents=True, exist_ok=True)

        (case_dir / "mpc_base_mva").write_text(str(base_mva))

        mpc_bus = pd.DataFrame(
            [
                {"bus": 1, "type": 3, "Pd": 0.0, "Qd": 0.0, "Vmax": 1.1, "Vmin": 0.9},
                {"bus": 2, "type": 1, "Pd": pd_bus2, "Qd": qd_bus2, "Vmax": 1.1, "Vmin": 0.9},
            ]
        )
        mpc_bus.to_csv(case_dir / "mpc_bus.csv", index=False)

        mpc_branch = pd.DataFrame(
            [
                {
                    "fbus": 1,
                    "tbus": 2,
                    "r": 0.01,
                    "x": 0.05,
                    "b": 0.0,
                    "rateA": 100.0,
                    "status": 1,
                }
            ]
        )
        mpc_branch.to_csv(case_dir / "mpc_branch.csv", index=False)

        p_cols = {"2": [p_factor_bus2]}
        q_cols = {"2": [q_factor_bus2]}
        if extra_load_bus is not None:
            p_cols[str(extra_load_bus)] = [0.3]
            q_cols[str(extra_load_bus)] = [0.2]

        p_load = pd.DataFrame(p_cols, index=["t0"])
        q_load = pd.DataFrame(q_cols, index=["t0"])
        p_load.to_csv(case_dir / "p_load.csv")
        q_load.to_csv(case_dir / "q_load.csv")

    def test_loader_keeps_legacy_mode_when_pd_qd_zero(self, tmp_path: Path):
        case_dir = tmp_path / "legacy_mode"
        self._write_case_files(
            case_dir=case_dir,
            base_mva=10.0,
            pd_bus2=0.0,
            qd_bus2=0.0,
            p_factor_bus2=0.4,
            q_factor_bus2=0.2,
        )

        case = load_case_from_folder(case_dir, enable_popup_warning=False)
        load2 = next(load for load in case.loads if load.bus_id == 2)

        assert load2.p_d == pytest.approx(0.4)
        assert load2.q_d == pytest.approx(0.2)

    def test_loader_forces_normalized_mode_when_pd_qd_nonzero(self, tmp_path: Path):
        case_dir = tmp_path / "normalized_mode"
        self._write_case_files(
            case_dir=case_dir,
            base_mva=10.0,
            pd_bus2=5.0,
            qd_bus2=2.0,
            p_factor_bus2=0.5,
            q_factor_bus2=0.25,
        )

        case = load_case_from_folder(case_dir, enable_popup_warning=False)
        load2 = next(load for load in case.loads if load.bus_id == 2)

        assert load2.p_d == pytest.approx((0.5 * 5.0) / 10.0)
        assert load2.q_d == pytest.approx((0.25 * 2.0) / 10.0)

    def test_loader_raises_if_load_bus_missing_in_mpc_bus(self, tmp_path: Path):
        case_dir = tmp_path / "missing_bus"
        self._write_case_files(
            case_dir=case_dir,
            base_mva=10.0,
            pd_bus2=5.0,
            qd_bus2=2.0,
            p_factor_bus2=0.5,
            q_factor_bus2=0.25,
            extra_load_bus=3,
        )

        with pytest.raises(ValueError) as excinfo:
            load_case_from_folder(case_dir, enable_popup_warning=False)

        assert "do not exist in" in str(excinfo.value)

    def test_loader_accepts_custom_file_names(self, tmp_path: Path):
        case_dir = tmp_path / "custom_names"
        self._write_case_files(
            case_dir=case_dir,
            base_mva=20.0,
            pd_bus2=10.0,
            qd_bus2=4.0,
            p_factor_bus2=0.5,
            q_factor_bus2=0.25,
        )

        (case_dir / "mpc_bus.csv").rename(case_dir / "bus_variant.csv")
        (case_dir / "mpc_branch.csv").rename(case_dir / "branch_variant.csv")
        (case_dir / "p_load.csv").rename(case_dir / "active_profile.csv")
        (case_dir / "q_load.csv").rename(case_dir / "reactive_profile.csv")

        case = load_case_from_folder(
            case_dir,
            enable_popup_warning=False,
            mpc_bus_filename="bus_variant.csv",
            mpc_branch_filename="branch_variant.csv",
            p_load_filename="active_profile.csv",
            q_load_filename="reactive_profile.csv",
        )
        load2 = next(load for load in case.loads if load.bus_id == 2)

        assert load2.p_d == pytest.approx((0.5 * 10.0) / 20.0)
        assert load2.q_d == pytest.approx((0.25 * 4.0) / 20.0)

    def test_loader_scales_absolute_loads_to_pu(self, tmp_path: Path):
        case_dir = tmp_path / "absolute_mode"
        self._write_case_files(
            case_dir=case_dir,
            base_mva=10.0,
            pd_bus2=0.0,
            qd_bus2=0.0,
            p_factor_bus2=5.0,
            q_factor_bus2=2.0,
        )

        case = load_case_from_folder(
            case_dir,
            load_interpretation_mode="absolute",
            enable_popup_warning=False,
        )
        load2 = next(load for load in case.loads if load.bus_id == 2)

        assert load2.p_d == pytest.approx(0.5)
        assert load2.q_d == pytest.approx(0.2)

    def test_loader_reads_custom_base_mva_csv_name(self, tmp_path: Path):
        case_dir = tmp_path / "custom_base_mva"
        self._write_case_files(
            case_dir=case_dir,
            base_mva=50.0,
            pd_bus2=5.0,
            qd_bus2=2.5,
            p_factor_bus2=1.0,
            q_factor_bus2=1.0,
        )

        (case_dir / "mpc_base_mva").unlink()
        pd.DataFrame({"base_mva": [25.0]}).to_csv(case_dir / "base_profile.csv", index=False)

        case = load_case_from_folder(
            case_dir,
            enable_popup_warning=False,
            mpc_base_mva_filename="base_profile.csv",
        )

        assert case.base_mva == pytest.approx(25.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
