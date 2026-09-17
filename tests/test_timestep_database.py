"""
Unit tests for TimestepDatabase module.

Tests database schema creation, result storage, and query functionality.
"""

import pytest
import tempfile
import sqlite3 as db
from pathlib import Path
import pandas as pd

from libs.shared.data import (
    PowerFlowCase, BusData, BranchData, GeneratorData, LoadData, OPFResult
)
from libs.shared.timestep_database import TimestepDatabase


@pytest.fixture
def temp_db():
    """Create a temporary database file path (not creating the file)."""
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".db", delete=True) as f:
        db_path = f.name
    # File is deleted by tempfile context manager
    # Ensure it doesn't exist before returning
    Path(db_path).unlink(missing_ok=True)
    yield db_path
    # Cleanup after test
    Path(db_path).unlink(missing_ok=True)
    Path(db_path).with_name(f"{Path(db_path).stem}_socp_duals.xlsx").unlink(missing_ok=True)


@pytest.fixture
def sample_case():
    """Create a sample PowerFlowCase for testing."""
    buses = [
        BusData(bus_id=1, bus_type=3, base_kv=35),
        BusData(bus_id=2, bus_type=1, base_kv=35),
        BusData(bus_id=3, bus_type=1, base_kv=35),
    ]
    
    branches = [
        BranchData(branch_id=1, from_bus=1, to_bus=2, r=0.01, x=0.03, rateA=100),
        BranchData(branch_id=2, from_bus=2, to_bus=3, r=0.02, x=0.05, rateA=80),
    ]
    
    generators = [
        GeneratorData(gen_id=1, bus_id=1, p_min=0, p_max=1.0, c_p=10.0),
    ]
    
    loads = [
        LoadData(bus_id=2, p_d=0.5, q_d=0.1),
        LoadData(bus_id=3, p_d=0.3, q_d=0.05),
    ]
    
    return PowerFlowCase(
        buses=buses,
        branches=branches,
        generators=generators,
        loads=loads,
        root_bus=1,
        base_mva=100.0,
    )


@pytest.fixture
def sample_result():
    """Create a sample OPFResult for testing."""
    return OPFResult(
        voltages={1: 1.0, 2: 0.98, 3: 0.95},
        flows={(1, 2): (0.8, 0.1), (2, 3): (0.3, 0.05)},
        duals_p={1: 50.0, 2: 55.0, 3: 60.0},
        duals_q={1: 5.0, 2: 6.0, 3: 7.0},
        generator_output={1: (0.8, 0.2)},
        cost=1000.0,
        convergence_info={"termination_message": "Optimal", "iterations": 7},
        solver_name="socp",
        solve_time=0.123,
        socp_duals=[
            {"dual_name": "lambda_v_ref", "entity_type": "system", "entity_id": None, "value": 1.25},
            {"dual_name": "mu_v_lv_plus", "entity_type": "bus", "entity_id": 2, "value": 0.5},
            {"dual_name": "mu_l", "entity_type": "branch", "entity_id": 1, "value": 0.75},
            {"dual_name": "mu_g_p_plus", "entity_type": "generator", "entity_id": 1, "value": 2.5},
        ],
    )


@pytest.fixture
def pq_case_counts_case():
    """Create a simple case with five branches so all pq cases can be exercised."""
    buses = [
        BusData(bus_id=1, bus_type=3, base_kv=35),
        BusData(bus_id=2, bus_type=1, base_kv=35),
        BusData(bus_id=3, bus_type=1, base_kv=35),
        BusData(bus_id=4, bus_type=1, base_kv=35),
        BusData(bus_id=5, bus_type=1, base_kv=35),
        BusData(bus_id=6, bus_type=1, base_kv=35),
    ]

    branches = [
        BranchData(branch_id=1, from_bus=1, to_bus=2, r=0.01, x=0.03, rateA=100),
        BranchData(branch_id=2, from_bus=2, to_bus=3, r=0.01, x=0.03, rateA=100),
        BranchData(branch_id=3, from_bus=3, to_bus=4, r=0.01, x=0.03, rateA=100),
        BranchData(branch_id=4, from_bus=4, to_bus=5, r=0.01, x=0.03, rateA=100),
        BranchData(branch_id=5, from_bus=5, to_bus=6, r=0.01, x=0.03, rateA=100),
    ]

    generators = [
        GeneratorData(gen_id=1, bus_id=1, p_min=0, p_max=2.0, c_p=10.0),
    ]

    loads = [
        LoadData(bus_id=2, p_d=0.5, q_d=0.1),
        LoadData(bus_id=3, p_d=0.3, q_d=0.05),
        LoadData(bus_id=4, p_d=0.2, q_d=0.05),
        LoadData(bus_id=5, p_d=0.1, q_d=0.02),
        LoadData(bus_id=6, p_d=0.1, q_d=0.02),
    ]

    return PowerFlowCase(
        buses=buses,
        branches=branches,
        generators=generators,
        loads=loads,
        root_bus=1,
        base_mva=100.0,
    )


@pytest.fixture
def pq_case_counts_results():
    """Create two timestep results with all four sign cases represented."""
    timestep_0 = OPFResult(
        voltages={i: 1.0 - 0.01 * i for i in range(1, 7)},
        flows={
            (1, 2): (1.0, 1.0),
            (2, 3): (-1.0, -1.0),
            (3, 4): (1.0, -1.0),
            (4, 5): (-1.0, 1.0),
            (5, 6): (0.0, 0.0),
        },
        duals_p={i: 1.0 for i in range(1, 7)},
        duals_q={i: 0.5 for i in range(1, 7)},
        generator_output={1: (1.5, 0.2)},
        cost=1000.0,
        convergence_info={"termination_message": "Optimal"},
        solver_name="bfsa",
        solve_time=0.1,
    )

    timestep_1 = OPFResult(
        voltages={i: 1.0 - 0.02 * i for i in range(1, 7)},
        flows={
            (1, 2): (2.0, 2.0),
            (2, 3): (3.0, -3.0),
            (3, 4): (4.0, 4.0),
            (4, 5): (-4.0, -4.0),
            (5, 6): (-2.0, 2.0),
        },
        duals_p={i: 2.0 for i in range(1, 7)},
        duals_q={i: 1.0 for i in range(1, 7)},
        generator_output={1: (2.5, 0.4)},
        cost=1100.0,
        convergence_info={"termination_message": "Optimal"},
        solver_name="socp",
        solve_time=0.2,
    )

    return timestep_0, timestep_1


class TestTimestepDatabaseInit:
    """Test database initialization and table creation."""

    def test_create_tables(self, temp_db, sample_case):
        """Test that create_tables creates schema and populates grid metadata."""
        db_mgr = TimestepDatabase(temp_db)
        db_mgr.create_tables(sample_case)

        # Verify database file exists
        assert Path(temp_db).exists()

        # Verify grid tables exist
        con = db.connect(temp_db)
        cur = con.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cur.fetchall()}
        con.close()

        expected_tables = {
            "Grid_Buses",
            "Grid_Branches",
            "Grid_Generators",
            "Res_Timesteps",
            "Res_Buses",
            "Res_Branches",
            "Res_Generators",
            "Res_SOCPDuals",
            "Agg_BusMetrics",
            "Agg_BranchMetrics",
        }
        assert expected_tables <= tables

    def test_create_tables_duplicate_overwrite(self, temp_db, sample_case):
        """Test that creating tables twice overwrites existing database."""
        db_mgr = TimestepDatabase(temp_db)
        db_mgr.create_tables(sample_case)

        db_mgr2 = TimestepDatabase(temp_db)
        db_mgr2.create_tables(sample_case)

        con = db.connect(temp_db)
        cur = con.cursor()
        cur.execute("SELECT COUNT(*) FROM Grid_Buses")
        bus_count = cur.fetchone()[0]
        con.close()

        assert bus_count == len(sample_case.buses)


class TestResultStorage:
    """Test storing and retrieving results."""

    def test_store_single_timestep(self, temp_db, sample_case, sample_result):
        """Test storing a single timestep result."""
        db_mgr = TimestepDatabase(temp_db)
        db_mgr.create_tables(sample_case)
        db_mgr.store_timestep_result(
            timestep=0,
            scenario_label="Test Scenario",
            solver_name="socp",
            result=sample_result,
            case=sample_case,
        )

        # Verify data was stored
        con = db.connect(temp_db)
        cur = con.cursor()

        cur.execute("SELECT * FROM Res_Timesteps WHERE timestep = 0 AND solver_name = 'socp'")
        timestep_row = cur.fetchone()
        assert timestep_row is not None
        assert timestep_row[3] == 1000.0  # cost
        assert timestep_row[5] == 7

        cur.execute("SELECT COUNT(*) FROM Res_Buses WHERE timestep = 0 AND solver_name = 'socp'")
        count = cur.fetchone()[0]
        assert count == 3  # 3 buses

        cur.execute(
            "SELECT p_d, q_d FROM Res_Buses WHERE timestep = 0 AND solver_name = 'socp' AND bus_id = 2"
        )
        p_d, q_d = cur.fetchone()
        assert p_d == 0.5
        assert q_d == 0.1

        cur.execute("SELECT COUNT(*) FROM Res_Branches WHERE timestep = 0 AND solver_name = 'socp'")
        count = cur.fetchone()[0]
        assert count == 2  # 2 branches

        cur.execute("SELECT COUNT(*) FROM Res_Generators WHERE timestep = 0 AND solver_name = 'socp'")
        count = cur.fetchone()[0]
        assert count == 1  # 1 generator

        cur.execute("SELECT COUNT(*) FROM Res_SOCPDuals WHERE timestep = 0 AND solver_name = 'socp'")
        count = cur.fetchone()[0]
        assert count == 4

        cur.execute(
            "SELECT value FROM Res_SOCPDuals "
            "WHERE timestep = 0 AND solver_name = 'socp' "
            "AND dual_name = 'mu_l' AND entity_type = 'branch' AND entity_id = 1"
        )
        assert cur.fetchone()[0] == 0.75

        con.close()

    def test_close_session_exports_socp_duals_excel_with_one_sheet_per_dual(
        self, temp_db, sample_case, sample_result
    ):
        """SOCP duals should be exported once at session close, one sheet per dual."""
        db_mgr = TimestepDatabase(temp_db)
        db_mgr.create_tables(sample_case)
        db_mgr.open_session()
        db_mgr.store_timestep_result(
            timestep=0,
            scenario_label="Test Scenario",
            solver_name="socp",
            result=sample_result,
            case=sample_case,
        )
        assert not db_mgr.get_socp_duals_excel_path().exists()
        db_mgr.close_session()

        excel_path = db_mgr.get_socp_duals_excel_path()
        assert excel_path.exists()

        with pd.ExcelFile(excel_path) as workbook:
            assert set(workbook.sheet_names) == {
                "lambda_v_ref",
                "mu_g_p_plus",
                "mu_l",
                "mu_v_lv_plus",
            }

        exported = pd.read_excel(excel_path, sheet_name="mu_l")
        assert set(exported.columns) == {
            "timestep",
            "solver_name",
            "dual_name",
            "entity_type",
            "entity_id",
            "value",
        }
        assert len(exported) == 1
        mu_l = exported.iloc[0]
        assert int(mu_l["timestep"]) == 0
        assert mu_l["solver_name"] == "socp"
        assert mu_l["dual_name"] == "mu_l"
        assert mu_l["entity_type"] == "branch"
        assert int(mu_l["entity_id"]) == 1
        assert float(mu_l["value"]) == 0.75

    def test_close_session_splits_large_socp_dual_sheets(
        self, temp_db, sample_case, sample_result, monkeypatch
    ):
        """SOCP dual exports should split rows that exceed Excel's sheet limit."""
        monkeypatch.setattr(TimestepDatabase, "EXCEL_MAX_ROWS", 4)

        oversized_result = OPFResult(
            voltages=sample_result.voltages,
            flows=sample_result.flows,
            duals_p=sample_result.duals_p,
            duals_q=sample_result.duals_q,
            generator_output=sample_result.generator_output,
            cost=sample_result.cost,
            convergence_info=sample_result.convergence_info,
            solver_name=sample_result.solver_name,
            solve_time=sample_result.solve_time,
            socp_duals=[
                {
                    "dual_name": "mu_large",
                    "entity_type": "bus",
                    "entity_id": entity_id,
                    "value": float(entity_id),
                }
                for entity_id in range(1, 6)
            ],
        )

        db_mgr = TimestepDatabase(temp_db)
        db_mgr.create_tables(sample_case)
        db_mgr.open_session()
        db_mgr.store_timestep_result(
            timestep=0,
            scenario_label="Test Scenario",
            solver_name="socp",
            result=oversized_result,
            case=sample_case,
        )
        db_mgr.close_session()

        excel_path = db_mgr.get_socp_duals_excel_path()
        with pd.ExcelFile(excel_path) as workbook:
            assert workbook.sheet_names == ["mu_large", "mu_large_2"]

        first_sheet = pd.read_excel(excel_path, sheet_name="mu_large")
        second_sheet = pd.read_excel(excel_path, sheet_name="mu_large_2")
        assert len(first_sheet) == 3
        assert len(second_sheet) == 2
        assert list(second_sheet["entity_id"]) == [4, 5]


class TestPqCaseCounts:
    """Test sign-case postprocessing across timesteps."""

    def test_get_bfsa_pq_case_counts_dataframe(self, temp_db, pq_case_counts_case, pq_case_counts_results):
        """Count branch sign cases per timestep for BFSA results."""
        db_mgr = TimestepDatabase(temp_db)
        db_mgr.create_tables(pq_case_counts_case)

        timestep_0, timestep_1 = pq_case_counts_results
        db_mgr.store_timestep_result(0, "Scenario", "bfsa", timestep_0, pq_case_counts_case)
        db_mgr.store_timestep_result(1, "Scenario", "bfsa", timestep_1, pq_case_counts_case)
        db_mgr.store_timestep_result(0, "Scenario", "socp", timestep_0, pq_case_counts_case)

        df = db_mgr.get_bfsa_pq_case_counts("bfsa")

        assert list(df.columns) == ["timestep", "case1", "case2", "case3", "case4"]
        assert df.shape == (2, 5)

        row0 = df.loc[df["timestep"] == 0].iloc[0]
        assert row0["case1"] == 2
        assert row0["case2"] == 1
        assert row0["case3"] == 1
        assert row0["case4"] == 1

        row1 = df.loc[df["timestep"] == 1].iloc[0]
        assert row1["case1"] == 2
        assert row1["case2"] == 1
        assert row1["case3"] == 1
        assert row1["case4"] == 1

    def test_get_bfsa_pq_case_counts_export_to_excel(self, temp_db, pq_case_counts_case, pq_case_counts_results):
        """Export pq case counts to Excel and verify the workbook is created."""
        db_mgr = TimestepDatabase(temp_db)
        db_mgr.create_tables(pq_case_counts_case)

        timestep_0, _ = pq_case_counts_results
        db_mgr.store_timestep_result(0, "Scenario", "socp", timestep_0, pq_case_counts_case)

        output_path = Path(temp_db).with_name("case_counts.xlsx")
        df = db_mgr.get_bfsa_pq_case_counts("socp", export_to_excel=True, excel_path=output_path)

        assert output_path.exists()
        exported = pd.read_excel(output_path)
        assert exported.equals(df)
        assert df.loc[0, "case1"] == 2

    def test_retrieve_socp_duals(self, temp_db, sample_case, sample_result):
        """Test SOCP dual rows round-trip through storage and retrieval."""
        db_mgr = TimestepDatabase(temp_db)
        db_mgr.create_tables(sample_case)
        db_mgr.store_timestep_result(
            timestep=0,
            scenario_label="Test Scenario",
            solver_name="socp",
            result=sample_result,
            case=sample_case,
        )

        retrieved = db_mgr.retrieve_timestep_result(0, "socp", sample_case)

        assert retrieved.socp_duals == [
            {"dual_name": "lambda_v_ref", "entity_type": "system", "entity_id": None, "value": 1.25},
            {"dual_name": "mu_g_p_plus", "entity_type": "generator", "entity_id": 1, "value": 2.5},
            {"dual_name": "mu_l", "entity_type": "branch", "entity_id": 1, "value": 0.75},
            {"dual_name": "mu_v_lv_plus", "entity_type": "bus", "entity_id": 2, "value": 0.5},
        ]

    def test_store_solver_failure_row(self, temp_db, sample_case):
        db_mgr = TimestepDatabase(temp_db)
        db_mgr.create_tables(sample_case)

        db_mgr.store_solver_failure(
            timestep=3,
            scenario_label="Range SOCP",
            solver_name="socp",
            convergence_status="infeasible_no_shedding",
            error_message="SOCP solve failed: termination_condition=infeasible",
        )

        con = db.connect(temp_db)
        cur = con.cursor()
        cur.execute(
            "SELECT timestep, solver_name, cost, solve_time, iterations, convergence_status "
            "FROM Res_Timesteps WHERE timestep = 3 AND solver_name = 'socp'"
        )
        row = cur.fetchone()
        con.close()

        assert row is not None
        assert row[0] == 3
        assert row[1] == "socp"
        assert row[2] is None
        assert row[3] is None
        assert row[4] is None
        assert "infeasible_no_shedding" in row[5]

    def test_retrieve_timestep_range_result_sums_cost_and_time(self, temp_db, sample_case):
        """Test range retrieval aggregates total horizon cost and solve time."""
        db_mgr = TimestepDatabase(temp_db)
        db_mgr.create_tables(sample_case)

        for timestep in range(3):
            result = OPFResult(
                voltages={1: 1.0, 2: 0.98},
                flows={(1, 2): (0.5, 0.1)},
                duals_p={1: 20.0, 2: 19.0},
                duals_q={1: 2.0, 2: 1.0},
                generator_output={1: (0.6, 0.1)},
                cost=100.0 + 10.0 * timestep,
                convergence_info={"termination_message": "Optimal"},
                solver_name="socp",
                solve_time=0.2 + 0.05 * timestep,
            )
            db_mgr.store_timestep_result(
                timestep=timestep,
                scenario_label="Range test",
                solver_name="socp",
                result=result,
                case=sample_case,
            )

        retrieved = db_mgr.retrieve_timestep_range_result(0, 2, "socp", sample_case)

        assert retrieved.cost == 330.0
        assert retrieved.solve_time == 0.75
        assert retrieved.voltages == {1: 1.0, 2: 0.98}

    def test_retrieve_timestep_range_result_includes_solver_timing_summary(self, temp_db, sample_case):
        """Test range retrieval includes persisted wall and internal timing details."""
        db_mgr = TimestepDatabase(temp_db)
        db_mgr.create_tables(sample_case)

        result = OPFResult(
            voltages={1: 1.0, 2: 0.98},
            flows={(1, 2): (0.5, 0.1)},
            duals_p={1: 20.0, 2: 19.0},
            duals_q={1: 2.0, 2: 1.0},
            generator_output={1: (0.6, 0.1)},
            cost=100.0,
            convergence_info={"termination_message": "Optimal"},
            solver_name="socp",
            solve_time=0.2,
        )
        db_mgr.store_timestep_result(
            timestep=0,
            scenario_label="Range test",
            solver_name="socp",
            result=result,
            case=sample_case,
        )
        db_mgr.store_solver_timing_summary(
            start_timestep=0,
            end_timestep=0,
            solver_name="socp",
            total_time=1.081,
            wall_time=1.287,
            internal_timing={
                "build": 0.032,
                "extract": 0.082,
                "optimize": 1.081,
                "total": 1.277,
            },
        )

        retrieved = db_mgr.retrieve_timestep_range_result(0, 0, "socp", sample_case)

        assert retrieved.solve_time == 1.081
        assert retrieved.convergence_info["wall_time"] == 1.287
        assert retrieved.convergence_info["timing"] == {
            "build": 0.032,
            "extract": 0.082,
            "optimize": 1.081,
            "total": 1.277,
        }

    def test_store_multiple_timesteps(self, temp_db, sample_case, sample_result):
        """Test storing multiple timestep results."""
        db_mgr = TimestepDatabase(temp_db)
        db_mgr.create_tables(sample_case)

        # Store results for timesteps 0-4
        for t in range(5):
            result = OPFResult(
                voltages={1: 1.0, 2: 0.98 - t*0.01, 3: 0.95 - t*0.02},
                flows={(1, 2): (0.8, 0.1), (2, 3): (0.3, 0.05)},
                duals_p={1: 50.0 + t, 2: 55.0 + t, 3: 60.0 + t},
                duals_q={1: 5.0, 2: 6.0, 3: 7.0},
                generator_output={1: (0.8, 0.2)},
                cost=1000.0 + t*10,
                convergence_info={"termination_message": "Optimal"},
                solver_name="socp",
                solve_time=0.123,
            )
            db_mgr.store_timestep_result(
                timestep=t,
                scenario_label="Multi-timestep",
                solver_name="socp",
                result=result,
                case=sample_case,
            )

        # Verify all timesteps stored
        con = db.connect(temp_db)
        cur = con.cursor()
        cur.execute("SELECT COUNT(DISTINCT timestep) FROM Res_Timesteps WHERE solver_name = 'socp'")
        count = cur.fetchone()[0]
        assert count == 5
        con.close()


class TestQueryAndAggregation:
    """Test query functions and aggregation."""

    def test_compute_aggregates(self, temp_db, sample_case, sample_result):
        """Test computing aggregates across range."""
        db_mgr = TimestepDatabase(temp_db)
        db_mgr.create_tables(sample_case)

        # Store results for timesteps 0-4 with varying voltage values
        for t in range(5):
            result = OPFResult(
                voltages={1: 1.0, 2: 0.98 - t*0.01, 3: 0.95 - t*0.02},
                flows={(1, 2): (0.8 + t*0.05, 0.1), (2, 3): (0.3, 0.05)},
                duals_p={1: 50.0, 2: 55.0, 3: 60.0},
                duals_q={1: 5.0, 2: 6.0, 3: 7.0},
                generator_output={1: (0.8, 0.2)},
                cost=1000.0,
                convergence_info={"termination_message": "Optimal"},
                solver_name="socp",
                solve_time=0.123,
            )
            db_mgr.store_timestep_result(t, "Test", "socp", result, sample_case)

        # Compute aggregates
        db_mgr.compute_aggregates(0, 4)

        # Query voltage aggregates
        stats = db_mgr.get_timestep_range_stats(0, 4, "socp", "voltage")

        assert "error" not in stats
        assert stats["metric"] == "voltage"
        assert stats["range"] == [0, 4]
        assert stats["entities"] == 3  # 3 buses
        assert stats["mean"] is not None
        assert stats["max"] is not None
        assert stats["min"] is not None

        demand_stats = db_mgr.get_timestep_range_stats(0, 4, "socp", "p_d")
        assert "error" not in demand_stats
        assert demand_stats["metric"] == "p_d"
        assert demand_stats["entities"] == 3

    def test_get_bus_timeseries(self, temp_db, sample_case):
        """Test querying bus timeseries."""
        db_mgr = TimestepDatabase(temp_db)
        db_mgr.create_tables(sample_case)

        # Store results with varying voltages for bus 2
        for t in range(5):
            result = OPFResult(
                voltages={1: 1.0, 2: 0.98 - t*0.01, 3: 0.95},
                flows={(1, 2): (0.8, 0.1), (2, 3): (0.3, 0.05)},
                duals_p={1: 50.0, 2: 55.0, 3: 60.0},
                duals_q={1: 5.0, 2: 6.0, 3: 7.0},
                generator_output={1: (0.8, 0.2)},
                cost=1000.0,
                convergence_info={"termination_message": "Optimal"},
                solver_name="socp",
                solve_time=0.123,
            )
            db_mgr.store_timestep_result(t, "Test", "socp", result, sample_case)

        # Query bus 2 voltage timeseries
        timeseries = db_mgr.get_bus_timeseries(2, 0, 4, "socp", "voltage")

        assert len(timeseries) == 5
        assert timeseries[0] == (0, 0.98)
        assert timeseries[1] == (1, 0.97)
        assert timeseries[2] == (2, 0.96)

        demand_timeseries = db_mgr.get_bus_timeseries(2, 0, 4, "socp", "p_d")
        assert demand_timeseries == [(0, 0.5), (1, 0.5), (2, 0.5), (3, 0.5), (4, 0.5)]

    def test_export_to_csv(self, temp_db, sample_case):
        """Test exporting results to CSV."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_mgr = TimestepDatabase(temp_db)
            db_mgr.create_tables(sample_case)

            # Store some results
            result = OPFResult(
                voltages={1: 1.0, 2: 0.98, 3: 0.95},
                flows={(1, 2): (0.8, 0.1), (2, 3): (0.3, 0.05)},
                duals_p={1: 50.0, 2: 55.0, 3: 60.0},
                duals_q={1: 5.0, 2: 6.0, 3: 7.0},
                generator_output={1: (0.8, 0.2)},
                cost=1000.0,
                convergence_info={"termination_message": "Optimal"},
                solver_name="socp",
                solve_time=0.123,
            )
            db_mgr.store_timestep_result(0, "Test", "socp", result, sample_case)

            # Export to CSV
            db_mgr.export_to_csv(0, 0, "socp", Path(temp_dir))

            # Verify CSV files created
            assert (Path(temp_dir) / "socp_buses.csv").exists()
            assert (Path(temp_dir) / "socp_branches.csv").exists()
            assert (Path(temp_dir) / "socp_generators.csv").exists()

            # Verify content
            with open(Path(temp_dir) / "socp_buses.csv") as f:
                lines = f.readlines()
                assert len(lines) == 4  # header + 3 buses

    def test_get_timerange(self, temp_db, sample_case):
        """Test getting all timesteps in database."""
        db_mgr = TimestepDatabase(temp_db)
        db_mgr.create_tables(sample_case)

        # Store results for non-sequential timesteps
        for t in [0, 2, 5, 10]:
            result = OPFResult(
                voltages={1: 1.0, 2: 0.98, 3: 0.95},
                flows={(1, 2): (0.8, 0.1), (2, 3): (0.3, 0.05)},
                duals_p={1: 50.0, 2: 55.0, 3: 60.0},
                duals_q={1: 5.0, 2: 6.0, 3: 7.0},
                generator_output={1: (0.8, 0.2)},
                cost=1000.0,
                convergence_info={"termination_message": "Optimal"},
                solver_name="socp",
                solve_time=0.123,
            )
            db_mgr.store_timestep_result(t, "Test", "socp", result, sample_case)

        # Get timerange
        timerange = db_mgr.get_timerange()
        assert timerange == [0, 2, 5, 10]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
