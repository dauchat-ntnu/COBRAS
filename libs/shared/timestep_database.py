"""
Timestep results database manager for COBRAS.

Stores multi-timestep solver results (BFSA, SOCP) in SQLite following a structure
similar to PowerGAMA's database schema. Provides query and aggregation functionality.
"""

import sqlite3 as db
import os
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

from libs.shared.data import PowerFlowCase, OPFResult



class TimestepDatabase:
    """
    SQLite database manager for multi-timestep COBRAS results.
    
    Schema inspired by PowerGAMA but adapted for COBRAS OPFResult format.
    Stores grid metadata, per-timestep results, and computed aggregates.
    """

    SQLITE_MAX_VARIABLE_NUMBER = 990
    EXCEL_MAX_ROWS = 1_048_576
    EXCEL_HEADER_ROWS = 1

    def __init__(self, filename: str):
        """Initialize database connection and file.
        
        Args:
            filename: Path to SQLite database file (will be created if missing)
        """
        self.filename = os.path.abspath(filename)
        self.sqlite_version = db.sqlite_version
        self._branches_map: Dict[Tuple[int, int], int] = {}
        self._con: Optional[db.Connection] = None

    def open_session(self) -> None:
        """Open a persistent write session for batched inserts."""
        if self._con is None:
            self._con = db.connect(self.filename)

    def close_session(self) -> None:
        """Commit and close persistent write session."""
        if self._con is not None:
            try:
                self._con.commit()
                self._export_socp_duals_to_excel(self._con)
            finally:
                self._con.close()
                self._con = None

    def create_tables(self, case: PowerFlowCase) -> None:
        """
        Create database schema and populate grid metadata tables.
        
        Args:
            case: PowerFlowCase with grid topology/parameters
            
        Note:
            If database file already exists, it will be overwritten.
        """
        if os.path.isfile(self.filename):
            os.remove(self.filename)
        self.get_socp_duals_excel_path().unlink(missing_ok=True)

        if self._con is not None:
            self._con.close()
            self._con = None

        con = db.connect(self.filename)
        try:
            with con:
                cur = con.cursor()

                cur.execute(
                    "CREATE TABLE Grid_Buses("
                    "indx INT PRIMARY KEY, "
                    "bus_id INT, "
                    "bus_type INT, "
                    "base_kv DOUBLE, "
                    "v_max DOUBLE, "
                    "v_min DOUBLE"
                    ")"
                )
                buses_data = tuple((i, b.bus_id, b.bus_type, b.base_kv, b.v_max, b.v_min) for i, b in enumerate(case.buses))
                cur.executemany("INSERT INTO Grid_Buses VALUES(?,?,?,?,?,?)", buses_data)

                cur.execute(
                    "CREATE TABLE Grid_Branches("
                    "indx INT PRIMARY KEY, "
                    "branch_id INT, "
                    "from_bus INT, "
                    "to_bus INT, "
                    "r DOUBLE, "
                    "x DOUBLE, "
                    "b DOUBLE, "
                    "s_max DOUBLE"
                    ")"
                )
                branches_data = tuple((i, br.branch_id, br.from_bus, br.to_bus, br.r, br.x, br.b, br.s_max) for i, br in enumerate(case.branches))
                cur.executemany("INSERT INTO Grid_Branches VALUES(?,?,?,?,?,?,?,?)", branches_data)
                self._branches_map = {(br.from_bus, br.to_bus): i for i, br in enumerate(case.branches)}

                cur.execute(
                    "CREATE TABLE Grid_Generators("
                    "indx INT PRIMARY KEY, "
                    "gen_id INT, "
                    "bus_id INT, "
                    "p_min DOUBLE, "
                    "p_max DOUBLE, "
                    "q_min DOUBLE, "
                    "q_max DOUBLE, "
                    "c_p DOUBLE, "
                    "c_q DOUBLE, "
                    "c_0 DOUBLE"
                    ")"
                )
                generators_data = tuple((i, g.gen_id, g.bus_id, g.p_min, g.p_max, g.q_min, g.q_max, g.c_p, g.c_q, g.c_0) for i, g in enumerate(case.generators))
                cur.executemany("INSERT INTO Grid_Generators VALUES(?,?,?,?,?,?,?,?,?,?)", generators_data)

                cur.execute(
                    "CREATE TABLE Grid_Metadata("
                    "key TEXT PRIMARY KEY, "
                    "value TEXT"
                    ")"
                )
                cur.executemany(
                    "INSERT INTO Grid_Metadata VALUES(?, ?)",
                    [
                        ("base_mva", str(case.base_mva)),
                        ("root_bus", str(case.root_bus)),
                    ],
                )

                cur.execute(
                    "CREATE TABLE Res_Timesteps("
                    "timestep INT, "
                    "scenario_label TEXT, "
                    "solver_name TEXT, "
                    "cost DOUBLE, "
                    "solve_time DOUBLE, "
                    "iterations INT, "
                    "convergence_status TEXT, "
                    "load_interpretation_mode TEXT, "
                    "allow_load_shedding INT, "
                    "warm_start_mode TEXT, "
                    "PRIMARY KEY (timestep, solver_name)"
                    ")"
                )
                cur.execute(
                    "CREATE TABLE Res_Buses("
                    "timestep INT, "
                    "solver_name TEXT, "
                    "bus_id INT, "
                    "voltage DOUBLE, "
                    "p_d DOUBLE, "
                    "q_d DOUBLE, "
                    "lmp_p DOUBLE, "
                    "lmp_q DOUBLE, "
                    "load_shed DOUBLE, "
                    "PRIMARY KEY (timestep, solver_name, bus_id)"
                    ")"
                )
                cur.execute(
                    "CREATE TABLE Res_Branches("
                    "timestep INT, "
                    "solver_name TEXT, "
                    "branch_id INT, "
                    "flow_p DOUBLE, "
                    "flow_q DOUBLE, "
                    "flow_loss DOUBLE, "
                    "ell DOUBLE, "
                    "PRIMARY KEY (timestep, solver_name, branch_id)"
                    ")"
                )
                cur.execute(
                    "CREATE TABLE Res_Generators("
                    "timestep INT, "
                    "solver_name TEXT, "
                    "gen_id INT, "
                    "dispatch_p DOUBLE, "
                    "dispatch_q DOUBLE, "
                    "available_p_max DOUBLE, "
                    "PRIMARY KEY (timestep, solver_name, gen_id)"
                    ")"
                )
                cur.execute(
                    "CREATE TABLE Res_SOCPDuals("
                    "timestep INT, "
                    "solver_name TEXT, "
                    "dual_name TEXT, "
                    "entity_type TEXT, "
                    "entity_id INT, "
                    "value DOUBLE, "
                    "PRIMARY KEY (timestep, solver_name, dual_name, entity_type, entity_id)"
                    ")"
                )
                cur.execute(
                    "CREATE TABLE Res_SolverTimingSummary("
                    "start_timestep INT, "
                    "end_timestep INT, "
                    "solver_name TEXT, "
                    "total_time DOUBLE, "
                    "wall_time DOUBLE, "
                    "internal_build_time DOUBLE, "
                    "internal_extract_time DOUBLE, "
                    "internal_optimize_time DOUBLE, "
                    "internal_total_time DOUBLE, "
                    "PRIMARY KEY (start_timestep, end_timestep, solver_name)"
                    ")"
                )
                cur.execute(
                    "CREATE TABLE Agg_BusMetrics("
                    "start_timestep INT, "
                    "end_timestep INT, "
                    "solver_name TEXT, "
                    "bus_id INT, "
                    "metric TEXT, "
                    "mean_value DOUBLE, "
                    "max_value DOUBLE, "
                    "min_value DOUBLE, "
                    "std_value DOUBLE, "
                    "count INT, "
                    "PRIMARY KEY (start_timestep, end_timestep, solver_name, bus_id, metric)"
                    ")"
                )
                cur.execute(
                    "CREATE TABLE Agg_BranchMetrics("
                    "start_timestep INT, "
                    "end_timestep INT, "
                    "solver_name TEXT, "
                    "branch_id INT, "
                    "metric TEXT, "
                    "mean_value DOUBLE, "
                    "max_value DOUBLE, "
                    "min_value DOUBLE, "
                    "std_value DOUBLE, "
                    "count INT, "
                    "PRIMARY KEY (start_timestep, end_timestep, solver_name, branch_id, metric)"
                    ")"
                )
        finally:
            con.close()

    def store_timestep_result(
        self,
        timestep: int,
        scenario_label: str,
        solver_name: str,
        result: OPFResult,
        case: Optional[PowerFlowCase] = None,
        load_interpretation_mode: Optional[str] = None,
        allow_load_shedding: Optional[bool] = None,
        warm_start_mode: Optional[str] = None,
    ) -> None:
        """Store OPFResult for a single timestep into database."""
        con = self._con if self._con is not None else db.connect(self.filename)
        owns_connection = self._con is None
        try:
            cur = con.cursor()

            cur.execute(
                "INSERT INTO Res_Timesteps "
                "(timestep, scenario_label, solver_name, cost, solve_time, iterations, convergence_status, load_interpretation_mode, allow_load_shedding, warm_start_mode) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    timestep,
                    scenario_label,
                    solver_name,
                    result.cost,
                    result.solve_time,
                    result.convergence_info.get("iterations", result.convergence_info.get("max_iterations")),
                    str(result.convergence_info.get("termination_message") or (
                        "converged" if result.convergence_info.get("converged") is True
                        else "not_converged" if result.convergence_info.get("converged") is False else "Unknown")),
                    load_interpretation_mode,
                    int(bool(allow_load_shedding)) if allow_load_shedding is not None else None,
                    warm_start_mode,
                ),
            )

            p_d_by_bus = {bus.bus_id: 0.0 for bus in case.buses} if case is not None else {}
            q_d_by_bus = {bus.bus_id: 0.0 for bus in case.buses} if case is not None else {}
            if case is not None:
                for load in case.loads:
                    p_d_by_bus[load.bus_id] = p_d_by_bus.get(load.bus_id, 0.0) + float(load.p_d)
                    q_d_by_bus[load.bus_id] = q_d_by_bus.get(load.bus_id, 0.0) + float(load.q_d)

            bus_rows = [
                (
                    timestep,
                    solver_name,
                    bus_id,
                    voltage,
                    float(p_d_by_bus.get(bus_id, 0.0)),
                    float(q_d_by_bus.get(bus_id, 0.0)),
                    result.duals_p.get(bus_id),
                    result.duals_q.get(bus_id),
                    result.convergence_info.get("p_curtailment_by_bus", {}).get(bus_id, 0.0),
                )
                for bus_id, voltage in result.voltages.items()
            ]
            cur.executemany(
                "INSERT INTO Res_Buses "
                "(timestep, solver_name, bus_id, voltage, p_d, q_d, lmp_p, lmp_q, load_shed) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                bus_rows,
            )

            branches_map = self._branches_map
            branch_currents = getattr(result, "branch_currents", {}) or {}
            line_state = result.convergence_info.get("line_state_bfsa", {}) if result.convergence_info else {}
            resistance_by_edge = {(br.from_bus, br.to_bus): br.r for br in case.branches} if case else {}
            branch_rows = []
            for (from_bus, to_bus), flow in result.flows.items():
                branch_id = branches_map.get((from_bus, to_bus))
                if branch_id is not None:
                    ell_value = branch_currents.get((from_bus, to_bus))
                    if ell_value is None:
                        ell_value = (line_state.get((from_bus, to_bus), {}) or {}).get("ell")
                    if ell_value is None:
                        voltage = result.voltages.get(from_bus)
                        if voltage is not None and float(voltage) > 0.0:
                            ell_value = (float(flow[0]) ** 2 + float(flow[1]) ** 2) / float(voltage)
                    loss = (line_state.get((from_bus, to_bus), {}) or {}).get("ploss")
                    if loss is None and ell_value is not None and (from_bus, to_bus) in resistance_by_edge:
                        loss = resistance_by_edge[(from_bus, to_bus)] * ell_value
                    branch_rows.append((timestep, solver_name, branch_id, flow[0], flow[1], loss, ell_value))

            cur.executemany(
                "INSERT INTO Res_Branches "
                "(timestep, solver_name, branch_id, flow_p, flow_q, flow_loss, ell) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                branch_rows,
            )

            if case is not None and case.generators:
                generator_rows = [
                    (
                        timestep,
                        solver_name,
                        gen.gen_id,
                        float(result.generator_output.get(gen.gen_id, (0.0, 0.0))[0]),
                        float(result.generator_output.get(gen.gen_id, (0.0, 0.0))[1]),
                        self._get_available_p_max(gen.gen_id, case),
                    )
                    for gen in case.generators
                ]
            else:
                generator_rows = [
                    (
                        timestep,
                        solver_name,
                        gen_id,
                        float(dispatch[0]),
                        float(dispatch[1]),
                        self._get_available_p_max(gen_id, case),
                    )
                    for gen_id, dispatch in result.generator_output.items()
                ]
            cur.executemany(
                "INSERT INTO Res_Generators "
                "(timestep, solver_name, gen_id, dispatch_p, dispatch_q, available_p_max) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                generator_rows,
            )

            socp_dual_rows = []
            for dual in getattr(result, "socp_duals", []) or []:
                if dual.get("value") is None:
                    continue
                socp_dual_rows.append(
                    (
                        timestep,
                        solver_name,
                        str(dual.get("dual_name", "")),
                        str(dual.get("entity_type", "")),
                        dual.get("entity_id"),
                        float(dual.get("value")),
                    )
                )
            if socp_dual_rows:
                cur.executemany(
                    "INSERT INTO Res_SOCPDuals "
                    "(timestep, solver_name, dual_name, entity_type, entity_id, value) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    socp_dual_rows,
                )

            if owns_connection:
                con.commit()
        finally:
            if owns_connection:
                con.close()

    def store_solver_timing_summary(
        self,
        start_timestep: int,
        end_timestep: int,
        solver_name: str,
        total_time: Optional[float] = None,
        wall_time: Optional[float] = None,
        internal_timing: Optional[Dict[str, float]] = None,
    ) -> None:
        """Store aggregate solver timing for a multi-timestep range."""
        con = self._con if self._con is not None else db.connect(self.filename)
        owns_connection = self._con is None
        timing = internal_timing or {}
        try:
            cur = con.cursor()
            try:
                cur.execute(
                    "INSERT OR REPLACE INTO Res_SolverTimingSummary "
                    "(start_timestep, end_timestep, solver_name, total_time, wall_time, "
                    "internal_build_time, internal_extract_time, internal_optimize_time, internal_total_time) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        start_timestep,
                        end_timestep,
                        solver_name,
                        total_time,
                        wall_time,
                        timing.get("build"),
                        timing.get("extract"),
                        timing.get("optimize"),
                        timing.get("total"),
                    ),
                )
            except db.OperationalError:
                cur.execute(
                    "CREATE TABLE IF NOT EXISTS Res_SolverTimingSummary("
                    "start_timestep INT, "
                    "end_timestep INT, "
                    "solver_name TEXT, "
                    "total_time DOUBLE, "
                    "wall_time DOUBLE, "
                    "internal_build_time DOUBLE, "
                    "internal_extract_time DOUBLE, "
                    "internal_optimize_time DOUBLE, "
                    "internal_total_time DOUBLE, "
                    "PRIMARY KEY (start_timestep, end_timestep, solver_name)"
                    ")"
                )
                cur.execute(
                    "INSERT OR REPLACE INTO Res_SolverTimingSummary "
                    "(start_timestep, end_timestep, solver_name, total_time, wall_time, "
                    "internal_build_time, internal_extract_time, internal_optimize_time, internal_total_time) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        start_timestep,
                        end_timestep,
                        solver_name,
                        total_time,
                        wall_time,
                        timing.get("build"),
                        timing.get("extract"),
                        timing.get("optimize"),
                        timing.get("total"),
                    ),
                )
            if owns_connection:
                con.commit()
        finally:
            if owns_connection:
                con.close()

    def store_solver_failure(
        self,
        timestep: int,
        scenario_label: str,
        solver_name: str,
        convergence_status: str,
        error_message: Optional[str] = None,
        load_interpretation_mode: Optional[str] = None,
        allow_load_shedding: Optional[bool] = None,
        warm_start_mode: Optional[str] = None,
    ) -> None:
        """
        Store a failed solve marker for a single timestep.

        Writes only Res_Timesteps with NULL cost/solve_time so range runs can continue.
        """
        con = self._con if self._con is not None else db.connect(self.filename)
        owns_connection = self._con is None
        try:
            cur = con.cursor()
            status = str(convergence_status or "failed")
            if error_message:
                status = f"{status}: {error_message}"

            cur.execute(
                "INSERT OR REPLACE INTO Res_Timesteps "
                "(timestep, scenario_label, solver_name, cost, solve_time, iterations, convergence_status, load_interpretation_mode, allow_load_shedding, warm_start_mode) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    timestep,
                    scenario_label,
                    solver_name,
                    None,
                    None,
                    None,
                    status,
                    load_interpretation_mode,
                    int(bool(allow_load_shedding)) if allow_load_shedding is not None else None,
                    warm_start_mode,
                ),
            )

            if owns_connection:
                con.commit()
        finally:
            if owns_connection:
                con.close()

    def _get_available_p_max(self, gen_id: int, case: Optional[PowerFlowCase]) -> float:
        """Helper: get available P capacity for generator at current timestep."""
        if case is None:
            return 0.0
        for gen in case.generators:
            if gen.gen_id == gen_id:
                return gen.get_p_max_available()
        return 0.0

    def get_socp_duals_excel_path(self) -> Path:
        """Return the automatic Excel export path for stored SOCP duals."""
        db_path = Path(self.filename)
        if db_path.parent.name.lower() in {"db", "results"}:
            return db_path.parent.parent / "excel" / f"{db_path.stem}_socp_duals.xlsx"
        return db_path.with_name(f"{db_path.stem}_socp_duals.xlsx")

    def _export_socp_duals_to_excel(self, con: db.Connection) -> None:
        """Export all stored SOCP duals to an Excel workbook next to the database."""
        import pandas as pd

        try:
            duals = pd.read_sql_query(
                "SELECT timestep, solver_name, dual_name, entity_type, entity_id, value "
                "FROM Res_SOCPDuals "
                "ORDER BY timestep, solver_name, dual_name, entity_type, entity_id",
                con,
            )
        except db.OperationalError:
            return

        if duals.empty:
            return

        def _sheet_name(dual_name: str, used_names: set[str]) -> str:
            invalid_sheet_chars = set('[]:*?/\\')
            sanitized = "".join("_" if ch in invalid_sheet_chars else ch for ch in str(dual_name))
            sanitized = sanitized.strip() or "dual"
            base = sanitized[:31]
            candidate = base
            suffix = 1
            while candidate in used_names:
                tail = f"_{suffix}"
                candidate = f"{base[:31 - len(tail)]}{tail}"
                suffix += 1
            used_names.add(candidate)
            return candidate

        output_path = self.get_socp_duals_excel_path()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        max_excel_data_rows = self.EXCEL_MAX_ROWS - self.EXCEL_HEADER_ROWS
        used_sheet_names: set[str] = set()
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            for dual_name in sorted(duals["dual_name"].dropna().unique()):
                dual_rows = duals.loc[duals["dual_name"] == dual_name].copy()
                for start in range(0, len(dual_rows), max_excel_data_rows):
                    chunk = dual_rows.iloc[start : start + max_excel_data_rows]
                    chunk_index = start // max_excel_data_rows
                    chunk_name = str(dual_name) if chunk_index == 0 else f"{dual_name}_{chunk_index + 1}"
                    sheet = _sheet_name(chunk_name, used_sheet_names)
                    chunk.to_excel(writer, sheet_name=sheet, index=False)

    def get_base_mva(self) -> float:
        """Return the system base MVA stored in the database, if available."""
        con = db.connect(self.filename)
        try:
            with con:
                cur = con.cursor()
                try:
                    cur.execute("SELECT value FROM Grid_Metadata WHERE key = ?", ("base_mva",))
                except db.OperationalError:
                    return 100.0

                row = cur.fetchone()
                if row and row[0] is not None:
                    try:
                        return float(row[0])
                    except (TypeError, ValueError):
                        return 100.0
                return 100.0
        finally:
            con.close()

    def get_generator_ids(self) -> List[int]:
        """Return generator IDs stored in the grid metadata."""
        con = db.connect(self.filename)
        try:
            with con:
                cur = con.cursor()
                cur.execute("SELECT gen_id FROM Grid_Generators ORDER BY gen_id")
                return [int(row[0]) for row in cur.fetchall()]
        finally:
            con.close()

    def get_generator_timeseries(
        self,
        gen_id: int,
        start_timestep: int,
        end_timestep: int,
        solver_name: str,
    ) -> List[Tuple[int, float]]:
        """Query active generator dispatch across a timestep range."""
        con = db.connect(self.filename)
        try:
            with con:
                cur = con.cursor()
                cur.execute(
                    "SELECT timestep, dispatch_p FROM Res_Generators "
                    "WHERE gen_id = ? AND timestep >= ? AND timestep <= ? AND solver_name = ? "
                    "ORDER BY timestep",
                    (gen_id, start_timestep, end_timestep, solver_name),
                )
                return cur.fetchall()
        finally:
            con.close()

    def get_solve_time_timeseries(
        self,
        start_timestep: int,
        end_timestep: int,
        solver_name: str,
    ) -> List[Tuple[int, float]]:
        """Query solver runtime across a timestep range."""
        con = db.connect(self.filename)
        try:
            with con:
                cur = con.cursor()
                cur.execute(
                    "SELECT timestep, solve_time FROM Res_Timesteps "
                    "WHERE timestep >= ? AND timestep <= ? AND solver_name = ? "
                    "ORDER BY timestep",
                    (start_timestep, end_timestep, solver_name),
                )
                return cur.fetchall()
        finally:
            con.close()

    def get_solve_time_and_iteration_timeseries(
        self,
        start_timestep: int,
        end_timestep: int,
        solver_name: str,
    ) -> List[Tuple[int, Optional[float], Optional[int]]]:
        """Query solver runtime and iteration count across a timestep range."""
        con = db.connect(self.filename)
        try:
            with con:
                cur = con.cursor()
                try:
                    cur.execute(
                        "SELECT timestep, solve_time, iterations FROM Res_Timesteps "
                        "WHERE timestep >= ? AND timestep <= ? AND solver_name = ? "
                        "ORDER BY timestep",
                        (start_timestep, end_timestep, solver_name),
                    )
                    return cur.fetchall()
                except db.OperationalError:
                    cur.execute(
                        "SELECT timestep, solve_time, NULL AS iterations FROM Res_Timesteps "
                        "WHERE timestep >= ? AND timestep <= ? AND solver_name = ? "
                        "ORDER BY timestep",
                        (start_timestep, end_timestep, solver_name),
                    )
                    return cur.fetchall()
        finally:
            con.close()

    def compute_aggregates(self, start_timestep: int, end_timestep: int) -> None:
        """
        Compute and store aggregate statistics for timestep range.
        
        Args:
            start_timestep: Start index (inclusive)
            end_timestep: End index (inclusive)
        """
        con = db.connect(self.filename)
        try:
            with con:
                cur = con.cursor()

                def _sample_std_from_moments(sum_x: float, sum_x2: float, n: int) -> float:
                    if n <= 1:
                        return 0.0
                    variance = (sum_x2 - (sum_x * sum_x) / n) / (n - 1)
                    if variance < 0.0:
                        variance = 0.0
                    return variance ** 0.5

                # Clear existing aggregates for the same range to support recomputation.
                cur.execute(
                    "DELETE FROM Agg_BusMetrics WHERE start_timestep = ? AND end_timestep = ?",
                    (start_timestep, end_timestep),
                )
                cur.execute(
                    "DELETE FROM Agg_BranchMetrics WHERE start_timestep = ? AND end_timestep = ?",
                    (start_timestep, end_timestep),
                )

                bus_insert_rows = []
                for metric in ["voltage", "p_d", "q_d", "lmp_p", "lmp_q"]:
                    cur.execute(
                        f"SELECT solver_name, bus_id, "
                        f"AVG({metric}), MAX({metric}), MIN({metric}), COUNT({metric}), "
                        f"SUM({metric}), SUM({metric} * {metric}) "
                        f"FROM Res_Buses "
                        f"WHERE timestep >= ? AND timestep <= ? AND {metric} IS NOT NULL "
                        f"GROUP BY solver_name, bus_id",
                        (start_timestep, end_timestep),
                    )
                    for solver_name, bus_id, mean_v, max_v, min_v, cnt, sum_x, sum_x2 in cur.fetchall():
                        n = int(cnt or 0)
                        if n == 0:
                            continue
                        std_v = _sample_std_from_moments(float(sum_x), float(sum_x2), n)
                        bus_insert_rows.append(
                            (
                                start_timestep,
                                end_timestep,
                                solver_name,
                                int(bus_id),
                                metric,
                                float(mean_v),
                                float(max_v),
                                float(min_v),
                                float(std_v),
                                n,
                            )
                        )

                if bus_insert_rows:
                    cur.executemany(
                        "INSERT INTO Agg_BusMetrics "
                        "(start_timestep, end_timestep, solver_name, bus_id, metric, "
                        "mean_value, max_value, min_value, std_value, count) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        bus_insert_rows,
                    )

                branch_insert_rows = []
                for metric in ["flow_p", "flow_q", "ell"]:
                    cur.execute(
                        f"SELECT solver_name, branch_id, "
                        f"AVG({metric}), MAX({metric}), MIN({metric}), COUNT({metric}), "
                        f"SUM({metric}), SUM({metric} * {metric}) "
                        f"FROM Res_Branches "
                        f"WHERE timestep >= ? AND timestep <= ? AND {metric} IS NOT NULL "
                        f"GROUP BY solver_name, branch_id",
                        (start_timestep, end_timestep),
                    )
                    for solver_name, branch_id, mean_v, max_v, min_v, cnt, sum_x, sum_x2 in cur.fetchall():
                        n = int(cnt or 0)
                        if n == 0:
                            continue
                        std_v = _sample_std_from_moments(float(sum_x), float(sum_x2), n)
                        branch_insert_rows.append(
                            (
                                start_timestep,
                                end_timestep,
                                solver_name,
                                int(branch_id),
                                metric,
                                float(mean_v),
                                float(max_v),
                                float(min_v),
                                float(std_v),
                                n,
                            )
                        )

                if branch_insert_rows:
                    cur.executemany(
                        "INSERT INTO Agg_BranchMetrics "
                        "(start_timestep, end_timestep, solver_name, branch_id, metric, "
                        "mean_value, max_value, min_value, std_value, count) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        branch_insert_rows,
                    )
        finally:
            con.close()

    def get_timestep_range_stats(
        self,
        start_timestep: int,
        end_timestep: int,
        solver_name: str,
        metric: str = "voltage",
    ) -> Dict[str, Any]:
        """
        Query aggregate statistics for a metric across range.
        
        Args:
            start_timestep: Start index
            end_timestep: End index
            solver_name: Solver name
            metric: Metric type ("voltage", "lmp_p", "lmp_q", "flow_p", "flow_q")
            
        Returns:
            Dict with keys: mean, max, min, std, count, by_entity (list of per-bus/branch stats)
        """
        con = db.connect(self.filename)
        try:
            with con:
                cur = con.cursor()

                # Determine table and ID column
                if metric in ["voltage", "p_d", "q_d", "lmp_p", "lmp_q"]:
                    table = "Agg_BusMetrics"
                    id_col = "bus_id"
                elif metric in ["flow_p", "flow_q", "ell"]:
                    table = "Agg_BranchMetrics"
                    id_col = "branch_id"
                else:
                    return {"error": f"Unknown metric: {metric}"}

                # Query aggregate values
                cur.execute(
                    f"SELECT {id_col}, mean_value, max_value, min_value, std_value, count "
                    f"FROM {table} "
                    f"WHERE start_timestep = ? AND end_timestep = ? AND solver_name = ? AND metric = ?",
                    (start_timestep, end_timestep, solver_name, metric),
                )
                rows = cur.fetchall()

                if not rows:
                    return {
                        "error": f"No aggregates found for range [{start_timestep}, {end_timestep}]",
                        "metric": metric,
                        "solver": solver_name,
                    }

                # Compute overall stats
                all_means = [row[1] for row in rows]
                all_maxes = [row[2] for row in rows]
                all_mins = [row[3] for row in rows]

                return {
                    "metric": metric,
                    "solver": solver_name,
                    "range": [start_timestep, end_timestep],
                    "mean": sum(all_means) / len(all_means) if all_means else 0.0,
                    "max": max(all_maxes) if all_maxes else 0.0,
                    "min": min(all_mins) if all_mins else 0.0,
                    "entities": len(rows),
                    "by_entity": [
                        {
                            "id": row[0],
                            "mean": row[1],
                            "max": row[2],
                            "min": row[3],
                            "std": row[4],
                            "count": row[5],
                        }
                        for row in rows
                    ],
                }
        finally:
            con.close()

    def get_bus_timeseries(
        self,
        bus_id: int,
        start_timestep: int,
        end_timestep: int,
        solver_name: str,
        metric: str = "voltage",
    ) -> List[Tuple[int, float]]:
        """
        Query timeseries for a bus metric across range.
        
        Args:
            bus_id: Bus ID
            start_timestep: Start index
            end_timestep: End index
            solver_name: Solver name
            metric: Metric type (voltage, p_d, q_d, lmp_p, lmp_q)
            
        Returns:
            List of (timestep, value) tuples
        """
        con = db.connect(self.filename)
        try:
            with con:
                cur = con.cursor()
                cur.execute(
                    f"SELECT timestep, {metric} FROM Res_Buses "
                    f"WHERE bus_id = ? AND timestep >= ? AND timestep <= ? AND solver_name = ? "
                    f"ORDER BY timestep",
                    (bus_id, start_timestep, end_timestep, solver_name),
                )
                return cur.fetchall()
        finally:
            con.close()

    def get_branch_timeseries(
        self,
        branch_id: int,
        start_timestep: int,
        end_timestep: int,
        solver_name: str,
        metric: str = "flow_p",
    ) -> List[Tuple[int, float]]:
        """
        Query timeseries for a branch metric across range.
        
        Args:
            branch_id: Branch ID
            start_timestep: Start index
            end_timestep: End index
            solver_name: Solver name
            metric: Metric type (flow_p, flow_q)
            
        Returns:
            List of (timestep, value) tuples
        """
        con = db.connect(self.filename)
        try:
            with con:
                cur = con.cursor()
                if metric == "ell":
                    try:
                        cur.execute(
                            "SELECT rb.timestep, "
                            "COALESCE(rb.ell, "
                            "CASE WHEN bus.voltage IS NOT NULL AND bus.voltage > 0 "
                            "THEN ((rb.flow_p * rb.flow_p) + (rb.flow_q * rb.flow_q)) / bus.voltage "
                            "ELSE NULL END) "
                            "FROM Res_Branches rb "
                            "JOIN Grid_Branches gb ON rb.branch_id = gb.indx "
                            "LEFT JOIN Res_Buses bus "
                            "ON bus.timestep = rb.timestep "
                            "AND bus.solver_name = rb.solver_name "
                            "AND bus.bus_id = gb.from_bus "
                            "WHERE rb.branch_id = ? AND rb.timestep >= ? AND rb.timestep <= ? AND rb.solver_name = ? "
                            "ORDER BY rb.timestep",
                            (branch_id, start_timestep, end_timestep, solver_name),
                        )
                    except db.OperationalError:
                        cur.execute(
                            "SELECT rb.timestep, "
                            "CASE WHEN bus.voltage IS NOT NULL AND bus.voltage > 0 "
                            "THEN ((rb.flow_p * rb.flow_p) + (rb.flow_q * rb.flow_q)) / bus.voltage "
                            "ELSE NULL END "
                            "FROM Res_Branches rb "
                            "JOIN Grid_Branches gb ON rb.branch_id = gb.indx "
                            "LEFT JOIN Res_Buses bus "
                            "ON bus.timestep = rb.timestep "
                            "AND bus.solver_name = rb.solver_name "
                            "AND bus.bus_id = gb.from_bus "
                            "WHERE rb.branch_id = ? AND rb.timestep >= ? AND rb.timestep <= ? AND rb.solver_name = ? "
                            "ORDER BY rb.timestep",
                            (branch_id, start_timestep, end_timestep, solver_name),
                        )
                    return cur.fetchall()

                cur.execute(
                    f"SELECT timestep, {metric} FROM Res_Branches "
                    f"WHERE branch_id = ? AND timestep >= ? AND timestep <= ? AND solver_name = ? "
                    f"ORDER BY timestep",
                    (branch_id, start_timestep, end_timestep, solver_name),
                )
                return cur.fetchall()
        finally:
            con.close()

    @staticmethod
    def _classify_pq_case(flow_p: float, flow_q: float) -> int:
        """Classify a branch flow into one of the four BFSA pq cases.

        The sign logic follows ``bfsa_pq_cases`` in ``libs.methods.physical.bfsa.algorithm``:
        zero-valued flows are assigned by the first matching branch, so a branch
        with ``flow_p == 0`` and ``flow_q == 0`` is counted as case 1.
        """
        if flow_p >= 0 and flow_q >= 0:
            return 1
        if flow_p <= 0 and flow_q <= 0:
            return 2
        if flow_p >= 0 and flow_q <= 0:
            return 3
        return 4

    def get_bfsa_pq_case_counts(
        self,
        solver_name: str,
        start_timestep: Optional[int] = None,
        end_timestep: Optional[int] = None,
        export_to_excel: bool = False,
        excel_path: Optional[Path] = None,
    ) -> pd.DataFrame:
        """Count branch-flow sign cases per timestep for BFSA or SOCP results.

        Args:
            solver_name: Solver to analyze, typically "bfsa" or "socp".
            start_timestep: Optional inclusive lower timestep bound.
            end_timestep: Optional inclusive upper timestep bound.
            export_to_excel: If True, write the dataframe to an Excel workbook.
            excel_path: Optional output path for the Excel file. If omitted, a
                file named ``<database_stem>_<solver_name>_pq_case_counts.xlsx``
                is written next to the database.

        Returns:
            A pandas DataFrame with columns ``timestep``, ``case1``, ``case2``,
            ``case3``, and ``case4``.
        """
        import pandas as pd

        solver = str(solver_name).strip().lower()
        if not solver:
            raise ValueError("solver_name must not be empty")

        con = db.connect(self.filename)
        try:
            with con:
                cur = con.cursor()
                query = (
                    "SELECT timestep, flow_p, flow_q FROM Res_Branches "
                    "WHERE solver_name = ?"
                )
                params: list[Any] = [solver]

                if start_timestep is not None:
                    query += " AND timestep >= ?"
                    params.append(int(start_timestep))
                if end_timestep is not None:
                    query += " AND timestep <= ?"
                    params.append(int(end_timestep))

                query += " ORDER BY timestep, branch_id"
                cur.execute(query, params)
                rows = cur.fetchall()
        finally:
            con.close()

        counts_by_timestep: Dict[int, Dict[str, int]] = {}
        for timestep, flow_p, flow_q in rows:
            timestep_int = int(timestep)
            case_index = self._classify_pq_case(
                float(flow_p) if flow_p is not None else 0.0,
                float(flow_q) if flow_q is not None else 0.0,
            )
            timestep_counts = counts_by_timestep.setdefault(
                timestep_int,
                {"case1": 0, "case2": 0, "case3": 0, "case4": 0},
            )
            timestep_counts[f"case{case_index}"] += 1

        records = [
            {
                "timestep": timestep,
                "case1": counts["case1"],
                "case2": counts["case2"],
                "case3": counts["case3"],
                "case4": counts["case4"],
            }
            for timestep, counts in sorted(counts_by_timestep.items())
        ]
        df = pd.DataFrame(records, columns=["timestep", "case1", "case2", "case3", "case4"])

        if export_to_excel:
            output_path = Path(excel_path) if excel_path is not None else self.get_pq_case_counts_excel_path(solver)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
                df.to_excel(writer, index=False, sheet_name="pq_case_counts")

        return df

    def get_pq_case_counts_excel_path(self, solver_name: str) -> Path:
        """Return the default Excel export path for pq case counts."""
        db_path = Path(self.filename)
        solver = str(solver_name).strip().lower() or "solver"
        if db_path.parent.name.lower() in {"db", "results"}:
            return db_path.parent.parent / "excel" / f"{db_path.stem}_{solver}_pq_case_counts.xlsx"
        return db_path.with_name(f"{db_path.stem}_{solver}_pq_case_counts.xlsx")

    def export_to_csv(
        self,
        start_timestep: int,
        end_timestep: int,
        solver_name: str,
        output_dir: Path,
    ) -> None:
        """
        Export timestep results to CSV files.
        
        Args:
            start_timestep: Start index
            end_timestep: End index
            solver_name: Solver name
            output_dir: Output directory (will be created)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        con = db.connect(self.filename)
        try:
            with con:
                cur = con.cursor()

                # Export buses
                cur.execute(
                    "SELECT timestep, bus_id, voltage, p_d, q_d, lmp_p, lmp_q FROM Res_Buses "
                    "WHERE timestep >= ? AND timestep <= ? AND solver_name = ? "
                    "ORDER BY timestep, bus_id",
                    (start_timestep, end_timestep, solver_name),
                )
                bus_rows = cur.fetchall()
                if bus_rows:
                    with open(output_dir / f"{solver_name}_buses.csv", "w") as f:
                        f.write("timestep,bus_id,voltage,p_d,q_d,lmp_p,lmp_q\n")
                        for row in bus_rows:
                            f.write(
                                f"{row[0]},{row[1]},{row[2]:.6f},{row[3]:.6f},{row[4]:.6f},{row[5]:.6f},{row[6]:.6f}\n"
                            )

                # Export branches
                cur.execute(
                    "SELECT timestep, branch_id, flow_p, flow_q FROM Res_Branches "
                    "WHERE timestep >= ? AND timestep <= ? AND solver_name = ? "
                    "ORDER BY timestep, branch_id",
                    (start_timestep, end_timestep, solver_name),
                )
                branch_rows = cur.fetchall()
                if branch_rows:
                    with open(output_dir / f"{solver_name}_branches.csv", "w") as f:
                        f.write("timestep,branch_id,flow_p,flow_q\n")
                        for row in branch_rows:
                            f.write(f"{row[0]},{row[1]},{row[2]:.6f},{row[3]:.6f}\n")

                # Export generators
                cur.execute(
                    "SELECT timestep, gen_id, dispatch_p, dispatch_q, available_p_max FROM Res_Generators "
                    "WHERE timestep >= ? AND timestep <= ? AND solver_name = ? "
                    "ORDER BY timestep, gen_id",
                    (start_timestep, end_timestep, solver_name),
                )
                gen_rows = cur.fetchall()
                if gen_rows:
                    with open(output_dir / f"{solver_name}_generators.csv", "w") as f:
                        f.write("timestep,gen_id,dispatch_p,dispatch_q,available_p_max\n")
                        for row in gen_rows:
                            f.write(f"{row[0]},{row[1]},{row[2]:.6f},{row[3]:.6f},{row[4]:.6f}\n")
        finally:
            con.close()

    def get_timerange(self) -> List[int]:
        """
        Get all timesteps in database.
        
        Returns:
            Sorted list of timestep indices
        """
        con = db.connect(self.filename)
        try:
            with con:
                cur = con.cursor()
                candidate_tables = ["Res_Timesteps", "Res_Buses", "Res_Branches", "Res_Generators"]
                timesteps: set[int] = set()

                for table_name in candidate_tables:
                    try:
                        cur.execute(
                            f"SELECT DISTINCT timestep FROM {table_name} "
                            "WHERE timestep IS NOT NULL ORDER BY timestep"
                        )
                    except db.OperationalError:
                        # Table may be missing for older databases or partial exports.
                        continue

                    for (timestep_value,) in cur.fetchall():
                        try:
                            timesteps.add(int(timestep_value))
                        except (TypeError, ValueError):
                            continue

                return sorted(timesteps)
        finally:
            con.close()

    def get_bus_ids(self) -> List[int]:
        """Get all bus IDs stored in the grid metadata."""
        con = db.connect(self.filename)
        try:
            with con:
                cur = con.cursor()
                cur.execute("SELECT bus_id FROM Grid_Buses ORDER BY bus_id")
                return [int(row[0]) for row in cur.fetchall()]
        finally:
            con.close()

    def get_branch_ids(self) -> List[int]:
        """Get all branch IDs stored in the grid metadata."""
        con = db.connect(self.filename)
        try:
            with con:
                cur = con.cursor()
                cur.execute("SELECT branch_id FROM Grid_Branches ORDER BY branch_id")
                return [int(row[0]) for row in cur.fetchall()]
        finally:
            con.close()

    def get_branch_endpoints(self) -> List[Tuple[int, int]]:
        """Get all branch endpoints (from_bus, to_bus) stored in grid metadata."""
        con = db.connect(self.filename)
        try:
            with con:
                cur = con.cursor()
                cur.execute("SELECT from_bus, to_bus FROM Grid_Branches")
                return [(int(row[0]), int(row[1])) for row in cur.fetchall()]
        finally:
            con.close()

    def retrieve_timestep_result(
        self, timestep: int, solver_name: str, case: PowerFlowCase
    ) -> OPFResult:
        """
        Reconstruct an OPFResult from database records for a specific timestep and solver.
        
        Args:
            timestep: Timestep index
            solver_name: Solver name
            case: PowerFlowCase for reference (needed for structure)
            
        Returns:
            OPFResult object with voltages, flows, duals, generator output
        """
        con = db.connect(self.filename)
        try:
            with con:
                cur = con.cursor()
                
                # Retrieve bus results
                cur.execute(
                    "SELECT bus_id, voltage, lmp_p, lmp_q, load_shed "
                    "FROM Res_Buses "
                    "WHERE timestep = ? AND solver_name = ? "
                    "ORDER BY bus_id",
                    (timestep, solver_name),
                )
                bus_rows = cur.fetchall()
                
                voltages = {}
                duals_p = {}
                duals_q = {}
                load_shed = {}
                
                for bus_id, v_sq, lmp_p, lmp_q, l_shed in bus_rows:
                    voltages[int(bus_id)] = float(v_sq) if v_sq is not None else 0.0
                    duals_p[int(bus_id)] = float(lmp_p) if lmp_p is not None else 0.0
                    duals_q[int(bus_id)] = float(lmp_q) if lmp_q is not None else 0.0
                    load_shed[int(bus_id)] = float(l_shed) if l_shed is not None else 0.0
                
                # Retrieve branch results
                cur.execute(
                    "SELECT gb.from_bus, gb.to_bus, flow_p, flow_q, flow_loss "
                    "FROM Res_Branches rb "
                    "JOIN Grid_Branches gb ON rb.branch_id = gb.indx "
                    "WHERE rb.timestep = ? AND rb.solver_name = ? "
                    "ORDER BY rb.branch_id",
                    (timestep, solver_name),
                )
                branch_rows = cur.fetchall()
                
                flows = {}
                for from_bus, to_bus, flow_p, flow_q, flow_loss in branch_rows:
                    flows[(int(from_bus), int(to_bus))] = (
                        float(flow_p) if flow_p is not None else 0.0,
                        float(flow_q) if flow_q is not None else 0.0,
                    )
                
                # Retrieve generator results
                cur.execute(
                    "SELECT gen_id, dispatch_p, dispatch_q "
                    "FROM Res_Generators "
                    "WHERE timestep = ? AND solver_name = ? "
                    "ORDER BY gen_id",
                    (timestep, solver_name),
                )
                gen_rows = cur.fetchall()
                
                generator_output = {}
                for gen_id, disp_p, disp_q in gen_rows:
                    generator_output[int(gen_id)] = (
                        float(disp_p) if disp_p is not None else 0.0,
                        float(disp_q) if disp_q is not None else 0.0,
                    )

                socp_duals = []
                try:
                    cur.execute(
                        "SELECT dual_name, entity_type, entity_id, value "
                        "FROM Res_SOCPDuals "
                        "WHERE timestep = ? AND solver_name = ? "
                        "ORDER BY dual_name, entity_type, entity_id",
                        (timestep, solver_name),
                    )
                    for dual_name, entity_type, entity_id, value in cur.fetchall():
                        socp_duals.append(
                            {
                                "dual_name": str(dual_name),
                                "entity_type": str(entity_type),
                                "entity_id": int(entity_id) if entity_id is not None else None,
                                "value": float(value) if value is not None else None,
                            }
                        )
                except db.OperationalError:
                    socp_duals = []
                
                # Retrieve metadata for this timestep
                cur.execute(
                    "SELECT cost, solve_time, iterations, convergence_status "
                    "FROM Res_Timesteps "
                    "WHERE timestep = ? AND solver_name = ?",
                    (timestep, solver_name),
                )
                meta_row = cur.fetchone()
                
                if meta_row:
                    cost = float(meta_row[0]) if meta_row[0] is not None else 0.0
                    solve_time = float(meta_row[1]) if meta_row[1] is not None else 0.0
                    iterations = int(meta_row[2]) if meta_row[2] is not None else None
                    conv_msg = str(meta_row[3]) if meta_row[3] is not None else "Unknown"
                else:
                    cost = 0.0
                    solve_time = 0.0
                    iterations = None
                    conv_msg = "Retrieved from database"
                
                # Create OPFResult
                result = OPFResult(
                    voltages=voltages,
                    flows=flows,
                    duals_p=duals_p,
                    duals_q=duals_q,
                    generator_output=generator_output,
                    cost=cost,
                    solve_time=solve_time,
                    convergence_info={"termination_message": conv_msg, "iterations": iterations},
                    socp_duals=socp_duals,
                )
                
                return result
        finally:
            con.close()

    def retrieve_timestep_range_result(
        self,
        start_timestep: int,
        end_timestep: int,
        solver_name: str,
        case: PowerFlowCase,
    ) -> OPFResult:
        """
        Reconstruct an OPFResult for a timestep range.

        The network state is taken from the first available timestep in the range,
        while cost and solve time are aggregated across all matching timesteps.
        """
        con = db.connect(self.filename)
        try:
            with con:
                cur = con.cursor()
                cur.execute(
                    "SELECT timestep FROM Res_Timesteps "
                    "WHERE timestep >= ? AND timestep <= ? AND solver_name = ? "
                    "ORDER BY timestep",
                    (start_timestep, end_timestep, solver_name),
                )
                rows = cur.fetchall()
                if not rows:
                    raise ValueError(
                        f"No results found for solver '{solver_name}' in range [{start_timestep}, {end_timestep}]"
                    )

                representative_timestep = int(rows[0][0])
                result = self.retrieve_timestep_result(representative_timestep, solver_name, case)

                cur.execute(
                    "SELECT SUM(cost), SUM(solve_time) FROM Res_Timesteps "
                    "WHERE timestep >= ? AND timestep <= ? AND solver_name = ?",
                    (start_timestep, end_timestep, solver_name),
                )
                cost_sum, solve_time_sum = cur.fetchone()

                result.cost = float(cost_sum) if cost_sum is not None else 0.0
                result.solve_time = float(solve_time_sum) if solve_time_sum is not None else 0.0
                if result.convergence_info is None:
                    result.convergence_info = {}

                try:
                    cur.execute(
                        "SELECT total_time, wall_time, internal_build_time, internal_extract_time, "
                        "internal_optimize_time, internal_total_time "
                        "FROM Res_SolverTimingSummary "
                        "WHERE start_timestep = ? AND end_timestep = ? AND solver_name = ?",
                        (start_timestep, end_timestep, solver_name),
                    )
                    timing_row = cur.fetchone()
                except db.OperationalError:
                    timing_row = None

                if timing_row:
                    total_time, wall_time, build_time, extract_time, optimize_time, internal_total_time = timing_row
                    result.solve_time = float(total_time) if total_time is not None else result.solve_time
                    if wall_time is not None:
                        result.convergence_info["wall_time"] = float(wall_time)
                    timing = result.convergence_info.setdefault("timing", {})
                    if isinstance(timing, dict):
                        timing_values = {
                            "build": build_time,
                            "extract": extract_time,
                            "optimize": optimize_time,
                            "total": internal_total_time,
                        }
                        for key, value in timing_values.items():
                            if value is not None:
                                timing[key] = float(value)
                return result
        finally:
            con.close()
