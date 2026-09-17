"""
Shared data structures for PowerSOC.

All solvers (SOCP, BFSA, Pandapower, etc.) use these structures
to ensure consistent input/output formats.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Optional
import math


@dataclass
class BusData:
    """Bus data point."""
    bus_id: int
    bus_type: int = 1  # 1=load, 2=gen, 3=slack
    p_d: float = 0.0
    q_d: float = 0.0
    g_s: float = 0.0
    b_s: float = 0.0
    area: int = 1
    v_m: float = 1.0
    v_a: float = 0.0
    base_kv: float = 1.0
    zone: int = 1
    v_max: float = 1.1
    v_min: float = 0.9


@dataclass
class BranchData:
    """Branch (line) data point."""
    branch_id: int
    from_bus: int
    to_bus: int
    r: float
    x: float
    b: float = 0.0
    rateA: float = 9999.0
    rateB: float = 9999.0
    rateC: float = 9999.0
    ratio: float = 0.0
    angle: float = 0.0
    status: int = 1
    angmin: float = -360.0
    angmax: float = 360.0
    lmax: float = 1
    original_from_bus: Optional[int] = None
    original_to_bus: Optional[int] = None
    
    @property
    def s_max(self) -> float:
        """Thermal limit (MVA); use rateA if available."""
        return self.rateA if self.rateA < 9999 else float('inf')



@dataclass
class GeneratorData:
    """Generator data point."""
    gen_id: int
    bus_id: int
    p_min: float = 0.0
    p_max: float = 1.0
    q_min: float = -0.33
    q_max: float = 0.33
    v_target: float = 1.0
    mbase: float = 100.0
    status: int = 1
    p_max_old: float = 1.0
    c_p: float = 1.0  # Cost coefficient (active power)
    c_q: float = 0.0  # Cost coefficient (reactive power)
    c_0: float = 0.0  # Constant cost
    profile_name: Optional[str] = None  # Optional profile key in network/profiles.csv
    p_max_available: Optional[float] = None  # Snapshot-available active power (same unit as p_max)
    p_min_available: Optional[float] = None  # Snapshot-available active power minimum (same unit as p_min)
    q_max_available: Optional[float] = None  # Snapshot-available reactive power maximum (same unit as q_max)
    q_min_available: Optional[float] = None  # Snapshot-available reactive power minimum (same unit as q_min)

    def get_p_max_available(self) -> float:
        """Return snapshot-available active power; fallback to static p_max."""
        if self.p_max_available is None:
            return self.p_max
        return self.p_max_available

    def get_q_max_available(self) -> float:
        """Return snapshot-available reactive power; fallback to static q_max."""
        if self.q_max_available is None:
            return self.q_max
        return self.q_max_available
    
    #IGNORE MINIMUM REQUIREMENTS FOR NOW, to do later


@dataclass
class LoadData:
    """Load data point."""
    bus_id: int
    p_d: float = 0.0
    q_d: float = 0.0


@dataclass
class PowerFlowCase:
    """
    Complete case for power flow/OPF analysis.
    
    Represents a single time snapshot of network + loads.
    Produced by input loaders (CSV, MATPOWER).
    Consumed by solvers (SOCP, BFSA).
    """
    
    buses: List[BusData]
    branches: List[BranchData]
    generators: List[GeneratorData]
    loads: List[LoadData]
    root_bus: int
    base_mva: float = 100.0 #FALLBACK, TODO make sure the user as declared a base_mva 
    
    # Topology info (computed by orient_radial_network)
    incoming_branches: Dict[int, int] = field(default_factory=dict)  # {branch_id: to_bus}
    outgoing_branches: Dict[int, List[int]] = field(default_factory=dict)  # {from_bus: [branch_id]}
    tree_edges: List[Tuple[int, int]] = field(default_factory=list)  # Directed edges (parent, child)
    
    def __post_init__(self):
        """Validate case on creation."""
        if self.root_bus not in [b.bus_id for b in self.buses]:
            raise ValueError(f"root_bus {self.root_bus} not found in bus list")
        if not math.isfinite(self.base_mva) or self.base_mva <= 0:
            raise ValueError(f"base_mva must be positive, got {self.base_mva}")


@dataclass
class OPFResult:
    """
    Standardized output from any OPF solver.
    
    All solvers (SOCP, BFSA, Pandapower) return this format.
    This enables plug-and-play comparison.
    
    Attributes:
        voltages: {bus_id: V²} (squared voltage in p.u.²)
        flows: {(from_bus, to_bus): (p, q)} (p.u.)
        duals_p: {bus_id: λ_p} (active power marginal price)
        duals_q: {bus_id: λ_q} (reactive power marginal price)
        generator_output: {gen_id: (p, q)} (p.u.)
        cost: Total objective cost ($)
        convergence_info: {
            "converged": bool,
            "iterations": int,
            "final_mismatch": float,
            "solver_time": float (seconds),
            "termination_message": str
        }
    """
    
    voltages: Dict[int, float]  # {bus_id: V²}
    flows: Dict[Tuple[int, int], Tuple[float, float]]  # {(i, j): (p, q)}
    duals_p: Dict[int, float]  # {bus_id: λ_p}
    duals_q: Dict[int, float]  # {bus_id: λ_q}
    generator_output: Dict[int, Tuple[float, float]]  # {gen_id: (p, q)}
    cost: float
    convergence_info: Dict = field(default_factory=dict)
    branch_currents: Dict[Tuple[int, int], float] = field(default_factory=dict)  # {(i, j): ell}
    
    # Optional metadata
    solver_name: Optional[str] = None
    solve_time: Optional[float] = None
    socp_duals: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert to dict for serialization."""
        return {
            "voltages": self.voltages,
            "flows": self.flows,
            "duals_p": self.duals_p,
            "duals_q": self.duals_q,
            "generator_output": self.generator_output,
            "cost": self.cost,
            "convergence_info": self.convergence_info,
            "branch_currents": self.branch_currents,
            "solver_name": self.solver_name,
            "solve_time": self.solve_time,
            "socp_duals": self.socp_duals,
        }


def validate_case(case: PowerFlowCase) -> None:
    """
    Validate case consistency.
    
    Raises:
        ValueError: If case is inconsistent
    """
    bus_ids = {b.bus_id for b in case.buses}
    for label, ids in (
        ("bus", [b.bus_id for b in case.buses]),
        ("branch", [b.branch_id for b in case.branches]),
        ("generator", [g.gen_id for g in case.generators]),
    ):
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate {label} IDs are not supported")
    for branch in case.branches:
        if branch.from_bus not in bus_ids or branch.to_bus not in bus_ids:
            raise ValueError(f"Branch {branch.branch_id} endpoints not in bus list")
    gen_bus_ids = {g.bus_id for g in case.generators}
    load_bus_ids = {l.bus_id for l in case.loads}
    
    if not gen_bus_ids <= bus_ids:
        missing = gen_bus_ids - bus_ids
        raise ValueError(f"Generator buses not in bus list: {missing}")
    
    if not load_bus_ids <= bus_ids:
        missing = load_bus_ids - bus_ids
        raise ValueError(f"Load buses not in bus list: {missing}")
    
    if case.root_bus not in bus_ids:
        raise ValueError(f"root_bus {case.root_bus} not in bus list")

