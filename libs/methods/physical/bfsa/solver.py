"""
BFSA Solver wrapper - returns standardized OPFResult.
"""

from __future__ import annotations

import time
from typing import Optional, Dict, Tuple, List
import networkx as nx

from libs.methods.defaults import get_method_defaults
from libs.shared import (
    DispatchResult,
    EconomicStateResult,
    MethodCapabilities,
    OPFResult,
    PhysicalStateResult,
    PowerFlowCase,
    economic_state_from_opf_result,
    orient_radial_network,
    physical_state_from_opf_result,
)
from .algorithm import run_electrical_bfsa_forest, run_dlmp_propagation_forest
from libs.methods.dispatch import DummyMeritOrderDispatchEstimator


class BFSASolver:
    """
    BFSA Solver wrapper.

    Backward-Forward Sweep Algorithm for fast OPF in radial networks.

    Example:
        solver = BFSAElectricalSolver(config={"max_iterations": 100})
        result = solver.estimate(case)
    """

    name = "BFSA"
    capabilities = MethodCapabilities(
        produces_physical_state=True,
        produces_economic_state=True,
        produces_dispatch=True,
        supports_warm_start=False,
        requires_physical_state=False,
    )

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize BFSA solver.

        Args:
            config: Dict with keys:
                - compute_losses: bool (default True)
                - tol: convergence tolerance (default 1e-3)
                - max_iterations: max iterations (default 100)
                - convergence_check: voltage, current, or both (default both)
                - v_root_sq: root voltage squared (default 1.0)
        """
        self.config = get_method_defaults("physical", "bfsa")
        self.config.update(config or {})

    @staticmethod
    def _opf_result_from_physical_state(physical_state: PhysicalStateResult) -> OPFResult:
        """Build the legacy electrical result shape needed by existing DLMP propagation."""
        return OPFResult(
            voltages=dict(physical_state.voltages),
            flows=dict(physical_state.flows),
            duals_p={},
            duals_q={},
            generator_output=dict(physical_state.diagnostics.get("generator_output", {})),
            cost=float(physical_state.diagnostics.get("cost", 0.0)),
            convergence_info=dict(physical_state.diagnostics),
            branch_currents=dict(physical_state.branch_currents),
            solver_name=physical_state.method_name,
            solve_time=physical_state.solve_time,
        )


    @staticmethod
    def _compute_bus_net_demand(
        case: PowerFlowCase,
        p_bus: Dict[int, float],
        line_state_bfsa: Dict[Tuple[int, int], Dict[str, float]],
        generator_output: Dict[int, Tuple[float, float]],
    ) -> Dict[int, float]:
        """Compute per-bus net active demand as demand - generation."""
        p_gen_by_bus = {b.bus_id: 0.0 for b in case.buses}

        # Use explicit generator outputs if available.
        for gen in case.generators:
            p_gen, _ = generator_output.get(gen.gen_id, (0.0, 0.0))
            p_gen_by_bus[gen.bus_id] = p_gen_by_bus.get(gen.bus_id, 0.0) + float(p_gen)

        # BFSA currently does not solve generator dispatch; infer root injection if needed.
        if all(abs(v) < 1e-12 for v in p_gen_by_bus.values()):
            root_outgoing = sum(
                float(state.get("ptrans", 0.0))
                for (i, _), state in line_state_bfsa.items()
                if int(i) == int(case.root_bus)
            )
            p_gen_by_bus[case.root_bus] = float(p_bus.get(case.root_bus, 0.0)) + root_outgoing

        return {
            bus_id: float(p_bus.get(bus_id, 0.0)) - float(p_gen_by_bus.get(bus_id, 0.0))
            for bus_id in p_bus
        }

    @staticmethod
    def _aggregate_generator_dispatch_by_bus(
        case: PowerFlowCase,
        generator_output: Dict[int, Tuple[float, float]],
    ) -> Tuple[Dict[int, float], Dict[int, float]]:
        """Aggregate generator dispatch per bus for active and reactive power."""
        return DummyMeritOrderDispatchEstimator._aggregate_generator_dispatch_by_bus(case, generator_output)


    @staticmethod
    def _build_net_bus_injections(
        p_load_bus: Dict[int, float],
        q_load_bus: Dict[int, float],
        p_gen_by_bus: Dict[int, float],
        q_gen_by_bus: Dict[int, float],
    ) -> Tuple[Dict[int, float], Dict[int, float]]:
        """Build net bus demand maps from original load and current dispatch."""
        p_net_bus = {
            bus_id: float(p_load_bus.get(bus_id, 0.0)) - float(p_gen_by_bus.get(bus_id, 0.0))
            for bus_id in p_load_bus
        }
        q_net_bus = {
            bus_id: float(q_load_bus.get(bus_id, 0.0)) - float(q_gen_by_bus.get(bus_id, 0.0))
            for bus_id in q_load_bus
        }
        return p_net_bus, q_net_bus

    @staticmethod
    def _sum_line_losses(result_bfsa: Dict) -> Tuple[float, float]:
        """Return total active and reactive line losses from an electrical BFSA result."""
        line_state = result_bfsa.get("line_state_bfsa", {})
        total_p_loss = sum(float(state.get("ploss", 0.0)) for state in line_state.values())
        total_q_loss = sum(float(state.get("qloss", 0.0)) for state in line_state.values())
        return total_p_loss, total_q_loss


    def _resolve_price_propagation_seeds(
        self,
        p_clearing: Dict[str, object],
        q_clearing: Dict[str, object],
    ) -> Tuple[float, float, str]:
        """Return the P/Q prices to seed DLMP propagation and their source."""
        price_mode = str(self.config.get("merit_order_mode", "dummy")).strip().lower()
        if price_mode == "mcp":
            if self.config.get("mcp_active_price") is None or self.config.get("mcp_reactive_price") is None:
                raise ValueError(
                    "BFSA merit_order_mode='mcp' requires mcp_active_price and mcp_reactive_price."
                )
            return (
                float(self.config["mcp_active_price"]),
                float(self.config["mcp_reactive_price"]),
                "mcp",
            )

        return (
            float(p_clearing["marginal_price"]),
            float(q_clearing["marginal_price"]),
            "dummy",
        )

    @staticmethod
    def _plot_bus_net_demand(net_demand_p: Dict[int, float]) -> None:
        """Plot line chart of net bus demand (P_d - P_g)."""
        if not net_demand_p:
            return

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed; skipping net bus demand/gen plot.")
            return

        bus_ids = sorted(net_demand_p.keys())
        values = [float(net_demand_p[b]) for b in bus_ids]

        fig, ax = plt.subplots(figsize=(12, 4.5))
        ax.plot(bus_ids, values, marker="o", linewidth=1.6)
        ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
        ax.set_title("BFSA Net Bus Active Power: demand - generation")
        ax.set_xlabel("Bus")
        ax.set_ylabel("P_d - P_g (p.u.)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        plt.show()


    @staticmethod
    def _plot_merit_order_component(
        offers_sorted: List[Dict[str, object]],
        total_load: float,
        marginal_price: float,
        component_label: str,
    ) -> None:
        """Plot merit-order staircase for one component (P or Q)."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed; skipping merit-order plot.")
            return

        if not offers_sorted:
            return

        x_left = []
        widths = []
        heights = []
        labels = []
        is_excess = []

        cumulative = 0.0
        for offer in offers_sorted:
            cap = float(offer["capacity"])
            x_left.append(cumulative)
            widths.append(cap)
            heights.append(float(offer["cost"]))
            labels.append(str(offer["name"]))
            is_excess.append(bool(offer["is_load_excess"]))
            cumulative += cap

        fig, ax = plt.subplots(figsize=(12, 4.8))

        for xl, w, h, lbl, excess in zip(x_left, widths, heights, labels, is_excess):
            if w <= 0.0:
                continue
            color = "#f4a261" if excess else "#457b9d"
            hatch = "//" if excess else None
            ax.bar(
                xl,
                h,
                width=w,
                align="edge",
                color=color,
                edgecolor="black",
                linewidth=0.8,
                alpha=0.85,
                hatch=hatch,
            )

        ax.axvline(float(total_load), color="#d62828", linestyle="--", linewidth=1.8, label="Total load")
        ax.axhline(float(marginal_price), color="#2a9d8f", linestyle=":", linewidth=1.8, label="Clearing price")



        ax.set_title(f"BFSA Merit Order Clearing ({component_label})")
        ax.set_xlabel(f"Cumulative offered {component_label} (p.u.)")
        ax.set_ylabel("Marginal cost")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

        if cumulative > 0.0:
            ax.set_xlim(0.0, cumulative * 1.05)
        fig.tight_layout()
        plt.show()

    def solve_electrical(
        self,
        case: PowerFlowCase,
        dispatch: Optional[DispatchResult] = None,
        orient: bool = True,
        show_net_bus_plot: bool = False,
        show_merit_order_plot: bool = False,
    ) -> OPFResult:
        """Solve only the electrical BFSA stage and return a warm-startable result."""
        if not case.incoming_branches:
            if orient:
                print("Orienting radial network for BFSA...")
                case = orient_radial_network(case)
            else:
                raise ValueError("Case not oriented; set orient=True or call orient_radial_network first")

        G = nx.DiGraph()
        G.add_nodes_from(b.bus_id for b in case.buses)
        G.add_edges_from(case.tree_edges)

        p_load_bus = {b.bus_id: 0.0 for b in case.buses}
        q_load_bus = {b.bus_id: 0.0 for b in case.buses}
        for load in case.loads:
            p_load_bus[load.bus_id] += load.p_d
            q_load_bus[load.bus_id] += load.q_d

        line_lookup = {}
        for br in case.branches:
            if br.status != 1:
                continue
            key_undir = tuple(sorted((br.from_bus, br.to_bus)))
            line_lookup[key_undir] = br

        base_total_p_load = float(sum(p_load_bus.values()))
        base_total_q_load = float(sum(q_load_bus.values()))
        start_time = time.time()

        if dispatch is None:
            dispatch = DummyMeritOrderDispatchEstimator(config=self.config).estimate(case)

        dispatch_market = dispatch.market_clearing or {}
        initial_p_clearing = dispatch_market.get("p_clearing")
        initial_q_clearing = dispatch_market.get("q_clearing")
        if initial_p_clearing is None or initial_q_clearing is None:
            raise ValueError("DispatchResult is missing p_clearing/q_clearing market data.")

        initial_generator_output = dict(dispatch.generator_output)
        initial_p_gen_by_bus = dict(dispatch_market.get("p_gen_by_bus", {}))
        initial_q_gen_by_bus = dict(dispatch_market.get("q_gen_by_bus", {}))
        if not initial_p_gen_by_bus or not initial_q_gen_by_bus:
            initial_p_gen_by_bus, initial_q_gen_by_bus = self._aggregate_generator_dispatch_by_bus(
                case=case,
                generator_output=initial_generator_output,
            )

        initial_net_p_bus, initial_net_q_bus = self._build_net_bus_injections(
            p_load_bus=p_load_bus,
            q_load_bus=q_load_bus,
            p_gen_by_bus=initial_p_gen_by_bus,
            q_gen_by_bus=initial_q_gen_by_bus,
        )

        result_bfsa = run_electrical_bfsa_forest(
            G,
            radial_roots=[case.root_bus],
            p_bus=initial_net_p_bus,
            q_bus=initial_net_q_bus,
            line_lookup_undirected=line_lookup,
            v_root_sq=self.config["v_root_sq"],
            config=self.config,
        )

        p_loss_feedback, q_loss_feedback = self._sum_line_losses(result_bfsa)
        total_p_load = base_total_p_load
        total_q_load = base_total_q_load
        lambda_p_clear, lambda_q_clear, price_source = self._resolve_price_propagation_seeds(
            p_clearing=initial_p_clearing,
            q_clearing=initial_q_clearing,
        )

        solve_time = time.time() - start_time
        voltages = result_bfsa["v_sq_bfsa"]
        flows = {}
        branch_currents = {}
        for (i, j), line_data in result_bfsa["line_state_bfsa"].items():
            flows[(i, j)] = (line_data["ptrans"], line_data["qtrans"])
            if "ell" in line_data:
                branch_currents[(i, j)] = float(line_data["ell"])

        convergence_info = dict(result_bfsa["convergence_info"])
        convergence_info["solver_time"] = solve_time
        convergence_info["price_time"] = 0.0
        convergence_info["solver_stage"] = "electrical"
        convergence_info["line_state_bfsa"] = result_bfsa["line_state_bfsa"]
        convergence_info["merit_order"] = {
            "total_p_load": total_p_load,
            "total_q_load": total_q_load,
            "base_total_p_load": base_total_p_load,
            "base_total_q_load": base_total_q_load,
            "loss_feedback_p": p_loss_feedback,
            "loss_feedback_q": q_loss_feedback,
            "lambda_p_clear": lambda_p_clear,
            "lambda_q_clear": lambda_q_clear,
            "p_setting_bus": int(initial_p_clearing["setting_bus"]),
            "q_setting_bus": int(initial_q_clearing["setting_bus"]),
            "p_setting_gen_id": int(initial_p_clearing.get("setting_gen_id", -1)),
            "q_setting_gen_id": int(initial_q_clearing.get("setting_gen_id", -1)),
            "loss_feedback_converged": False,
            "loss_feedback_iterations": 0,
            "loss_feedback_tol": float(self.config.get("loss_feedback_tol", 0.05)),
            "loss_feedback_relative_change": None,
            "loss_feedback_history": [
                {
                    "iteration": 0,
                    "total_p_load": total_p_load,
                    "total_q_load": total_q_load,
                    "loss_feedback_p": p_loss_feedback,
                    "loss_feedback_q": q_loss_feedback,
                    "p_setting_gen_id": int(initial_p_clearing.get("setting_gen_id", -1)),
                    "p_clearing_generator_output": float(initial_p_clearing.get("load_excess_capacity", 0.0)),
                    "relative_change": None,
                    "converged": False,
                }
            ],
            "initial_total_p_load": float(initial_p_clearing["total_load"]),
            "initial_total_q_load": float(initial_q_clearing["total_load"]),
            "initial_lambda_p_clear": float(initial_p_clearing["marginal_price"]),
            "initial_lambda_q_clear": float(initial_q_clearing["marginal_price"]),
            "price_source": price_source,
            "merit_order_mode": str(self.config.get("merit_order_mode", "dummy")),
            "mcp_active_price": None if self.config.get("mcp_active_price") is None else float(self.config["mcp_active_price"]),
            "mcp_reactive_price": None if self.config.get("mcp_reactive_price") is None else float(self.config["mcp_reactive_price"]),
            "initial_p_setting_gen_id": int(initial_p_clearing.get("setting_gen_id", -1)),
            "initial_q_setting_gen_id": int(initial_q_clearing.get("setting_gen_id", -1)),
            "initial_generator_output": initial_generator_output,
            "load_excess_p": float(initial_p_clearing["load_excess_capacity"]),
            "load_excess_q": float(initial_q_clearing["load_excess_capacity"]),
            "p_curve": initial_p_clearing["offers_sorted"],
            "q_curve": initial_q_clearing["offers_sorted"],
            "line_state_bfsa": result_bfsa["line_state_bfsa"],
            "radial_roots": [int(case.root_bus)],
        }

        active_cost = 0.0 if total_p_load <= 0.0 else float(lambda_p_clear * total_p_load * case.base_mva)
        reactive_cost = 0.0 if total_q_load <= 0.0 else float(lambda_q_clear * total_q_load * case.base_mva)
        total_cost = float(dispatch.objective) if dispatch.objective is not None else active_cost + reactive_cost

        if show_net_bus_plot:
            self._plot_bus_net_demand(self._compute_bus_net_demand(
                case, p_load_bus, result_bfsa["line_state_bfsa"], initial_generator_output,
            ))
        if show_merit_order_plot:
            for clearing, label in ((initial_p_clearing, "P"), (initial_q_clearing, "Q")):
                self._plot_merit_order_component(
                    clearing["offers_sorted"], clearing["total_load"],
                    clearing["marginal_price"], label,
                )

        return OPFResult(
            voltages=voltages,
            flows=flows,
            duals_p={},
            duals_q={},
            generator_output=initial_generator_output,
            cost=total_cost,
            convergence_info=convergence_info,
            branch_currents=branch_currents,
            solver_name="BFSA-ELECTRICAL",
            solve_time=solve_time,
        )

    def propagate_dlmp(
        self,
        case: PowerFlowCase,
        electrical_result: OPFResult,
        dlmp_seed_bus: Optional[int] = None,
    ) -> Dict[int, Dict[str, float]]:
        """Propagate DLMPs from a previously computed electrical BFSA result."""
        if not electrical_result.convergence_info:
            raise ValueError("electrical_result must include convergence_info from solve_electrical")

        if not case.incoming_branches:
            case = orient_radial_network(case)

        G = nx.DiGraph()
        G.add_nodes_from(b.bus_id for b in case.buses)
        G.add_edges_from(case.tree_edges)

        line_lookup = {}
        for br in case.branches:
            if br.status != 1:
                continue
            key_undir = tuple(sorted((br.from_bus, br.to_bus)))
            line_lookup[key_undir] = br

        merit = electrical_result.convergence_info.get("merit_order", {})
        lambda_p_seed = float(merit.get("lambda_p_clear", 0.0))
        lambda_q_seed = float(merit.get("lambda_q_clear", 0.0))
        seed_bus = int(dlmp_seed_bus) if dlmp_seed_bus is not None else int(merit.get("p_setting_bus", case.root_bus))
        line_state_bfsa = electrical_result.convergence_info.get("line_state_bfsa")
        if line_state_bfsa is None:
            line_state_bfsa = merit.get("line_state_bfsa")
        if line_state_bfsa is None:
            raise KeyError("electrical_result.convergence_info is missing line_state_bfsa")

        return run_dlmp_propagation_forest(
            G=G,
            radial_roots=electrical_result.convergence_info.get("radial_roots", [case.root_bus]),
            line_state_bfsa=line_state_bfsa,
            line_lookup_undirected=line_lookup,
            lambda_p_seed=lambda_p_seed,
            lambda_q_seed=lambda_q_seed,
            dlmp_seed_bus=seed_bus,
        )
    
    def solve(
        self,
        case: PowerFlowCase,
        orient: bool = True,
        show_net_bus_plot: bool = False,
        show_merit_order_plot: bool = False,
    ) -> OPFResult:
        """
        Solve BFSA for given case.
        
        Args:
            case: PowerFlowCase object
            orient: If True, auto-orient network if not already done
            show_net_bus_plot: If True, show line plot of per-bus net active demand (P_d - P_g)
            show_merit_order_plot: If True, show dummy merit-order plots for active and reactive clearing
        
        Returns:
            OPFResult with voltages, flows, duals, and convergence info
        """
        electrical_result = self.solve_electrical(
            case=case,
            orient=orient,
            show_net_bus_plot=show_net_bus_plot,
            show_merit_order_plot=show_merit_order_plot,
        )

        price_comp_start = time.time()
        dlmp = self.propagate_dlmp(case=case, electrical_result=electrical_result)
        price_time = time.time() - price_comp_start

        solve_time = float(electrical_result.solve_time or 0.0) + price_time
        # print(f"BFSA total solve time: {solve_time*1000:.3f} ms (electrical: {electrical_result.solve_time*1000:.3f}, DLMP: {price_time*1000:.3f})")

        convergence_info = dict(electrical_result.convergence_info)
        convergence_info["solver_time"] = solve_time
        convergence_info["price_time"] = price_time
        convergence_info["solver_stage"] = "electrical+dlmp"

        return OPFResult(
            voltages=dict(electrical_result.voltages),
            flows=dict(electrical_result.flows),
            duals_p={int(bus_id): float(data["lambda_p"]) for bus_id, data in dlmp.items()},
            duals_q={int(bus_id): float(data["lambda_q"]) for bus_id, data in dlmp.items()},
            generator_output=dict(electrical_result.generator_output),
            cost=float(electrical_result.cost),
            convergence_info=convergence_info,
            branch_currents=dict(electrical_result.branch_currents),
            solver_name="BFSA",
            solve_time=solve_time,
        )


class BFSAElectricalSolver(BFSASolver):
    """Public solver that exposes the electrical BFSA stage only."""

    name = "BFSA-ELECTRICAL"
    capabilities = MethodCapabilities(
        produces_physical_state=True,
        produces_economic_state=False,
        produces_dispatch=False,
        supports_warm_start=False,
        requires_physical_state=False,
    )

    def estimate(
        self,
        case: PowerFlowCase,
        *,
        dispatch: Optional[DispatchResult] = None,
        economic_state: Optional[EconomicStateResult] = None,
        previous_physical_state: Optional[PhysicalStateResult] = None,
        orient: bool = True,
        show_net_bus_plot: bool = False,
        show_merit_order_plot: bool = False,
    ) -> PhysicalStateResult:
        """Estimate the physical network state using BFSA."""
        result = self.solve_electrical(
            case=case,
            dispatch=dispatch,
            orient=orient,
            show_net_bus_plot=show_net_bus_plot,
            show_merit_order_plot=show_merit_order_plot,
        )
        physical_state = physical_state_from_opf_result(result)
        physical_state.diagnostics["generator_output"] = dict(result.generator_output)
        physical_state.diagnostics["cost"] = result.cost
        return physical_state

    def solve(
        self,
        case: PowerFlowCase,
        orient: bool = True,
        show_net_bus_plot: bool = False,
        show_merit_order_plot: bool = False,
    ) -> OPFResult:
        return self.solve_electrical(
            case=case,
            orient=orient,
            show_net_bus_plot=show_net_bus_plot,
            show_merit_order_plot=show_merit_order_plot,
        )
