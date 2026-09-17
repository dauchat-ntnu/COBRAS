"""Dummy merit-order dispatch estimator method."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

from libs.methods.base import MethodSpec
from libs.methods.defaults import get_method_defaults, get_method_option_schema
from libs.shared import (
    DispatchResult,
    EconomicStateResult,
    MethodCapabilities,
    PhysicalStateResult,
    PowerFlowCase,
)


class DummyMeritOrderDispatchEstimator:
    """Clear independent active/reactive dummy merit orders."""

    name = "DUMMY-MERIT-ORDER"
    capabilities = MethodCapabilities(
        produces_physical_state=False,
        produces_economic_state=False,
        produces_dispatch=True,
        supports_warm_start=False,
        requires_physical_state=False,
        requires_dispatch=False,
    )

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}

    @staticmethod
    def _get_offer_quantity(gen, quantity_attr: str) -> float:
        """Resolve offer quantity with profile-aware p_max/q_max support."""
        if quantity_attr == "p_max" and hasattr(gen, "get_p_max_available"):
            quantity = float(gen.get_p_max_available())
        elif quantity_attr == "q_max" and hasattr(gen, "get_q_max_available"):
            quantity = float(gen.get_q_max_available())
        else:
            quantity = float(getattr(gen, quantity_attr, 0.0))
        return float(max(quantity, 0.0))

    @classmethod
    def _clear_component(
        cls,
        case: PowerFlowCase,
        total_load: float,
        quantity_attr: str,
        cost_attr: str,
    ) -> Dict[str, object]:
        """Clear one merit-order component and return clearing metadata."""
        dispatch = {gen.gen_id: 0.0 for gen in case.generators}
        total_load = float(total_load)

        active_generators = [
            gen
            for gen in case.generators
            if int(gen.status) > 0 and cls._get_offer_quantity(gen, quantity_attr) > 0.0
        ]

        offers = []
        for gen in active_generators:
            offers.append(
                {
                    "gen_id": int(gen.gen_id),
                    "bus_id": int(gen.bus_id),
                    "name": f"G{gen.gen_id}@B{gen.bus_id}",
                    "capacity": cls._get_offer_quantity(gen, quantity_attr),
                    "cost": float(getattr(gen, cost_attr, 0.0)),
                    "is_load_excess": False,
                }
            )

        total_capacity = float(sum(float(offer["capacity"]) for offer in offers))
        load_excess_capacity = max(total_load - total_capacity, 0.0)
        offers.append(
            {
                "gen_id": -1,
                "bus_id": int(case.root_bus),
                "name": "LOAD_EXCESS",
                "capacity": float(load_excess_capacity),
                "cost": 0.0,
                "is_load_excess": True,
            }
        )

        merit = sorted(
            offers,
            key=lambda offer: (float(offer["cost"]), -float(offer["capacity"]), int(offer["gen_id"])),
        )

        if total_load <= 0.0:
            return {
                "marginal_price": 0.0,
                "setting_bus": int(case.root_bus),
                "setting_gen_id": -1,
                "dispatch": dispatch,
                "offers_sorted": merit,
                "total_capacity": total_capacity,
                "load_excess_capacity": load_excess_capacity,
                "total_load": total_load,
            }

        remaining = total_load
        setting_offer = merit[-1]
        for offer in merit:
            available = float(offer["capacity"])
            take = min(available, max(remaining, 0.0))
            gen_id = int(offer["gen_id"])
            if gen_id >= 0:
                dispatch[gen_id] = take
            if take > 0.0:
                setting_offer = offer
            remaining -= take
            if remaining <= 1e-12:
                break

        return {
            "marginal_price": float(setting_offer["cost"]),
            "setting_bus": int(setting_offer["bus_id"]),
            "setting_gen_id": int(setting_offer["gen_id"]),
            "dispatch": dispatch,
            "offers_sorted": merit,
            "total_capacity": total_capacity,
            "load_excess_capacity": load_excess_capacity,
            "total_load": total_load,
        }

    @staticmethod
    def _aggregate_generator_dispatch_by_bus(
        case: PowerFlowCase,
        generator_output: Dict[int, Tuple[float, float]],
    ) -> Tuple[Dict[int, float], Dict[int, float]]:
        p_gen_by_bus = {bus.bus_id: 0.0 for bus in case.buses}
        q_gen_by_bus = {bus.bus_id: 0.0 for bus in case.buses}
        for gen in case.generators:
            p_gen, q_gen = generator_output.get(gen.gen_id, (0.0, 0.0))
            p_gen_by_bus[gen.bus_id] = p_gen_by_bus.get(gen.bus_id, 0.0) + float(p_gen)
            q_gen_by_bus[gen.bus_id] = q_gen_by_bus.get(gen.bus_id, 0.0) + float(q_gen)
        return p_gen_by_bus, q_gen_by_bus

    def estimate(
        self,
        case: PowerFlowCase,
        *,
        physical_state: Optional[PhysicalStateResult] = None,
        economic_state: Optional[EconomicStateResult] = None,
        previous_dispatch: Optional[DispatchResult] = None,
    ) -> DispatchResult:
        """Estimate dispatch through independent P/Q dummy merit-order clearing."""
        total_p_load = float(sum(load.p_d for load in case.loads))
        total_q_load = float(sum(load.q_d for load in case.loads))

        p_clearing = self._clear_component(
            case=case,
            total_load=total_p_load,
            quantity_attr="p_max",
            cost_attr="c_p",
        )
        q_clearing = self._clear_component(
            case=case,
            total_load=total_q_load,
            quantity_attr="q_max",
            cost_attr="c_q",
        )

        dispatch_p = p_clearing["dispatch"]
        dispatch_q = q_clearing["dispatch"]
        generator_output = {
            gen.gen_id: (
                float(dispatch_p.get(gen.gen_id, 0.0)),
                float(dispatch_q.get(gen.gen_id, 0.0)),
            )
            for gen in case.generators
        }
        p_gen_by_bus, q_gen_by_bus = self._aggregate_generator_dispatch_by_bus(case, generator_output)

        lambda_p_clear = float(p_clearing["marginal_price"])
        lambda_q_clear = float(q_clearing["marginal_price"])
        active_cost = 0.0 if total_p_load <= 0.0 else float(lambda_p_clear * total_p_load * case.base_mva)
        reactive_cost = 0.0 if total_q_load <= 0.0 else float(lambda_q_clear * total_q_load * case.base_mva)

        return DispatchResult(
            generator_output=generator_output,
            objective=active_cost + reactive_cost,
            market_clearing={
                "p_clearing": p_clearing,
                "q_clearing": q_clearing,
                "total_p_load": total_p_load,
                "total_q_load": total_q_load,
                "lambda_p_clear": lambda_p_clear,
                "lambda_q_clear": lambda_q_clear,
                "p_gen_by_bus": p_gen_by_bus,
                "q_gen_by_bus": q_gen_by_bus,
                "price_source": "dummy",
            },
            diagnostics={
                "dispatch_stage": "dummy_merit_order",
                "p_gen_by_bus": p_gen_by_bus,
                "q_gen_by_bus": q_gen_by_bus,
            },
            method_name=self.name,
        )


def build_dummy_merit_order_dispatch(**options):
    """Build the dummy merit-order dispatch estimator."""
    return DummyMeritOrderDispatchEstimator(config=dict(options))


DUMMY_MERIT_ORDER_METHOD = MethodSpec(
    family="dispatch",
    key="dummy_merit_order",
    label="Dummy merit-order dispatch",
    description="Simple merit-order dispatch estimator used as the baseline dispatch method.",
    builder=build_dummy_merit_order_dispatch,
    capabilities=DummyMeritOrderDispatchEstimator.capabilities,
    default_options=get_method_defaults("dispatch", "dummy_merit_order"),
    option_schema=get_method_option_schema("dispatch", "dummy_merit_order"),
)
