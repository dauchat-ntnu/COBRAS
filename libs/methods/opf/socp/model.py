"""
RadialSOCPModel: Pyomo-based SOCP formulation for radial OPF.
"""

from __future__ import annotations

import math
from typing import Any
import pyomo.kernel as pmo
from libs.shared import PowerFlowCase


class RadialSOCPModel:
    """
    Radial branch-flow SOCP OPF model with explicit directional branch flows.

    Variables
    ---------
    v[i]        : squared voltage magnitude at bus i
    ell[e]      : squared current magnitude on branch e
    p[e], q[e]  : forward active/reactive branch flow P_ij, Q_ij on branch e
    p_rev[e]    : reverse active branch flow P_ji on branch e
    q_rev[e]    : reverse reactive branch flow Q_ji on branch e
    pg[g], qg[g]: active/reactive generation of generator g
    """

    def __init__(
        self,
        case: PowerFlowCase,
        allow_load_shedding: bool = False,
        load_curtailment_penalty_p: float = 1e3,
        load_curtailment_penalty_q: float = 1e3,
    ):
        self.case = case
        self.allow_load_shedding = bool(allow_load_shedding)
        self.load_curtailment_penalty_p = float(load_curtailment_penalty_p)
        self.load_curtailment_penalty_q = float(load_curtailment_penalty_q)
        self.model = pmo.block()
        self._topology_signature: tuple[Any, ...] | None = None

    @staticmethod
    def _case_signature(case: PowerFlowCase, allow_load_shedding: bool) -> tuple[Any, ...]:
        """Return a lightweight signature for detecting topology changes."""
        return (
            bool(allow_load_shedding),
            int(case.root_bus),
            tuple((b.bus_id, b.bus_type, b.base_kv, b.v_max, b.v_min) for b in case.buses),
            tuple((br.branch_id, br.from_bus, br.to_bus, br.status, br.r, br.x, br.b, br.rateA, br.lmax) for br in case.branches),
            tuple((g.gen_id, g.bus_id, g.status) for g in case.generators),
        )

    @staticmethod
    def _make_parameter_dict(values: dict[int, float]) -> pmo.parameter_dict:
        params = pmo.parameter_dict()
        for key, value in values.items():
            params[key] = pmo.parameter(value=float(value))
        return params

    @staticmethod
    def _update_parameter_dict(params: pmo.parameter_dict, values: dict[int, float]) -> None:
        for key, value in values.items():
            params[key].value = float(value)

    @staticmethod
    def _incoming_by_bus(case: PowerFlowCase, bus_ids: list[int]) -> dict[int, list[int]]:
        """Return incoming branches in canonical form: {bus_id: [branch_id, ...]}."""
        incoming_raw = case.incoming_branches or {}
        incoming: dict[int, list[int]] = {i: [] for i in bus_ids}

        # Current shared topology format: {branch_id: to_bus}
        if incoming_raw and all(isinstance(v, int) for v in incoming_raw.values()):
            for branch_id, to_bus in incoming_raw.items():
                if to_bus in incoming:
                    incoming[to_bus].append(branch_id)
            return incoming

        # Alternate format support: {bus_id: branch_id} or {bus_id: [branch_id, ...]}
        for bus_id, value in incoming_raw.items():
            if bus_id not in incoming:
                continue
            if isinstance(value, list):
                incoming[bus_id].extend(value)
            elif value is not None:
                incoming[bus_id].append(value)

        return incoming

    @staticmethod
    def _is_finite_number(value: object) -> bool:
        """Return True when value can be converted to a finite float."""
        try:
            return math.isfinite(float(value))
        except (TypeError, ValueError):
            return False

    @classmethod
    def _validate_scalar_bound(
        cls,
        *,
        symbol: str,
        index: int,
        side: str,
        value: object,
        context: str,
    ) -> None:
        if cls._is_finite_number(value):
            return
        raise ValueError(
            "Invalid SOCP bound detected before solver call: "
            f"{symbol}[{index}] {side}={value!r} is not a finite number. "
            f"Context: {context}"
        )

    @classmethod
    def _validate_interval_bounds(
        cls,
        *,
        symbol: str,
        index: int,
        lb: object,
        ub: object,
        context: str,
    ) -> None:
        cls._validate_scalar_bound(symbol=symbol, index=index, side="lb", value=lb, context=context)
        cls._validate_scalar_bound(symbol=symbol, index=index, side="ub", value=ub, context=context)

        lb_f = float(lb)
        ub_f = float(ub)
        if lb_f > ub_f:
            raise ValueError(
                "Inconsistent SOCP bounds detected before solver call: "
                f"{symbol}[{index}] has lb={lb_f} > ub={ub_f}. "
                f"Context: {context}"
            )

    @classmethod
    def _validate_finite_value(
        cls,
        *,
        symbol: str,
        index: int,
        value: object,
        context: str,
    ) -> None:
        if cls._is_finite_number(value):
            return
        raise ValueError(
            "Invalid SOCP data detected before solver call: "
            f"{symbol}[{index}]={value!r} is not a finite number. "
            f"Context: {context}"
        )

    def build(self, case: PowerFlowCase | None = None):
        if case is not None:
            self.case = case

        case    = self.case
        m       = self.model
        m.allow_load_shedding = self.allow_load_shedding

        bus_ids = [b.bus_id for b in case.buses]

        branch_ids = [br.branch_id for br in case.branches if br.status == 1]

        # Only active generators participate in SOCP dispatch.
        active_gens = [g for g in case.generators if int(getattr(g, "status", 0)) == 1]
        gen_ids     = [g.gen_id for g in active_gens]

        if case.root_bus not in bus_ids:
            raise ValueError("root_bus is not present in bus list.")

        m.BUS       = bus_ids
        m.BRANCH    = branch_ids
        m.GEN       = gen_ids

        bus_map     = {b.bus_id: b for b in case.buses}
        branch_map  = {br.branch_id: br for br in case.branches}
        gen_map     = {g.gen_id: g for g in active_gens}

        loads_by_bus_p = {i: 0.0 for i in bus_ids}
        loads_by_bus_q = {i: 0.0 for i in bus_ids}
        for ld in case.loads:
            if ld.bus_id not in loads_by_bus_p:
                raise ValueError(
                    "Invalid SOCP data detected before solver call: "
                    f"load bus_id={ld.bus_id} is not present in case buses. "
                    f"Known buses: {sorted(bus_ids)}"
                )
            self._validate_finite_value(
                symbol="load_p",
                index=ld.bus_id,
                value=ld.p_d,
                context=f"bus_id={ld.bus_id}, p_d_raw={ld.p_d!r}, q_d_raw={ld.q_d!r}",
            )
            self._validate_finite_value(
                symbol="load_q",
                index=ld.bus_id,
                value=ld.q_d,
                context=f"bus_id={ld.bus_id}, p_d_raw={ld.p_d!r}, q_d_raw={ld.q_d!r}",
            )
            loads_by_bus_p[ld.bus_id] += ld.p_d
            loads_by_bus_q[ld.bus_id] += ld.q_d

        gens_at_bus = {i: [] for i in bus_ids}
        for g in active_gens:
            gens_at_bus[g.bus_id].append(g.gen_id)

        incoming    = self._incoming_by_bus(case, bus_ids)
        outgoing    = case.outgoing_branches
        v_min       = {i: bus_map[i].v_min for i in bus_ids}
        v_max       = {i: bus_map[i].v_max for i in bus_ids}
        r           = {e: branch_map[e].r for e in branch_ids}
        x           = {e: branch_map[e].x for e in branch_ids}
        s_max       = {e: branch_map[e].s_max for e in branch_ids}
        l_max       = {e: branch_map[e].lmax for e in branch_ids}         
        p_g_min     = {g: gen_map[g].p_min for g in gen_ids}
        p_g_max     = {g: gen_map[g].get_p_max_available() for g in gen_ids}
        q_g_min     = {g: gen_map[g].q_min if gen_map[g].q_min_available is None else gen_map[g].q_min_available for g in gen_ids}
        q_g_max     = {g: gen_map[g].get_q_max_available() for g in gen_ids}
        c_p         = {g: gen_map[g].c_p for g in gen_ids}
        c_q         = {g: gen_map[g].c_q for g in gen_ids}

        # Preflight validation: fail early with IDs + source fields instead of opaque solver errors.
        for i in bus_ids:
            bus = bus_map[i]
            cls_context = (
                f"bus_id={i}, root_bus={case.root_bus}, v_min_raw={bus.v_min!r}, "
                f"v_max_raw={bus.v_max!r}"
            )
            self._validate_interval_bounds(
                symbol="v",
                index=i,
                lb=v_min[i],
                ub=v_max[i],
                context=cls_context,
            )
            self._validate_finite_value(
                symbol="load_sum_p",
                index=i,
                value=loads_by_bus_p[i],
                context=f"bus_id={i}, aggregated_p_load={loads_by_bus_p[i]!r}",
            )
            self._validate_finite_value(
                symbol="load_sum_q",
                index=i,
                value=loads_by_bus_q[i],
                context=f"bus_id={i}, aggregated_q_load={loads_by_bus_q[i]!r}",
            )

        for e in branch_ids:
            br = branch_map[e]
            self._validate_finite_value(
                symbol="branch_r",
                index=e,
                value=r[e],
                context=(
                    f"branch_id={e}, from_bus={br.from_bus}, to_bus={br.to_bus}, "
                    f"r_raw={br.r!r}, x_raw={br.x!r}, rateA_raw={br.rateA!r}"
                ),
            )
            self._validate_finite_value(
                symbol="branch_x",
                index=e,
                value=x[e],
                context=(
                    f"branch_id={e}, from_bus={br.from_bus}, to_bus={br.to_bus}, "
                    f"r_raw={br.r!r}, x_raw={br.x!r}, rateA_raw={br.rateA!r}"
                ),
            )
            if float(s_max[e]) < 0.0:
                raise ValueError(
                    "Invalid SOCP data detected before solver call: "
                    f"branch_smax[{e}]={float(s_max[e])} is negative. "
                    f"Context: branch_id={e}, from_bus={br.from_bus}, to_bus={br.to_bus}, rateA_raw={br.rateA!r}"
                )
            self._validate_finite_value(
                symbol="branch_lmax",
                index=e,
                value=l_max[e],
                context=(
                    f"branch_id={e}, from_bus={br.from_bus}, to_bus={br.to_bus}, "
                    f"lmax_raw={br.lmax!r}"
                ),
            )
            if float(l_max[e]) < 0.0:
                raise ValueError(
                    "Invalid SOCP data detected before solver call: "
                    f"branch_lmax[{e}]={float(l_max[e])} is negative. "
                    f"Context: branch_id={e}, from_bus={br.from_bus}, to_bus={br.to_bus}"
                )

        for g in gen_ids:
            gen = gen_map[g]
            p_context = (
                f"gen_id={g}, bus_id={gen.bus_id}, status={gen.status}, "
                f"p_min_raw={gen.p_min!r}, p_max_raw={gen.p_max!r}, "
                f"p_max_available_raw={gen.p_max_available!r}"
            )
            self._validate_interval_bounds(
                symbol="pg",
                index=g,
                lb=p_g_min[g],
                ub=p_g_max[g],
                context=p_context,
            )
            self._validate_finite_value(
                symbol="gen_cost_p",
                index=g,
                value=c_p[g],
                context=f"gen_id={g}, c_p_raw={gen.c_p!r}",
            )

            q_context = (
                f"gen_id={g}, bus_id={gen.bus_id}, status={gen.status}, "
                f"q_min_raw={gen.q_min!r}, q_max_raw={gen.q_max!r}, "
                f"q_max_available_raw={gen.q_max_available!r}"
            )
            self._validate_interval_bounds(
                symbol="qg",
                index=g,
                lb=q_g_min[g],
                ub=q_g_max[g],
                context=q_context,
            )
            self._validate_finite_value(
                symbol="gen_cost_q",
                index=g,
                value=c_q[g],
                context=f"gen_id={g}, c_q_raw={gen.c_q!r}",
            )

        
        root = case.root_bus
        non_root_bus_ids =  [b.bus_id for b in case.buses if b.bus_id != root]


        m.load_p = self._make_parameter_dict(loads_by_bus_p)
        m.load_q = self._make_parameter_dict(loads_by_bus_q)
        m.p_g_min = self._make_parameter_dict(p_g_min)
        m.p_g_max = self._make_parameter_dict(p_g_max)
        m.q_g_min = self._make_parameter_dict(q_g_min)
        m.q_g_max = self._make_parameter_dict(q_g_max)
        m.c_p = self._make_parameter_dict(c_p)
        m.c_q = self._make_parameter_dict(c_q)

        if self.allow_load_shedding:
            m.p_curt_cap = self._make_parameter_dict({i: max(float(loads_by_bus_p.get(i, 0.0)), 0.0) for i in bus_ids})
            m.q_curt_cap = self._make_parameter_dict({i: max(float(loads_by_bus_q.get(i, 0.0)), 0.0) for i in bus_ids})

        m.v = pmo.variable_dict()
        for i in bus_ids:
            m.v[i] = pmo.variable(lb=0.0)

        m.ell = pmo.variable_dict()
        for e in branch_ids:
            m.ell[e] = pmo.variable()

        m.p = pmo.variable_dict()
        m.q = pmo.variable_dict()
        for e in branch_ids:
            m.p[e] = pmo.variable()
            m.q[e] = pmo.variable()

        m.pg = pmo.variable_dict()
        m.qg = pmo.variable_dict()
        for g in gen_ids:
            m.pg[g] = pmo.variable()
            m.qg[g] = pmo.variable()

        if self.allow_load_shedding:
            # Nodal load-curtailment variables (active/reactive).
            m.p_curt = pmo.variable_dict()
            m.q_curt = pmo.variable_dict()
            for i in bus_ids:
                m.p_curt[i] = pmo.variable(lb=0.0)
                m.q_curt[i] = pmo.variable(lb=0.0)
            m.p_curt_upper = pmo.constraint_dict()
            m.q_curt_upper = pmo.constraint_dict()
            for i in bus_ids:
                m.p_curt_upper[i] = pmo.constraint(m.p_curt[i] <= m.p_curt_cap[i])
                m.q_curt_upper[i] = pmo.constraint(m.q_curt[i] <= m.q_curt_cap[i])

        m.p_rev = pmo.expression_dict()
        m.q_rev = pmo.expression_dict()
        for e in branch_ids:
            m.p_rev[e] = pmo.expression(m.p[e] - r[e] * m.ell[e])
            m.q_rev[e] = pmo.expression(m.q[e] - x[e] * m.ell[e])

        m.dual = pmo.suffix(direction=pmo.suffix.IMPORT)

        objective_expr = sum(m.c_p[g] * m.pg[g] + m.c_q[g] * m.qg[g] for g in gen_ids)
        if self.allow_load_shedding:
            objective_expr += self.load_curtailment_penalty_p * sum(m.p_curt[i] for i in bus_ids)
            objective_expr += self.load_curtailment_penalty_q * sum(m.q_curt[i] for i in bus_ids)

        m.obj = pmo.objective(objective_expr, sense=pmo.minimize)

        m.v_ref = pmo.constraint(m.v[root] == 1.0) #TODO in the future when cascaded  radial grids are used only the root of the top-level should be set to vref

        m.v_upper = pmo.constraint_dict()
        m.v_lower = pmo.constraint_dict()
        for i in non_root_bus_ids:
            m.v_upper[i] = pmo.constraint(m.v[i] <= v_max[i])
            m.v_lower[i] = pmo.constraint(m.v[i] >= v_min[i])

        m.ell_upper = pmo.constraint_dict()
        m.ell_lower = pmo.constraint_dict()
        for e in branch_ids:
            m.ell_upper[e] = pmo.constraint(m.ell[e] <= l_max[e])
            m.ell_lower[e] = pmo.constraint(m.ell[e] >= 0.0)

        m.pg_lower = pmo.constraint_dict()
        m.pg_upper = pmo.constraint_dict()
        m.qg_lower = pmo.constraint_dict()
        m.qg_upper = pmo.constraint_dict()
        for g in gen_ids:
            m.pg_lower[g] = pmo.constraint(m.pg[g] >= m.p_g_min[g])
            m.pg_upper[g] = pmo.constraint(m.pg[g] <= m.p_g_max[g])
            m.qg_lower[g] = pmo.constraint(m.qg[g] >= m.q_g_min[g])
            m.qg_upper[g] = pmo.constraint(m.qg[g] <= m.q_g_max[g])

        m.p_balance = pmo.constraint_dict()
        m.q_balance = pmo.constraint_dict()
        for i in bus_ids:
            gen_term_p = sum(m.pg[g] for g in gens_at_bus[i])
            gen_term_q = sum(m.qg[g] for g in gens_at_bus[i])
            incoming_term_p = sum(m.p_rev[e] for e in incoming.get(i, []))
            incoming_term_q = sum(m.q_rev[e] for e in incoming.get(i, []))
            outgoing_term_p = sum(m.p[e] for e in outgoing.get(i, []))
            outgoing_term_q = sum(m.q[e] for e in outgoing.get(i, []))

            if self.allow_load_shedding:
                m.p_balance[i] = pmo.constraint(
                    gen_term_p - m.load_p[i] + m.p_curt[i] + incoming_term_p - outgoing_term_p == 0
                )
                m.q_balance[i] = pmo.constraint(
                    gen_term_q - m.load_q[i] + m.q_curt[i] + incoming_term_q - outgoing_term_q == 0
                )
            else:
                m.p_balance[i] = pmo.constraint(
                    gen_term_p - m.load_p[i] + incoming_term_p - outgoing_term_p == 0
                )
                m.q_balance[i] = pmo.constraint(
                    gen_term_q - m.load_q[i] + incoming_term_q - outgoing_term_q == 0
                )

        m.voltage_drop = pmo.constraint_dict()
        for e in branch_ids:
            br = branch_map[e]
            i = br.from_bus
            j = br.to_bus
            m.voltage_drop[e] = pmo.constraint(
                m.v[j] - m.v[i]
                + 2.0 * (r[e] * m.p[e] + x[e] * m.q[e])
                - (r[e] ** 2 + x[e] ** 2) * m.ell[e]
                == 0
            )

        # p^2 + q^2 <= ell * v encoded as a rotated quadratic cone.
        sqrt2 = 2.0 ** 0.5
        m.soc = pmo.block_dict()
        for e in branch_ids:
            i = branch_map[e].from_bus
            m.soc[e] = pmo.conic.rotated_quadratic.as_domain(
                r1=m.ell[e],
                r2=m.v[i],
                x=[sqrt2 * m.p[e], sqrt2 * m.q[e]],
            )

        # # p^2 + q^2 <= s_max^2 encoded as a quadratic cone.
        # m.branch_limit = pmo.block_dict()
        # for e in branch_ids:
        #     m.branch_limit[e] = pmo.conic.quadratic.as_domain(
        #         r=s_max[e],
        #         x=[m.p[e], m.q[e]],
        #     )

        self._topology_signature = self._case_signature(case, self.allow_load_shedding)
        return m

    def update_case(self, case: PowerFlowCase):
        """Update mutable parameters for a new timestep without rebuilding the model."""
        if self._topology_signature != self._case_signature(case, self.allow_load_shedding):
            return self.build(case)

        self.case = case
        m = self.model

        bus_ids = [b.bus_id for b in case.buses]
        active_gens = [g for g in case.generators if int(getattr(g, "status", 0)) == 1]

        loads_by_bus_p = {i: 0.0 for i in bus_ids}
        loads_by_bus_q = {i: 0.0 for i in bus_ids}
        for ld in case.loads:
            loads_by_bus_p[ld.bus_id] += float(ld.p_d)
            loads_by_bus_q[ld.bus_id] += float(ld.q_d)

        p_g_min = {g.gen_id: float(g.p_min) for g in active_gens}
        p_g_max = {g.gen_id: float(g.get_p_max_available()) for g in active_gens}
        q_g_min = {g.gen_id: float(g.q_min if g.q_min_available is None else g.q_min_available) for g in active_gens}
        q_g_max = {g.gen_id: float(g.get_q_max_available()) for g in active_gens}
        c_p = {g.gen_id: float(g.c_p) for g in active_gens}
        c_q = {g.gen_id: float(g.c_q) for g in active_gens}

        self._update_parameter_dict(m.load_p, loads_by_bus_p)
        self._update_parameter_dict(m.load_q, loads_by_bus_q)
        self._update_parameter_dict(m.p_g_min, p_g_min)
        self._update_parameter_dict(m.p_g_max, p_g_max)
        self._update_parameter_dict(m.q_g_min, q_g_min)
        self._update_parameter_dict(m.q_g_max, q_g_max)
        self._update_parameter_dict(m.c_p, c_p)
        self._update_parameter_dict(m.c_q, c_q)

        if self.allow_load_shedding:
            self._update_parameter_dict(m.p_curt_cap, {i: max(float(loads_by_bus_p.get(i, 0.0)), 0.0) for i in bus_ids})
            self._update_parameter_dict(m.q_curt_cap, {i: max(float(loads_by_bus_q.get(i, 0.0)), 0.0) for i in bus_ids})

        return m
