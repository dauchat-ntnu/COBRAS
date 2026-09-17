"""
Core BFSA algorithm - no external solver dependency.

Implements backward-forward sweep for power flow computation
and DLMP propagation in radial networks.
"""

import time
import numpy as np
import networkx as nx
from collections import deque
from typing import Dict, List, Tuple, Optional


_DLMP_EPS = 1e-10


def _normalize_bfsa_convergence_check(value: object) -> str:
    """Return the configured BFSA convergence criterion."""
    check = str(value or "both").strip().lower()
    if check in {"current", "currents", "ell", "loss", "losses"}:
        return "current"
    if check in {"voltage", "voltages", "v"}:
        return "voltage"
    if check in {"both", "all"}:
        return "both"
    raise ValueError(
        "BFSA convergence_check must be one of 'voltage', 'current', or 'both'."
    )


def bfsa_pq_case1(cphv: float, cqhv: float, pint: float, qint: float, xpu: float, rpu: float, vpu: float = 1.0) -> Tuple[float, float]:
    """
    Compute DLMPs in case 1 (pint, qint >= 0).
    
    Backward propagation of locational marginal prices through line impedances.
    
    Parameters:
        cphv, cqhv (float): Parent node costs (LMPs)
        pint, qint (float): Line power flows (p.u.)
        xpu, rpu (float): Line reactance and resistance (p.u.)
        vpu (float): Squared Parent voltage (p.u.)
    
    Returns:
        tuple: (lambda_p, lambda_q) child node costs (DLMPs)
    """
    d = (vpu - 2*xpu*qint - 2*rpu*pint)
    if abs(d) < 1e-10:
        # Avoid division by zero; return parent costs
        return cphv, cqhv
    
    lambda_p = (cphv*(vpu-2*xpu*qint) + cqhv*2*xpu*pint) / d
    lambda_q = (cphv*(2*rpu*qint) + cqhv*(vpu-2*rpu*pint)) / d
    
    return lambda_p, lambda_q


def bfsa_inverse_pq_case1(
    lambda_p_child: float,
    lambda_q_child: float,
    pint: float,
    qint: float,
    xpu: float,
    rpu: float,
    vpu: float = 1.0,
) -> Tuple[float, float]:
    """
    Recover parent DLMPs from child DLMPs for case 1.

    This is the inverse mapping of bfsa_pq_case1 and is used when propagating
    prices upstream from an arbitrary seed bus.
    """
    d = (vpu - 2*xpu*qint - 2*rpu*pint)

    a11 = (vpu - 2*xpu*qint)
    a12 = 2*xpu*pint
    a21 = 2*rpu*qint
    a22 = (vpu - 2*rpu*pint)

    det = a11*a22 - a12*a21
    if abs(det) < 1e-10:
        return lambda_p_child, lambda_q_child

    rhs_p = d * lambda_p_child
    rhs_q = d * lambda_q_child

    lambda_p_parent = (a22*rhs_p - a12*rhs_q) / det
    lambda_q_parent = (-a21*rhs_p + a11*rhs_q) / det
    return lambda_p_parent, lambda_q_parent


def bfsa_pq_cases(cp: float, cq: float, pint: float, qint: float, xpu: float, rpu: float, vp:float) -> Tuple[float, float]:
    """
    Compute DLMPs in general 
    Backward propagation of locational marginal prices through line impedances.
    
    Parameters:
        cp, cq (float): Parent node costs (LMPs)
        pint, qint (float): Line power flows (p.u.)
        xpu, rpu (float): Line reactance and resistance (p.u.)
        vp: parent voltage
    
    Returns:
        tuple: (lambda_p, lambda_q) child node costs (DLMPs)
    """
    #case 1
    if pint==0 and qint==0:  
        lambda_p = cp 
        lambda_q = cq 

    elif pint>=0 and qint >=0:
        d = (vp - 2*xpu*qint - 2*rpu*pint)
        if abs(d) < 1e-10:
            # Avoid division by zero; return parent costs
            return cp,cq 
        else:
            lambda_p = (cp*(vp-2*xpu*qint) + cq*2*xpu*pint) / d
            lambda_q = (cp*(2*rpu*qint) + cq*(vp-2*rpu*pint)) / d

    #case 2
    elif pint<=0 and qint <=0:
        lambda_p = (cp*(vp-2*rpu*pint) - cq*2*xpu*pint) / vp
        lambda_q = (cq*(vp-2*xpu*qint) - cp*(2*rpu*qint)) / vp
    #case 3:
    elif pint>=0 and qint <=0:
        d = (vp - 2*rpu*pint)
        lambda_p = (cp*vp + cq*2*xpu*pint) / d
        lambda_q = (cq*(vp-2*xpu*qint-2*rpu*pint) - cp*(2*rpu*qint)) / d 
    #case 4
    elif pint<=0 and qint >=0:
        d = (vp - 2*xpu*qint)
        lambda_p =  (cp*(vp-2*xpu*qint-2*rpu*pint) - cq*(2*xpu*pint)) / d 
        lambda_q =  (cq*vp + cp*2*rpu*qint) / d

    

    return lambda_p, lambda_q


def _linear_radial_path(T: nx.DiGraph, root: int) -> Optional[List[int]]:
    """Return root-to-leaf node order when the directed radial is a single path."""
    root = int(root)
    if root not in T.nodes:
        return None

    path = [root]
    current = root
    seen = {root}

    while True:
        successors = [int(node) for node in T.successors(current)]
        if len(successors) == 0:
            break
        if len(successors) > 1:
            return None

        current = successors[0]
        if current in seen:
            return None
        seen.add(current)
        path.append(current)

    return path if len(path) == len(T.nodes) else None


def _radial_case_for_path(
    path: List[int],
    line_state: Dict[Tuple[int, int], Dict[str, float]],
) -> Optional[int]:
    """Classify a path as all-case-3 or all-case-4 from branch P/Q signs."""
    edge_cases = []
    for i, j in zip(path[:-1], path[1:]):
        state = line_state[(int(i), int(j))]
        pint = float(state.get("ptrans", 0.0))
        qint = float(state.get("qtrans", 0.0))

        if pint >= -_DLMP_EPS and qint <= _DLMP_EPS:
            edge_cases.append(3)
        elif pint <= _DLMP_EPS and qint >= -_DLMP_EPS:
            edge_cases.append(4)
        else:
            return None

    if edge_cases and all(case == 3 for case in edge_cases):
        return 3
    if edge_cases and all(case == 4 for case in edge_cases):
        return 4
    return None


def _case3_blocks(
    state: Dict[str, float],
    rpu: float,
    xpu: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build U/L blocks for case 3 with vector order [lambda_p, lambda_q]."""
    pint = float(state.get("ptrans", 0.0))
    qint = float(state.get("qtrans", 0.0))
    v_i = float(state.get("v_start_sq", 1.0))
    d = v_i - 2.0 * rpu * pint

    u = np.array(
        [
            [d, -2.0 * xpu * pint],
            [0.0, d - 2.0 * xpu * qint],
        ],
        dtype=float,
    )
    l = np.array(
        [
            [-v_i, 0.0],
            [2.0 * rpu * qint, d],
        ],
        dtype=float,
    )
    return u, l


def _case4_blocks(
    state: Dict[str, float],
    rpu: float,
    xpu: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build U/L blocks for case 4 with vector order [lambda_q, lambda_p]."""
    pint = float(state.get("ptrans", 0.0))
    qint = float(state.get("qtrans", 0.0))
    v_i = float(state.get("v_start_sq", 1.0))
    d = v_i - 2.0 * rpu * pint
    d_prime = v_i - 2.0 * xpu * qint

    u = np.array(
        [
            [d_prime, -2.0 * rpu * qint],
            [0.0, d - 2.0 * xpu * qint],
        ],
        dtype=float,
    )
    l = np.array(
        [
            [-v_i, 0.0],
            [2.0 * xpu * pint, d_prime],
        ],
        dtype=float,
    )
    return u, l


def _solve_split_boundary_bidiagonal(
    path: List[int],
    blocks: List[Tuple[np.ndarray, np.ndarray]],
    root_known_index: int,
    root_known_value: float,
    leaf_known_index: int,
    leaf_known_value: float,
) -> Optional[Dict[int, np.ndarray]]:
    """
    Solve U_k x_{k+1} + L_k x_k = 0 with one known root component
    and the complementary known leaf component.

    The unknown root component is kept as an affine scalar through the forward
    sweep, then fixed by the leaf boundary condition.
    """
    root_unknown_index = 1 - root_known_index

    base_vectors = {
        int(path[0]): np.array([0.0, 0.0], dtype=float),
    }
    basis_vectors = {
        int(path[0]): np.array([0.0, 0.0], dtype=float),
    }
    base_vectors[int(path[0])][root_known_index] = float(root_known_value)
    basis_vectors[int(path[0])][root_unknown_index] = 1.0

    try:
        for edge_index, child in enumerate(path[1:]):
            parent = int(path[edge_index])
            child = int(child)
            u, l = blocks[edge_index]

            base_rhs = -l @ base_vectors[parent]
            basis_rhs = -l @ basis_vectors[parent]
            base_vectors[child] = np.linalg.solve(u, base_rhs)
            basis_vectors[child] = np.linalg.solve(u, basis_rhs)
    except np.linalg.LinAlgError:
        return None

    leaf = int(path[-1])
    denominator = float(basis_vectors[leaf][leaf_known_index])
    if abs(denominator) < _DLMP_EPS:
        return None

    alpha = (float(leaf_known_value) - float(base_vectors[leaf][leaf_known_index])) / denominator
    return {
        int(bus): base_vectors[int(bus)] + alpha * basis_vectors[int(bus)]
        for bus in path
    }


def solve_case3_case4_radial_dlmp(
    T: nx.DiGraph,
    line_state: Dict[Tuple[int, int], Dict[str, float]],
    line_lookup_undirected: Dict[Tuple[int, int], object],
    root_bus: int,
    lambda_p_seed: float,
    lambda_q_seed: float,
) -> Optional[Dict[int, Dict[str, float]]]:
    """
    Solve case-3/case-4 DLMP equations on a linear radial using a forward
    block-bidiagonal sweep. Returns None when the radial is not a supported
    all-case-3/all-case-4 path.
    """
    path = _linear_radial_path(T, root_bus)
    if path is None or len(path) < 2:
        return None

    radial_case = _radial_case_for_path(path, line_state)
    if radial_case is None:
        return None

    blocks = []
    for i, j in zip(path[:-1], path[1:]):
        key_undir = tuple(sorted((int(i), int(j))))
        br = line_lookup_undirected[key_undir]
        rpu = float(br.r)
        xpu = float(br.x)
        state = line_state[(int(i), int(j))]

        if radial_case == 3:
            blocks.append(_case3_blocks(state, rpu=rpu, xpu=xpu))
        else:
            blocks.append(_case4_blocks(state, rpu=rpu, xpu=xpu))

    if radial_case == 3:
        vectors = _solve_split_boundary_bidiagonal(
            path=path,
            blocks=blocks,
            root_known_index=0,
            root_known_value=lambda_p_seed,
            leaf_known_index=1,
            leaf_known_value=lambda_q_seed,
        )
        if vectors is None:
            return None
        return {
            int(bus): {"lambda_p": float(vec[0]), "lambda_q": float(vec[1])}
            for bus, vec in vectors.items()
        }

    vectors = _solve_split_boundary_bidiagonal(
        path=path,
        blocks=blocks,
        root_known_index=0,
        root_known_value=lambda_q_seed,
        leaf_known_index=1,
        leaf_known_value=lambda_p_seed,
    )
    if vectors is None:
        return None
    return {
        int(bus): {"lambda_p": float(vec[1]), "lambda_q": float(vec[0])}
        for bus, vec in vectors.items()
    }


def propagate_dlmp_from_seed(
    T: nx.DiGraph,
    line_state: Dict[Tuple[int, int], Dict[str, float]],
    line_lookup_undirected: Dict[Tuple[int, int], object],
    seed_bus: int,
    lambda_p_seed: float,
    lambda_q_seed: float,
) -> Dict[int, Dict[str, float]]:
    """
    Propagate DLMPs over a radial tree from an arbitrary seed bus.

    Handles both downstream and upstream traversals by using the direct
    bfsa_pq_case1 mapping on forward edges and case2 on reverse edges.
    """
    if seed_bus not in T.nodes:
        raise ValueError(f"Seed bus {seed_bus} is not in radial component")

    dlmp = {int(seed_bus): {"lambda_p": float(lambda_p_seed), "lambda_q": float(lambda_q_seed)}}
    queue = deque([int(seed_bus)])
    visited = {int(seed_bus)}

    while queue:
        current = int(queue.popleft())

        # Traverse the radial component as an undirected graph.
        for neighbor in T.predecessors(current):
            neighbor = int(neighbor)
            if neighbor in visited:
                continue

            key_undir = tuple(sorted((neighbor, current)))
            br = line_lookup_undirected[key_undir]
            xpu = float(br.x)
            rpu = float(br.r)

            st = line_state[(neighbor, current)]
            lambda_p_neighbor, lambda_q_neighbor = bfsa_inverse_pq_case1(
                lambda_p_child=dlmp[current]["lambda_p"],
                lambda_q_child=dlmp[current]["lambda_q"],
                pint=st["ptrans"],
                qint=st["qtrans"],
                xpu=xpu,
                rpu=rpu,
                vpu=st["v_start_sq"],

            )
            dlmp[neighbor] = {"lambda_p": lambda_p_neighbor, "lambda_q": lambda_q_neighbor}
            visited.add(neighbor)
            queue.append(neighbor)

        for neighbor in T.successors(current):
            neighbor = int(neighbor)
            if neighbor in visited:
                continue

            key_undir = tuple(sorted((current, neighbor)))
            br = line_lookup_undirected[key_undir]
            xpu = float(br.x)
            rpu = float(br.r)

            st = line_state[(current, neighbor)]
            # print (f"{current} -> {neighbor}: cphv={dlmp[current]['lambda_p']:.4f}, cqhv={dlmp[current]['lambda_q']:.4f}, pint={st['ptrans']:.4f}, qint={st['qtrans']:.4f}, xpu={xpu:.4f}, rpu={rpu:.4f}, vpu={st['v_end_sq']:.4f}")
            lambda_p_neighbor, lambda_q_neighbor = bfsa_pq_case1(
                cphv=dlmp[current]["lambda_p"],
                cqhv=dlmp[current]["lambda_q"],
                pint=st["ptrans"],
                qint=st["qtrans"],
                xpu=xpu,
                rpu=rpu,
                vpu=st["v_end_sq"]
            )
            dlmp[neighbor] = {"lambda_p": lambda_p_neighbor, "lambda_q": lambda_q_neighbor}
            visited.add(neighbor)
            queue.append(neighbor)

    return dlmp


def bfsa_electrical(
    radial_root: int,
    G: nx.DiGraph,
    p_bus: Dict[int, float],
    q_bus: Dict[int, float],
    line_lookup_undirected: Dict[Tuple[int, int], object],
    v_root_sq: float,
    config: Dict,
) -> Dict:
    """
    Run BFSA on a single radial component.
    
    Parameters:
        radial_root (int): Root bus ID of this radial tree
        G (nx.DiGraph): Full rooted forest (directed)
        p_bus, q_bus (dict): Net bus demands {bus_id: demand - generation}
        line_lookup_undirected (dict): Line parameters {(min_bus, max_bus): BranchData}
        v_root_sq (float): Root voltage squared (p.u.²), typically 1.0
        config (dict): BFSA config with keys:
            - compute_losses: bool (default True)
            - tol: convergence tolerance (default 1e-3)
            - max_iterations: max BFS iterations (default 100)
            - convergence_check: voltage, current, or both (default both)

    
    Returns:
        dict: Results with keys:
            - 'v_sq_local': {bus: V²}
            - 'line_state': {(i,j): {ptrans, qtrans, ploss, qloss, ...}}
            - 'converged': bool
            - 'iterations': int
            - 'final_mismatch': selected convergence mismatch
            - 'voltage_mismatch': final voltage mismatch
            - 'current_mismatch': final current-squared mismatch
    """
    
    root_int    = int(radial_root)
    T           = nx.dfs_tree(G, source=root_int)
    edges_dir   = [(int(i), int(j)) for i, j in T.edges()]
    nodes_radial= [int(n) for n in T.nodes()]
    
    preorder    = list(nx.topological_sort(T))
    postorder   = list(reversed(preorder))
    
    v_sq_local  = {n: v_root_sq for n in nodes_radial}
    compute_losses = config.get("compute_losses", True)
    max_iter    = int(config.get("max_iterations", 2))
    tol         = config.get("tol", 1e-3)
    convergence_check = _normalize_bfsa_convergence_check(
        config.get("convergence_check", config.get("bfsa_convergence_check", "both"))
    )
    

    
    p_edge = {(i, j): float(p_bus.get(j, 0.0)) for (i, j) in edges_dir}
    q_edge = {(i, j): float(q_bus.get(j, 0.0)) for (i, j) in edges_dir}
    l_edge = {(i, j): 0.0 for (i, j) in edges_dir}
    
    converged = False
    final_mismatch = np.nan
    voltage_mismatch = np.nan
    current_mismatch = np.nan
    n_iter = 0
    
    for it in range(1, max_iter + 1):
        n_iter = it
        v_sq_prev = v_sq_local.copy()
        l_edge_prev = l_edge.copy()
        
        # Backward sweep: accumulate net demand + line losses.
        # Negative p_bus / q_bus values represent net injection and naturally
        # reverse the branch flow direction toward the parent.
        for j in postorder:
            if j == root_int:
                continue
            
            parent = int(next(T.predecessors(j)))
            children = [int(c) for c in T.successors(j)]
            
            p_children = sum(p_edge.get((j, c), 0.0) for c in children)
            q_children = sum(q_edge.get((j, c), 0.0) for c in children)
            
            key_undir = tuple(sorted((parent, j)))
            if key_undir not in line_lookup_undirected:
                raise KeyError(f"Line parameters missing for edge {key_undir}")
            
            br = line_lookup_undirected[key_undir]
            rpu, xpu = float(br.r), float(br.x)
            
            p_edge[(parent, j)] = float(p_bus.get(j, 0.0)) + p_children + rpu * l_edge.get((parent, j), 0.0)
            q_edge[(parent, j)] = float(q_bus.get(j, 0.0)) + q_children + xpu * l_edge.get((parent, j), 0.0)
        
        # Forward sweep: update losses and voltages
        v_sq_local[root_int] = v_root_sq
        l_new = {}
        
        for i in preorder:
            for j in T.successors(i):
                i, j = int(i), int(j)
                key_undir = tuple(sorted((i, j)))
                
                if key_undir not in line_lookup_undirected:
                    raise KeyError(f"Line parameters missing for edge {key_undir}")
                
                br = line_lookup_undirected[key_undir]
                rpu, xpu = float(br.r), float(br.x)
                pij = float(p_edge.get((i, j), 0.0))
                qij = float(q_edge.get((i, j), 0.0))
                
                v_start_sq = float(v_sq_local[i])
                v_safe = max(v_start_sq, 1e-9)
                ell = (pij**2 + qij**2) / v_safe
                v_end_sq = v_start_sq - 2.0 * (rpu*pij + xpu*qij) + (rpu**2 + xpu**2)*ell
                
                l_new[(i, j)] = ell
                v_sq_local[j] = max(v_end_sq, 0.1)
        
        # Check convergence
        if compute_losses:
            voltage_rel_errs = [
                abs(v_sq_local[n] - v_sq_prev[n]) / max(abs(v_sq_prev[n]), 1e-9)
                for n in nodes_radial
            ]
            current_rel_errs = [
                abs(l_new[e] - l_edge_prev.get(e, 0.0))
                / max(abs(l_edge_prev.get(e, 0.0)), 1e-9)
                for e in l_new
            ]
            voltage_mismatch = max(voltage_rel_errs) if voltage_rel_errs else 0.0
            current_mismatch = max(current_rel_errs) if current_rel_errs else 0.0

            if convergence_check == "voltage":
                final_mismatch = voltage_mismatch
            elif convergence_check == "current":
                final_mismatch = current_mismatch
            else:
                final_mismatch = max(voltage_mismatch, current_mismatch)
            
            if final_mismatch < tol:
                converged = True
                l_edge = l_new
                break
        else:
            converged = True
            final_mismatch = 0.0
            voltage_mismatch = 0.0
            current_mismatch = 0.0
            l_edge = l_new
            break

        l_edge = l_new
    
    
    
    # Build line_state dict
    line_state = {}
    for (i, j) in edges_dir:
        key_undir = tuple(sorted((i, j)))
        br = line_lookup_undirected[key_undir]
        
        rpu, xpu = float(br.r), float(br.x)
        pij = float(p_edge.get((i, j), 0.0))
        qij = float(q_edge.get((i, j), 0.0))
        ell = float(l_edge.get((i, j), 0.0))
        v_start_sq = float(v_sq_local[i])
        v_end_sq = float(v_sq_local[j])
        ploss, qloss = rpu*ell, xpu*ell
        
        line_state[(i, j)] = {
            "ptrans": pij,
            "qtrans": qij,
            "strans": np.sqrt(pij**2 + qij**2),
            "ell": ell,
            "v_start_sq": v_start_sq,
            "v_end_sq": v_end_sq,
            "ploss": ploss,
            "qloss": qloss,
        }
    
   
    return {
        "v_sq_local": v_sq_local,
        "line_state": line_state,
        "converged": converged,
        "iterations": n_iter,
        "final_mismatch": final_mismatch,
        "voltage_mismatch": voltage_mismatch,
        "current_mismatch": current_mismatch,
        "convergence_check": convergence_check,
    }


def run_electrical_bfsa_forest(
    G: nx.DiGraph,
    radial_roots: List[int],
    p_bus: Dict[int, float],
    q_bus: Dict[int, float],
    line_lookup_undirected: Dict[Tuple[int, int], object],
    v_root_sq: float,
    config: Dict,
) -> Dict:
    """
    Run BFSA on an entire rooted forest (multiple radials).
    
    Parameters:
        G: Rooted forest as directed graph
        radial_roots: List of root bus IDs
        p_bus, q_bus: Bus loads
        line_lookup_undirected: Line parameters
        v_root_sq: Root voltage squared
        config: BFSA configuration
    
    Returns:
        dict: Consolidated results
    """
    
    start_time = time.time()
    
    v_sq_bfsa = {}
    line_state_bfsa = {}
    dlmp_bfsa = {}
    all_converged = True
    max_iterations = 1 
    final_mismatch = 0.0
    voltage_mismatch = 0.0
    current_mismatch = 0.0
    convergence_check = _normalize_bfsa_convergence_check(
        config.get("convergence_check", config.get("bfsa_convergence_check", "both"))
    )

    
    for root in radial_roots:
        result = bfsa_electrical(root, G, p_bus, q_bus, line_lookup_undirected,v_root_sq, config)
        
        v_sq_bfsa.update(result["v_sq_local"])
        line_state_bfsa.update(result["line_state"])
        
        all_converged = all_converged and result["converged"]
        max_iterations = max(max_iterations, result["iterations"])
        final_mismatch = max(final_mismatch, float(result.get("final_mismatch", 0.0)))
        voltage_mismatch = max(voltage_mismatch, float(result.get("voltage_mismatch", 0.0)))
        current_mismatch = max(current_mismatch, float(result.get("current_mismatch", 0.0)))

        
    
    
    elapsed = time.time() - start_time

    bus_losses = {}

    for (i, j), state in line_state_bfsa.items():
        if i not in bus_losses:
            bus_losses[i] = {
                "ploss": 0.0,
                "qloss": 0.0,
            }

        bus_losses[i]["ploss"] += state["ploss"]
        bus_losses[i]["qloss"] += state["qloss"]

    
    
    return {
        "v_sq_bfsa": v_sq_bfsa,
        "line_state_bfsa": line_state_bfsa,
        "dlmp_bfsa": dlmp_bfsa,
        "bus_losses": bus_losses,
        "convergence_info": {
            "converged": all_converged,
            "iterations": max_iterations,
            "max_iterations": max_iterations,
            "final_mismatch": final_mismatch,
            "voltage_mismatch": voltage_mismatch,
            "current_mismatch": current_mismatch,
            "convergence_check": convergence_check,
        },
        "elapsed_time": elapsed,
    }


def run_dlmp_propagation_forest(
    G: nx.DiGraph,
    radial_roots: List[int],
    line_state_bfsa: Dict[Tuple[int, int], Dict[str, float]],
    line_lookup_undirected: Dict[Tuple[int, int], object],
    lambda_p_seed: float,
    lambda_q_seed: float,
    dlmp_seed_bus: Optional[int] = None,
) -> Dict[int, Dict[str, float]]:
    """
    Propagate DLMPs across a rooted forest from a seed price pair.

    This reuses an already-computed electrical state and avoids rerunning
    backward-forward sweeps when only seed prices/buses change.
    """

    dlmp_bfsa = {}
    for root in radial_roots:
        root_int = int(root)
        T = nx.dfs_tree(G, source=root_int)
        nodes_radial = {int(n) for n in T.nodes()}

        seed_bus = int(dlmp_seed_bus) if (dlmp_seed_bus is not None and int(dlmp_seed_bus) in nodes_radial) else root_int
        dlmp_local = propagate_dlmp_from_seed(
            T=T,
            line_state=line_state_bfsa,
            line_lookup_undirected=line_lookup_undirected,
            seed_bus=seed_bus,
            lambda_p_seed=lambda_p_seed,
            lambda_q_seed=lambda_q_seed,
        )
        dlmp_bfsa.update(dlmp_local)

    return dlmp_bfsa
