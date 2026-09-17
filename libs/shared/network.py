"""
Network topology and tree utilities.

Radial network analysis (enforces tree structure, computes parent/child relationships).
"""

from typing import List, Tuple, Set
import networkx as nx
from .data import PowerFlowCase, validate_case


def orient_radial_network(case: PowerFlowCase) -> PowerFlowCase:
    """
    Orient branches according to radial tree structure.
    
    For a radial network (tree topology), compute:
    - Which bus is parent/child for each branch
    - Incoming and outgoing branches for each bus
    - DFS order for forward/backward sweeps
    
    Assumes exactly one root bus (slack bus).
    Enforces that all buses are reachable from root.
    
    Args:
        case: PowerFlowCase with buses and branches
    
    Returns:
        Modified case with incoming_branches and outgoing_branches populated
        
    Raises:
        ValueError: If network is not connected or has cycles
    """
    
    validate_case(case)
    bus_ids = [b.bus_id for b in case.buses]
    root = case.root_bus
    
    if root not in bus_ids:
        raise ValueError(f"Root bus {root} not in bus list")
    
    # Build undirected graph
    G = nx.Graph()
    G.add_nodes_from(bus_ids)
    
    branch_map = {}
    for br in case.branches:
        branch_map[br.branch_id] = br
        if br.status == 1:  # Only add active branches
            if G.has_edge(br.from_bus, br.to_bus):
                raise ValueError("Parallel active branches are not radial")
            G.add_edge(br.from_bus, br.to_bus, branch_id=br.branch_id)
    
    # Check connectivity
    if not nx.is_connected(G):
        raise ValueError("Network is not connected (not radial)")
    
    # Check for cycles
    if not nx.is_tree(G):
        raise ValueError("Network has cycles (not radial)")
    
    # Build directed tree from root using BFS
    T = nx.bfs_tree(G, root)
    
    # Compute incoming/outgoing branches
    incoming_branches = {}  # {branch_id: to_bus}
    outgoing_branches = {}  # {bus: [branch_ids]}
    
    for bus in bus_ids:
        outgoing_branches[bus] = []
    
    tree_edges = []
    
    for parent, child in T.edges():
        branch_id = G.edges[parent, child]["branch_id"]
        branch = branch_map[branch_id]
        if branch.original_from_bus is None:
            branch.original_from_bus = branch.from_bus
            branch.original_to_bus = branch.to_bus
        branch.from_bus, branch.to_bus = parent, child
        incoming_branches[branch_id] = child
        outgoing_branches[parent].append(branch_id)
        tree_edges.append((parent, child))
    
    # Update case
    case.incoming_branches = incoming_branches
    case.outgoing_branches = outgoing_branches
    case.tree_edges = tree_edges
    
    return case

