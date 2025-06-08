# src/digester/graph_structures.py
from typing import Dict, Set, Tuple, List, NewType
from pathlib import Path

# Using NewType for more semantic meaning, though they are still strings at runtime.
NodeID = NewType('NodeID', str) # Represents a unique ID for a node in a graph, e.g., FQN or file:line:col:type

# Call Graph: Maps a caller's FQN (NodeID) to a set of callees' FQNs (NodeID).
CallGraph = Dict[NodeID, Set[NodeID]]

# Program Dependence Graph (PDG)
# For Control Dependencies: Maps a controlling node's ID (e.g., an if-condition)
# to a set of node IDs that are control-dependent on it.
ControlDependenceGraph = Dict[NodeID, Set[NodeID]]

# For Data Dependencies (more complex, placeholder for now):
# Might map a variable instance (e.g., var_name@line:col) to a set of
# definition sites (var_name_def@line:col) it depends on, or vice-versa.
DataDependenceGraph = Dict[NodeID, Set[NodeID]] # Placeholder structure

# Combined PDG for a file or project might aggregate these.
# For now, these are separate concepts.
