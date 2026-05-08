"""
A graph simulation engine for the REWIRE drug repurposing project.

This module provides functions to:
1. Load a protein-protein interaction (PPI) network from a CSV file.
2. Simulate the effect of a drug on the network by perturbing edge weights
   around its target proteins.

The core principle is to treat the human interactome as a dynamic graph and
model drug action as a topological transformation from a baseline state (G₀)
to a drug-influenced state (G_drug).
"""
import copy
import logging
import sys

import networkx as nx
import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def build_graph(filepath: str) -> nx.Graph | None:
    """
    Loads a PPI network from a CSV file into a NetworkX graph.

    The CSV file must contain 'gene1', 'gene2', and 'weight' columns.
    Weights are normalized from a 0-1000 scale to a 0.0-1.0 scale.

    Args:
        filepath: The path to the PPI CSV file.

    Returns:
        A NetworkX weighted undirected graph (G₀), or None if loading fails.
    """
    try:
        logging.info(f"Loading PPI data from '{filepath}'...")
        df = pd.read_csv(filepath)
        # Standardize column names for consistency
        df.rename(columns={
            'protein1': 'gene1', 'protein2': 'gene2',
            'source': 'gene1', 'target': 'gene2'
        }, inplace=True)

        if not {'gene1', 'gene2', 'weight'}.issubset(df.columns):
            logging.error("CSV must contain columns: 'gene1', 'gene2', 'weight'")
            return None

        # Normalize weights
        df['weight'] = df['weight'] / 1000.0

        # Use from_pandas_edgelist for efficient loading
        G = nx.from_pandas_edgelist(df, 'gene1', 'gene2', edge_attr='weight')

        logging.info("Graph built successfully.")
        logging.info(f"  - Nodes: {G.number_of_nodes():,}")
        logging.info(f"  - Edges: {G.number_of_edges():,}")
        return G

    except FileNotFoundError:
        logging.error(f"File not found at '{filepath}'.")
        return None
    except Exception as e:
        logging.error(f"An error occurred while building the graph: {e}")
        return None

def _validate_deep_copy(original_graph: nx.Graph, copied_graph: nx.Graph):
    """
    Helper to validate that the graph was deep copied.
    Modifies an edge in the copy and asserts the original is unchanged.
    """
    logging.info("Validating deep copy...")
    # Pick a random edge from the graph to modify
    try:
        edge_to_modify = list(copied_graph.edges())[0]
        node1, node2 = edge_to_modify
        original_weight = original_graph[node1][node2]['weight']
        
        # Modify the weight in the copied graph
        copied_graph[node1][node2]['weight'] = -999.0
        
        # Assert that the original graph's weight is unchanged
        assert original_graph[node1][node2]['weight'] == original_weight, "Deep copy failed: Original graph was modified!"

        # Restore the copied graph so validation does not contaminate results.
        copied_graph[node1][node2]['weight'] = original_weight
        logging.info("Deep copy validation successful.")
    except (IndexError, KeyError):
        logging.warning("Could not validate deep copy (graph may be too small or empty).")


def simulate_binding(
    G0: nx.Graph,
    targets: list[str],
    affinities: dict[str, float] | None = None,
    actions: dict[str, str] | None = None
) -> tuple[nx.Graph, list[str]]:
    """
    Simulates drug binding by creating a perturbed copy of the baseline graph.

    This function creates a deep copy of G₀. For each target protein, it
    adjusts the weights of incident edges based on the drug's affinity and
    action (inhibitor/activator).

    Args:
        G0: The baseline NetworkX graph. This graph is never modified.
        targets: A list of gene symbols for the drug's targets (e.g., ['ABL1', 'KIT']).
        affinities: A dict mapping target gene -> affinity (Ki/Kd in nM).
                    If None, a default inhibition strength is used.
        actions: A dict mapping target gene -> 'inhibitor' or 'activator'.
                 If None, all targets are assumed to be inhibitors.

    Returns:
        A tuple containing:
        - G_drug (nx.Graph): The new, modified graph.
        - failed_targets (list[str]): A list of targets not found in the graph.
    """
    if affinities is None:
        affinities = {}
    if actions is None:
        actions = {}

    G_drug = copy.deepcopy(G0)
    _validate_deep_copy(G0, G_drug) # Internal check to ensure safety
    
    failed_targets = []
    
    for target in targets:
        if target not in G_drug:
            logging.warning(f"Target '{target}' not found in the graph. Skipping.")
            failed_targets.append(target)
            continue

        # Convert affinity (nM) to inhibition/activation strength (0-1)
        # Low nM (high affinity) -> high strength
        affinity_nM = affinities.get(target)
        if affinity_nM is not None:
            strength = 1 / (1 + affinity_nM)
        else:
            strength = 0.5  # Default strength if no affinity is provided

        action = actions.get(target, 'inhibitor').lower()

        for neighbor in G_drug.neighbors(target):
            original_weight = G_drug[target][neighbor]['weight']
            
            if action == 'inhibitor':
                # Weaken the connection
                new_weight = original_weight * (1 - strength)
                # Clamp to a minimum value to prevent edge disappearance
                G_drug[target][neighbor]['weight'] = np.clip(new_weight, 0.001, 1.0)
            elif action == 'activator':
                # Strengthen the connection
                new_weight = original_weight * (1 + strength)
                # Clamp to a maximum of 1.0
                G_drug[target][neighbor]['weight'] = np.clip(new_weight, 0.001, 1.0)

    logging.info(f"Simulation complete. {len(failed_targets)} targets were not found.")
    return G_drug, failed_targets


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python ppi_graph.py <path_to_ppi_csv>")
        sys.exit(1)

    ppi_file_path = sys.argv[1]
    
    # 1. Load the real PPI graph
    G0 = build_graph(ppi_file_path)

    if G0:
        # 2. Define a sample simulation
        # Targets for Imatinib, a multi-target cancer drug
        imatinib_targets = ['ABL1', 'KIT', 'PDGFRA', 'FAKE_GENE']
        
        logging.info("\n--- Running Simulation Test ---")
        logging.info(f"Simulating binding for targets: {imatinib_targets} (no affinities)")
        
        # 3. Run the simulation
        G_imatinib, not_found = simulate_binding(G0, imatinib_targets)
        
        print(f"\nTargets not found in graph: {not_found}")

        # 4. Verify results
        print("\n--- Verification ---")
        try:
            # Pick a known neighbor of ABL1 to check its weight change
            neighbor_to_check = 'CRKL' # CRKL is a known interactor of ABL1
            if G0.has_edge('ABL1', neighbor_to_check):
                before_weight = G0['ABL1'][neighbor_to_check]['weight']
                after_weight = G_imatinib['ABL1'][neighbor_to_check]['weight']
                
                print(f"Edge weight of ('ABL1', '{neighbor_to_check}') BEFORE: {before_weight:.4f}")
                print(f"Edge weight of ('ABL1', '{neighbor_to_check}') AFTER:  {after_weight:.4f}")
                
                # Final check that G0 is pristine
                assert G0['ABL1'][neighbor_to_check]['weight'] == before_weight, "CRITICAL ERROR: G0 was modified!"
                print("\nSUCCESS: Baseline graph G₀ remains unchanged after simulation.")
            else:
                print(f"INFO: Verification edge ('ABL1', '{neighbor_to_check}') not found.")

        except KeyError:
            print("INFO: Could not perform verification check for ABL1's neighbor.")
