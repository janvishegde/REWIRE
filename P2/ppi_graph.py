"""Graph builder utilities for REWIRE Week 1.

This module provides a stable interface for loading a weighted protein-protein
interaction graph and simulating drug binding perturbations.
"""

from __future__ import annotations

import copy
import csv
from typing import Dict, Iterable

import networkx as nx
import pandas as pd


def build_graph(filepath: str) -> nx.Graph:
    """Build a weighted undirected PPI graph from a CSV file.

    Args:
        filepath: Path to a CSV with columns: source,target,weight.

    Returns:
        A weighted undirected NetworkX graph (G0).
    """
    # Read the data using pandas
    df = pd.read_csv(filepath)
    required = {"source", "target", "weight"}
    if not required.issubset(df.columns):
        raise ValueError("CSV must contain columns: source,target,weight")

    # Scale the weight column
    df["weight"] = df["weight"] / 1000.0

    # Verify the weights
    min_weight = df["weight"].min()
    max_weight = df["weight"].max()
    print(f"Min weight: {min_weight}")
    print(f"Max weight: {max_weight}")
    if min_weight > 0 and max_weight <= 1:
        print("Weight scaling successful: min > 0 and max <= 1.")
    else:
        print("Weight scaling check failed.")


    # Build the graph
    graph = nx.Graph()
    for _, row in df.iterrows():
        source = row["source"].strip()
        target = row["target"].strip()
        if source == target:
            continue
        graph.add_edge(source, target, weight=row["weight"])

    return graph


def simulate_binding(
    G: nx.Graph,
    targets: Iterable[str],
    affinities: Dict[str, float],
) -> nx.Graph:
    """Simulate drug perturbation by reducing weights near target proteins.

    The input graph is deep-copied so the original graph remains unchanged.
    For each target node found in the graph, all incident edge weights are
    scaled by a target-specific factor derived from affinity.

    Args:
        G: Base graph (G0).
        targets: Protein targets for the drug.
        affinities: Mapping target -> affinity in [0, 1] (values are clamped).

    Returns:
        A perturbed graph (G_drug) with modified edge weights.
    """
    G_drug = copy.deepcopy(G)

    for target in targets:
        if target not in G_drug:
            continue

        affinity = max(0.0, min(1.0, float(affinities.get(target, 0.5))))
        factor = 1.0 - (0.5 * affinity)

        for neighbor in G_drug.neighbors(target):
            old_weight = float(G_drug[target][neighbor].get("weight", 1.0))
            new_weight = max(0.01, old_weight * factor)
            G_drug[target][neighbor]["weight"] = new_weight

    return G_drug


if __name__ == "__main__":
    G0 = build_graph("fake_ppi.csv")
    sample_targets = ["P53", "EGFR", "AKT1"]
    sample_affinities = {"P53": 0.7, "EGFR": 0.4, "AKT1": 0.9}
    Gdrug = simulate_binding(G0, sample_targets, sample_affinities)

    print("Loaded graph:", G0.number_of_nodes(), "nodes,", G0.number_of_edges(), "edges")
    print("\nBefore/after edge weights for affected edges:")

    shown = 0
    for target in sample_targets:
        if target not in G0:
            continue
        for neighbor in G0.neighbors(target):
            before = G0[target][neighbor]["weight"]
            after = Gdrug[target][neighbor]["weight"]
            if abs(before - after) > 1e-12:
                print(f"{target} -- {neighbor}: {before:.3f} -> {after:.3f}")
                shown += 1
            if shown >= 8:
                break
        if shown >= 8:
            break
