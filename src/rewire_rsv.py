"""
REWIRE: Drug Repurposing via Protein Network Rewiring
Stage 2 — Graph Simulation + RSV (Rewiring Signature Vector) Engine
======================================================================
Inputs  : ppi_genes.csv, drug_targets_symbols.csv (or drug_50.csv)
Outputs : rsv_results.csv  — one RSV vector per drug
          rsv_matrix.npy   — numpy matrix for downstream ML/clustering
"""

import time
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# 1. NETWORK LOADER
# ─────────────────────────────────────────────────────────────────────────────

def load_ppi_network(ppi_path: str, weight_col: str = "weight") -> nx.Graph:
    """
    Load a PPI network from CSV into a weighted undirected NetworkX graph.

    Parameters
    ----------
    ppi_path   : Path to ppi_genes.csv  (columns: gene1, gene2, weight)
    weight_col : Name of the weight column (default 'weight')

    Returns
    -------
    G : nx.Graph — weighted PPI graph

    Notes
    -----
    - Edges with missing weights are assigned a default of 0.5
    - Self-loops are removed (not biologically meaningful here)
    - Duplicate edges are coalesced by taking the max weight
    """
    df = pd.read_csv(ppi_path)
    df.columns = df.columns.str.strip().str.lower()

    # Normalise column names defensively
    rename = {}
    for col in df.columns:
        if "gene1" in col or col == "protein1":
            rename[col] = "gene1"
        elif "gene2" in col or col == "protein2":
            rename[col] = "gene2"
        elif "weight" in col or "score" in col or "combined" in col:
            rename[col] = "weight"
    df.rename(columns=rename, inplace=True)

    if "weight" not in df.columns:
        print("[WARN] No weight column found — assigning uniform weight 0.5")
        df["weight"] = 0.5

    # Drop self-loops, fill NaN weights
    df = df[df["gene1"] != df["gene2"]]
    df["weight"] = df["weight"].fillna(0.5)

    # If STRING scores (0–1000), normalise to 0–1
    if df["weight"].max() > 1.0:
        df["weight"] = df["weight"] / 1000.0

    G = nx.Graph()
    for _, row in df.iterrows():
        u, v, w = row["gene1"], row["gene2"], row["weight"]
        if G.has_edge(u, v):
            G[u][v]["weight"] = max(G[u][v]["weight"], w)  # keep strongest evidence
        else:
            G.add_edge(u, v, weight=w)

    print(f"[PPI] Loaded: {G.number_of_nodes():,} nodes | {G.number_of_edges():,} edges")
    return G


# ─────────────────────────────────────────────────────────────────────────────
# 2. DRUG TARGET LOADER
# ─────────────────────────────────────────────────────────────────────────────

def load_drug_targets(drug_path: str) -> dict[str, list[str]]:
    """
    Load drug→target mappings from CSV.

    Parameters
    ----------
    drug_path : Path to drug_targets_symbols.csv or drug_50.csv
                Expected columns: drug_name, target_gene  (or gene_symbol)

    Returns
    -------
    drug_map : { drug_name: [GENE1, GENE2, ...] }
    """
    df = pd.read_csv(drug_path)
    df.columns = df.columns.str.strip().str.lower()

    # Accept both 'target_gene' and 'gene_symbol'
    for alt in ["gene_symbol", "target", "gene"]:
        if alt in df.columns and "target_gene" not in df.columns:
            df.rename(columns={alt: "target_gene"}, inplace=True)

    df = df.dropna(subset=["drug_name", "target_gene"])
    df["drug_name"]   = df["drug_name"].str.strip()
    df["target_gene"] = df["target_gene"].str.strip().str.upper()

    drug_map = df.groupby("drug_name")["target_gene"].apply(list).to_dict()
    print(f"[DRUGS] Loaded: {len(drug_map)} drugs | "
          f"{df['target_gene'].nunique()} unique targets")
    return drug_map

#_____________________________________________
#LOCALSUBGRAPH
#__________________________________________________

def extract_local_subgraph(G, target_genes, hops=2):

    nodes = set()

    for gene in target_genes:

        if gene not in G:
            continue

        current = {gene}

        for _ in range(hops):

            neighbors = set()

            for n in current:
                neighbors.update(G.neighbors(n))

            current.update(neighbors)

        nodes.update(current)

    return G.subgraph(nodes).copy()

# ─────────────────────────────────────────────────────────────────────────────
# 3. DRUG BINDING SIMULATOR
# ─────────────────────────────────────────────────────────────────────────────

def simulate_drug_binding(
    G: nx.Graph,
    target_genes: list[str],
    inhibition_strength: float = 0.8,
    neighbor_dampening: float = 0.3,
) -> nx.Graph:
    """
    Simulate the effect of a drug binding to its target genes by rewiring
    edge weights in a copy of the PPI graph.

    Strategy
    --------
    1. DIRECT edges  (target–target or target–any):
       weight ← weight × (1 − inhibition_strength)
       → Models competitive inhibition / steric blocking

    2. INDIRECT edges (neighbors of targets, 1 hop away):
       weight ← weight × (1 − neighbor_dampening × 0.5)
       → Models allosteric signal attenuation

    Parameters
    ----------
    G                   : Original PPI graph (not modified)
    target_genes        : List of gene symbols the drug hits
    inhibition_strength : Fraction of edge weight removed at direct targets (0–1)
    neighbor_dampening  : Fraction of weight reduced for 1-hop neighbors (0–1)

    Returns
    -------
    G_sim : nx.Graph — rewired graph (deep copy, original untouched)
    """
    targets_in_graph = [g for g in target_genes if G.has_node(g)]
    if not targets_in_graph:
        return G.copy()  # drug has no network presence → unchanged graph

    import copy
    G_sim = copy.deepcopy(G)

    # Collect 1-hop neighbors (excluding the target nodes themselves)
    neighbors_1hop = set()
    for t in targets_in_graph:
        neighbors_1hop.update(G_sim.neighbors(t))
    neighbors_1hop -= set(targets_in_graph)

    # Apply direct inhibition on all edges touching a target
    for t in targets_in_graph:
        for nbr in list(G_sim.neighbors(t)):
            old_w = G_sim[t][nbr]["weight"]
            G_sim[t][nbr]["weight"] = old_w * (1.0 - inhibition_strength)

    # Apply indirect dampening on edges between neighbors (not touching target)
    for n in neighbors_1hop:
        for nbr in list(G_sim.neighbors(n)):
            if nbr not in targets_in_graph:   # direct edges already handled
                old_w = G_sim[n][nbr]["weight"]
                G_sim[n][nbr]["weight"] = old_w * (1.0 - neighbor_dampening * 0.5)

    return G_sim


# ─────────────────────────────────────────────────────────────────────────────
# 4. GRAPH FEATURE EXTRACTOR  (per-node, localised to targets + neighborhood)
# ─────────────────────────────────────────────────────────────────────────────

def _subgraph_around_targets(
    G: nx.Graph,
    targets: list[str],
    hops: int = 2
) -> set[str]:
    """
    Return the k-hop neighborhood of target genes.
    Used to localise centrality computation → massive speed-up on large graphs.
    """
    nodes = set(targets)
    frontier = set(targets)
    for _ in range(hops):
        next_frontier = set()
        for n in frontier:
            if G.has_node(n):
                next_frontier.update(G.neighbors(n))
        nodes |= next_frontier
        frontier = next_frontier
    return nodes


def compute_graph_features(
    G: nx.Graph,
    targets: list[str],
    local_hops: int = 2,
) -> dict[str, float]:
    """
    Compute graph topology features on the k-hop subgraph around targets.

    Features computed
    -----------------
    1. mean_degree_centrality      — average degree centrality of target nodes
    2. mean_betweenness_centrality — average betweenness (approximated)
    3. mean_clustering_coefficient — average local clustering of targets
    4. mean_weighted_degree        — average sum of edge weights at targets
    5. subgraph_density            — edge density of the local subgraph
    6. mean_closeness_centrality   — average closeness of targets in subgraph

    Parameters
    ----------
    G          : Graph to analyse (original or simulated)
    targets    : Target gene list (used to localise the computation)
    local_hops : Neighbourhood radius for local subgraph extraction

    Returns
    -------
    features : dict of feature_name → float value
    """
    valid_targets = [t for t in targets if G.has_node(t)]
    if not valid_targets:
        return {k: 0.0 for k in [
            "mean_degree_centrality", "mean_betweenness_centrality",
            "mean_clustering_coefficient", "mean_weighted_degree",
            "subgraph_density", "mean_closeness_centrality"
        ]}

    # Extract local subgraph for speed
    local_nodes = _subgraph_around_targets(G, valid_targets, hops=local_hops)
    SG = G.subgraph(local_nodes).copy()

    if SG.number_of_nodes() < 2:
        return {k: 0.0 for k in [
            "mean_degree_centrality", "mean_betweenness_centrality",
            "mean_clustering_coefficient", "mean_weighted_degree",
            "subgraph_density", "mean_closeness_centrality"
        ]}

    # Centraliy metrics (on local subgraph — much faster)
    deg_centrality  = nx.degree_centrality(SG)
    # k=min(50, n) samples for betweenness → O(k * n * log n) instead of O(n³)
    k_samples = min(50, SG.number_of_nodes())
    btwn_centrality = nx.betweenness_centrality(
        SG, k=k_samples, weight="weight", normalized=True, seed=42
    )
    clustering      = nx.clustering(SG, weight="weight")
    closeness = nx.closeness_centrality(SG)
    def _mean_for_targets(metric_dict):
        vals = [metric_dict.get(t, 0.0) for t in valid_targets]
        return float(np.mean(vals)) if vals else 0.0

    # Weighted degree (strength) of targets
    weighted_degrees = []
    for t in valid_targets:
        wd = sum(d.get("weight", 1.0) for _, d in SG[t].items())
        weighted_degrees.append(wd)

    return {
        "mean_degree_centrality":      _mean_for_targets(deg_centrality),
        "mean_betweenness_centrality": _mean_for_targets(btwn_centrality),
        "mean_clustering_coefficient": _mean_for_targets(clustering),
        "mean_weighted_degree":        float(np.mean(weighted_degrees)) if weighted_degrees else 0.0,
        "subgraph_density":            nx.density(SG),
        "mean_closeness_centrality":   _mean_for_targets(closeness),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. RSV COMPUTATION CORE
# ─────────────────────────────────────────────────────────────────────────────

RSV_FEATURE_NAMES = [
    "Δdegree_centrality",
    "Δbetweenness_centrality",
    "Δclustering_coefficient",
    "Δweighted_degree",
    "Δsubgraph_density",
    "Δcloseness_centrality",
]


def compute_rsv(
    G_original: nx.Graph,
    G_simulated: nx.Graph,
    target_genes: list[str],
    local_hops: int = 2,
) -> np.ndarray:
    """
    Compute the Rewiring Signature Vector (RSV) for one drug.

    RSV = feature_vector(G_simulated) − feature_vector(G_original)

    Each element represents the CHANGE in a graph topology feature
    caused by the drug's simulated binding. This delta encodes how
    much the drug reshapes the network — the "rewiring fingerprint".

    Returns
    -------
    rsv : np.ndarray of shape (6,)  — one float per feature
    """
    feats_before = compute_graph_features(G_original,  target_genes, local_hops)
    feats_after  = compute_graph_features(G_simulated, target_genes, local_hops)

    rsv = np.array([
        feats_after[k] - feats_before[k]
        for k in [
            "mean_degree_centrality",
            "mean_betweenness_centrality",
            "mean_clustering_coefficient",
            "mean_weighted_degree",
            "subgraph_density",
            "mean_closeness_centrality",
        ]
    ], dtype=np.float32)

    return rsv


# ─────────────────────────────────────────────────────────────────────────────
# 6. BATCH PIPELINE — all drugs → RSV matrix
# ─────────────────────────────────────────────────────────────────────────────

def run_rewire_pipeline(
    ppi_path:   str,
    drug_path:  str,
    output_csv: str = "rsv_results.csv",
    output_npy: str = "rsv_matrix.npy",
    inhibition_strength: float = 0.8,
    neighbor_dampening:  float = 0.3,
    local_hops:          int   = 2,
    verbose:             bool  = True,
) -> pd.DataFrame:
    """
    Full REWIRE pipeline:
        CSV files → PPI graph → per-drug simulation → RSV vectors → output files

    Parameters
    ----------
    ppi_path            : ppi_genes.csv
    drug_path           : drug_50.csv or drug_targets_symbols.csv
    output_csv          : Path for the RSV results CSV
    output_npy          : Path for the RSV numpy matrix
    inhibition_strength : How strongly the drug inhibits direct target edges
    neighbor_dampening  : Signal attenuation at 1-hop neighbors
    local_hops          : Neighborhood radius for feature extraction

    Returns
    -------
    results_df : DataFrame with drug_name, targets, RSV components, rsv_norm
    """
    # ── Step 1: Load data ────────────────────────────────────────────────────
    G = load_ppi_network(ppi_path)
    drug_map = load_drug_targets(drug_path)

    results = []
    rsv_matrix = []
    total = len(drug_map)

    # ── Step 2: Per-drug simulation + RSV ────────────────────────────────────
    for i, (drug, targets) in enumerate(drug_map.items(), 1):
        t0 = time.time()

        targets_found = [t for t in targets if G.has_node(t)]
        coverage = len(targets_found) / len(targets) if targets else 0.0

        if not targets_found:
            if verbose:
                print(f"[{i:>3}/{total}] {drug:<25} — ✗ no targets in PPI graph, skipping")
            continue

        # Simulate binding
        G_sim = simulate_drug_binding(
            G, targets_found,
            inhibition_strength=inhibition_strength,
            neighbor_dampening=neighbor_dampening,
        )

        # Compute RSV
        rsv = compute_rsv(G, G_sim, targets_found, local_hops=local_hops)
        rsv_norm = float(np.linalg.norm(rsv))   # magnitude = overall rewiring strength

        elapsed = time.time() - t0

        row = {
            "drug_name":       drug,
            "targets_input":   ";".join(targets),
            "targets_matched": ";".join(targets_found),
            "target_coverage": round(coverage, 3),
            **{name: round(float(val), 6) for name, val in zip(RSV_FEATURE_NAMES, rsv)},
            "rsv_norm":        round(rsv_norm, 6),
        }
        results.append(row)
        rsv_matrix.append(rsv)

        if verbose:
            print(f"[{i:>3}/{total}] {drug:<25} | targets={len(targets_found)} "
                  f"| ‖RSV‖={rsv_norm:.4f} | {elapsed:.2f}s")

    # ── Step 3: Save outputs ─────────────────────────────────────────────────
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("rsv_norm", ascending=False).reset_index(drop=True)
    results_df.to_csv(output_csv, index=False)

    np.save(output_npy, np.vstack(rsv_matrix).astype(np.float32))

    print(f"\n[DONE] Processed {len(results_df)} drugs")
    print(f"       RSV CSV    → {output_csv}")
    print(f"       RSV matrix → {output_npy}  shape={np.vstack(rsv_matrix).shape}")

    return results_df


# ─────────────────────────────────────────────────────────────────────────────
# 7. UTILITY: Inspect a single drug's RSV in detail
# ─────────────────────────────────────────────────────────────────────────────

def inspect_drug(
    drug_name: str,
    G: nx.Graph,
    drug_map: dict[str, list[str]],
    inhibition_strength: float = 0.8,
    neighbor_dampening:  float = 0.3,
    local_hops:          int   = 2,
) -> dict:
    """
    Detailed inspection of a single drug's network rewiring.
    Useful for debugging and visualisation prep.

    Returns a dict with:
    - targets (matched / unmatched)
    - before/after feature vectors
    - RSV vector
    - top 10 most-affected edges (sorted by weight delta)
    """
    targets = drug_map.get(drug_name, [])
    targets_found = [t for t in targets if G.has_node(t)]

    if not targets_found:
        print(f"[WARN] No targets for '{drug_name}' found in PPI graph.")
        return {}

    G_sim = simulate_drug_binding(
        G, targets_found,
        inhibition_strength=inhibition_strength,
        neighbor_dampening=neighbor_dampening,
    )

    feats_before = compute_graph_features(G, targets_found, local_hops)
    feats_after  = compute_graph_features(G_sim, targets_found, local_hops)
    rsv          = compute_rsv(G, G_sim, targets_found, local_hops)

    # Find most rewired edges (largest absolute weight change)
    edge_deltas = []
    for t in targets_found:
        for nbr in G.neighbors(t):
            w_before = G[t][nbr]["weight"]
            w_after  = G_sim[t][nbr]["weight"]
            edge_deltas.append((t, nbr, w_before, w_after, w_after - w_before))

    edge_deltas.sort(key=lambda x: abs(x[4]), reverse=True)
    top_edges = pd.DataFrame(
        edge_deltas[:10],
        columns=["gene_a", "gene_b", "weight_before", "weight_after", "delta"]
    )

    result = {
        "drug":              drug_name,
        "targets_in_graph":  targets_found,
        "targets_missing":   [t for t in targets if t not in G],
        "features_before":   feats_before,
        "features_after":    feats_after,
        "rsv":               rsv,
        "rsv_norm":          float(np.linalg.norm(rsv)),
        "top_rewired_edges": top_edges,
    }

    # Pretty print
    print(f"\n{'='*55}")
    print(f"  DRUG: {drug_name}")
    print(f"{'='*55}")
    print(f"  Targets matched : {targets_found}")
    print(f"  Targets missing : {result['targets_missing']}")
    print(f"\n  {'Feature':<32} {'Before':>10} {'After':>10} {'Delta':>10}")
    print(f"  {'-'*62}")
    for k in feats_before:
        delta = feats_after[k] - feats_before[k]
        print(f"  {k:<32} {feats_before[k]:>10.4f} {feats_after[k]:>10.4f} {delta:>+10.4f}")
    print(f"\n  RSV vector : {np.round(rsv, 5).tolist()}")
    print(f"  ‖RSV‖      : {result['rsv_norm']:.5f}")
    print(f"\n  Top rewired edges:")
    print(top_edges.to_string(index=False))

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 8. ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── Paths — adjust to your project structure ────────────────────────────
    PPI_PATH   = "data/ppi_genes.csv"
    DRUG_PATH  = "data/drug_50.csv"          # or drug_targets_symbols.csv
    OUT_CSV    = "output/rsv_results.csv"
    OUT_NPY    = "output/rsv_matrix.npy"

    # ── Run full pipeline ────────────────────────────────────────────────────
    results = run_rewire_pipeline(
        ppi_path            = PPI_PATH,
        drug_path           = DRUG_PATH,
        output_csv          = OUT_CSV,
        output_npy          = OUT_NPY,
        inhibition_strength = 0.8,   # tweak: higher = stronger effect at targets
        neighbor_dampening  = 0.3,   # tweak: lower = less signal propagation
        local_hops          = 2,     # tweak: 3 for larger context, slower
        verbose             = True,
    )

    print("\nTop 10 drugs by rewiring magnitude:")
    print(results[["drug_name", "targets_matched", "rsv_norm"]].head(10).to_string(index=False))

    # ── Inspect one drug in detail ───────────────────────────────────────────
    import networkx as nx
    G        = load_ppi_network(PPI_PATH)
    drug_map = load_drug_targets(DRUG_PATH)
    inspect_drug("Imatinib", G, drug_map)
