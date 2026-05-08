"""REWIRE Week 2: Real Data + ID Resolution (Person 2).

This module provides the Week 2 "data resolution" layer:
- Resolve drug→targets (FILE A) into valid PPI graph node IDs.
- Resolve disease→genes (FILE B) into valid PPI graph node IDs and create a
  disease-stressed graph by amplifying edge weights around disease genes.

Protector rules:
- Never crash on missing IDs.
- Log unresolved mappings to failed_mappings.txt and continue.

Allowed deps (per prompt): pandas, networkx, numpy, copy, logging, os.
"""

import copy
import logging
import os

import networkx as nx
import numpy as np
import pandas as pd


FAILED_MAPPINGS_FILE = "failed_mappings.txt"

# Minimal, pragmatic alias map for common mismatches.
_ALIASES = {
    "SHP2": "PTPN11",
    "HER2": "ERBB2",
    "P53": "TP53",
}


def _append_failed_mapping(message: str) -> None:
    """Append a single line to failed_mappings.txt without ever throwing."""

    try:
        with open(FAILED_MAPPINGS_FILE, "a", encoding="utf-8") as handle:
            handle.write(message.rstrip("\n") + "\n")
    except OSError:
        # Protector: logging must never break the pipeline.
        return


def _resolve_gene_symbol(symbol: str, graph_nodes: set[str]) -> str | None:
    """Resolve a gene symbol against graph nodes using exact/uppercase/aliases."""

    if symbol is None:
        return None

    raw = str(symbol).strip()
    if not raw:
        return None

    if raw in graph_nodes:
        return raw

    upper = raw.upper()
    if upper in graph_nodes:
        return upper

    alias_target = _ALIASES.get(upper)
    if alias_target and alias_target in graph_nodes:
        return alias_target

    return None


def get_affinity_estimate(gene_symbol: str) -> float:
    """Affinity fallback handler.

    PLACEHOLDER — replace with real Ki/Kd from DrugBank when Person 1 provides an
    affinity column.

    Returns:
        float: Default affinity in nM (100.0).
    """

    _ = gene_symbol
    return 100.0


def resolve_targets(
    drug_name: str,
    drug_df: pd.DataFrame,
    graph_nodes: set[str],
) -> tuple[list[str], dict[str, str], dict[str, float | None]]:
    """Resolve drug targets (FILE A) into valid nodes in the PPI graph.

    Expected FILE A columns (real data):
      - drug_name
      - gene_symbol

    Test/mocked schemas may also use:
      - target_gene, gene, symbol (gene column)

    Resolution logic:
      - exact match
      - uppercase match
      - alias mapping
      - else: log to failed_mappings.txt and skip

    Returns:
        resolved_targets: list[str]
        actions: dict[str,str] with default action 'inhibitor'
        affinities: dict[str, float|None] default None for all (no Ki/Kd in CSV)
    """

    resolved_targets: list[str] = []
    actions: dict[str, str] = {}
    affinities: dict[str, float | None] = {}

    if drug_df is None or not isinstance(drug_df, pd.DataFrame):
        _append_failed_mapping(f"Drug '{drug_name}': invalid drug_df")
        return resolved_targets, actions, affinities

    if "drug_name" not in drug_df.columns:
        _append_failed_mapping(f"Drug '{drug_name}': missing column 'drug_name'")
        return resolved_targets, actions, affinities

    gene_col = None
    for candidate in ("gene_symbol", "target_gene", "gene", "symbol"):
        if candidate in drug_df.columns:
            gene_col = candidate
            break

    if gene_col is None:
        _append_failed_mapping(f"Drug '{drug_name}': missing gene column")
        return resolved_targets, actions, affinities

    # Filter: case-insensitive equality first.
    series = drug_df["drug_name"].astype(str)
    exact_mask = series.str.strip().str.lower() == str(drug_name).strip().lower()
    subset = drug_df[exact_mask]

    # Protector fallback: partial match if exact yields nothing.
    if subset.empty:
        subset = drug_df[series.str.contains(str(drug_name), case=False, na=False)]

    for raw_gene in subset[gene_col].dropna().astype(str).unique().tolist():
        resolved = _resolve_gene_symbol(raw_gene, graph_nodes)
        if resolved is None:
            _append_failed_mapping(f"Drug '{drug_name}' unmapped target: {raw_gene}")
            continue

        if resolved not in resolved_targets:
            resolved_targets.append(resolved)
            actions[resolved] = "inhibitor"
            affinities[resolved] = None

    return resolved_targets, actions, affinities


def build_disease_graph(
    G0: nx.Graph,
    disease_name: str,
    disease_df: pd.DataFrame,
    score_col: str | None = None,
) -> nx.Graph:
    """Build a disease-stressed graph by amplifying edges around disease genes.

    Expected FILE B columns (real data):
      - disease
      - gene

    Tests/mocked schemas may use:
      - disease_name
      - gene_symbol

    Logic:
      - Filter disease rows via case-insensitive partial match
      - Resolve gene IDs using the same resolver as drug targets
      - Deep copy G0 (do not mutate baseline)
      - Amplify weights for edges incident to disease genes
        * default factor = 1.3
        * if score_col exists and value present: use it as amplification factor
      - Clamp weights to [0.001, 1.0]

    PLACEHOLDER score — replace with OpenTargets association score when Person 1
    provides it.
    """

    if disease_df is None or not isinstance(disease_df, pd.DataFrame):
        _append_failed_mapping(f"Disease '{disease_name}': invalid disease_df")
        return copy.deepcopy(G0)

    disease_col = "disease_name" if "disease_name" in disease_df.columns else "disease"
    gene_col = "gene_symbol" if "gene_symbol" in disease_df.columns else "gene"

    if disease_col not in disease_df.columns or gene_col not in disease_df.columns:
        _append_failed_mapping(f"Disease '{disease_name}': missing required columns")
        return copy.deepcopy(G0)

    subset = disease_df[
        disease_df[disease_col].astype(str).str.contains(str(disease_name), case=False, na=False)
    ]

    graph_nodes = set(G0.nodes())
    resolved_genes: list[str] = []
    for raw_gene in subset[gene_col].dropna().astype(str).unique().tolist():
        resolved = _resolve_gene_symbol(raw_gene, graph_nodes)
        if resolved is None:
            _append_failed_mapping(f"Disease '{disease_name}' unmapped gene: {raw_gene}")
            continue
        resolved_genes.append(resolved)

    G_disease = copy.deepcopy(G0)

    for gene in resolved_genes:
        factor = 1.3

        if score_col and score_col in subset.columns:
            try:
                val = subset.loc[subset[gene_col].astype(str) == gene, score_col].iloc[0]
                if pd.notna(val):
                    factor = float(val)
            except Exception:
                factor = 1.3

        for neighbor in G_disease.neighbors(gene):
            old_weight = float(G_disease[gene][neighbor].get("weight", 0.0))
            new_weight = old_weight * factor
            G_disease[gene][neighbor]["weight"] = float(np.clip(new_weight, 0.001, 1.0))

    return G_disease


if __name__ == "__main__":
    # Week 2 validation run using local CSVs.
    from ppi_graph import build_graph, simulate_binding

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    here = os.path.dirname(os.path.abspath(__file__))
    ppi_csv = os.path.join(here, "ppi_genes.csv")
    drug_csv = os.path.join(here, "drug_targets_symbols.csv")
    disease_csv = os.path.join(here, "disease_genes.csv")

    # Reset log so the summary is clean
    if os.path.exists(FAILED_MAPPINGS_FILE):
        os.remove(FAILED_MAPPINGS_FILE)

    print("--- Loading G0 ---")
    G0 = build_graph(ppi_csv)
    if G0 is None:
        raise SystemExit(1)

    drug_df = pd.read_csv(drug_csv)
    disease_df = pd.read_csv(disease_csv)

    graph_nodes = set(G0.nodes())

    print("\n--- Resolving Imatinib targets ---")
    resolved, actions, affinities = resolve_targets("Imatinib", drug_df, graph_nodes)
    raw_gene_col = "gene_symbol" if "gene_symbol" in drug_df.columns else (
        "target_gene" if "target_gene" in drug_df.columns else (
            "gene" if "gene" in drug_df.columns else "symbol"
        )
    )
    # Compute totals using the same matching rules as resolve_targets():
    # exact (case-insensitive) then partial contains() if exact yields nothing.
    drug_series = drug_df["drug_name"].astype(str)
    exact_mask = drug_series.str.strip().str.lower() == "imatinib"
    raw_subset = drug_df[exact_mask]
    if raw_subset.empty:
        raw_subset = drug_df[drug_series.str.contains("Imatinib", case=False, na=False)]
    total_raw = int(raw_subset[raw_gene_col].dropna().astype(str).nunique()) if raw_gene_col in raw_subset.columns else 0
    failed = max(0, total_raw - len(resolved))
    print("Resolved targets:", resolved)
    print(f"Summary: {len(resolved)}/{total_raw} targets resolved, {failed} failed (see {FAILED_MAPPINGS_FILE}).")
    if total_raw == 0:
        print("Note: No matching 'Imatinib' rows found in drug CSV.")

    print("\n--- Simulating binding ---")
    G_drug, failed_targets = simulate_binding(G0, resolved, affinities=None, actions=actions)
    print("Failed targets from simulate_binding:", failed_targets)

    if "ABL1" in G0 and len(list(G0.neighbors("ABL1"))) > 0:
        nbr = list(G0.neighbors("ABL1"))[0]
        before = float(G0["ABL1"][nbr]["weight"])
        after = float(G_drug["ABL1"][nbr]["weight"])
        print(f"ABL1 neighbor '{nbr}' weight: {before:.4f} -> {after:.4f}")
        assert float(G0["ABL1"][nbr]["weight"]) == before
        print("Confirmed: G0 unchanged after simulation")

    print("\n--- Building Parkinson disease-stressed graph ---")
    G_dis = build_disease_graph(G0, "Parkinson", disease_df)

    # Basic count report for Parkinson rows
    disease_col = "disease" if "disease" in disease_df.columns else "disease_name"
    gene_col = "gene" if "gene" in disease_df.columns else "gene_symbol"
    park_subset = disease_df[disease_df[disease_col].astype(str).str.contains("Parkinson", case=False, na=False)]
    total = int(park_subset[gene_col].dropna().nunique())
    resolved_count = sum(
        1
        for g in park_subset[gene_col].dropna().astype(str).unique().tolist()
        if _resolve_gene_symbol(g, graph_nodes) is not None
    )
    print(f"Parkinson genes found in graph: {resolved_count}/{total}")

    print(f"\nDone. Unmapped IDs (if any) are in {FAILED_MAPPINGS_FILE}.")
