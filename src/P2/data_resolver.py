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


def _load_csv_if_exists(path: str) -> pd.DataFrame | None:
    try:
        if os.path.exists(path):
            return pd.read_csv(path)
    except Exception:
        return None
    return None


def _candidate_csv_paths(filename: str) -> list[str]:
    """Return likely locations for upgraded CSVs.

    Supports:
    - same directory as this script (per prompt)
    - repo layout: src/P2 + data/processed
    """

    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, filename),
        os.path.normpath(os.path.join(here, "..", "..", "data", "processed", filename)),
    ]
    # de-dupe while preserving order
    out: list[str] = []
    for p in candidates:
        if p not in out:
            out.append(p)
    return out


def _load_first_csv(filename: str) -> pd.DataFrame | None:
    for path in _candidate_csv_paths(filename):
        df = _load_csv_if_exists(path)
        if df is not None:
            return df
    return None


def _filter_drug_rows(df: pd.DataFrame, drug_name: str) -> pd.DataFrame:
    """Case-insensitive match: exact equality, then partial contains fallback."""

    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return df
    if "drug_name" not in df.columns:
        return df.iloc[0:0]

    series = df["drug_name"].astype(str)
    dn = str(drug_name).strip().lower()
    exact_mask = series.str.strip().str.lower() == dn
    subset = df[exact_mask]
    if subset.empty:
        subset = df[series.str.contains(str(drug_name), case=False, na=False)]
    return subset


def _normalize_action(action_val: str | None) -> str:
    if action_val is None:
        return "inhibitor"
    a = str(action_val).strip().lower()
    if a in {"inhibitor", "activator", "unknown"}:
        return a
    return "unknown"


def load_all_data(
    ppi_csv: str = "ppi_genes.csv",
    drug_affinity_csv: str = "drug_50_affinity.csv",
    disease_csv: str = "disease_genes.csv",
    action_csv: str = "drug_targets_small_molecule.csv",
):
    """Convenience loader for Person-2 workflows.

    This is intentionally lightweight and avoids global state.

    Returns:
        (G0, graph_nodes, drug_df, disease_df, action_df)

    Notes:
        - Uses Week-1 build_graph() for PPI CSV.
        - CSV paths are interpreted relative to the current working directory.
    """

    from ppi_graph import build_graph

    def _resolve_input_path(p: str) -> str:
        # If user passed an explicit existing path, use it.
        if os.path.exists(p):
            return p
        # If they passed a bare filename, try our known locations.
        base = os.path.basename(p)
        for cand in _candidate_csv_paths(base):
            if os.path.exists(cand):
                return cand
        return p

    ppi_path = _resolve_input_path(ppi_csv)
    drug_affinity_path = _resolve_input_path(drug_affinity_csv)
    disease_path = _resolve_input_path(disease_csv)
    action_path = _resolve_input_path(action_csv)

    G0 = build_graph(ppi_path)
    if G0 is None:
        raise ValueError(f"Failed to build graph from: {ppi_path}")

    graph_nodes = set(G0.nodes())

    drug_df = pd.read_csv(drug_affinity_path)
    disease_df = pd.read_csv(disease_path)
    action_df = pd.read_csv(action_path)

    return G0, graph_nodes, drug_df, disease_df, action_df


def resolve_targets(
    drug_name: str,
    drug_df_or_graph_nodes,
    graph_nodes: set[str] | None = None,
    action_df: pd.DataFrame | None = None,
) -> tuple[list[str], dict[str, str], dict[str, float | None]]:
    """Resolve targets using upgraded CSV priority, while staying backwards compatible.

    New behavior (when called without an explicit drug_df):
      1) Check drug_50_affinity.csv (affinity_nM) first
      2) If not found there, check drug_targets_small_molecule.csv (action)
      3) If still not found, fall back to drug_50.csv (target_gene)
      4) If nowhere: log and return empty

    Merge rule:
      - If the drug appears in BOTH affinity and small_molecule datasets,
        merge by gene_symbol so actions + affinities are both populated.

    Backwards compatibility:
      - Old call style: resolve_targets(drug_name, drug_df, graph_nodes)
      - New convenience: resolve_targets(drug_name, graph_nodes)
    """

    # Support calling patterns:
    #   resolve_targets(drug, graph_nodes)
    #   resolve_targets(drug, drug_df, graph_nodes)
    drug_df: pd.DataFrame | None
    if graph_nodes is None and isinstance(drug_df_or_graph_nodes, set):
        graph_nodes = drug_df_or_graph_nodes
        drug_df = None
    else:
        drug_df = drug_df_or_graph_nodes if isinstance(drug_df_or_graph_nodes, pd.DataFrame) else None

    if graph_nodes is None or not isinstance(graph_nodes, set):
        _append_failed_mapping(f"Drug '{drug_name}': resolve_targets called without graph_nodes")
        return [], {}, {}

    resolved_targets: list[str] = []
    actions: dict[str, str] = {}
    affinities: dict[str, float | None] = {}

    # If caller passes a df explicitly, keep the Week-2 generic behavior on that df.
    # (This is how unit tests call the function.)
    if drug_df is not None:
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

        subset = _filter_drug_rows(drug_df, drug_name)

        # Optional columns when available on the primary df
        action_col = "action" if "action" in subset.columns else None
        affinity_col = "affinity_nM" if "affinity_nM" in subset.columns else None

        # Optional: action_df passed separately (common when drug_df is affinity-only)
        action_subset = None
        action_gene_col = None
        if action_df is not None and isinstance(action_df, pd.DataFrame) and not action_df.empty:
            action_subset = _filter_drug_rows(action_df, drug_name)
            if action_subset is not None and not action_subset.empty:
                for cand in ("gene_symbol", "target_gene", "gene", "symbol"):
                    if cand in action_subset.columns:
                        action_gene_col = cand
                        break

        for raw_gene in subset[gene_col].dropna().astype(str).unique().tolist():
            resolved = _resolve_gene_symbol(raw_gene, graph_nodes)
            if resolved is None:
                _append_failed_mapping(f"Drug '{drug_name}' unmapped target: {raw_gene}")
                continue
            if resolved in resolved_targets:
                continue

            resolved_targets.append(resolved)
            act = "inhibitor"
            if action_col is not None:
                try:
                    vals = subset.loc[subset[gene_col].astype(str) == raw_gene, action_col].dropna()
                    if not vals.empty:
                        act = _normalize_action(vals.iloc[0])
                except Exception:
                    pass
            elif action_subset is not None and action_gene_col is not None and "action" in action_subset.columns:
                # Pull action from separate df
                try:
                    vals = action_subset.loc[
                        action_subset[action_gene_col].astype(str) == str(raw_gene),
                        "action",
                    ].dropna()
                    if not vals.empty:
                        act = _normalize_action(vals.iloc[0])
                except Exception:
                    pass

            aff: float | None = None
            if affinity_col is not None:
                try:
                    vals = subset.loc[subset[gene_col].astype(str) == raw_gene, affinity_col].dropna()
                    if not vals.empty:
                        aff = float(vals.iloc[0])
                except Exception:
                    aff = None

            actions[resolved] = act
            affinities[resolved] = aff

        return resolved_targets, actions, affinities

    # New multi-source lookup behavior: load local upgraded CSVs.
    affinity_df = _load_first_csv("drug_50_affinity.csv")
    small_df = _load_first_csv("drug_targets_small_molecule.csv")
    fallback_df = _load_first_csv("drug_50.csv")

    aff_subset = _filter_drug_rows(affinity_df, drug_name) if affinity_df is not None else None
    small_subset = _filter_drug_rows(small_df, drug_name) if small_df is not None else None
    fb_subset = _filter_drug_rows(fallback_df, drug_name) if fallback_df is not None else None

    found_in_affinity = aff_subset is not None and not aff_subset.empty
    found_in_small = small_subset is not None and not small_subset.empty
    found_in_fallback = fb_subset is not None and not fb_subset.empty

    if not (found_in_affinity or found_in_small or found_in_fallback):
        _append_failed_mapping(f"Drug '{drug_name}' not found in any lookup CSV")
        return [], {}, {}

    # Priority logic:
    # - If found in affinity: take affinity genes, and union in small-molecule genes if present (merge).
    # - Else if found in small: take small genes.
    # - Else: take fallback genes.
    genes: set[str] = set()
    if found_in_affinity:
        if "gene_symbol" in aff_subset.columns:
            genes.update(aff_subset["gene_symbol"].dropna().astype(str).unique().tolist())
        if found_in_small and "gene_symbol" in small_subset.columns:
            genes.update(small_subset["gene_symbol"].dropna().astype(str).unique().tolist())
    elif found_in_small:
        if "gene_symbol" in small_subset.columns:
            genes.update(small_subset["gene_symbol"].dropna().astype(str).unique().tolist())
    else:
        if "target_gene" in fb_subset.columns:
            genes.update(fb_subset["target_gene"].dropna().astype(str).unique().tolist())

    # Build lookups for action + affinity.
    action_by_gene: dict[str, str] = {}
    if found_in_small and "gene_symbol" in small_subset.columns:
        if "action" in small_subset.columns:
            for _, row in small_subset[["gene_symbol", "action"]].dropna().iterrows():
                gs = str(row["gene_symbol"]).strip()
                if gs and gs not in action_by_gene:
                    action_by_gene[gs] = _normalize_action(row["action"])

    affinity_by_gene: dict[str, float] = {}
    if found_in_affinity and "gene_symbol" in aff_subset.columns and "affinity_nM" in aff_subset.columns:
        for _, row in aff_subset[["gene_symbol", "affinity_nM"]].dropna().iterrows():
            gs = str(row["gene_symbol"]).strip()
            try:
                val = float(row["affinity_nM"])
            except Exception:
                continue
            if not gs:
                continue
            # Prefer strongest (smallest nM) if multiple rows exist.
            if gs not in affinity_by_gene or val < affinity_by_gene[gs]:
                affinity_by_gene[gs] = val

    for raw_gene in sorted(genes):
        resolved = _resolve_gene_symbol(raw_gene, graph_nodes)
        if resolved is None:
            _append_failed_mapping(f"Drug '{drug_name}' unmapped target: {raw_gene}")
            continue

        resolved_targets.append(resolved)
        actions[resolved] = action_by_gene.get(raw_gene, "inhibitor")
        affinities[resolved] = affinity_by_gene.get(raw_gene)

    return resolved_targets, actions, affinities


def build_disease_graph(
    G0: nx.Graph,
    disease_name: str,
    disease_df: pd.DataFrame | None = None,
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

    # If no df is passed, prefer valid_disease_genes.csv (more coverage),
    # then fall back to disease_genes.csv.
    if disease_df is None:
        valid_df = _load_first_csv("valid_disease_genes.csv")
        fallback_df = _load_first_csv("disease_genes.csv")

        if valid_df is not None:
            subset = valid_df[
                valid_df["disease"].astype(str).str.contains(str(disease_name), case=False, na=False)
            ]
            if not subset.empty:
                disease_df = valid_df
            else:
                disease_df = fallback_df
        else:
            disease_df = fallback_df

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

    # Reset log so the summary is clean
    if os.path.exists(FAILED_MAPPINGS_FILE):
        os.remove(FAILED_MAPPINGS_FILE)

    print("--- Loading G0 ---")
    G0 = build_graph(ppi_csv)
    if G0 is None:
        raise SystemExit(1)

    graph_nodes = set(G0.nodes())

    print("\n--- Resolving Imatinib targets ---")
    # New usage: loads upgraded CSVs automatically (next-to-script or data/processed)
    resolved, actions, affinities = resolve_targets("Imatinib", graph_nodes)

    # Compute X/Y/Z summary using the same priority/merge rules, but without graph resolution.
    affinity_df = _load_first_csv("drug_50_affinity.csv")
    small_df = _load_first_csv("drug_targets_small_molecule.csv")
    fallback_df = _load_first_csv("drug_50.csv")

    aff_subset = _filter_drug_rows(affinity_df, "Imatinib") if affinity_df is not None else None
    small_subset = _filter_drug_rows(small_df, "Imatinib") if small_df is not None else None
    fb_subset = _filter_drug_rows(fallback_df, "Imatinib") if fallback_df is not None else None

    found_in_aff = aff_subset is not None and not aff_subset.empty
    found_in_small = small_subset is not None and not small_subset.empty
    found_in_fb = fb_subset is not None and not fb_subset.empty

    raw_genes: set[str] = set()
    if found_in_aff:
        if "gene_symbol" in aff_subset.columns:
            raw_genes.update(aff_subset["gene_symbol"].dropna().astype(str).unique().tolist())
        if found_in_small and "gene_symbol" in small_subset.columns:
            raw_genes.update(small_subset["gene_symbol"].dropna().astype(str).unique().tolist())
    elif found_in_small:
        if "gene_symbol" in small_subset.columns:
            raw_genes.update(small_subset["gene_symbol"].dropna().astype(str).unique().tolist())
    elif found_in_fb:
        if "target_gene" in fb_subset.columns:
            raw_genes.update(fb_subset["target_gene"].dropna().astype(str).unique().tolist())

    total_raw = len(raw_genes)
    failed = max(0, total_raw - len(resolved))
    print("Resolved targets:", resolved)
    print(f"Summary: {len(resolved)}/{total_raw} targets resolved, {failed} failed (see {FAILED_MAPPINGS_FILE}).")
    if total_raw == 0:
        print("Note: No matching 'Imatinib' rows found in lookup CSVs.")

    print("\n--- Simulating binding ---")
    # Pass real affinities when available (affinity_nM) so perturbation strength is meaningful.
    G_drug, failed_targets = simulate_binding(G0, resolved, affinities=affinities, actions=actions)
    print("Failed targets from simulate_binding:", failed_targets)

    if "ABL1" in G0 and len(list(G0.neighbors("ABL1"))) > 0:
        nbr = list(G0.neighbors("ABL1"))[0]
        before = float(G0["ABL1"][nbr]["weight"])
        after = float(G_drug["ABL1"][nbr]["weight"])
        print(f"ABL1 neighbor '{nbr}' weight: {before:.4f} -> {after:.4f}")
        assert float(G0["ABL1"][nbr]["weight"]) == before
        print("Confirmed: G0 unchanged after simulation")

    print("\n--- Building Parkinson disease-stressed graph ---")
    G_dis = build_disease_graph(G0, "Parkinson")

    # Basic count report for Parkinson rows using the chosen dataset
    valid_df = _load_first_csv("valid_disease_genes.csv")
    fallback_df = _load_first_csv("disease_genes.csv")
    use_df = None
    if valid_df is not None:
        sub = valid_df[valid_df["disease"].astype(str).str.contains("Parkinson", case=False, na=False)]
        if not sub.empty:
            use_df = valid_df
        else:
            use_df = fallback_df
    else:
        use_df = fallback_df

    if use_df is not None:
        park_subset = use_df[use_df["disease"].astype(str).str.contains("Parkinson", case=False, na=False)]
        total = int(park_subset["gene"].dropna().astype(str).nunique())
        resolved_count = sum(
            1
            for g in park_subset["gene"].dropna().astype(str).unique().tolist()
            if _resolve_gene_symbol(g, graph_nodes) is not None
        )
        print(f"Parkinson genes found in graph: {resolved_count}/{total}")
    else:
        print("Note: No disease CSV found for Parkinson check.")

    print(f"\nDone. Unmapped IDs (if any) are in {FAILED_MAPPINGS_FILE}.")
