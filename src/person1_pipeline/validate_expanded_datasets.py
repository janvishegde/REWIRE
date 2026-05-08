"""
validate_expanded_datasets.py  —  Person 1 / REWIRE pipeline
=============================================================
Validates:
  1. disease_genes_expanded.csv  vs  ppi_400.csv
     — Which disease genes are present in the PPI graph?
     — Which diseases have enough network coverage?

  2. ppi_400.csv graph statistics
     — Node/edge counts, degree distribution, density
     — Comparison vs previous ppi_genes.csv (if available)

Run after both build_ppi_400.py and fetch_disease_genes_expanded.py complete.
"""

import pandas as pd
import networkx as nx
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
DISEASE_PATH     = "data/processed/disease_genes_expanded.csv"
PPI_400_PATH     = "data/processed/ppi_400.csv"
PPI_OLD_PATH     = "data/processed/ppi_genes.csv"          # optional comparison
DRUG_TARGET_PATH = "data/processed/canonical_drug_targets.csv"


# ── Loaders ───────────────────────────────────────────────────────────────────
def load_ppi(path: str) -> tuple[pd.DataFrame, nx.Graph]:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()
    for col in df.columns:
        if "gene1" in col or col == "protein1": df.rename(columns={col: "gene1"}, inplace=True)
        if "gene2" in col or col == "protein2": df.rename(columns={col: "gene2"}, inplace=True)
        if "weight" in col or "score" in col:   df.rename(columns={col: "weight"}, inplace=True)
    if "weight" not in df.columns:
        df["weight"] = 0.5
    if df["weight"].max() > 1.0:
        df["weight"] = df["weight"] / 1000.0
    G = nx.from_pandas_edgelist(df, "gene1", "gene2", edge_attr="weight")
    return df, G


def load_disease(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()
    if "gene" not in df.columns and "gene_symbol" in df.columns:
        df.rename(columns={"gene_symbol": "gene"}, inplace=True)
    return df


def sep(title: str = "") -> None:
    print(f"\n{'─' * 58}")
    if title:
        print(f"  {title}")
        print(f"{'─' * 58}")


# ── Validation 1: PPI graph stats ─────────────────────────────────────────────
def validate_ppi_graph(ppi_path: str, old_ppi_path: str = None) -> nx.Graph:
    sep("PPI GRAPH STATISTICS  (ppi_400.csv)")

    if not Path(ppi_path).exists():
        print(f"  [ERROR] {ppi_path} not found. Run build_ppi_400.py first.")
        return None

    df, G = load_ppi(ppi_path)
    degrees    = [d for _, d in G.degree()]
    avg_degree = sum(degrees) / len(degrees) if degrees else 0
    components = list(nx.connected_components(G))

    print(f"  Nodes (unique genes)  : {G.number_of_nodes():>10,}")
    print(f"  Edges                 : {G.number_of_edges():>10,}")
    print(f"  Average degree        : {avg_degree:>10.2f}")
    print(f"  Max degree            : {max(degrees):>10,}")
    print(f"  Graph density         : {nx.density(G):>10.6f}")
    print(f"  Connected components  : {len(components):>10,}")
    print(f"  Largest component     : {max(len(c) for c in components):>10,} nodes")
    print(f"  Weight mean           : {df['weight'].mean():>10.4f}")
    print(f"  Weight std            : {df['weight'].std():>10.4f}")

    # Top 10 hub genes
    top_hubs = sorted(G.degree(), key=lambda x: x[1], reverse=True)[:10]
    print(f"\n  Top 10 hub genes:")
    print(f"  {'Gene':<12} {'Degree':>8}")
    print(f"  {'─'*12}  {'─'*8}")
    for gene, deg in top_hubs:
        print(f"  {gene:<12} {deg:>8,}")

    # Comparison with old PPI if available
    if old_ppi_path and Path(old_ppi_path).exists():
        sep("PPI COMPARISON  (ppi_400 vs ppi_genes / cutoff 700)")
        df_old, G_old = load_ppi(old_ppi_path)
        print(f"  {'Metric':<28} {'Old (700)':>10} {'New (400)':>10} {'Delta':>10}")
        print(f"  {'─'*28}  {'─'*10}  {'─'*10}  {'─'*10}")
        metrics = [
            ("Nodes",  G_old.number_of_nodes(),  G.number_of_nodes()),
            ("Edges",  G_old.number_of_edges(),  G.number_of_edges()),
            ("Avg degree",
             sum(d for _, d in G_old.degree()) / G_old.number_of_nodes(),
             sum(d for _, d in G.degree())     / G.number_of_nodes()),
        ]
        for label, old_val, new_val in metrics:
            if isinstance(old_val, int):
                delta = f"+{new_val - old_val:,}" if new_val > old_val else f"{new_val - old_val:,}"
                print(f"  {label:<28} {old_val:>10,} {new_val:>10,} {delta:>10}")
            else:
                delta = new_val - old_val
                print(f"  {label:<28} {old_val:>10.2f} {new_val:>10.2f} {delta:>+10.2f}")

    return G


# ── Validation 2: Disease gene vs PPI coverage ─────────────────────────────────
def validate_disease_coverage(disease_path: str, G: nx.Graph) -> None:
    sep("DISEASE GENE COVERAGE  (disease_genes_expanded vs ppi_400)")

    if not Path(disease_path).exists():
        print(f"  [ERROR] {disease_path} not found. "
              f"Run fetch_disease_genes_expanded.py first.")
        return

    if G is None:
        print("  [SKIP] PPI graph not available.")
        return

    df = load_disease(disease_path)
    ppi_genes = set(G.nodes())

    df["in_ppi"] = df["gene"].isin(ppi_genes)
    matched   = df[df["in_ppi"]]
    unmatched = df[~df["in_ppi"]]

    n_genes         = df["gene"].nunique()
    n_matched_genes = matched["gene"].nunique()
    pct             = n_matched_genes / n_genes * 100 if n_genes else 0

    print(f"  Disease genes total   : {n_genes:>8,}")
    print(f"  Matched in PPI graph  : {n_matched_genes:>8,}  ({pct:.1f}%)")
    print(f"  Unmatched             : {df['gene'].nunique() - n_matched_genes:>8,}  ({100 - pct:.1f}%)")

    # Per-disease breakdown
    sep("PER-DISEASE PPI COVERAGE")
    per_disease = (
        df.groupby("disease")
        .apply(lambda g: pd.Series({
            "total_genes":   g["gene"].nunique(),
            "matched_genes": g[g["in_ppi"]]["gene"].nunique(),
            "avg_score":     g["score"].mean() if "score" in g.columns else 0,
        }))
        .reset_index()
    )
    per_disease["coverage_pct"] = (
        per_disease["matched_genes"] / per_disease["total_genes"] * 100
    ).round(1)
    per_disease = per_disease.sort_values("coverage_pct", ascending=False)

    print(f"\n  {'Disease':<38} {'Genes':>6} {'Matched':>8} {'Cover%':>8}")
    print(f"  {'─'*38}  {'─'*6}  {'─'*8}  {'─'*8}")
    for _, row in per_disease.iterrows():
        flag = "✅" if row["coverage_pct"] >= 60 else ("⚠️ " if row["coverage_pct"] >= 30 else "❌")
        print(f"  {flag} {row['disease']:<36} "
              f"{int(row['total_genes']):>6} "
              f"{int(row['matched_genes']):>8} "
              f"{row['coverage_pct']:>7.1f}%")

    # Unmatched genes
    unmatched_genes = sorted(unmatched["gene"].unique())
    print(f"\n  Unmatched genes ({len(unmatched_genes)}):")
    for g in unmatched_genes[:20]:
        print(f"    – {g}")
    if len(unmatched_genes) > 20:
        print(f"    … and {len(unmatched_genes) - 20} more")


# ── Validation 3: Drug target vs PPI coverage ─────────────────────────────────
def validate_drug_coverage(drug_path: str, G: nx.Graph) -> None:
    sep("DRUG TARGET COVERAGE  (canonical_drug_targets vs ppi_400)")

    if not Path(drug_path).exists():
        print(f"  [SKIP] {drug_path} not found.")
        return
    if G is None:
        print("  [SKIP] PPI graph not available.")
        return

    df = pd.read_csv(drug_path)
    df.columns = df.columns.str.strip().str.lower()
    if "target_gene" in df.columns and "gene_symbol" not in df.columns:
        df.rename(columns={"target_gene": "gene_symbol"}, inplace=True)

    ppi_genes = set(G.nodes())
    df["in_ppi"] = df["gene_symbol"].isin(ppi_genes)

    matched_drugs = df[df["in_ppi"]]["drug_name"].nunique()
    total_drugs   = df["drug_name"].nunique()
    matched_genes = df[df["in_ppi"]]["gene_symbol"].nunique()
    total_genes   = df["gene_symbol"].nunique()

    print(f"  Drugs with ≥1 PPI target : {matched_drugs:>6} / {total_drugs}")
    print(f"  Genes matched in PPI     : {matched_genes:>6} / {total_genes}")

    # Drugs with zero PPI targets → will produce flat RSV
    zero_coverage = df.groupby("drug_name")["in_ppi"].any()
    missing_drugs = zero_coverage[~zero_coverage].index.tolist()
    if missing_drugs:
        print(f"\n  ❌ Drugs with NO PPI targets ({len(missing_drugs)}):")
        for d in missing_drugs:
            print(f"     – {d}")
    else:
        print(f"\n  ✅ All drugs have ≥1 target in ppi_400 graph")


# ── Readiness gate ────────────────────────────────────────────────────────────
def readiness_gate(G: nx.Graph, disease_df: pd.DataFrame = None) -> None:
    sep("PIPELINE READINESS GATE")

    checks = []

    # Check 1: Graph size
    if G is not None:
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()
        checks.append(("PPI nodes > 10,000",   n_nodes > 10_000,   f"{n_nodes:,}"))
        checks.append(("PPI edges > 100,000",  n_edges > 100_000,  f"{n_edges:,}"))
        checks.append(("Graph is connected",
                        nx.number_connected_components(G) == 1,
                        f"{nx.number_connected_components(G)} component(s)"))
    else:
        checks.append(("PPI graph loaded", False, "file missing"))

    # Check 2: Disease coverage
    if disease_df is not None and G is not None:
        ppi_genes  = set(G.nodes())
        n_diseases = disease_df["disease"].nunique()
        pct_matched = (disease_df["gene"].isin(ppi_genes).sum()
                       / len(disease_df) * 100)
        checks.append(("≥ 40 diseases loaded",      n_diseases >= 40,    f"{n_diseases}"))
        checks.append(("Disease gene coverage ≥ 50%", pct_matched >= 50, f"{pct_matched:.1f}%"))

    print()
    for label, passed, value in checks:
        icon = "✅" if passed else "❌"
        print(f"  {icon}  {label:<40}  {value}")

    all_passed = all(p for _, p, _ in checks)
    print()
    if all_passed:
        print("  → All checks passed. Person 2 can proceed with RSV simulation.")
    else:
        failed = [l for l, p, _ in checks if not p]
        print(f"  → {len(failed)} check(s) failed. Resolve before handing off to Person 2.")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("REWIRE — Expanded Dataset Validation")
    print(f"  PPI (400)  : {PPI_400_PATH}")
    print(f"  Disease    : {DISEASE_PATH}")
    print(f"  Drug       : {DRUG_TARGET_PATH}")

    G          = validate_ppi_graph(PPI_400_PATH, PPI_OLD_PATH)
    validate_disease_coverage(DISEASE_PATH, G)
    validate_drug_coverage(DRUG_TARGET_PATH, G)

    disease_df = None
    if Path(DISEASE_PATH).exists():
        disease_df = load_disease(DISEASE_PATH)

    readiness_gate(G, disease_df)
