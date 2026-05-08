"""
validate_targets.py  —  Person 1 / REWIRE pipeline
====================================================
Cross-checks drug_targets_small_molecule.csv against ppi_genes.csv
to find which drug targets are actually present in the STRING
interaction graph.

Why this matters
----------------
A drug target that doesn't appear in the PPI graph produces no
network rewiring — its G_drug will be identical to G₀ and its
RSV will be all zeros. You need to know this BEFORE Person 2
runs the batch simulation.

Prints
------
  1. Total matched targets  (gene exists in PPI graph)
  2. Total unmatched targets
  3. Matched / unmatched gene lists
  4. Top 10 drugs ranked by number of valid PPI targets
  5. Coverage per must-have drug  (Imatinib, Nilotinib, …)
"""

import sys
import pandas as pd
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
DRUG_PATH = "data/processed/drug_targets_small_molecule.csv"
PPI_PATH  = "data/processed/ppi_genes.csv"

MUST_HAVE = {
    "Imatinib", "Nilotinib", "Gefitinib", "Erlotinib",
    "Metformin", "Atorvastatin", "Aspirin",
}


# ── Loaders ───────────────────────────────────────────────────────────────────
def load_drug_targets(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        sys.exit(f"[ERROR] Drug file not found: {path}\n"
                 f"        Run parse_drugbank.py first.")
    df = pd.read_csv(p)
    df.columns = df.columns.str.strip().str.lower()

    # Accept gene_symbol or target_gene
    if "gene_symbol" not in df.columns and "target_gene" in df.columns:
        df.rename(columns={"target_gene": "gene_symbol"}, inplace=True)

    df = df.dropna(subset=["drug_name", "gene_symbol"])
    df["gene_symbol"] = df["gene_symbol"].str.strip().str.upper()
    df["drug_name"]   = df["drug_name"].str.strip()
    return df[["drug_name", "gene_symbol"]].drop_duplicates()


def load_ppi_genes(path: str) -> set:
    p = Path(path)
    if not p.exists():
        sys.exit(f"[ERROR] PPI file not found: {path}\n"
                 f"        Run map_ppi_genes.py first.")
    ppi = pd.read_csv(p)
    ppi.columns = ppi.columns.str.strip().str.lower()

    # Support both gene1/gene2 and protein1/protein2 naming
    col_map = {}
    for col in ppi.columns:
        if "gene1" in col or col == "protein1": col_map[col] = "gene1"
        elif "gene2" in col or col == "protein2": col_map[col] = "gene2"
    ppi.rename(columns=col_map, inplace=True)

    genes = set(ppi["gene1"].dropna().str.strip().str.upper()) | \
            set(ppi["gene2"].dropna().str.strip().str.upper())
    return genes


# ── Validation ────────────────────────────────────────────────────────────────
def validate(drug_df: pd.DataFrame, ppi_genes: set) -> pd.DataFrame:
    """
    Annotate each drug-gene row with a match flag.
    Returns the annotated DataFrame.
    """
    drug_df = drug_df.copy()
    drug_df["in_ppi"] = drug_df["gene_symbol"].isin(ppi_genes)
    return drug_df


def print_section(title: str) -> None:
    width = 60
    print(f"\n{'─' * width}")
    print(f"  {title}")
    print(f"{'─' * width}")


# ── Report ────────────────────────────────────────────────────────────────────
def run_report(drug_df: pd.DataFrame, ppi_genes: set) -> None:

    annotated  = validate(drug_df, ppi_genes)
    matched    = annotated[annotated["in_ppi"]]
    unmatched  = annotated[~annotated["in_ppi"]]

    n_matched_genes   = matched["gene_symbol"].nunique()
    n_unmatched_genes = unmatched["gene_symbol"].nunique()
    n_total_genes     = annotated["gene_symbol"].nunique()
    pct = (n_matched_genes / n_total_genes * 100) if n_total_genes else 0

    # ── 1. Global counts ──────────────────────────────────────────────────
    print_section("TARGET GENE COVERAGE vs PPI GRAPH")
    print(f"  PPI graph unique genes      : {len(ppi_genes):>7,}")
    print(f"  Drug target genes (total)   : {n_total_genes:>7,}")
    print(f"  ✅  Matched  (in PPI graph)  : {n_matched_genes:>7,}  ({pct:.1f}%)")
    print(f"  ❌  Unmatched (not in PPI)   : {n_unmatched_genes:>7,}  ({100-pct:.1f}%)")

    # ── 2. Matched gene list ──────────────────────────────────────────────
    print_section("MATCHED GENE SYMBOLS (first 30)")
    matched_list = sorted(matched["gene_symbol"].unique())
    for i, g in enumerate(matched_list[:30], 1):
        print(f"  {i:>3}. {g}")
    if len(matched_list) > 30:
        print(f"       ... and {len(matched_list) - 30} more")

    # ── 3. Unmatched gene list ────────────────────────────────────────────
    print_section("UNMATCHED GENE SYMBOLS (first 30)")
    unmatched_list = sorted(unmatched["gene_symbol"].unique())
    if not unmatched_list:
        print("  (none — perfect coverage)")
    for i, g in enumerate(unmatched_list[:30], 1):
        print(f"  {i:>3}. {g}")
    if len(unmatched_list) > 30:
        print(f"       ... and {len(unmatched_list) - 30} more")

    # ── 4. Top 10 drugs by matched PPI target count ───────────────────────
    print_section("TOP 10 DRUGS — most valid PPI targets")
    top_drugs = (
        matched.groupby("drug_name")["gene_symbol"]
        .nunique()
        .reset_index()
        .rename(columns={"gene_symbol": "ppi_targets"})
        .sort_values("ppi_targets", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )
    top_drugs.index += 1
    print(top_drugs.to_string())

    # ── 5. Must-have drug breakdown ───────────────────────────────────────
    print_section("MUST-HAVE DRUG COVERAGE")
    print(f"  {'Drug':<20} {'Total targets':>14} {'In PPI':>8} {'Missing':>8}")
    print(f"  {'─'*20}  {'─'*13}  {'─'*7}  {'─'*7}")

    drugs_in_data = set(annotated["drug_name"].unique())

    for drug in sorted(MUST_HAVE):
        if drug not in drugs_in_data:
            print(f"  {drug:<20} {'NOT IN CSV':>14}")
            continue
        sub      = annotated[annotated["drug_name"] == drug]
        total    = sub["gene_symbol"].nunique()
        in_ppi   = sub[sub["in_ppi"]]["gene_symbol"].nunique()
        missing  = total - in_ppi
        flag     = "✅" if in_ppi > 0 else "❌"
        print(f"  {flag} {drug:<18} {total:>14,} {in_ppi:>8,} {missing:>8,}")

        # Print which specific genes matched / missed for these critical drugs
        matched_genes  = sorted(sub[sub["in_ppi"]]["gene_symbol"].unique())
        missing_genes  = sorted(sub[~sub["in_ppi"]]["gene_symbol"].unique())
        if matched_genes:
            print(f"       matched : {', '.join(matched_genes)}")
        if missing_genes:
            print(f"       missing : {', '.join(missing_genes)}")

    # ── 6. Readiness gate ─────────────────────────────────────────────────
    print_section("PIPELINE READINESS")
    must_have_in_data   = MUST_HAVE & drugs_in_data
    must_have_with_hits = {
        d for d in must_have_in_data
        if matched[matched["drug_name"] == d]["gene_symbol"].nunique() > 0
    }

    if len(must_have_with_hits) == len(MUST_HAVE):
        print("  ✅  All must-have drugs have ≥1 PPI target. Person 2 can proceed.")
    else:
        missing_must = MUST_HAVE - must_have_with_hits
        print(f"  ❌  {len(missing_must)} must-have drug(s) have zero PPI targets:")
        for d in sorted(missing_must):
            print(f"       – {d}")
        print("  Action: check DrugBank XML parsing OR STRING version mismatch.")

    if n_matched_genes < 20:
        print(f"\n  ⚠️   Only {n_matched_genes} unique genes matched — "
              f"may not be enough for meaningful RSV vectors.")
        print("       Consider lowering STRING confidence cutoff below 700.")

    print()


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("REWIRE — Target Validation")
    print(f"  Drug file : {DRUG_PATH}")
    print(f"  PPI file  : {PPI_PATH}")

    drug_df   = load_drug_targets(DRUG_PATH)
    ppi_genes = load_ppi_genes(PPI_PATH)

    print(f"\n  Loaded {len(drug_df):,} drug-gene rows "
          f"({drug_df['drug_name'].nunique():,} unique drugs)")

    run_report(drug_df, ppi_genes)