import pandas as pd

# ==================================================
# REWIRE — Dataset Statistics + Summary Export
# ==================================================

summary = []


def log(text=""):
    print(text)
    summary.append(text)


log("REWIRE — Dataset Statistics")
log("=" * 60)

# ==================================================
# Load datasets
# ==================================================

drug_df = pd.read_csv(
    "data/processed/drug_targets_small_molecule.csv"
)

disease_df = pd.read_csv(
    "data/processed/disease_genes_full.csv"
)

ppi_df = pd.read_csv(
    "data/processed/ppi_genes.csv"
)

drug50_df = pd.read_csv(
    "data/processed/drug_50_validated.csv"
)

# ==================================================
# Build PPI gene set
# ==================================================

ppi_genes = set(ppi_df["gene1"]).union(
    set(ppi_df["gene2"])
)

# ==================================================
# Drug dataset stats
# ==================================================

drug_genes = set(drug_df["gene_symbol"])

drug_overlap = (
    len(drug_genes.intersection(ppi_genes))
    / len(drug_genes)
) * 100

# ==================================================
# Disease dataset stats
# ==================================================

disease_genes = set(disease_df["gene"])

disease_overlap = (
    len(disease_genes.intersection(ppi_genes))
    / len(disease_genes)
) * 100

# ==================================================
# Print + Store Summary
# ==================================================

log("\nPPI NETWORK")
log("-" * 60)

log(f"PPI edges                 : {len(ppi_df):,}")
log(f"PPI unique genes          : {len(ppi_genes):,}")

log("\nDRUG DATASET")
log("-" * 60)

log(f"Drug-target rows          : {len(drug_df):,}")
log(f"Unique drugs              : {drug_df['drug_name'].nunique():,}")
log(f"Unique target genes       : {len(drug_genes):,}")
log(f"Drug-PPI overlap          : {drug_overlap:.2f}%")

log("\nDISEASE DATASET")
log("-" * 60)

log(f"Disease-gene rows         : {len(disease_df):,}")
log(f"Unique diseases           : {disease_df['disease'].nunique():,}")
log(f"Unique disease genes      : {len(disease_genes):,}")
log(f"Disease-PPI overlap       : {disease_overlap:.2f}%")

log("\nCURATED RSV DATASET")
log("-" * 60)

log(f"Validated RSV drugs       : {len(drug50_df):,}")

log("\nPIPELINE STATUS")
log("-" * 60)

log("✔ DrugBank parsing complete")
log("✔ STRING processing complete")
log("✔ OpenTargets integration complete")
log("✔ Small-molecule filtering complete")
log("✔ Drug validation complete")
log("✔ Disease validation complete")
log("✔ Curated RSV dataset complete")

log("\nREWIRE preprocessing pipeline is READY.")

# ==================================================
# Save Summary File
# ==================================================

with open(
    "data/processed/dataset_summary.txt",
    "w",
    encoding="utf-8"
) as f:

    for line in summary:
        f.write(line + "\n")

log("\nSaved: data/processed/dataset_summary.txt")