import pandas as pd

print("REWIRE — Drug Target Validation")
print("-" * 50)

# Load drug targets
drug_df = pd.read_csv(
    "data/processed/drug_targets_small_molecule.csv"
)

# Load PPI genes
ppi_df = pd.read_csv(
    "data/processed/ppi_genes.csv"
)

# Build PPI gene set
ppi_genes = set(ppi_df["gene1"]).union(
    set(ppi_df["gene2"])
)

# Keep only valid graph-compatible targets
valid_df = drug_df[
    drug_df["gene_symbol"].isin(ppi_genes)
]
# Keep cleaner human-readable drug names only
valid_df = valid_df[
    ~valid_df["drug_name"].str.startswith("(")
]

valid_df = valid_df[
    valid_df["drug_name"].str.len() < 40
]

valid_df = valid_df[
    ~valid_df["drug_name"].str.contains(
        r"[A-Z]{8,}",
        regex=True,
        na=False
    )
]
# Save validated dataset
valid_df.to_csv(
    "data/processed/valid_drug_targets.csv",
    index=False
)

print(f"Original rows : {len(drug_df):,}")
print(f"Valid rows    : {len(valid_df):,}")

overlap = (
    len(set(valid_df['gene_symbol']))
    / len(set(drug_df['gene_symbol']))
) * 100

print(f"Overlap       : {overlap:.2f}%")

print("\nSaved:")
print("data/processed/valid_drug_targets.csv")