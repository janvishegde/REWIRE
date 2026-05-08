import pandas as pd

# Load files
drug_df = pd.read_csv("data/processed/drug_targets_symbols.csv")
disease_df = pd.read_csv("data/processed/disease_genes.csv")

# Keep useful columns
drug_df = drug_df[["drug_id", "drug_name", "gene_symbol"]]

# Remove blanks
drug_df = drug_df.dropna()
disease_df = disease_df.dropna()

# Merge on gene symbol
overlap = pd.merge(
    drug_df,
    disease_df,
    left_on="gene_symbol",
    right_on="gene",
    how="inner"
)

# Remove duplicates
overlap = overlap.drop_duplicates()

# Save
overlap.to_csv("data/processed/drug_disease_overlaps.csv", index=False)

print("Saved drug_disease_overlaps.csv")
print("Rows:", len(overlap))
print(overlap.head(20))