import pandas as pd

valid = pd.read_csv("data/processed/valid_drug_targets_clean.csv")
affinity = pd.read_csv("data/processed/drug_50_affinity.csv")

valid.columns = valid.columns.str.strip().str.lower()
affinity.columns = affinity.columns.str.strip().str.lower()

merged = affinity.merge(valid[["drug_name", "gene_symbol", "action"]], on=["drug_name", "gene_symbol"], how="left")

merged = merged.drop_duplicates(subset=["drug_name", "gene_symbol","action"])
merged = merged.sort_values(["drug_name", "gene_symbol"]).reset_index(drop=True)
merged["action"] = merged["action"].fillna("unknown")
merged = merged[["drug_name", "gene_symbol", "action", "affinity_nm", "source"]]

merged.to_csv("data/processed/canonical_drug_targets.csv", index=False)

print(f"Total rows    : {len(merged)}")
print(f"Unique drugs  : {merged['drug_name'].nunique()}")
print(f"Unique genes  : {merged['gene_symbol'].nunique()}")
print(f"\nFirst 20 rows:")
print(merged.head(20).to_string(index=False))