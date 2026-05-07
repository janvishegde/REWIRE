import pandas as pd

# Load curated drug set
drug_df = pd.read_csv("data/processed/drug_50.csv")

# Load PPI genes
ppi_df = pd.read_csv("data/processed/ppi_genes.csv")

# Build set of valid PPI genes
ppi_genes = set(ppi_df["gene1"]).union(set(ppi_df["gene2"]))

print("REWIRE — Batch Drug Validation")
print("-" * 50)

valid_rows = []
skipped = []

for _, row in drug_df.iterrows():

    drug = row["drug_name"]
    target = row["target_gene"]

    if target in ppi_genes:

        valid_rows.append({
            "drug_name": drug,
            "target_gene": target
        })

        print(f"✅ {drug:<20} -> {target}")

    else:

        skipped.append((drug, target))

        print(f"❌ {drug:<20} -> {target} NOT IN PPI")


# Save valid set
valid_df = pd.DataFrame(valid_rows)

valid_df.to_csv(
    "data/processed/drug_50_validated.csv",
    index=False
)

print("\n" + "-" * 50)
print("SUMMARY")
print("-" * 50)

print(f"Total drugs      : {len(drug_df)}")
print(f"Valid drugs      : {len(valid_df)}")
print(f"Skipped drugs    : {len(skipped)}")

print("\nSaved:")
print("data/processed/drug_50_validated.csv")