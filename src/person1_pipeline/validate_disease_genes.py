import pandas as pd

print("REWIRE — Disease Gene Validation")
print("-" * 50)

# Load disease genes
disease_df = pd.read_csv(
    "data/processed/disease_genes_full.csv"
)

# Load PPI graph genes
ppi_df = pd.read_csv(
    "data/processed/ppi_genes.csv"
)

# Build unique PPI gene set
ppi_genes = set(ppi_df["gene1"]).union(
    set(ppi_df["gene2"])
)

# Unique disease genes
disease_genes = set(disease_df["gene"])

matched = disease_genes.intersection(ppi_genes)
missing = disease_genes.difference(ppi_genes)

print(f"PPI genes           : {len(ppi_genes):,}")
print(f"Disease genes       : {len(disease_genes):,}")
print(f"Matched genes       : {len(matched):,}")
print(f"Missing genes       : {len(missing):,}")

overlap = (len(matched) / len(disease_genes)) * 100

print(f"\nOverlap percentage  : {overlap:.2f}%")

print("\n" + "-" * 50)
print("MISSING GENES")
print("-" * 50)

if len(missing) == 0:
    print("None")
else:
    for gene in sorted(missing):
        print("-", gene)

print("\n" + "-" * 50)
print("PER-DISEASE COVERAGE")
print("-" * 50)

for disease in disease_df["disease"].unique():

    genes = set(
        disease_df[disease_df["disease"] == disease]["gene"]
    )

    matched_genes = genes.intersection(ppi_genes)

    pct = (len(matched_genes) / len(genes)) * 100

    print(
        f"{disease:<30} "
        f"{len(matched_genes):>2}/{len(genes):<2} "
        f"({pct:.1f}%)"
    )