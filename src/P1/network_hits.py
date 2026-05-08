import pandas as pd

ppi = pd.read_csv("data/processed/ppi_genes.csv")
drug = pd.read_csv("data/processed/drug_targets_symbols.csv")
disease = pd.read_csv("data/processed/disease_genes.csv")

drug = drug.dropna(subset=["gene_symbol"])

# Match target -> gene1
hits1 = pd.merge(
    drug,
    ppi,
    left_on="gene_symbol",
    right_on="gene1",
    how="inner"
)

result1 = pd.merge(
    hits1,
    disease,
    left_on="gene2",
    right_on="gene",
    how="inner"
)

# Match target -> gene2
hits2 = pd.merge(
    drug,
    ppi,
    left_on="gene_symbol",
    right_on="gene2",
    how="inner"
)

result2 = pd.merge(
    hits2,
    disease,
    left_on="gene1",
    right_on="gene",
    how="inner"
)

final = pd.concat([result1, result2], ignore_index=True)

final = final[[
    "drug_name",
    "gene_symbol",
    "disease",
    "gene"
]].drop_duplicates()

final.to_csv("data/processed/drug_disease_network_hits.csv", index=False)

print("Saved drug_disease_network_hits.csv")
print("Rows:", len(final))
print(final.head(20))