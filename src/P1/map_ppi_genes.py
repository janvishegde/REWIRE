import pandas as pd

print("Loading files...")

# Load processed PPI
ppi = pd.read_csv("data/processed/ppi.csv")

# Load STRING mapping file
info = pd.read_csv(
    "data/raw/9606.protein.info.v12.0.txt",
    sep="\t"
)

# Keep useful columns
info = info[["#string_protein_id", "preferred_name"]]

# Rename columns
info.columns = ["string_id", "gene_name"]

print("Mapping protein1...")

ppi = pd.merge(
    ppi,
    info,
    left_on="protein1",
    right_on="string_id",
    how="left"
)

ppi.rename(columns={"gene_name": "gene1"}, inplace=True)
ppi.drop(columns=["string_id"], inplace=True)

print("Mapping protein2...")

ppi = pd.merge(
    ppi,
    info,
    left_on="protein2",
    right_on="string_id",
    how="left"
)

ppi.rename(columns={"gene_name": "gene2"}, inplace=True)
ppi.drop(columns=["string_id"], inplace=True)

# Keep rows where both mapped
ppi = ppi.dropna(subset=["gene1", "gene2"])

# Final columns
ppi_final = ppi[["gene1", "gene2", "weight"]]

# Save
ppi_final.to_csv("data/processed/ppi_genes.csv", index=False)

print("Saved ppi_genes.csv")
print("Rows:", len(ppi_final))
print(ppi_final.head())