import pandas as pd

# Load the datasets
ppi_genes = pd.read_csv('ppi_genes.csv')
drug_targets = pd.read_csv('drug_targets_symbols.csv')

# Print shapes
print("Shape of ppi_genes.csv:", ppi_genes.shape)
print("Shape of drug_targets_symbols.csv:", drug_targets.shape)
print("-" * 30)

# Print first 3 rows
print("First 3 rows of ppi_genes.csv:")
print(ppi_genes.head(3))
print("-" * 30)
print("First 3 rows of drug_targets_symbols.csv:")
print(drug_targets.head(3))
print("-" * 30)

# Print stats for 'weight' column
print("Statistics for 'weight' column in ppi_genes.csv:")
print("Min:", ppi_genes['weight'].min())
print("Max:", ppi_genes['weight'].max())
print("Mean:", ppi_genes['weight'].mean())
print("-" * 30)

# Print unique counts
print("Number of unique drugs:", drug_targets['drug_name'].nunique())
print("Number of unique gene symbols:", drug_targets['gene_symbol'].nunique())
