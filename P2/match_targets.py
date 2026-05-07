import pandas as pd

# Load the datasets
try:
    drug_targets = pd.read_csv('drug_targets_symbols.csv')
    ppi_genes = pd.read_csv('ppi_genes.csv')
except FileNotFoundError as e:
    print(f"Error loading data: {e}. Please make sure the CSV files are in the correct directory.")
    exit()

# Get unique gene symbols from both files
drug_gene_symbols = set(drug_targets['gene_symbol'].unique())
ppi_gene1_symbols = set(ppi_genes['gene1'].unique())
ppi_gene2_symbols = set(ppi_genes['gene2'].unique())
ppi_all_symbols = ppi_gene1_symbols.union(ppi_gene2_symbols)

# Find the intersection of gene symbols
matched_genes = drug_gene_symbols.intersection(ppi_all_symbols)

# Print the match statistics
print(f"Found {len(matched_genes)} matching gene symbols out of {len(drug_gene_symbols)} total drug targets.")
print("-" * 30)

# Filter the drug targets dataframe to only include drugs with targets in the PPI network
matched_drug_targets = drug_targets[drug_targets['gene_symbol'].isin(matched_genes)]

# Find the top 10 drugs with the most targets in the PPI network
top_10_drugs = matched_drug_targets['drug_name'].value_counts().nlargest(10)

print("Top 10 drugs with the most targets present in the PPI network:")
print(top_10_drugs)
