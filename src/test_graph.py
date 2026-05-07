import pandas as pd

from rewire_rsv import (
    load_ppi_network,
    extract_local_subgraph,
    simulate_drug_binding,
    compute_rsv
)

print("Loading graph...")

G = load_ppi_network("data/processed/ppi_genes.csv")

print("\nLoading mini drug dataset...")

df = pd.read_csv("data/processed/drug_mini.csv")

# Take first drug only
drug_name = df.iloc[0]["drug_name"]
target_gene = df.iloc[0]["target_gene"]

print(f"\nDrug: {drug_name}")
print(f"Target: {target_gene}")

print("\nSimulating binding...")

G_local = extract_local_subgraph(G, [target_gene], hops=2)

G_sim = simulate_drug_binding(G_local, [target_gene])
print("\nComputing RSV...")

rsv = compute_rsv(G, G_sim, [target_gene])

print("\nRSV Vector:")
print(rsv)