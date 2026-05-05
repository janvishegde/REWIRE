import pandas as pd

# Load network hits
df = pd.read_csv("data/processed/drug_disease_network_hits.csv")

# Count unique genes hit by each drug for each disease
ranked = (
    df.groupby(["disease", "drug_name"])["gene"]
    .nunique()
    .reset_index()
)

ranked.rename(columns={"gene": "score"}, inplace=True)

# Sort highest score first
ranked = ranked.sort_values(
    by=["disease", "score"],
    ascending=[True, False]
)

# Save full ranking
ranked.to_csv("data/processed/ranked_candidates.csv", index=False)

print("Saved ranked_candidates.csv\n")

# Show top 5 per disease
for disease in ranked["disease"].unique():
    print("=" * 60)
    print("Disease:", disease)

    top = ranked[ranked["disease"] == disease].head(5)

    print(top.to_string(index=False))