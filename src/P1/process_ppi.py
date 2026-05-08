import pandas as pd

try:
    file_path = "data/raw/string_links.txt"

    df = pd.read_csv(file_path, sep=r"\s+")

    df.columns = ["protein1", "protein2", "weight"]

    # keep strong interactions
    df = df[df["weight"] >= 700]

    # save
    df.to_csv("data/processed/ppi.csv", index=False)

    # stats
    proteins = set(df["protein1"]).union(set(df["protein2"]))

    print("Saved ppi.csv")
    print("Rows:", len(df))
    print("Unique proteins:", len(proteins))
    print(df.head())

except FileNotFoundError:
    print("File not found.")