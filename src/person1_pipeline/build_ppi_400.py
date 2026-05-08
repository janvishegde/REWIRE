"""
build_ppi_400.py  —  Person 1 / REWIRE pipeline
=================================================
Regenerates the STRING PPI graph at confidence cutoff = 400
(relaxed from the previous 700) for the human proteome.

Downloads STRING human protein links if not already cached,
maps Ensembl protein IDs to HGNC gene symbols via the
9606.protein.info.v12.0.txt alias file, then filters and
normalises weights.

Output: data/processed/ppi_400.csv
Columns: gene1, gene2, weight (0.0–1.0)

Files downloaded (cached in data/raw/string/):
  9606.protein.links.v12.0.txt.gz   (~450 MB compressed)
  9606.protein.info.v12.0.txt.gz    (~30 MB compressed)

Runtime: ~3–6 min on first run (download), ~45s on re-runs.
"""

import gzip
import requests
import pandas as pd
import networkx as nx
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
CONFIDENCE_CUTOFF = 400          # relaxed from 700
OUTPUT_PATH       = "data/processed/ppi_400.csv"
CACHE_DIR         = Path("data/raw/string")

STRING_VERSION = "12.0"
TAXON_ID       = "9606"          # Homo sapiens

LINKS_URL = (f"https://stringdb-downloads.org/download/"
             f"protein.links.v{STRING_VERSION}/"
             f"{TAXON_ID}.protein.links.v{STRING_VERSION}.txt.gz")

INFO_URL  = (f"https://stringdb-downloads.org/download/"
             f"protein.info.v{STRING_VERSION}/"
             f"{TAXON_ID}.protein.info.v{STRING_VERSION}.txt.gz")


# ── Downloader ────────────────────────────────────────────────────────────────
def download_if_missing(url: str, dest: Path, label: str) -> Path:
    if dest.exists():
        print(f"  [cache] {label} already present — skipping download")
        return dest

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  [download] {label} ...", end="", flush=True)

    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):   # 1 MB chunks
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"\r  [download] {label}  {pct:.0f}%    ", end="", flush=True)

    print(f"\r  [download] {label}  done ({downloaded / 1e6:.0f} MB)          ")
    return dest


# ── Gene symbol mapper ────────────────────────────────────────────────────────
def build_protein_to_gene_map(info_path: Path) -> dict[str, str]:
    """
    Parse STRING protein info file to map  9606.ENSP... → HGNC gene symbol.
    The preferred_name column is the HGNC gene symbol for human proteins.
    """
    print("  Building protein → gene symbol map ...", end="", flush=True)

    opener = gzip.open if str(info_path).endswith(".gz") else open
    with opener(info_path, "rt") as f:
        info_df = pd.read_csv(f, sep="\t", usecols=["#string_protein_id", "preferred_name"])

    mapping = dict(zip(
        info_df["#string_protein_id"],
        info_df["preferred_name"]
    ))
    print(f"  {len(mapping):,} proteins mapped")
    return mapping


# ── PPI loader + filter ───────────────────────────────────────────────────────
def load_and_filter_ppi(
    links_path: Path,
    protein_map: dict[str, str],
    cutoff: int,
) -> pd.DataFrame:
    """
    Load STRING links, apply confidence cutoff, translate to gene symbols,
    remove self-loops and duplicates, normalise weights to 0–1.
    """
    print(f"  Loading STRING links (cutoff ≥ {cutoff}) ...", end="", flush=True)

    opener = gzip.open if str(links_path).endswith(".gz") else open
    with opener(links_path, "rt") as f:
        links = pd.read_csv(f, sep=r"\s+")

    print(f"  {len(links):,} raw edges loaded")

    # Apply confidence cutoff
    links = links[links["combined_score"] >= cutoff].copy()
    print(f"  {len(links):,} edges after cutoff={cutoff}")

    # Map to gene symbols
    links["gene1"] = links["protein1"].map(protein_map)
    links["gene2"] = links["protein2"].map(protein_map)

    # Drop unmapped proteins (no HGNC symbol)
    before = len(links)
    links = links.dropna(subset=["gene1", "gene2"])
    print(f"  {before - len(links):,} edges dropped (unmapped proteins)")

    # Remove self-loops
    links = links[links["gene1"] != links["gene2"]]

    # Normalise STRING score (0–1000) to weight (0.0–1.0)
    links["weight"] = links["combined_score"] / 1000.0

    # Deduplicate: keep canonical order (sort gene pair), take max weight
    links["_a"] = links[["gene1", "gene2"]].min(axis=1)
    links["_b"] = links[["gene1", "gene2"]].max(axis=1)
    links = (
        links.groupby(["_a", "_b"])["weight"]
        .max()
        .reset_index()
        .rename(columns={"_a": "gene1", "_b": "gene2"})
    )

    return links[["gene1", "gene2", "weight"]].reset_index(drop=True)


# ── Graph statistics ──────────────────────────────────────────────────────────
def print_graph_stats(df: pd.DataFrame) -> None:
    G = nx.from_pandas_edgelist(df, "gene1", "gene2", edge_attr="weight")

    degrees    = [d for _, d in G.degree()]
    avg_degree = sum(degrees) / len(degrees) if degrees else 0
    avg_weight = df["weight"].mean()

    print()
    print("=" * 55)
    print("  PPI Graph Statistics  (cutoff = 400)")
    print("=" * 55)
    print(f"  Total nodes           : {G.number_of_nodes():>10,}")
    print(f"  Total edges           : {G.number_of_edges():>10,}")
    print(f"  Average degree        : {avg_degree:>10.2f}")
    print(f"  Max degree            : {max(degrees):>10,}")
    print(f"  Min degree            : {min(degrees):>10,}")
    print(f"  Average edge weight   : {avg_weight:>10.4f}")
    print(f"  Weight range          : {df['weight'].min():.3f} – {df['weight'].max():.3f}")
    print(f"  Connected components  : {nx.number_connected_components(G):>10,}")
    print(f"  Largest component     : {len(max(nx.connected_components(G), key=len)):>10,} nodes")


# ── Main ──────────────────────────────────────────────────────────────────────
def run(
    cutoff:      int  = CONFIDENCE_CUTOFF,
    output_path: str  = OUTPUT_PATH,
) -> pd.DataFrame:

    print("REWIRE — STRING PPI Graph Expansion")
    print(f"  Confidence cutoff : {cutoff}")
    print(f"  Species           : Homo sapiens ({TAXON_ID})")
    print(f"  STRING version    : v{STRING_VERSION}")
    print()

    links_file = CACHE_DIR / f"{TAXON_ID}.protein.links.v{STRING_VERSION}.txt.gz"
    info_file  = CACHE_DIR / f"{TAXON_ID}.protein.info.v{STRING_VERSION}.txt.gz"

    download_if_missing(LINKS_URL, links_file, "protein.links")
    download_if_missing(INFO_URL,  info_file,  "protein.info")

    protein_map = build_protein_to_gene_map(info_file)
    ppi_df      = load_and_filter_ppi(links_file, protein_map, cutoff)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    ppi_df.to_csv(output_path, index=False)
    print(f"\n  Saved → {output_path}")

    print_graph_stats(ppi_df)

    return ppi_df


if __name__ == "__main__":
    run()
