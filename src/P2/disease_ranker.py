"""Disease-driven drug ranking for the REWIRE demo (May 7).

Given a disease name, this module builds a disease network fingerprint (stub
RSV vector) and ranks available drug-perturbed graphs by cosine similarity.

Weeks status (project context):
- Week 1: PPI graph + simulate_binding (done)
- Week 2: data_resolver (drug/disease mapping + disease graph) (done)
- Week 3: this ranker + stub RSV interface for Person 3 (this module)
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import networkx as nx

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None

from batch_simulate import list_completed_drugs, load_graph
from data_resolver import build_disease_graph, resolve_targets


def _resolve_csv_path(here: Path, filename: str) -> Path:
    """Resolve a CSV path from either the local folder or data/processed."""

    direct = here / filename
    if direct.exists():
        return direct
    processed = (here / ".." / ".." / "data" / "processed" / filename).resolve()
    if processed.exists():
        return processed
    return direct


def _read_csv_best_effort(here: Path, filename: str, require_cols: list[str] | None = None) -> pd.DataFrame:
    """Read CSV from best candidate location.

    If a local file exists but lacks required columns, try data/processed.
    """

    direct = here / filename
    processed = (here / ".." / ".." / "data" / "processed" / filename).resolve()

    def _try(path: Path) -> pd.DataFrame | None:
        if not path.exists():
            return None
        df = pd.read_csv(path)
        if require_cols and not set(require_cols).issubset(df.columns):
            return None
        return df

    df = _try(direct)
    if df is not None:
        return df
    df = _try(processed)
    if df is not None:
        return df

    # Last resort: attempt direct path to raise a useful error.
    return pd.read_csv(direct)


def extract_rsv_stub(G0: nx.Graph, G_drug: nx.Graph, targets: list[str]) -> np.ndarray:
    """STUB — Person 3 replace this with compute_rsv(G0, G_drug, targets) returning real betweenness/community/spectral/entropy vector.

    Proxy vector (shape (4,)):
      [
        mean edge weight around targets in G_drug,
        std of edge weights around targets in G_drug,
        mean edge weight around targets in G0,
        len(targets) / G0.number_of_nodes()
      ]
    """

    if G0 is None or G_drug is None or G0.number_of_nodes() == 0:
        return np.zeros((4,), dtype=float)

    # Collect incident edge weights around targets.
    weights_drug: list[float] = []
    weights_g0: list[float] = []

    for t in targets:
        if t not in G0 or t not in G_drug:
            continue
        for nbr in G_drug.neighbors(t):
            try:
                weights_drug.append(float(G_drug[t][nbr].get("weight", 0.0)))
            except Exception:
                pass
        for nbr in G0.neighbors(t):
            try:
                weights_g0.append(float(G0[t][nbr].get("weight", 0.0)))
            except Exception:
                pass

    if weights_drug:
        mean_drug = float(np.mean(weights_drug))
        std_drug = float(np.std(weights_drug))
    else:
        mean_drug = 0.0
        std_drug = 0.0

    mean_g0 = float(np.mean(weights_g0)) if weights_g0 else 0.0
    density = float(len(targets)) / float(G0.number_of_nodes())

    return np.array([mean_drug, std_drug, mean_g0, density], dtype=float)


def get_disease_signature(
    disease_name: str,
    G0: nx.Graph,
    disease_df: pd.DataFrame,
) -> tuple[nx.Graph, np.ndarray, list[str]]:
    """Build a disease-stressed graph and its stub RSV signature.

    - Case-insensitive partial match on disease name.
    - Amplifies edges around matched disease genes by 1.3x.
    - Prints: "Found X/Y disease genes in graph for [disease_name]".
    """

    if disease_df is None or disease_df.empty:
        print(f"No disease genes provided for '{disease_name}'.")
        return G0.copy(), np.zeros((4,), dtype=float), []

    disease_col = "disease" if "disease" in disease_df.columns else "disease_name"
    gene_col = "gene" if "gene" in disease_df.columns else "gene_symbol"

    subset = disease_df[disease_df[disease_col].astype(str).str.contains(str(disease_name), case=False, na=False)]
    raw_genes = sorted(subset[gene_col].dropna().astype(str).unique().tolist())

    graph_nodes = set(G0.nodes())
    matched = []
    for g in raw_genes:
        gs = str(g).strip()
        if not gs:
            continue
        if gs in graph_nodes:
            matched.append(gs)
        elif gs.upper() in graph_nodes:
            matched.append(gs.upper())

    print(f"Found {len(matched)}/{len(raw_genes)} disease genes in graph for [{disease_name}]")

    # Build disease-stressed graph (1.3x amplification around disease genes).
    G_disease = build_disease_graph(G0, disease_name, disease_df)
    disease_rsv = extract_rsv_stub(G0, G_disease, matched)

    return G_disease, disease_rsv, matched


def _sanitize_drug_name(name: str) -> str:
    s = str(name).strip().lower()
    s = s.replace(" ", "_")
    s = s.replace("/", "")
    s = s.replace("\\", "")
    return s


def _load_targets_table(path: Path) -> dict[str, list[str]]:
    """Load drug→targets mapping from drug_targets_symbols.csv."""

    if not path.exists():
        return {}

    df = pd.read_csv(path)
    if "drug_name" not in df.columns or "gene_symbol" not in df.columns:
        return {}

    mapping: dict[str, list[str]] = {}
    for drug_name, sub in df.groupby(df["drug_name"].astype(str)):
        genes = sub["gene_symbol"].dropna().astype(str).unique().tolist()
        mapping[str(drug_name)] = genes
    return mapping


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
    if denom <= 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def rank_drugs(
    disease_name: str,
    G0: nx.Graph,
    disease_df: pd.DataFrame,
    drug_graphs_dir: str | Path,
    top_k: int = 10,
) -> pd.DataFrame:
    """Rank drugs by cosine similarity between stub RSV vectors."""

    drug_graphs_dir = Path(drug_graphs_dir)

    # Graceful empty-dir handling.
    if not drug_graphs_dir.exists():
        print(f"Error: drug graphs directory not found: {drug_graphs_dir}")
        return pd.DataFrame(columns=["rank", "drug_name", "cosine_score", "n_targets"])

    pkl_files = list(drug_graphs_dir.glob("*.pkl"))
    if not pkl_files:
        print(
            "Error: drug_graphs_dir has no .pkl files. Run batch simulation first (run_batch_simple.py)."
        )
        return pd.DataFrame(columns=["rank", "drug_name", "cosine_score", "n_targets"])

    _G_disease, disease_rsv, _disease_genes = get_disease_signature(disease_name, G0, disease_df)

    # Load targets lookup once (per spec).
    here = Path(__file__).resolve().parent
    targets_csv = here / "drug_targets_symbols.csv"
    targets_lookup = _load_targets_table(targets_csv)

    # Also load the small-molecule datasets once for fallback target resolution.
    drug_df = None
    action_df = None
    try:
        drug_df = _read_csv_best_effort(here, "drug_50_affinity.csv", require_cols=["drug_name", "gene_symbol"])
        # Prefer a version with an 'action' column if available.
        action_df = _read_csv_best_effort(
            here,
            "drug_targets_small_molecule.csv",
            require_cols=["drug_name", "gene_symbol", "action"],
        )
    except Exception:
        drug_df = None
        action_df = None

    graph_nodes = set(G0.nodes())

    drug_names = list_completed_drugs(drug_graphs_dir)
    if not drug_names:
        # Fallback: infer from *.pkl stems.
        drug_names = [p.stem.replace("_", " ") for p in pkl_files]

    iterator = drug_names
    if tqdm is not None:
        iterator = tqdm(drug_names, desc="Ranking drugs")

    rows: list[dict] = []
    for drug_name in iterator:
        pkl_path = drug_graphs_dir / f"{_sanitize_drug_name(drug_name)}.pkl"
        if not pkl_path.exists():
            # Skip missing output.
            continue

        G_drug = load_graph(pkl_path)

        targets = targets_lookup.get(str(drug_name), [])
        if not targets and drug_df is not None:
            # Fallback to Week-2 resolver so the demo still works for small molecules.
            targets, _actions, _aff = resolve_targets(str(drug_name), drug_df, graph_nodes, action_df)

        drug_rsv = extract_rsv_stub(G0, G_drug, targets)
        score = _cosine(drug_rsv, disease_rsv)

        rows.append(
            {
                "drug_name": str(drug_name),
                "cosine_score": float(score),
                "n_targets": int(len(targets)),
            }
        )

    if not rows:
        return pd.DataFrame(columns=["rank", "drug_name", "cosine_score", "n_targets"])

    df = pd.DataFrame(rows).sort_values("cosine_score", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))

    return df[["rank", "drug_name", "cosine_score", "n_targets"]].head(int(top_k))


def run_terminal_demo(
    disease_name: str,
    G0: nx.Graph,
    disease_df: pd.DataFrame,
    drug_df: pd.DataFrame,
    drug_graphs_dir: str | Path,
    top_k: int = 5,
) -> None:
    """Run the live terminal demo table."""

    _ = drug_df  # kept for future compatibility with the demo interface

    ranked = rank_drugs(disease_name, G0, disease_df, drug_graphs_dir, top_k=top_k)
    if ranked.empty:
        print("No ranking results (missing graphs or no drugs found).")
        return

    print(f"\n=== REWIRE: Top drugs for [{disease_name}] ===")

    # Manual formatting (no tabulate dependency).
    header = f"{'Rank':<5} {'Drug Name':<22} {'Score':<8} {'Targets':<7}"
    print(header)

    for _, row in ranked.iterrows():
        r = int(row["rank"])
        dn = str(row["drug_name"])[:22]
        sc = float(row["cosine_score"])
        nt = int(row["n_targets"])
        print(f"{r:<5} {dn:<22} {sc:<8.3f} {nt:<7}")


def main(argv: list[str]) -> int:
    disease_name = argv[1] if len(argv) > 1 else "Parkinson"
    drug_graphs_dir = argv[2] if len(argv) > 2 else "drug_graphs/"

    here = Path(__file__).resolve().parent

    # Graceful empty-dir handling.
    dg = Path(drug_graphs_dir)
    if not dg.exists() or not list(dg.glob("*.pkl")):
        print(
            f"Error: '{drug_graphs_dir}' has no .pkl files. Run batch simulation first (run_batch_simple.py)."
        )
        return 1

    from ppi_graph import build_graph

    G0 = build_graph(str(_resolve_csv_path(here, "ppi_genes.csv")))
    if G0 is None:
        print("Error: failed to load ppi_genes.csv")
        return 1

    disease_df = _read_csv_best_effort(here, "disease_genes.csv", require_cols=["disease", "gene"])
    drug_df = _read_csv_best_effort(here, "drug_50_affinity.csv", require_cols=["drug_name", "gene_symbol"])

    run_terminal_demo(disease_name, G0, disease_df, drug_df, drug_graphs_dir, top_k=5)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
