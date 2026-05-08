"""Verify that batch-generated drug graphs are consistent and RSV-ready.

This is a Person-2 deliverable to unblock Person-3:
- Confirms graph integrity (node/edge counts match baseline)
- Confirms edge weights are within [0.0, 1.0] and not negative
- Confirms each pickled graph has embedded metadata under G.graph['rewire']
- Prints a compact summary and exits non-zero on failure

Usage:
  python verify_drug_graphs.py ppi_genes.csv drug_graphs

Defaults:
  ppi_genes.csv, drug_graphs/
"""

from __future__ import annotations

from pathlib import Path
import pickle
import sys

import networkx as nx

from ppi_graph import build_graph


def _iter_edge_weights(G: nx.Graph):
    for _u, _v, data in G.edges(data=True):
        w = data.get("weight")
        if w is None:
            continue
        try:
            yield float(w)
        except Exception:
            continue


def main(argv: list[str]) -> int:
    ppi_csv = Path(argv[1]) if len(argv) > 1 else Path("ppi_genes.csv")
    out_dir = Path(argv[2]) if len(argv) > 2 else Path("drug_graphs")

    G0 = build_graph(str(ppi_csv))
    if G0 is None:
        print(f"FAIL: could not build graph from {ppi_csv}")
        return 2

    expected_nodes = int(G0.number_of_nodes())
    expected_edges = int(G0.number_of_edges())

    pkls = sorted(out_dir.glob("*.pkl"))
    if not pkls:
        print(f"FAIL: no .pkl files found in {out_dir}")
        return 2

    ok = True
    bad_counts = 0
    bad_weights = 0
    missing_meta = 0

    for pkl_path in pkls:
        try:
            with pkl_path.open("rb") as handle:
                G = pickle.load(handle)
        except Exception as exc:
            ok = False
            print(f"FAIL: could not load {pkl_path.name}: {exc}")
            continue

        if int(G.number_of_nodes()) != expected_nodes or int(G.number_of_edges()) != expected_edges:
            ok = False
            bad_counts += 1

        # Weight sanity: allow very small positive clamp (>= 0.0), max <= 1.0
        w_min = None
        w_max = None
        for w in _iter_edge_weights(G):
            w_min = w if w_min is None else min(w_min, w)
            w_max = w if w_max is None else max(w_max, w)
        if w_min is None or w_max is None:
            ok = False
            bad_weights += 1
        else:
            if w_min < 0.0 or w_max > 1.0:
                ok = False
                bad_weights += 1

        meta = getattr(G, "graph", {}).get("rewire") if hasattr(G, "graph") else None
        if not isinstance(meta, dict) or "drug_name" not in meta or "targets" not in meta:
            ok = False
            missing_meta += 1

    print("=== verify_drug_graphs summary ===")
    print(f"Baseline: {expected_nodes:,} nodes, {expected_edges:,} edges")
    print(f"Outputs : {len(pkls)} pkls in {out_dir}")
    print(f"Bad counts     : {bad_counts}")
    print(f"Bad weights    : {bad_weights}")
    print(f"Missing metadata: {missing_meta}")

    if not ok:
        print("FAIL: one or more checks failed")
        return 1

    print("OK: all graphs look consistent and RSV-ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
