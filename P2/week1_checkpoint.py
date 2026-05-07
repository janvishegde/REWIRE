"""Week 1 integration checkpoint for Person 2 (Graph Builder)."""

from __future__ import annotations

from ppi_graph import build_graph, simulate_binding


def summarize_changes(G0, Gdrug, limit: int = 8):
    changed = []
    for u, v, data in G0.edges(data=True):
        before = float(data.get("weight", 1.0))
        after = float(Gdrug[u][v].get("weight", 1.0))
        if abs(before - after) > 1e-12:
            changed.append((u, v, before, after))
    changed.sort(key=lambda row: abs(row[2] - row[3]), reverse=True)
    return changed[:limit], len(changed)


def main() -> None:
    print("=== REWIRE Week 1 Checkpoint: Graph Builder ===")

    G0 = build_graph("fake_ppi.csv")
    targets = ["P53", "EGFR", "MISSING_TARGET"]
    affinities = {"P53": 0.7, "EGFR": 0.4}
    Gdrug = simulate_binding(G0, targets, affinities)

    if G0 is None or Gdrug is None:
        print("FAIL: graph objects were not created")
        return

    top_changed, changed_count = summarize_changes(G0, Gdrug)

    print(f"Base graph:   nodes={G0.number_of_nodes()} edges={G0.number_of_edges()}")
    print(f"Drug graph:   nodes={Gdrug.number_of_nodes()} edges={Gdrug.number_of_edges()}")
    print(f"Targets used: {targets}")
    print(f"Changed edges: {changed_count}")

    print("\nTop changed edges (before -> after):")
    if not top_changed:
        print("- None")
    else:
        for u, v, before, after in top_changed:
            print(f"- {u} -- {v}: {before:.3f} -> {after:.3f}")

    print("\nStatus: PASS (ran without crash, missing target handled safely)")


if __name__ == "__main__":
    main()
