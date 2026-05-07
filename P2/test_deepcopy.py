"""Deep-copy safety check for ppi_graph.py."""

from __future__ import annotations

from ppi_graph import build_graph, simulate_binding


def main() -> None:
    G0 = build_graph("fake_ppi.csv")
    targets = ["P53", "EGFR"]
    affinities = {"P53": 0.7, "EGFR": 0.4}
    Gdrug = simulate_binding(G0, targets, affinities)

    if G0 is Gdrug:
        print("FAIL: simulate_binding returned the same graph object")
        return

    # Pick an edge affected by P53 when available.
    affected_neighbor = None
    if "P53" in G0:
        for nbr in G0.neighbors("P53"):
            affected_neighbor = nbr
            break

    if affected_neighbor is None:
        print("FAIL: could not find an edge incident to P53 for validation")
        return

    base_before = G0["P53"][affected_neighbor]["weight"]

    # Mutate Gdrug directly and verify G0 remains unchanged.
    Gdrug["P53"][affected_neighbor]["weight"] = 0.001
    base_after = G0["P53"][affected_neighbor]["weight"]

    if abs(base_before - base_after) < 1e-12:
        print("PASS: G0 is unchanged after mutating Gdrug (deep copy safe)")
    else:
        print("FAIL: G0 changed when Gdrug was mutated")


if __name__ == "__main__":
    main()
