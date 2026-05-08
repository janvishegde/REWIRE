# Person‑2 (Graph Builder) — May‑7 Handoff Contract

This doc is the **best‑case** “Person‑2 is done” checklist so Person‑3 can compute RSV smoothly.

## What Person‑3 gets (guarantees)

After you run the batch, `drug_graphs/` contains:

- `*.pkl` — one pickled NetworkX graph per drug (drug‑perturbed graph)
- `completed.txt` — checkpoint list of drug names successfully generated
- `failed_drugs.txt` — drugs that had no resolvable targets or crashed
- `drug_ids.txt` — deterministic full drug list (stable order)
- `drug_ids_completed.txt` — deterministic completed list (subset of `drug_ids.txt`)
- `manifest.csv` / `manifest.json` — per‑drug status + counts + file names

Each pickled graph has embedded metadata in `G.graph['rewire']`:

- `drug_name`
- `targets` (resolved, in‑graph)
- `actions` (per target)
- `affinities_nM` (per target, may be `None`)
- `failed_targets` (targets not found in the PPI graph)
- `ppi.nodes`, `ppi.edges`

Person‑3 can compute RSV directly from `(G0, G_drug)` and read `targets` from metadata.

## Commands (Person‑2)

From `src/P2/` (PowerShell):

- Batch generate (resume‑safe):
  - `python run_batch_simple.py`

- Verify outputs before handoff:
  - `python verify_drug_graphs.py ppi_genes.csv drug_graphs`

- Quick demo sanity (edge weight changes exist):
  - `python tests/test_ppi_graph.py ppi_genes.csv`

## Sprint deliverables (Person‑2) → where it is satisfied

- Real PPI loads + counts printed:
  - `ppi_graph.build_graph()` logs node/edge counts on load.

- Deep copy protection:
  - `ppi_graph._validate_deep_copy()` asserts baseline is unchanged.

- Batch simulation for all 50 drugs + checkpoint:
  - `run_batch_simple.py` writes `drug_graphs/*.pkl` and `completed.txt`.

- Handles missing targets (skip + warn):
  - `run_batch_simple.py` logs a warning and records status in `manifest.*`.

- Person‑3 unblocked (stable interface + deterministic order):
  - `drug_ids_completed.txt` + embedded `G.graph['rewire']` metadata.

## 10‑line verbal summary (what to say on May‑7)

1. We load the STRING PPI network as a weighted undirected graph $G_0$.
2. STRING weights are normalized to $[0,1]$.
3. For each drug, we resolve its target genes to nodes in $G_0$.
4. We create a deep copy $G_{drug}$ so $G_0$ is never mutated.
5. For each target, we perturb incident edge weights.
6. Inhibitors attenuate edges; activators strengthen edges.
7. Perturbation strength is proportional to binding affinity (nM).
8. Targets missing from the graph are skipped and logged.
9. We precompute and save 50 drug graphs as `.pkl` for fast downstream RSV.
10. Person‑3 computes RSV by comparing $G_0$ vs $G_{drug}$ for each drug.
