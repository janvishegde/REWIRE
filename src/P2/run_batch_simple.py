"""Batch-run graph simulations for all drugs.

Reads:
- ppi_genes.csv
- drug_50_affinity.csv
- drug_targets_small_molecule.csv

Writes:
- drug_graphs/{sanitized_drug_name}.pkl
- drug_graphs/completed.txt
- drug_graphs/failed_drugs.txt

Behavior:
- Loads baseline graph (G0) once.
- Resolves targets for each drug with real affinities + actions.
- Simulates binding and pickles the resulting graph.
- Checkpoints progress so reruns resume where they left off.
- Never crashes the batch on per-drug failures.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import logging
import os
import pickle
import random
from typing import Iterable

import pandas as pd

from data_resolver import resolve_targets
from ppi_graph import build_graph, simulate_binding


@dataclass
class ManifestRow:
    drug_name: str
    pkl_file: str | None
    status: str  # ok | failed | skipped
    n_targets: int
    n_failed_targets: int
    nodes: int
    edges: int
    error: str | None = None


def _try_get_tqdm():
    try:
        from tqdm import tqdm  # type: ignore

        return tqdm
    except Exception:
        return None


def _resolve_input_path(here: str, filename: str) -> str:
    """Resolve an input CSV path.

    Spec says inputs live next to this script; the repo also has
    `data/processed/` for upgraded CSVs, so we fall back to that.
    """

    direct = os.path.join(here, filename)
    if os.path.exists(direct):
        return direct

    processed = os.path.normpath(os.path.join(here, "..", "..", "data", "processed", filename))
    if os.path.exists(processed):
        return processed

    return direct


def _candidate_csv_paths(here: str, filename: str) -> list[str]:
    direct = os.path.join(here, filename)
    processed = os.path.normpath(os.path.join(here, "..", "..", "data", "processed", filename))
    out: list[str] = []
    for p in (direct, processed):
        if p not in out:
            out.append(p)
    return out


def _read_csv_best_effort(here: str, filename: str, require_cols: list[str] | None = None) -> pd.DataFrame:
    """Read a CSV from either the script dir or data/processed.

    If a candidate exists but lacks required columns, try the next candidate.
    """

    last_error: Exception | None = None
    for path in _candidate_csv_paths(here, filename):
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path)
            if require_cols and not set(require_cols).issubset(df.columns):
                continue
            return df
        except Exception as exc:
            last_error = exc
            continue
    if last_error is not None:
        raise last_error
    # Final fallback: preserve prior behavior (will raise FileNotFoundError)
    return pd.read_csv(os.path.join(here, filename))


def sanitize_drug_name(name: str) -> str:
    """Lowercase, spaces→underscores, remove slashes."""

    s = str(name).strip().lower()
    s = s.replace(" ", "_")
    s = s.replace("/", "")
    s = s.replace("\\", "")
    # keep it filesystem-friendly without inventing extra rules
    return s


def _read_lines_set(path: str) -> set[str]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return {line.strip().lower() for line in handle if line.strip()}
    except FileNotFoundError:
        return set()
    except OSError:
        return set()


def _append_line(path: str, line: str) -> None:
    try:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(line.rstrip("\n") + "\n")
    except OSError:
        return


def _iter_drugs(drug_df: pd.DataFrame) -> list[str]:
    if "drug_name" not in drug_df.columns:
        return []
    return sorted(drug_df["drug_name"].astype(str).dropna().unique().tolist())


def _progress_iter(items: Iterable[str], desc: str):
    tqdm = _try_get_tqdm()
    if tqdm is None:
        logging.warning("tqdm not installed; running without progress bar")
        return items
    return tqdm(list(items), desc=desc)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    here = os.path.dirname(os.path.abspath(__file__))

    ppi_csv = _resolve_input_path(here, "ppi_genes.csv")

    output_dir = os.path.join(here, "drug_graphs")
    os.makedirs(output_dir, exist_ok=True)

    completed_path = os.path.join(output_dir, "completed.txt")
    failed_path = os.path.join(output_dir, "failed_drugs.txt")
    manifest_csv_path = os.path.join(output_dir, "manifest.csv")
    manifest_json_path = os.path.join(output_dir, "manifest.json")
    drug_ids_path = os.path.join(output_dir, "drug_ids.txt")
    completed_ids_path = os.path.join(output_dir, "drug_ids_completed.txt")

    # 1) Load baseline graph ONCE.
    G0 = build_graph(ppi_csv)
    if G0 is None:
        raise SystemExit(f"Failed to build graph from: {ppi_csv}")

    graph_nodes = set(G0.nodes())

    # 2) Load drug affinity + action tables.
    drug_df = _read_csv_best_effort(here, "drug_50_affinity.csv", require_cols=["drug_name", "gene_symbol"])

    # Action table is optional.
    # If missing (or doesn't include an `action` column), we default all actions to inhibitor.
    action_df = None
    try:
        action_df = _read_csv_best_effort(
            here,
            "drug_targets_small_molecule.csv",
            require_cols=["drug_name", "gene_symbol", "action"],
        )
    except Exception:
        action_df = None
        logging.warning(
            "Optional action table not found (or missing required columns). "
            "Proceeding with default actions='inhibitor' for all targets."
        )

    drugs = _iter_drugs(drug_df)
    if not drugs:
        raise SystemExit("No drugs found in drug_50_affinity.csv (missing drug_name?)")

    # Deterministic drug order for Person-3 RSV computation.
    try:
        with open(drug_ids_path, "w", encoding="utf-8") as handle:
            for dn in drugs:
                handle.write(str(dn).strip() + "\n")
    except OSError:
        logging.warning("Could not write drug_ids.txt")

    completed = _read_lines_set(completed_path)

    succeeded = 0
    failed = 0
    manifest_rows: list[ManifestRow] = []

    for drug_name in _progress_iter(drugs, desc="Simulating drugs"):
        if str(drug_name).strip().lower() in completed:
            continue

        try:
            targets, actions, affinities = resolve_targets(drug_name, drug_df, graph_nodes, action_df)

            if not targets:
                logging.warning("No resolvable targets for drug '%s' (skipping)", drug_name)
                _append_line(failed_path, str(drug_name))
                manifest_rows.append(
                    ManifestRow(
                        drug_name=str(drug_name),
                        pkl_file=None,
                        status="skipped",
                        n_targets=0,
                        n_failed_targets=0,
                        nodes=int(G0.number_of_nodes()),
                        edges=int(G0.number_of_edges()),
                        error="no_resolvable_targets",
                    )
                )
                failed += 1
                continue

            G_drug, _failed_targets = simulate_binding(
                G0,
                targets,
                affinities=affinities,
                actions=actions,
            )

            # Embed metadata so downstream RSV code doesn't need to re-resolve.
            try:
                G_drug.graph["rewire"] = {
                    "drug_name": str(drug_name),
                    "targets": list(targets),
                    "actions": {str(k): str(v) for k, v in (actions or {}).items()},
                    "affinities_nM": {
                        str(k): (None if v is None else float(v)) for k, v in (affinities or {}).items()
                    },
                    "failed_targets": list(_failed_targets or []),
                    "ppi": {
                        "nodes": int(G0.number_of_nodes()),
                        "edges": int(G0.number_of_edges()),
                    },
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                }
            except Exception:
                # Protector: metadata must never break the batch.
                pass

            out_name = sanitize_drug_name(drug_name) + ".pkl"
            out_path = os.path.join(output_dir, out_name)
            with open(out_path, "wb") as handle:
                pickle.dump(G_drug, handle, protocol=pickle.HIGHEST_PROTOCOL)

            manifest_rows.append(
                ManifestRow(
                    drug_name=str(drug_name),
                    pkl_file=out_name,
                    status="ok",
                    n_targets=int(len(targets)),
                    n_failed_targets=int(len(_failed_targets or [])),
                    nodes=int(G_drug.number_of_nodes()),
                    edges=int(G_drug.number_of_edges()),
                )
            )

            _append_line(completed_path, str(drug_name))
            completed.add(str(drug_name).strip().lower())
            succeeded += 1

        except Exception as exc:
            _append_line(failed_path, str(drug_name))
            logging.exception("Exception while processing drug '%s': %s", drug_name, exc)
            manifest_rows.append(
                ManifestRow(
                    drug_name=str(drug_name),
                    pkl_file=None,
                    status="failed",
                    n_targets=0,
                    n_failed_targets=0,
                    nodes=int(G0.number_of_nodes()),
                    edges=int(G0.number_of_edges()),
                    error=repr(exc),
                )
            )
            failed += 1
            continue

    print(f"Batch complete: {succeeded} succeeded, {failed} failed")
    try:
        total_pkls = len([p for p in os.listdir(output_dir) if p.endswith(".pkl")])
    except OSError:
        total_pkls = 0
    try:
        total_completed = len(_read_lines_set(completed_path))
    except Exception:
        total_completed = 0
    print(f"Output: drug_graphs/ ({total_pkls} .pkl files, {total_completed} completed)")
    print(f"Failed drugs logged to: drug_graphs/failed_drugs.txt")

    # 3) Sanity check (automatic)
    _sanity_check(G0, graph_nodes, drug_df, action_df, output_dir, completed_path)

    # 3.1) Backfill metadata into already-generated graphs (important on resume/no-op runs).
    _ensure_metadata_for_completed(G0, graph_nodes, drug_df, action_df, output_dir, completed_path)

    # 4) Write manifest + deterministic completed-drug list for Person-3.
    _write_manifest_and_completed_lists(
        manifest_rows,
        manifest_csv_path,
        manifest_json_path,
        drug_ids_path,
        completed_path,
        completed_ids_path,
    )

    return 0


def _write_manifest_and_completed_lists(
    manifest_rows: list[ManifestRow],
    manifest_csv_path: str,
    manifest_json_path: str,
    drug_ids_path: str,
    completed_path: str,
    completed_ids_path: str,
) -> None:
    # If this run was a no-op resume, rebuild a full manifest from disk.
    rows = list(manifest_rows)
    if not rows:
        rows = _rebuild_manifest_from_outputs(drug_ids_path, completed_path, os.path.dirname(manifest_csv_path))

    # Manifest CSV (overwrite each run).
    try:
        df = pd.DataFrame([asdict(r) for r in rows])
        df.to_csv(manifest_csv_path, index=False)
    except Exception:
        logging.warning("Could not write manifest.csv")

    # Manifest JSON for easier inspection.
    try:
        with open(manifest_json_path, "w", encoding="utf-8") as handle:
            json.dump([asdict(r) for r in rows], handle, indent=2)
    except Exception:
        logging.warning("Could not write manifest.json")

    # Completed-drug list in deterministic order: filter drug_ids.txt by completed.txt.
    try:
        wanted = []
        all_ids = []
        if os.path.exists(drug_ids_path):
            with open(drug_ids_path, "r", encoding="utf-8") as handle:
                all_ids = [ln.strip() for ln in handle if ln.strip()]
        completed = set()
        if os.path.exists(completed_path):
            with open(completed_path, "r", encoding="utf-8") as handle:
                completed = {ln.strip().lower() for ln in handle if ln.strip()}
        for dn in all_ids:
            if dn.strip().lower() in completed:
                wanted.append(dn)
        with open(completed_ids_path, "w", encoding="utf-8") as handle:
            for dn in wanted:
                handle.write(dn + "\n")
    except Exception:
        logging.warning("Could not write drug_ids_completed.txt")


def _ensure_metadata_for_completed(
    G0,
    graph_nodes: set[str],
    drug_df: pd.DataFrame,
    action_df: pd.DataFrame,
    output_dir: str,
    completed_path: str,
) -> None:
    """Ensure every completed .pkl contains G.graph['rewire'] metadata.

    This is crucial for Person-3: they can compute RSV without re-resolving
    targets from CSVs.
    """

    completed_names = []
    try:
        with open(completed_path, "r", encoding="utf-8") as handle:
            completed_names = [ln.strip() for ln in handle if ln.strip()]
    except OSError:
        return

    for drug_name in completed_names:
        pkl_path = os.path.join(output_dir, sanitize_drug_name(drug_name) + ".pkl")
        if not os.path.exists(pkl_path):
            continue

        try:
            with open(pkl_path, "rb") as handle:
                G_loaded = pickle.load(handle)
        except Exception:
            continue

        meta = None
        try:
            meta = getattr(G_loaded, "graph", {}).get("rewire")
        except Exception:
            meta = None

        # If metadata exists and looks complete, leave it.
        if isinstance(meta, dict) and meta.get("drug_name") and isinstance(meta.get("targets"), list):
            continue

        try:
            targets, actions, affinities = resolve_targets(drug_name, drug_df, graph_nodes, action_df)
            G_loaded.graph["rewire"] = {
                "drug_name": str(drug_name),
                "targets": list(targets),
                "actions": {str(k): str(v) for k, v in (actions or {}).items()},
                "affinities_nM": {
                    str(k): (None if v is None else float(v)) for k, v in (affinities or {}).items()
                },
                "failed_targets": [],
                "ppi": {
                    "nodes": int(G0.number_of_nodes()),
                    "edges": int(G0.number_of_edges()),
                },
                "created_utc": datetime.now(timezone.utc).isoformat(),
            }
            with open(pkl_path, "wb") as handle:
                pickle.dump(G_loaded, handle, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            continue


def _rebuild_manifest_from_outputs(drug_ids_path: str, completed_path: str, output_dir: str) -> list[ManifestRow]:
    """Rebuild a manifest by scanning existing outputs.

    This supports resume/no-op batch runs while still producing a full handoff
    artifact set.
    """

    # Load intended drug list
    all_drugs: list[str] = []
    try:
        with open(drug_ids_path, "r", encoding="utf-8") as handle:
            all_drugs = [ln.strip() for ln in handle if ln.strip()]
    except OSError:
        all_drugs = []

    completed = set()
    try:
        with open(completed_path, "r", encoding="utf-8") as handle:
            completed = {ln.strip().lower() for ln in handle if ln.strip()}
    except OSError:
        completed = set()

    rows: list[ManifestRow] = []
    for dn in all_drugs:
        pkl_name = sanitize_drug_name(dn) + ".pkl"
        pkl_path = os.path.join(output_dir, pkl_name)

        if dn.strip().lower() not in completed:
            rows.append(
                ManifestRow(
                    drug_name=dn,
                    pkl_file=(pkl_name if os.path.exists(pkl_path) else None),
                    status="pending",
                    n_targets=0,
                    n_failed_targets=0,
                    nodes=0,
                    edges=0,
                )
            )
            continue

        if not os.path.exists(pkl_path):
            rows.append(
                ManifestRow(
                    drug_name=dn,
                    pkl_file=None,
                    status="failed",
                    n_targets=0,
                    n_failed_targets=0,
                    nodes=0,
                    edges=0,
                    error="completed_but_missing_pkl",
                )
            )
            continue

        # Best-effort read: pull counts + targets from embedded metadata.
        n_targets = 0
        nodes = 0
        edges = 0
        try:
            with open(pkl_path, "rb") as handle:
                G = pickle.load(handle)
            nodes = int(G.number_of_nodes())
            edges = int(G.number_of_edges())
            meta = getattr(G, "graph", {}).get("rewire")
            if isinstance(meta, dict) and isinstance(meta.get("targets"), list):
                n_targets = int(len(meta.get("targets") or []))
        except Exception:
            pass

        rows.append(
            ManifestRow(
                drug_name=dn,
                pkl_file=pkl_name,
                status="ok",
                n_targets=n_targets,
                n_failed_targets=0,
                nodes=nodes,
                edges=edges,
            )
        )

    return rows


def _sanity_check(
    G0,
    graph_nodes: set[str],
    drug_df: pd.DataFrame,
    action_df: pd.DataFrame,
    output_dir: str,
    completed_path: str,
) -> None:
    print("\n=== Sanity check (3 random outputs) ===")

    completed_names = []
    try:
        with open(completed_path, "r", encoding="utf-8") as handle:
            completed_names = [line.strip() for line in handle if line.strip()]
    except OSError:
        completed_names = []

    # Prefer sampling based on completed.txt so we can recover the original drug name.
    candidates = []
    for name in completed_names:
        pkl_path = os.path.join(output_dir, sanitize_drug_name(name) + ".pkl")
        if os.path.exists(pkl_path):
            candidates.append((name, pkl_path))

    if not candidates:
        print("No completed .pkl outputs found for sanity check.")
        return

    sample = random.sample(candidates, k=min(3, len(candidates)))

    for drug_name, pkl_path in sample:
        try:
            with open(pkl_path, "rb") as handle:
                G_loaded = pickle.load(handle)

            assert G_loaded.number_of_nodes() == G0.number_of_nodes(), "node count mismatch"
            assert G_loaded.number_of_edges() == G0.number_of_edges(), "edge count mismatch"

            targets, actions, affinities = resolve_targets(drug_name, drug_df, graph_nodes, action_df)
            if not targets:
                print(f"{drug_name}: OK counts, but no targets to inspect")
                continue

            t = targets[0]
            nbrs = list(G0.neighbors(t)) if t in G0 else []
            if not nbrs:
                print(f"{drug_name}: OK counts, but target {t} has no neighbors")
                continue

            nbr = nbrs[0]
            before = G0[t][nbr]["weight"]
            after = G_loaded[t][nbr]["weight"]
            print(f"{drug_name}: {t} -> {nbr}: {before:.4f} -> {after:.4f}")

        except Exception as exc:
            print(f"{drug_name}: Sanity check FAILED for {os.path.basename(pkl_path)} ({exc})")


if __name__ == "__main__":
    raise SystemExit(main())
