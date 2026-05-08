"""
parse_drugbank.py  —  Person 1 / REWIRE pipeline
=================================================
Extracts SMALL MOLECULE drugs only from DrugBank full XML.

Key fixes over v1
-----------------
1. Reads drug `type` from the XML *attribute* (not a child tag).
   In DrugBank XML the opening tag looks like:
       <drug type="small molecule" created="..." updated="...">
   elem.attrib["type"] is how you access it with ElementTree.

2. Removed the `count >= 1000` hard-stop.
   That cap was cutting off drugs alphabetically — Imatinib,
   Nilotinib, Metformin etc. all appear past position 1000 in
   the XML and were never reached.

3. Filters to human targets only  (organism == "Humans").
   Biologics target extracellular / non-human proteins that
   would never appear in a human intracellular PPI graph.

4. Skips rows with a missing gene symbol — blank targets
   are useless for graph loading.

5. Output columns: drug_name, gene_symbol only.
   Downstream scripts (simulate_binding, RSV) only need these two.

Output
------
data/processed/drug_targets_small_molecule.csv
    drug_name   : DrugBank preferred name
    gene_symbol : HGNC gene symbol  (e.g. ABL1, EGFR, HMGCR)
"""

import xml.etree.ElementTree as ET
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────
FILE_PATH   = r"data/raw/full database.xml"
OUTPUT_PATH = "data/processed/drug_targets_small_molecule.csv"

# Drugs we must confirm are present — used for a final sanity check
MUST_HAVE = {
    "Imatinib", "Nilotinib", "Gefitinib", "Erlotinib",
    "Metformin", "Atorvastatin", "Aspirin",
}


# ── Parser ────────────────────────────────────────────────────────────────────
def parse_small_molecule_drugs(file_path: str) -> pd.DataFrame:
    """
    Stream-parse DrugBank XML and return a DataFrame of
    small-molecule drug → human gene target pairs.

    Uses iterparse (event-driven) so the full ~1 GB XML is
    never loaded into memory at once.
    """
    records      = []
    total_seen   = 0      # all <drug> elements
    total_sm     = 0      # small molecules only
    total_skip   = 0      # small molecules skipped (no valid human gene target)

    print("Parsing DrugBank XML — small molecules only ...\n")

    for event, elem in ET.iterparse(file_path, events=("end",)):

        # Only process when a full <drug> element has been closed
        if not elem.tag.endswith("drug"):
            continue

        total_seen += 1

        # ── 1. TYPE FILTER ────────────────────────────────────────────────
        # `type` is an XML attribute: <drug type="small molecule" ...>
        # elem.attrib returns a plain dict — safe to .get()
        drug_type = elem.attrib.get("type", "").strip().lower()

        if drug_type != "small molecule":
            elem.clear()
            continue

        total_sm += 1

        # ── 2. IDENTIFIERS ───────────────────────────────────────────────
        drug_id   = ""
        drug_name = ""

        for child in elem:
            tag = child.tag

            if tag.endswith("drugbank-id") and child.attrib.get("primary") == "true":
                # Prefer the primary ID  (e.g. DB00619 for Imatinib)
                drug_id = child.text or ""

            elif tag.endswith("name") and not drug_name:
                drug_name = (child.text or "").strip()

        # Fallback: grab first drugbank-id if primary flag wasn't set
        if not drug_id:
            for child in elem:
                if child.tag.endswith("drugbank-id"):
                    drug_id = child.text or ""
                    break

        # ── 3. TARGETS ───────────────────────────────────────────────────
        drug_had_valid_target = False

        for child in elem:
            if not child.tag.endswith("targets"):
                continue

            for target in child:
                gene_symbol  = ""
                organism     = ""

                for sub in target:

                    if sub.tag.endswith("organism"):
                        organism = (sub.text or "").strip()

                    elif sub.tag.endswith("polypeptide"):
                        for poly in sub:
                            if poly.tag.endswith("gene-name"):
                                gene_symbol = (poly.text or "").strip().upper()

                # ── HUMAN + NON-EMPTY GENE FILTER ────────────────────────
                if organism == "Humans" and gene_symbol:
                    records.append({
                        "drug_name":   drug_name,
                        "gene_symbol": gene_symbol,
                    })
                    drug_had_valid_target = True

        if not drug_had_valid_target:
            total_skip += 1

        # Free memory — critical for large XML files
        elem.clear()

        # Progress heartbeat every 500 small molecules
        if total_sm % 500 == 0:
            print(f"  Small molecules processed: {total_sm:,}  "
                  f"(total <drug> tags seen: {total_seen:,})")

    print(f"\nDone.\n"
          f"  Total <drug> tags        : {total_seen:,}\n"
          f"  Small molecules found    : {total_sm:,}\n"
          f"  Skipped (no human target): {total_skip:,}\n"
          f"  Target rows collected    : {len(records):,}")

    return pd.DataFrame(records)


# ── Post-processing ───────────────────────────────────────────────────────────
def clean_and_save(df: pd.DataFrame, output_path: str) -> pd.DataFrame:
    """
    Deduplicate, sort, run sanity checks, and write CSV.
    """
    # Drop exact duplicates (same drug + same gene appearing twice)
    df = df.drop_duplicates(subset=["drug_name", "gene_symbol"]).reset_index(drop=True)

    # Sort for reproducibility and readability
    df = df.sort_values(["drug_name", "gene_symbol"]).reset_index(drop=True)

    # ── Sanity: confirm must-have drugs are present ───────────────────────
    drugs_found = set(df["drug_name"].unique())
    print("\n── Must-have drug check ─────────────────────────────")
    for drug in sorted(MUST_HAVE):
        status = "✅ FOUND" if drug in drugs_found else "❌ MISSING"
        print(f"  {status}  {drug}")

    # ── Quick summary ─────────────────────────────────────────────────────
    print(f"\n── Output summary ───────────────────────────────────")
    print(f"  Unique drugs   : {df['drug_name'].nunique():,}")
    print(f"  Unique genes   : {df['gene_symbol'].nunique():,}")
    print(f"  Total rows     : {len(df):,}")

    df.to_csv(output_path, index=False)
    print(f"\n  Saved → {output_path}")

    return df


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = parse_small_molecule_drugs(FILE_PATH)
    df = clean_and_save(df, OUTPUT_PATH)

    print("\nSample output:")
    # Show a few rows for the must-have drugs if present
    sample = df[df["drug_name"].isin(MUST_HAVE)]
    print(sample.head(20).to_string(index=False))