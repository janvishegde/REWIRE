"""
clean_drug_names.py  —  Person 1 / REWIRE pipeline
====================================================
Problem
-------
valid_drug_targets.csv was parsed directly from DrugBank XML which stores
ALL ligands — including crystallography fragments, research tool compounds,
IUPAC-named metabolites, and experimental code-names — as "drugs".

This script removes those and keeps only rows where drug_name looks like
a real drug (approved or in clinical development), then renames known
aliases to their standard INN/brand names.

What gets removed
-----------------
1. IUPAC/chemical systematic names
   → start with '(' e.g. "(4R,5R)-1,2-dithiane..."
   → start with a digit e.g. "1-Acetylindole"
   → contain Greek letters α β γ δ (crystallography ligands)
   → contain CoA, myo-, phospho-, deoxy-, -yl, -ane, -ene (chemistry suffixes)
   → contain chemical bracket patterns [, {
   → names longer than 55 characters (almost always IUPAC)

2. Plain chemical class words used as names
   → ends with " acid", " amine", " anol", " oate", " group", " ester"
   → fragment labels like "... group", "... moiety"

What is KEPT
------------
- Approved INN names: Imatinib, Metformin, Atorvastatin, Cisplatin ...
- Brand names where used: Gleevec, Herceptin ...
- Clinical code names: ABT-737, AMG-900 etc. (flag optionally — see REMOVE_CODES)
- Known aliases renamed to INN: Acetylsalicylic acid → Aspirin

Outputs
-------
data/processed/valid_drug_targets_clean.csv
    drug_name   : cleaned INN/approved name
    gene_symbol : HGNC gene symbol (unchanged)

data/processed/cleaning_report.txt
    Summary of what was removed and why
"""

import re
import pandas as pd
from pathlib import Path
from datetime import datetime

# ── Paths ─────────────────────────────────────────────────────────────────────
INPUT_PATH  = "data/processed/valid_drug_targets.csv"
OUTPUT_PATH = "data/processed/valid_drug_targets_clean.csv"
REPORT_PATH = "data/processed/cleaning_report.txt"

# ── Config ────────────────────────────────────────────────────────────────────
# Set True to also remove experimental code-names like ABT-737, AMG-208.
# Set False to keep them (they are valid for RSV if they have PPI targets).
REMOVE_CODES = False

# ── Known alias → INN rename map ──────────────────────────────────────────────
# Add more here if you discover other aliases in the data.
ALIAS_MAP = {
    "Acetylsalicylic acid":             "Aspirin",
    "Acetylsalicylate":                 "Aspirin",
    "Paracetamol":                      "Acetaminophen",
    "Adrenaline":                       "Epinephrine",
    "Noradrenaline":                    "Norepinephrine",
    "Thyroxine":                        "Levothyroxine",
    "Folic acid":                       "Folate",
    "Vitamin D3":                       "Cholecalciferol",
    "Vitamin C":                        "Ascorbic acid",
    "Retinol":                          "Vitamin A",
    "Hydrocortisone":                   "Cortisol",
    "Niacin":                           "Nicotinic acid",
    "Riboflavin":                       "Vitamin B2",
    "Cyanocobalamin":                   "Vitamin B12",
    "Pyridoxine":                       "Vitamin B6",
    "Thiamine":                         "Vitamin B1",
    "Mecobalamin":                      "Methylcobalamin",
    "Immunoglobulin G":                 "IgG",
    "Dextrose":                         "Glucose",
    "Ferrous sulfate":                  "Iron sulfate",
    "L-Dopa":                           "Levodopa",
    "L-DOPA":                           "Levodopa",
    "(R)-Atenolol":                     "Atenolol",
    "(S)-Atenolol":                     "Atenolol",
    "(S)-Metoprolol":                   "Metoprolol",
    "(R,S)-Metoprolol":                 "Metoprolol",
    "dl-Propranolol":                   "Propranolol",
    "(S)-Propranolol":                  "Propranolol",
    "Acetaminophen":                    "Acetaminophen",  # already correct
    "(6R)-Folinic acid":                "Leucovorin",
    "5-Fluorouracil":                   "Fluorouracil",
    "all-trans retinoic acid":          "Tretinoin",
    "All-trans retinoic acid":          "Tretinoin",
    "9-cis Retinoic Acid":              "Alitretinoin",
    "13-cis Retinoic Acid":             "Isotretinoin",
    "Methylprednisolone acetate":       "Methylprednisolone",
    "Prednisolone acetate":             "Prednisolone",
    "Dexamethasone acetate":            "Dexamethasone",
    "Testosterone propionate":          "Testosterone",
    "Estradiol valerate":               "Estradiol",
    "Morphine sulfate":                 "Morphine",
    "Codeine phosphate":                "Codeine",
    "Ampicillin sodium":                "Ampicillin",
    "Penicillin G":                     "Benzylpenicillin",
    "Epinephrine bitartrate":           "Epinephrine",
    "Carbidopa-Levodopa":               "Levodopa",
    "Amoxicillin trihydrate":           "Amoxicillin",
}


# ── IUPAC / chemical name detector ───────────────────────────────────────────
# Each pattern is annotated with what it catches.
DIRTY_PATTERNS = [
    (r"^\(",                            "starts with parenthesis"),
    (r"^\d",                            "starts with digit"),
    (r"[α-ωΑ-Ω]",                       "contains Greek letter"),
    (r"\bCoA\b",                        "contains CoA (coenzyme fragment)"),
    (r"\bmyo-",                         "contains myo- (inositol chemistry)"),
    (r"\bphospho",                      "contains phospho-"),
    (r"\b(deoxy|dideoxy)\b",            "contains deoxy"),
    (r"\[|\{",                          "contains chemical brackets"),
    (r"-\d+\(",                         "contains stereo numbering like -1("),
    (r"\b\d{4,}\b",                     "contains 4+ digit number"),
    # Chemical suffix words (word-boundary anchored to avoid false positives)
    (r"\b(acid|amine|anol|oate|ester|ylene|ylene|anone|arene|azide|"
     r"amide|aldehyde|ketone|lactam|lactone|sulfide|sulfate|sulfonate|"
     r"phosphate|phosphonate|oxide|chloride|bromide|iodide|fluoride|"
     r"moiety|fragment|group|residue|adduct|analog|analogue|"
     r"glucuronide|glucoside|glycoside)\s*$",
                                        "ends with chemical suffix"),
    # Embedded chemistry words mid-name
    (r"\b(hydroxy|methyl|ethyl|propyl|butyl|pentyl|hexyl|heptyl|octyl|"
     r"nonyl|decyl|fluoro|chloro|bromo|iodo|nitro|amino|thio|oxo|"
     r"oxy|sulfo|sulfonyl|carbonyl|carboxyl)\b",
                                        "contains substituent prefix"),
    # Name length — almost all IUPAC names are long
    (r"^.{56,}$",                       "name longer than 55 characters"),
    # Hyphenated stereo descriptors
    (r"\b[RSEZ]-\b",                    "contains stereo descriptor R/S/E/Z-"),
    (r"\b(1H|2H|3H|4H)-",              "contains ring-numbering prefix"),
    # Salt/ester forms with multiple words that include chemistry
    (r"\b(sodium|potassium|calcium|magnesium|ammonium|zinc|iron)\s+"
     r"(salt|chloride|sulfate|gluconate|acetate|phosphate|carbonate)\b",
                                        "inorganic salt descriptor"),
]

# Compile once for speed
DIRTY_COMPILED = [(re.compile(pat, re.IGNORECASE), reason)
                  for pat, reason in DIRTY_PATTERNS]

# Experimental code-name pattern: 2-5 uppercase letters + hyphen + digits
CODE_PATTERN = re.compile(r"^[A-Z]{1,6}[-\s]\d{2,}", re.IGNORECASE)


def classify_name(name: str) -> tuple[bool, str]:
    """
    Return (is_dirty, reason).
    is_dirty=True means the row should be removed.
    """
    name = str(name).strip()

    # 1. Try alias map first — if it has an alias it's a real drug
    if name in ALIAS_MAP:
        return False, "alias (will be renamed)"

    # 2. Check dirty patterns
    for pattern, reason in DIRTY_COMPILED:
        if pattern.search(name):
            return True, reason

    # 3. Experimental code names (optional removal)
    if REMOVE_CODES and CODE_PATTERN.match(name):
        return True, "experimental code-name (ABT-/AMG- style)"

    return False, ""


# ── Main ──────────────────────────────────────────────────────────────────────
def clean(input_path: str, output_path: str, report_path: str) -> pd.DataFrame:

    df = pd.read_csv(input_path)
    df.columns = df.columns.str.strip().str.lower()

    # Accept gene_symbol or target_gene
    if "target_gene" in df.columns and "gene_symbol" not in df.columns:
        df.rename(columns={"target_gene": "gene_symbol"}, inplace=True)

    original_rows  = len(df)
    original_drugs = df["drug_name"].nunique()

    # ── Step 1: Apply alias renaming BEFORE classification ────────────────────
    df["drug_name"] = df["drug_name"].apply(
        lambda n: ALIAS_MAP.get(str(n).strip(), str(n).strip())
    )

    # ── Step 2: Classify each unique drug name ────────────────────────────────
    unique_names   = df["drug_name"].unique()
    classification = {}  # name → (is_dirty, reason)
    for name in unique_names:
        classification[name] = classify_name(name)

    df["_dirty"]  = df["drug_name"].map(lambda n: classification[n][0])
    df["_reason"] = df["drug_name"].map(lambda n: classification[n][1])

    # ── Step 3: Split clean vs removed ────────────────────────────────────────
    clean_df   = df[~df["_dirty"]].copy()
    removed_df = df[df["_dirty"]].copy()

    clean_df = clean_df[
    ["drug_name", "gene_symbol", "action"]].drop_duplicates(
    subset=["drug_name", "gene_symbol", "action"]).reset_index(drop=True)
    clean_df = clean_df.sort_values(["drug_name", "gene_symbol"]).reset_index(drop=True)

    # ── Step 4: Save output ───────────────────────────────────────────────────
    clean_df.to_csv(output_path, index=False)

    # ── Step 5: Build report ──────────────────────────────────────────────────
    removed_by_reason = (
        removed_df.groupby("_reason")["drug_name"]
        .nunique()
        .sort_values(ascending=False)
    )

    aliased = [n for n in ALIAS_MAP if n in df["drug_name"].values or
               ALIAS_MAP[n] in df["drug_name"].values]

    lines = [
        f"REWIRE — Drug Name Cleaning Report",
        f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Input     : {input_path}",
        f"Output    : {output_path}",
        "",
        "=" * 60,
        "SUMMARY",
        "=" * 60,
        f"Original rows        : {original_rows:>7,}",
        f"Original unique drugs: {original_drugs:>7,}",
        f"Removed rows         : {len(removed_df):>7,}",
        f"Removed unique drugs : {removed_df['drug_name'].nunique():>7,}",
        f"Clean rows           : {len(clean_df):>7,}",
        f"Clean unique drugs   : {clean_df['drug_name'].nunique():>7,}",
        f"Aliases renamed      : {len(aliased):>7,}",
        "",
        "=" * 60,
        "REMOVED — by reason",
        "=" * 60,
    ]
    for reason, count in removed_by_reason.items():
        lines.append(f"  {count:>5,}  {reason}")

    lines += [
        "",
        "=" * 60,
        "REMOVED DRUG NAMES (first 100)",
        "=" * 60,
    ]
    for name in sorted(removed_df["drug_name"].unique())[:100]:
        lines.append(f"  {name}")

    lines += [
        "",
        "=" * 60,
        "ALIASES APPLIED",
        "=" * 60,
    ]
    for old, new in ALIAS_MAP.items():
        if old != new:
            lines.append(f"  {old!r:45s} → {new!r}")

    report_text = "\n".join(lines)

    Path(report_path).write_text(report_text, encoding="utf-8")

    # ── Step 6: Print summary to terminal ─────────────────────────────────────
    print("REWIRE — Drug Name Cleaning")
    print("=" * 55)
    print(f"  Input rows           : {original_rows:,}")
    print(f"  Rows removed         : {len(removed_df):,}  ({len(removed_df)/original_rows*100:.1f}%)")
    print(f"  Rows kept            : {len(clean_df):,}")
    print(f"  Unique drugs (clean) : {clean_df['drug_name'].nunique():,}")
    print(f"  Aliases renamed      : {len(aliased)}")
    print()
    print("Removed by reason:")
    for reason, count in removed_by_reason.items():
        print(f"  {count:>5,}  {reason}")
    print()

    # Must-have check
    must_have = {
        "Imatinib","Nilotinib","Gefitinib","Erlotinib","Metformin",
        "Atorvastatin","Aspirin","Cisplatin","Bortezomib","Tamoxifen",
        "Sorafenib","Dasatinib","Celecoxib","Donepezil","Fluoxetine",
        "Dexamethasone","Methotrexate","Cyclophosphamide","Letrozole","Etoposide",
    }
    found_drugs = set(clean_df["drug_name"].unique())
    print("Must-have drug check:")
    for drug in sorted(must_have):
        print(f"  {'✅' if drug in found_drugs else '❌'}  {drug}")

    print(f"\n  Saved → {output_path}")
    print(f"  Report → {report_path}")

    return clean_df


if __name__ == "__main__":
    result = clean(INPUT_PATH, OUTPUT_PATH, REPORT_PATH)
    print(f"\nSample output (first 20 rows):")
    print(result.head(20).to_string(index=False))