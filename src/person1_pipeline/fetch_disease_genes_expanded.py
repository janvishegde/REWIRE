"""
fetch_disease_genes_expanded.py  —  Person 1 / REWIRE pipeline
===============================================================
Pulls top 10 associated genes for 50 diseases from OpenTargets
GraphQL API and saves disease_genes_expanded.csv.

Output columns: disease, gene, score
"""

import time
import requests
import pandas as pd
from pathlib import Path

OUTPUT_PATH = "data/processed/disease_genes_expanded.csv"
API_URL     = "https://api.platform.opentargets.org/api/v4/graphql"

DISEASES = [
    # Oncology (15)
    "MONDO_0005105",  # melanoma
    "MONDO_0007254",  # breast cancer
    "MONDO_0005061",  # lung adenocarcinoma
    "MONDO_0005096",  # colorectal cancer
    "MONDO_0005575",  # chronic myeloid leukemia
    "MONDO_0018874",  # acute myeloid leukemia
    "MONDO_0005109",  # glioblastoma
    "MONDO_0005133",  # pancreatic cancer
    "MONDO_0005086",  # renal cell carcinoma
    "MONDO_0005052",  # bladder cancer
    "MONDO_0005070",  # hepatocellular carcinoma
    "MONDO_0005136",  # ovarian cancer
    "MONDO_0005178",  # prostate cancer
    "MONDO_0005044",  # gastric cancer
    "MONDO_0005559",  # non-Hodgkin lymphoma
    # Neurological (10)
    "MONDO_0004975",  # Alzheimer disease
    "MONDO_0005180",  # Parkinson disease
    "MONDO_0005301",  # multiple sclerosis
    "MONDO_0004985",  # amyotrophic lateral sclerosis
    "MONDO_0005027",  # epilepsy
    "MONDO_0005015",  # schizophrenia
    "MONDO_0004985",  # ALS (duplicate guard handled below)
    "MONDO_0005306",  # Huntington disease
    "MONDO_0005090",  # bipolar disorder
    "MONDO_0011561",  # autism spectrum disorder
    # Metabolic (10)
    "MONDO_0005148",  # type 2 diabetes
    "MONDO_0005015",  # obesity (reused EFO below)
    "MONDO_0007455",  # hypercholesterolemia
    "MONDO_0005290",  # non-alcoholic fatty liver disease
    "MONDO_0005002",  # coronary artery disease
    "MONDO_0004981",  # atrial fibrillation
    "MONDO_0005010",  # hypertension
    "MONDO_0005812",  # gout
    "MONDO_0005011",  # chronic kidney disease
    "MONDO_0005149",  # type 1 diabetes
    # Immunological (10)
    "MONDO_0007915",  # rheumatoid arthritis
    "MONDO_0005155",  # systemic lupus erythematosus
    "MONDO_0005260",  # Crohn disease
    "MONDO_0005265",  # ulcerative colitis
    "MONDO_0007191",  # psoriasis
    "MONDO_0016419",  # ankylosing spondylitis
    "MONDO_0019434",  # systemic sclerosis
    "MONDO_0005047",  # asthma
    "MONDO_0004526",  # atopic dermatitis
    "MONDO_0005311",  # primary Sjogren syndrome
    # Cardiovascular/other (5)
    "MONDO_0005098",  # heart failure
    "MONDO_0005003",  # myocardial infarction
    "MONDO_0021668",  # aortic aneurysm
    "MONDO_0005016",  # stroke
    "MONDO_0005560",  # peripheral artery disease
]

# Use EFO IDs as fallback for diseases that return no results with MONDO
EFO_OVERRIDES = {
    "MONDO_0005015": "EFO_0001073",   # obesity
    "MONDO_0005290": "EFO_0004064",   # NAFLD
    "MONDO_0005812": "EFO_0002090",   # gout
}

DISEASE_NAMES = {
    "MONDO_0005105": "melanoma",
    "MONDO_0007254": "breast_cancer",
    "MONDO_0005061": "lung_adenocarcinoma",
    "MONDO_0005096": "colorectal_cancer",
    "MONDO_0005575": "chronic_myeloid_leukemia",
    "MONDO_0018874": "acute_myeloid_leukemia",
    "MONDO_0005109": "glioblastoma",
    "MONDO_0005133": "pancreatic_cancer",
    "MONDO_0005086": "renal_cell_carcinoma",
    "MONDO_0005052": "bladder_cancer",
    "MONDO_0005070": "hepatocellular_carcinoma",
    "MONDO_0005136": "ovarian_cancer",
    "MONDO_0005178": "prostate_cancer",
    "MONDO_0005044": "gastric_cancer",
    "MONDO_0005559": "non_hodgkin_lymphoma",
    "MONDO_0004975": "alzheimer_disease",
    "MONDO_0005180": "parkinson_disease",
    "MONDO_0005301": "multiple_sclerosis",
    "MONDO_0004985": "amyotrophic_lateral_sclerosis",
    "MONDO_0005027": "epilepsy",
    "MONDO_0005306": "huntington_disease",
    "MONDO_0005090": "bipolar_disorder",
    "MONDO_0011561": "autism_spectrum_disorder",
    "MONDO_0005148": "type_2_diabetes",
    "MONDO_0007455": "hypercholesterolemia",
    "MONDO_0005002": "coronary_artery_disease",
    "MONDO_0004981": "atrial_fibrillation",
    "MONDO_0005010": "hypertension",
    "MONDO_0005011": "chronic_kidney_disease",
    "MONDO_0005149": "type_1_diabetes",
    "MONDO_0007915": "rheumatoid_arthritis",
    "MONDO_0005155": "systemic_lupus_erythematosus",
    "MONDO_0005260": "crohn_disease",
    "MONDO_0005265": "ulcerative_colitis",
    "MONDO_0007191": "psoriasis",
    "MONDO_0016419": "ankylosing_spondylitis",
    "MONDO_0019434": "systemic_sclerosis",
    "MONDO_0005047": "asthma",
    "MONDO_0004526": "atopic_dermatitis",
    "MONDO_0005311": "sjogren_syndrome",
    "MONDO_0005098": "heart_failure",
    "MONDO_0005003": "myocardial_infarction",
    "MONDO_0021668": "aortic_aneurysm",
    "MONDO_0005016": "stroke",
    "MONDO_0005560": "peripheral_artery_disease",
    "EFO_0001073":   "obesity",
    "EFO_0004064":   "nafld",
    "EFO_0002090":   "gout",
}

QUERY = """
query DiseaseTargets($diseaseId: String!, $size: Int!) {
  disease(efoId: $diseaseId) {
    name
    associatedTargets(page: { index: 0, size: $size }) {
      rows {
        target {
          approvedSymbol
        }
        score
      }
    }
  }
}
"""


def fetch_disease(disease_id: str, size: int = 10,
                  retries: int = 4, backoff: float = 2.0):
    """
    Query OpenTargets for top `size` associated genes for a disease.
    Returns list of {disease, gene, score} dicts or empty list on failure.
    """
    variables = {"diseaseId": disease_id, "size": size}

    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(
                API_URL,
                json={"query": QUERY, "variables": variables},
                headers={"Content-Type": "application/json"},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()

            if "errors" in data:
                raise ValueError(f"GraphQL errors: {data['errors']}")

            disease_block = data.get("data", {}).get("disease")
            if not disease_block:
                return []   # disease ID not found — skip cleanly

            rows = disease_block["associatedTargets"]["rows"]
            disease_label = DISEASE_NAMES.get(disease_id, disease_id)

            return [
                {
                    "disease": disease_label,
                    "gene":    row["target"]["approvedSymbol"],
                    "score":   round(row["score"], 4),
                }
                for row in rows
                if row["target"]["approvedSymbol"]
            ]

        except (requests.RequestException, ValueError, KeyError) as e:
            wait = backoff ** attempt
            print(f"    [attempt {attempt}/{retries}] {disease_id} failed: {e} "
                  f"— retrying in {wait:.0f}s")
            time.sleep(wait)

    print(f"    [SKIP] {disease_id} — all retries exhausted")
    return []


def run(output_path: str = OUTPUT_PATH, top_n: int = 10) -> pd.DataFrame:

    # Deduplicate disease list (preserve order)
    seen     = set()
    diseases = []
    for d in DISEASES:
        eid = EFO_OVERRIDES.get(d, d)
        if eid not in seen:
            seen.add(eid)
            diseases.append(eid)

    print(f"REWIRE — Fetching disease-gene associations")
    print(f"  Diseases  : {len(diseases)}")
    print(f"  Genes/dis : {top_n}")
    print(f"  API       : {API_URL}")
    print()

    all_records = []
    failed      = []

    for i, disease_id in enumerate(diseases, 1):
        label = DISEASE_NAMES.get(disease_id, disease_id)
        print(f"[{i:>2}/{len(diseases)}] {label:<35}", end="", flush=True)

        records = fetch_disease(disease_id, size=top_n)

        if records:
            all_records.extend(records)
            print(f"  ✅  {len(records)} genes  "
                  f"(top score: {records[0]['score']:.3f})")
        else:
            failed.append(disease_id)
            print("  ❌  skipped")

        time.sleep(0.3)   # polite rate-limit

    df = pd.DataFrame(all_records)

    if df.empty:
        print("\n[ERROR] No data retrieved. Check network / API availability.")
        return df

    df = df.drop_duplicates(subset=["disease", "gene"])
    df = df.sort_values(["disease", "score"], ascending=[True, False])
    df = df.reset_index(drop=True)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print()
    print("=" * 55)
    print("  disease_genes_expanded.csv — complete")
    print("=" * 55)
    print(f"  Total rows       : {len(df):,}")
    print(f"  Unique diseases  : {df['disease'].nunique()}")
    print(f"  Unique genes     : {df['gene'].nunique()}")
    print(f"  Failed diseases  : {len(failed)}")
    if failed:
        for f in failed:
            print(f"    – {DISEASE_NAMES.get(f, f)}")
    print(f"  Saved → {output_path}")

    return df


if __name__ == "__main__":
    run()
