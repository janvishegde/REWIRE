import requests
import time
import pandas as pd

url = "https://api.platform.opentargets.org/api/v4/graphql"

diseases = {
    "Parkinson disease": "MONDO_0005180",
    "Alzheimer disease": "MONDO_0004975",
    "Breast cancer": "MONDO_0007254",
    "Type 2 diabetes mellitus": "MONDO_0005148",
    "Asthma": "MONDO_0004979"
}

query = """
query DiseaseTargets($id: String!) {
  disease(efoId: $id) {
    name
    associatedTargets(page: {index: 0, size: 10}) {
      rows {
        target {
          approvedSymbol
        }
      }
    }
  }
}
"""

results = []

for disease_name, disease_id in diseases.items():
    print(f"\nFetching: {disease_name}")

    success = False

    for attempt in range(3):
        try:
            response = requests.post(
                url,
                json={
                    "query": query,
                    "variables": {"id": disease_id}
                },
                timeout=45
            )

            data = response.json()
            rows = data["data"]["disease"]["associatedTargets"]["rows"]

            for row in rows:
                gene = row["target"]["approvedSymbol"]
                print("-", gene)

                results.append({
                    "disease": disease_name,
                    "gene": gene
                })

            success = True
            break

        except Exception as e:
            print("Retrying...", attempt + 1)
            time.sleep(3)

    if not success:
        print("Skipped:", disease_name)

df = pd.DataFrame(results)
df.to_csv("data/processed/disease_genes.csv", index=False)

print("\nSaved disease_genes.csv")
print("Rows:", len(df))