import xml.etree.ElementTree as ET
import pandas as pd

file_path = r"data/raw/full database.xml"

records = []
count = 0

print("Extracting DrugBank targets with gene symbols...\n")

for event, elem in ET.iterparse(file_path, events=("end",)):

    if elem.tag.endswith("drug"):
        count += 1

        if count % 50 == 0:
            print("Processed drugs:", count)

        drug_id = ""
        drug_name = ""

        for child in elem:

            if child.tag.endswith("drugbank-id") and not drug_id:
                drug_id = child.text

            elif child.tag.endswith("name"):
                drug_name = child.text

            elif child.tag.endswith("targets"):

                for target in child:

                    target_name = ""
                    gene_symbol = ""
                    organism = ""

                    for sub in target:

                        if sub.tag.endswith("name"):
                            target_name = sub.text

                        elif sub.tag.endswith("organism"):
                            organism = sub.text

                        elif sub.tag.endswith("polypeptide"):

                            for poly in sub:
                                if poly.tag.endswith("gene-name"):
                                    gene_symbol = poly.text

                    records.append({
                        "drug_id": drug_id,
                        "drug_name": drug_name,
                        "target_name": target_name,
                        "gene_symbol": gene_symbol,
                        "organism": organism
                    })

        elem.clear()

        if count >= 1000:
            break

df = pd.DataFrame(records)

df.to_csv("data/processed/drug_targets_symbols.csv", index=False)

print("\nSaved drug_targets_symbols.csv")
print("Rows:", len(df))
print(df.head())