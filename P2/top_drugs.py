import pandas as pd

print(pd.read_csv('drug_targets_symbols.csv')['drug_name'].value_counts().nlargest(10))
