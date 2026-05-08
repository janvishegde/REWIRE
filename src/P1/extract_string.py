# download_string.py
import gzip
import shutil

input_file = r"C:\Users\janvi\Downloads\9606.protein.links.v12.0.txt.gz"
output_file = r"C:\Users\janvi\OneDrive\Documents\IV SEM AIML\REWIRE\data\raw\string_links.txt"

with gzip.open(input_file, "rb") as f_in:
    with open(output_file, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

print("Extraction complete!")