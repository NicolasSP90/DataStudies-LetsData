#%%
# Importing Libraries
import pandas as pd
import numpy as np
import fitz
# %%
file_names = ["pdf.pdf", "foo.pdf", "d2l-en-pytorch.pdf"]
#%%
# Open pdf
with fitz.open(f"./data/{file_names[2]}") as doc:
    page = doc.load_page(10)
    print(page.get_text())