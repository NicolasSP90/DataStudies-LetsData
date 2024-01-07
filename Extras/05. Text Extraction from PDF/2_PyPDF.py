#%%
# Importing Libraries
import pandas as pd
import numpy as np
import PyPDF2
from PyPDF2 import PdfReader
# %%
file_names = ["pdf.pdf", "foo.pdf", "d2l-en-pytorch.pdf"]
#%%
# Open pdf
pdf_file = open(f"./data/{file_names[1]}", "rb")
pdf_file
# %%
# PyPDF reader
pdf_reader = PdfReader(pdf_file)
pdf_reader
# %%
# Checking Cryptography
pdf_reader.is_encrypted
# %%
# Checking MetaData
pdf_metadata = pdf_reader.metadata
print(pdf_metadata.author)
print(pdf_metadata.creator)
print(pdf_metadata.producer)
print(pdf_metadata.title)
# %%
# Checking Number of Pages
pdf_pages = len(pdf_reader.pages)
pdf_pages
# %%
# Checking Data in a Specific Page (MetaData)
pdf_reader.pages[2]
# %%
# Checking Data in a Specific Page (Content)
pdf_reader.pages[2].extract_text()
# %%
