# %%
# Extracting data from pdf
# %%
# Libraries
import re
import PyPDF2
from pathlib import Path
import pandas as pd
# %%
# Import PDF
pdf_path = Path("./data/pdf.pdf")
pdf_file = open(pdf_path, "rb")
pdf_read = PyPDF2.PdfReader(pdf_file)
# %%
# Get Page Content
pages = len(pdf_read.pages)
pdf_text = ""
for i in range(pages):
    page = pdf_read.pages[i]
    page_content = page.extract_text()
    # page_text = str(page_content.encode("latin1", errors="ignore"))
    page_text = str(page_content.encode("utf-8"))

    pdf_text = pdf_text + page_text

pdf_text
# %%
# Storing only the needed text
pdf_text = re.sub(r".*10089355", "10089355", pdf_text)
pdf_text
#%%
pdf_text = re.sub(r"\\n", "", pdf_text)
pdf_text
#%%
pdf_text = re.sub(r"/", "\n", pdf_text)
pdf_text
#%%
pdf_text = re.sub(pattern=r"\s+(\n)\s+(\d{8})", repl=r"\1\2", string=pdf_text)
pdf_text
# %%
pdf_text = re.sub(r"'\w'\d", "", pdf_text)
pdf_text
# %%
text_ajusted = pdf_text.split("\n")
text_ajusted
# %%
# Dataframe structure:
# Inscription: 8 numeric digits
# Name: The only letters that must be separated
# Grade 1: 3 or 4 numeric digits in the format 0.00 or 00.00
# Grade 2: 3 or 4 numeric digits in the format 0.00 or 00.00
# Grade 3: 4 or 5 numeric digits in the format 00.00 or 000.00
df0 = pd.DataFrame(data={"text":text_ajusted})
df0
#%%
df0 = df0.drop(index=285)
df0
# %%
df0["text"].str.extract(r"(\d{8})")
# %%
df0["text"].str.extract(r"(\d{8})").isna().sum()
# %%
df0["text"].str.extract(r'([\d\s]+)')
# %%
df0["text"].str.extract(r'([\d\s]+)').isna().sum()
# %%
df0["text"].str.replace("\s+", "", regex=True).str.extract(r"(\d+)")
#%%
df0["text"].str.replace("\s+", "", regex=True).str.extract(r"(\d+)").isna().sum()
# %%
df0["text"].str.extract(r'([\d\s]+)') == df0["text"].str.replace("\s+", "", regex=True).str.extract(r"(\d+)")
# %%
(df0["text"].str.extract(r'([\d\s]+)') == df0["text"].str.replace("\s+", "", regex=True).str.extract(r"(\d+)")).value_counts()
# %%
df0["Inscription"] = df0["text"].str.replace("\s+", "", regex=True).str.extract(r"(\d+)")
df0
# %%
df0["Name"] = df0["text"].str.replace("\s+", " ", regex=True).str.extract(r"[\s\d]+,\s*([\w\s]+)")
df0
# %%
df0["Name"].isna().sum()
# %%
df0["Grade1"] = df0["text"].str.replace("\s*", "", regex=True).str.extract(r"[\d]+,[\w\.]+,([\d\.]+)")
df0
#%%
df0["Grade1"].isna().sum()
# %%
df0["Grade2"] = df0["text"].str.replace("\s*", "", regex=True).str.extract(r"[\d]+,[\w]+,[\d\.]+,([\d\.]+),")
df0
#%%
df0["Grade2"].isna().sum()
# %%
df0["Grade3"] = df0["text"].str.replace("\s*", "", regex=True).str.extract(r"[\d]+,[\w]+,[\d\.]+,[\d\.]+,([\d\.]+)")
df0
# %%
df0["Grade3"].isna().sum()
# %%
df0 = df0.drop(columns=["text"], axis=1)
df0
# %%
