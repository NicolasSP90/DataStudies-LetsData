#%%
#%%
# Libraries
import pandas as pd
import PyPDF2
from PyPDF2 import PdfReader
import re
# %%
def extracttext(firstpage, lastpage, filename):
    pdftext = ""
    
    #Open file
    pdf_file = open(f"./data/{filename}", "rb")

    #Instance the Reader
    pdf_reader = PdfReader(pdf_file)

    # Starting page is 0, not 1
    firstpage = firstpage - 1
    lastpage = lastpage -1
    currentpage = firstpage

    # Add text of the page to the string
    while currentpage <= lastpage:
        pdftext = pdftext + pdf_reader.pages[currentpage].extract_text()
        currentpage += 1

    return pdftext
#%%
#Pages 4 to 73
relevant_text = extracttext(4, 73, "edital_prograd_9.pdf")
relevant_text
#%%
relevant_text = re.sub("Serviço Público Federal", "", relevant_text)
relevant_text = re.sub("Ministério da Educação", "", relevant_text)
relevant_text = re.sub("Fundação Universidade Federal de Mato Grosso do Sul", "", relevant_text)
relevant_text = re.sub("ANEXO I – RESULTADO PRELIMINAR DA PROVA OBJETIVA E REDAÇÃO", "", relevant_text)
relevant_text = re.sub("(Edital nº 9/2024-Prograd/UFMS)", "", relevant_text)
relevant_text = re.sub("INSCRIÇÃO TIPO DE VAGA VL VH VN VM RED", "", relevant_text)
relevant_text = re.sub(r"\n\n\n\n()\n\n", "936821", relevant_text)
relevant_text
#%%
relevant_text.encode("utf-8", "ignore")
# %%
relevant_text = relevant_text.split("\n")
relevant_text
# %%
for i in relevant_text:
    if i == r"()":
        relevant_text.remove(i)
    elif "pg." in i:
        relevant_text.remove(i)
    else:
        pass

for i in relevant_text:
    if i == r"()":
        relevant_text.remove(i)
    elif "pg." in i:
        relevant_text.remove(i)
    else:
        pass
relevant_text
# %%
df_grades = pd.DataFrame(data=relevant_text)
df_grades.columns = ["full_text"]
# "id", "type", "VL", "VH", "VN", "VM", "RED"
# %%
df_grades.head()
#%%
df_grades.tail()
# %%
df_grades.shape
# %%
df_grades[df_grades["full_text"] == ""].value_counts()
#%%
condition = (df_grades["full_text"] == "")
df_grades = df_grades[~condition]
df_grades.shape
# %%
df_grades["id"] = df_grades["full_text"].str.extract(r"(\d+)")
df_grades.head()
# %%
df_grades["type"] = df_grades["full_text"].str.extract(r"[\d]+\s([\w]+)\s")
df_grades.head()
# %%
df_grades["VL"] = df_grades["full_text"].str.extract(r"[\d]+\s[\w]+\s([\d\,]+)\s")
df_grades.head()
# %%
df_grades["VH"] = df_grades["full_text"].str.extract(r"[\d]+\s[\w]+\s[\d\,]+\s([\d\,]+)\s")
df_grades.head()
# %%
df_grades["VN"] = df_grades["full_text"].str.extract(r"[\d]+\s[\w]+\s[\d\,]+\s[\d\,]+\s([\d\,]+)\s")
df_grades.head()
# %%
df_grades["VM"] = df_grades["full_text"].str.extract(r"[\d]+\s[\w]+\s[\d\,]+\s[\d\,]+\s[\d\,]+\s([\d\,]+)\s")
df_grades.head()
# %%
df_grades["RED"] = df_grades["full_text"].str.extract(r"[\d]+\s[\w]+\s[\d\,]+\s[\d\,]+\s[\d\,]+\s[\d\,]+\s(.*)")
df_grades.head()
# %%
df_grades = df_grades.drop(["full_text"], axis=1)
df_grades.head()
# %%
df_grades["VL"] = df_grades["VL"].str.replace("," , ".")
df_grades["VL"] = df_grades["VL"].astype(float)
df_grades["VH"] = df_grades["VH"].str.replace("," , ".")
df_grades["VH"] = df_grades["VH"].astype(float)
df_grades["VN"] = df_grades["VN"].str.replace("," , ".")
df_grades["VN"] = df_grades["VN"].astype(float)
df_grades["VM"] = df_grades["VM"].str.replace("," , ".")
df_grades["VM"] = df_grades["VM"].astype(float)
df_grades["RED"] = df_grades["RED"].str.replace("," , ".")
df_grades["RED"] = df_grades["RED"].astype(float)
df_grades.info()
#%%
df_grades.head()
# %%
df_grades.to_csv("./data/df_grades.csv")
# %%
